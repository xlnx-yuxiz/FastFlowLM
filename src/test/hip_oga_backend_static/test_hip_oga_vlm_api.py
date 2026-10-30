#!/usr/bin/env python3
"""Bounded real-HTTP qualification for image chat, streaming and recovery.

The server is owned by the caller. This script only sends requests and retains
all request/response evidence in result.json.
"""
import argparse
import base64
import binascii
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import struct
import threading
import time
import traceback
import zlib
from urllib.parse import urlsplit


HTTP_TIMEOUT_SECONDS = 120
CONTROL_TIMEOUT_SECONDS = 5
FIRST_CONTENT_TIMEOUT_SECONDS = 60
CANCEL_WAIT_SECONDS = 20
CONTEXT_LENGTH = 2048
COLOR_QUESTION = "What color is the LEFT half? Answer with one color word."
TEXT_BASELINE_PROMPT = "Say only the word READY."


def _png_chunk(kind, payload):
    kind_bytes = kind.encode("ascii")
    body = kind_bytes + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xffffffff)


def make_split_png(left_rgb, right_rgb, width=224, height=224):
    """Build a deterministic RGB PNG with equal-sized solid-color halves."""
    if width % 2:
        raise ValueError("PNG width must be even")
    row = bytes(left_rgb) * (width // 2) + bytes(right_rgb) * (width // 2)
    pixels = b"".join(b"\x00" + row for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk("IHDR", ihdr) +
            _png_chunk("IDAT", zlib.compress(pixels, level=9)) + _png_chunk("IEND", b""))


def save_assets(output):
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "left-red-right-blue": make_split_png((255, 0, 0), (0, 0, 255)),
        "left-blue-right-red": make_split_png((0, 0, 255), (255, 0, 0)),
        # Valid base64 and a PNG signature, but intentionally truncated/corrupt.
        "corrupt-png": b"\x89PNG\r\n\x1a\nnot-a-valid-png-payload",
    }
    names = {
        "left-red-right-blue": "left-red-right-blue.png",
        "left-blue-right-red": "left-blue-right-red.png",
        "corrupt-png": "corrupt.png",
    }
    metadata = {}
    for key, payload in payloads.items():
        path = image_dir / names[key]
        path.write_bytes(payload)
        metadata[key] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return payloads, metadata


def extract_words(content):
    return [word.lower() for word in re.findall(r"(?<![A-Za-z])([A-Za-z]+)(?![A-Za-z])", content)]


def assert_color_answer(record, expected, region='left'):
    content = record.get("content", "")
    opposite = "blue" if expected == "red" else "red"
    words = extract_words(content)
    assert expected in words, "expected whole color word %r missing from %r" % (expected, content)
    assert opposite not in words, "opposite whole color word %r present in %r" % (opposite, content)
    record["visual_check"] = {
        "expected_color": expected,
        "region": region,
        "observed_color_words": [word for word in words if word in ("red", "blue")],
        "passed": True,
    }


class VlmProbe:
    def __init__(self, args):
        self.args = args
        self.url = urlsplit(args.base_url)
        if self.url.scheme not in ("http", "https") or not self.url.hostname:
            raise ValueError("--base-url must be an absolute http:// or https:// URL")
        self.records = []
        self.cases = []
        self._lock = threading.Lock()
        self._asset_data = {}
        self._image_hashes = {}
        self._baseline_red = None
        self._baseline_blue = None
        self._text_baseline = None
        self._last_history_request = None
        self._last_history_record = None
        self._known_defects = []

    def _endpoint(self, path):
        base_path = self.url.path.rstrip("/")
        return (base_path + path) if base_path else path

    def body(self, messages, stream=False, max_tokens=8, request_id=None):
        body = {
            "model": self.args.model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "options": {"num_ctx": CONTEXT_LENGTH, "num_predict": max_tokens},
            "messages": messages,
        }
        if request_id is not None:
            body["request_id"] = request_id
        return body

    def text_messages(self):
        return [{"role": "user", "content": TEXT_BASELINE_PROMPT}]

    def image_url(self, image_key):
        encoded = base64.b64encode(self._asset_data[image_key]).decode("ascii")
        return "data:image/png;base64," + encoded

    def image_message(self, image_key, prompt=COLOR_QUESTION):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self.image_url(image_key)}},
            ],
        }

    def call(self, name, path, body, timeout=HTTP_TIMEOUT_SECONDS, stream=False, record=None):
        if record is None:
            record = {
                "name": name,
                "method": "POST",
                "path": path,
                "request": body,
                "timeout_seconds": timeout,
            }
            with self._lock:
                self.records.append(record)
        connection = None
        started = time.monotonic()
        raw_bytes = bytearray()
        try:
            if self.url.scheme == "https":
                connection = http.client.HTTPSConnection(self.url.hostname, self.url.port, timeout=timeout)
            else:
                connection = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=timeout)
            record["_connection"] = connection
            encoded_body = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            connection.request("POST", self._endpoint(path), encoded_body,
                               {"Content-Type": "application/json", "Accept": "text/event-stream, application/json",
                                "Connection": "close"})
            response = connection.getresponse()
            record["status"] = response.status
            record["headers"] = {key: value for key, value in response.getheaders()}
            if stream and response.status == 200:
                events = []
                saw_done = False
                terminal_seen = False
                content_parts = []
                reasoning_parts = []
                response_ids = []
                for raw_line in iter(response.readline, b""):
                    raw_bytes.extend(raw_line)
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        continue
                    assert not saw_done, "SSE data followed [DONE]"
                    assert line.startswith("data: "), "unexpected SSE line: %r" % line
                    value = line[6:]
                    if value == "[DONE]":
                        saw_done = True
                        continue
                    assert not terminal_seen, "SSE data followed the terminal finish/usage event before [DONE]"
                    event = json.loads(value)
                    events.append(event)
                    response_id = event.get("id")
                    assert isinstance(response_id, str) and response_id.strip(), event
                    response_ids.append(response_id)
                    assert event.get("model") == self.args.model, event
                    assert event.get("object") == "chat.completion.chunk", event
                    assert len(event.get("choices", [])) == 1, event
                    choice = event["choices"][0]
                    assert choice.get("index") == 0, event
                    delta = choice.get("delta", {})
                    assert isinstance(delta, dict), event
                    assert not delta.get("tool_calls"), event
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    assert isinstance(content, str) and isinstance(reasoning, str), event
                    if content:
                        content_parts.append(content)
                        first_content_event = record.get("_first_content_event")
                        if first_content_event is not None and content.strip():
                            first_content_event.set()
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if choice.get("finish_reason") is not None or event.get("usage") is not None:
                        terminal_seen = True
                raw_text = bytes(raw_bytes).decode("utf-8")
                record["raw"] = raw_text
                assert saw_done, "SSE response did not contain [DONE]"
                assert response_ids and len(set(response_ids)) == 1, response_ids
                finishes = [event["choices"][0]["finish_reason"] for event in events
                            if event["choices"][0].get("finish_reason") is not None]
                usages = [event["usage"] for event in events if event.get("usage") is not None]
                assert len(finishes) == 1, "expected one terminal finish_reason, got %r" % finishes
                assert len(usages) == 1, "expected one terminal usage event, got %r" % usages
                assert events[-1]["choices"][0].get("finish_reason") is not None, events[-1]
                assert events[-1].get("usage") is not None, events[-1]
                content = "".join(content_parts)
                reasoning = "".join(reasoning_parts)
                assert content, "stream contained no assistant content"
                record.update(content=content, reasoning=reasoning, finish=finishes[0], usage=usages[0],
                              event_count=len(events), done_count=1,
                              sse_validation={
                                  "response_id": response_ids[0],
                                  "model": self.args.model,
                                  "choice_index": 0,
                                  "tool_calls": False,
                                  "terminal_finish_count": len(finishes),
                                  "terminal_usage_count": len(usages),
                                  "terminal_event_is_last": True,
                                  "data_after_terminal_before_done": False,
                                  "data_after_done": False,
                              })
            else:
                raw_bytes.extend(response.read())
                raw_text = bytes(raw_bytes).decode("utf-8")
                record["raw"] = raw_text
                try:
                    data = json.loads(raw_text) if raw_text else None
                except json.JSONDecodeError:
                    data = None
                record["data"] = data
                if isinstance(data, dict) and data.get("object") == "chat.completion":
                    choices = data.get("choices", [])
                    assert len(choices) == 1, data
                    message = choices[0].get("message", {})
                    content = message.get("content") or ""
                    assert isinstance(content, str), data
                    record.update(content=content,
                                  reasoning=message.get("reasoning_content") or "",
                                  finish=choices[0].get("finish_reason"),
                                  usage=data.get("usage"))
            return record
        except Exception as exc:
            if raw_bytes and "raw" not in record:
                record["raw"] = bytes(raw_bytes).decode("utf-8", errors="replace")
            record["exception"] = repr(exc)
            record["traceback"] = traceback.format_exc()
            raise
        finally:
            if connection is not None:
                connection.close()
            record.pop("_connection", None)
            record["seconds"] = round(time.monotonic() - started, 3)
            if raw_bytes and "raw" not in record:
                record["raw"] = bytes(raw_bytes).decode("utf-8", errors="replace")

    def _chat_ok(self, record, expected_status=200, allow_cancel=False, expected_cache='miss'):
        assert record.get("status") == expected_status, record
        assert "exception" not in record, record
        if expected_status == 200:
            assert record.get("content"), record
            allowed_finishes = ("stop", "length", "cancel") if allow_cancel else ("stop", "length")
            assert record.get("finish") in allowed_finishes, record
            usage = record.get("usage")
            assert isinstance(usage, dict), record
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                assert isinstance(usage.get(field), int) and usage[field] >= 0, record
            assert usage["completion_tokens"] > 0, record
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], record
            assert 0 < usage["prompt_tokens"] < CONTEXT_LENGTH, record
            details = usage.get("prompt_tokens_details")
            assert isinstance(details, dict), record
            cached = details.get("cached_tokens")
            assert type(cached) is int, "cached_tokens must be an integer (excluding bool): %r" % record
            assert expected_cache in ("miss", "hit"), expected_cache
            effective_expectation = "miss" if self.args.cache == "off" else expected_cache
            record["expected_cache"] = effective_expectation
            if effective_expectation == "hit":
                assert 0 < cached < usage["prompt_tokens"], (
                    "expected cache hit with 0 < cached_tokens < prompt_tokens: %r" % record)
            else:
                assert cached == 0, "expected cache miss with cached_tokens=0: %r" % record
            assert usage["completion_tokens"] <= record["request"]["max_tokens"], record
        return record

    def _image_call(self, name, image_key, prompt=COLOR_QUESTION, stream=False, max_tokens=8,
                    request_id=None, expected_cache='miss'):
        body = self.body([self.image_message(image_key, prompt)], stream=stream,
                         max_tokens=max_tokens, request_id=request_id)
        record = self.call(name, "/v1/chat/completions", body, stream=stream)
        record["has_image"] = True
        record["image"] = {"key": image_key, "sha256": self._image_hashes[image_key]}
        return self._chat_ok(record, expected_cache=expected_cache)

    def _add_case(self, name, fn):
        case = {"name": name, "status": "RUNNING"}
        self.cases.append(case)
        try:
            fn(case)
            if case["status"] == "RUNNING":
                case["status"] = "PASS"
        except Exception as exc:
            case.update(status="FAIL", failure=repr(exc), traceback=traceback.format_exc())
            raise

    def _same_content_usage(self, actual, expected, compare_reasoning=False):
        fields = ("content", "finish", "reasoning") if compare_reasoning else ("content", "finish")
        for field in fields:
            assert actual[field] == expected[field], (field, actual, expected)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert actual["usage"][field] == expected["usage"][field], (field, actual, expected)

    def _record_text_baseline(self, case):
        body = self.body(self.text_messages(), max_tokens=8)
        record = self.call("text-baseline", "/v1/chat/completions", body)
        self._chat_ok(record)
        assert "ready" in extract_words(record["content"]), record
        case["request"] = "text-baseline"
        self._text_baseline = record

    def _record_image(self, case, name, image_key, expected):
        record = self._image_call(name, image_key)
        assert_color_answer(record, expected)
        case["request"] = name
        case["observed_visual_fact"] = record["visual_check"]
        if expected == "red" and self._baseline_red is None:
            self._baseline_red = record
        if expected == "blue" and self._baseline_blue is None:
            self._baseline_blue = record

    def _record_red_repeat(self, case):
        record = self._image_call("image-red-repeat", "left-red-right-blue")
        assert_color_answer(record, "red")
        self._same_content_usage(record, self._baseline_red)
        case["request"] = "image-red-repeat"
        case["compared_with"] = "image-red"
        case["observed_visual_fact"] = record["visual_check"]

    def _record_history(self, case, image_key, baseline, expected, expected_cache='miss'):
        baseline_content = baseline["content"]
        messages = [
            self.image_message(image_key),
            {"role": "assistant", "content": baseline_content},
            {"role": "user", "content": "What color is the RIGHT half of the original image? Answer with one color word."},
        ]
        body = self.body(messages)
        record = self.call(case['name'], "/v1/chat/completions", body)
        record["has_image"] = True
        record["image"] = {"key": image_key, "sha256": self._image_hashes[image_key]}
        self._chat_ok(record, expected_cache=expected_cache)
        assert_color_answer(record, expected, region='right')
        case["request"] = case['name']
        case["history_contains_original_image"] = True
        case["observed_visual_fact"] = record["visual_check"]
        self._last_history_request = body
        self._last_history_record = record

    def _record_history_repeat(self, case):
        assert self._last_history_request is not None, "history repeat has no saved request"
        assert self._last_history_record is not None, "history repeat has no preceding result"
        record = self.call("image-red-history-repeat", "/v1/chat/completions", self._last_history_request)
        record["has_image"] = True
        record["image"] = {"key": "left-red-right-blue",
                            "sha256": self._image_hashes["left-red-right-blue"]}
        self._chat_ok(record, expected_cache='miss')
        assert_color_answer(record, "blue", region='right')
        self._same_content_usage(record, self._last_history_record)
        case["request"] = "image-red-history-repeat"
        case["replayed_request"] = "image-red-history-followup"
        case["history_contains_original_image"] = True
        case["observed_visual_fact"] = record["visual_check"]

    def _record_stream_parity(self, case):
        record = self._image_call("image-red-sse-parity", "left-red-right-blue", stream=True)
        assert_color_answer(record, "red")
        self._same_content_usage(record, self._baseline_red, compare_reasoning=True)
        case["request"] = "image-red-sse-parity"
        case["compared_with"] = "image-red"
        case["sse"] = {"terminal_finish_count": 1, "terminal_usage_count": 1, "done_count": 1,
                       "post_done_data": False}
        case["observed_visual_fact"] = record["visual_check"]

    def _cancel_request(self, name, request_id):
        body = {"request_id": request_id}
        record = self.call(name, "/api/cancel", body, timeout=CONTROL_TIMEOUT_SECONDS)
        assert record.get("status") == 200 and "exception" not in record, record
        data = record.get("data")
        if data is None:
            data = json.loads(record.get("raw", ""))
            record["data"] = data
        assert isinstance(data, dict) and isinstance(data.get("cancelled"), bool), record
        record["cancelled"] = data["cancelled"]
        return record

    def _cancel_after_first_content(self, request_id, done_event, error_box):
        deadline = time.monotonic() + CANCEL_WAIT_SECONDS
        number = 0
        while time.monotonic() < deadline:
            if done_event.is_set():
                if error_box:
                    raise error_box[0]
                raise AssertionError("image stream ended before /api/cancel accepted request_id")
            number += 1
            record = self._cancel_request("cancel-poll-%03d" % number, request_id)
            if record["cancelled"]:
                return record, number
            time.sleep(0.05)
        raise AssertionError("/api/cancel did not accept the active request within %d seconds" % CANCEL_WAIT_SECONDS)

    def _record_cancel_and_recovery(self, case):
        request_id = "t05-vlm-cancel-%d-%d" % (os.getpid(), int(time.time() * 1000))
        long_prompt = ("What color is the LEFT half? Output only that color repeated with spaces "
                       "until generation is stopped.")
        body = self.body([self.image_message("left-red-right-blue", long_prompt)], stream=True,
                         max_tokens=256, request_id=request_id)
        stream_record = {
            "_first_content_event": threading.Event(),
        }
        # call() normally creates its record. Attach the event to that request
        # record before the worker enters call() so content cannot be missed.
        ready = threading.Event()
        result_box = []
        error_box = []
        done_event = threading.Event()

        def worker_call():
            try:
                ready.set()
                item = self.call("image-red-cancel-stream", "/v1/chat/completions", body,
                                 stream=True, record=stream_record)
                item["has_image"] = True
                item["image"] = {"key": "left-red-right-blue",
                                 "sha256": self._image_hashes["left-red-right-blue"]}
                result_box.append(item)
            except BaseException as exc:
                error_box.append(exc)
            finally:
                done_event.set()

        # Register the record synchronously, so the worker can signal its first
        # content chunk without racing record creation.
        stream_record.update({"name": "image-red-cancel-stream", "method": "POST",
                              "path": "/v1/chat/completions", "request": body,
                              "timeout_seconds": HTTP_TIMEOUT_SECONDS})
        with self._lock:
            self.records.append(stream_record)

        thread = threading.Thread(target=worker_call, name="t05-vlm-image-cancel", daemon=True)
        accepted = False
        try:
            thread.start()
            assert ready.wait(5), "stream worker did not start"
            first_deadline = time.monotonic() + FIRST_CONTENT_TIMEOUT_SECONDS
            while not stream_record["_first_content_event"].wait(0.05):
                if done_event.is_set():
                    if error_box:
                        raise error_box[0]
                    raise AssertionError("image stream ended without nonempty assistant content")
                if time.monotonic() >= first_deadline:
                    raise AssertionError("no nonempty assistant content within %d seconds" % FIRST_CONTENT_TIMEOUT_SECONDS)
            assert not done_event.is_set(), "stream completed before cancellation could be requested"
            cancel_record, poll_count = self._cancel_after_first_content(
                request_id, done_event, error_box)
            accepted = cancel_record["cancelled"]
            assert accepted is True, cancel_record
            thread.join(HTTP_TIMEOUT_SECONDS + 5)
            assert not thread.is_alive(), "cancelled streaming request thread did not exit"
            if error_box:
                raise error_box[0]
            assert result_box, stream_record
            completed_stream = result_box[0]
            self._chat_ok(completed_stream, allow_cancel=True)
            assert completed_stream["finish"] == "cancel", completed_stream
            case.update(request="image-red-cancel-stream", accepted_cancel=True,
                        cancel_poll_count=poll_count,
                        cancellation_records=[record["name"] for record in self.records
                                              if record.get("path") == "/api/cancel"],
                        observed_content_before_cancel=True)
        finally:
            # Every path closes the worker's HTTPConnection in call(); this
            # cleanup asks the server to release it before joining the thread.
            if thread.is_alive():
                cleanup_deadline = time.monotonic() + CANCEL_WAIT_SECONDS
                cleanup_index = 0
                while thread.is_alive() and time.monotonic() < cleanup_deadline:
                    cleanup_index += 1
                    try:
                        cleanup = self._cancel_request("cancel-cleanup-poll-%03d" % cleanup_index, request_id)
                        accepted = accepted or cleanup["cancelled"]
                        if cleanup["cancelled"]:
                            break
                    except Exception:
                        pass
                    time.sleep(0.05)
                thread.join(HTTP_TIMEOUT_SECONDS + 5)
                if thread.is_alive():
                    connection = stream_record.get("_connection")
                    if connection is not None:
                        connection.close()
                    thread.join(CONTROL_TIMEOUT_SECONDS + 1)
                assert not thread.is_alive(), "stream worker survived cancellation cleanup"

    def _record_corrupt_rejection(self, case):
        body = self.body([self.image_message("corrupt-png")])
        record = self.call("invalid-image-corrupt-png", "/v1/chat/completions", body)
        record["has_image"] = True
        record["image"] = {"key": "corrupt-png", "sha256": self._image_hashes["corrupt-png"]}
        expected_status = self.args.invalid_image_http_status
        case["request"] = "invalid-image-corrupt-png"
        case["expected_http_status"] = 400
        case["characterization_expected_http_status"] = expected_status
        if expected_status == 400:
            if record.get("status") == 400:
                data = record.get("data")
                assert isinstance(data, dict) and isinstance(data.get("error"), dict), record
                assert data["error"].get("type") == "invalid_request_error", record
                assert data["error"].get("code") == 400, record
                case["structured_error"] = "invalid_request_error/400"
                return
            if record.get("status") == 200 and record.get("data"):
                data = record["data"]
                assert isinstance(data, dict) and data.get("error"), record
                defect = {
                    "name": "corrupt-image-http-status",
                    "expected": "HTTP 400 structured invalid_request_error",
                    "observed_status": 200,
                    "observed_error": data.get("error"),
                    "request": record["name"],
                }
                case.update(status="KNOWN_DEFECT", knownDefect=defect)
                self._known_defects.append(defect)
                return
            raise AssertionError("expected HTTP 400 structured invalid_request_error: %r" % record)
        assert expected_status == 200
        assert record.get("status") == 200 and "exception" not in record, record
        data = record.get("data")
        assert isinstance(data, dict) and data.get("error"), record
        defect = {
            "name": "corrupt-image-http-status",
            "expected": "HTTP 400 structured invalid_request_error",
            "observed_status": 200,
            "observed_error": data.get("error"),
            "request": record["name"],
            "characterization_mode": True,
        }
        case.update(status="KNOWN_DEFECT", knownDefect=defect)
        self._known_defects.append(defect)

    def run_smoke(self):
        self._add_case("text-baseline", self._record_text_baseline)
        self._add_case("image-red", lambda case: self._record_image(case, "image-red", "left-red-right-blue", "red"))
        self._add_case("image-blue", lambda case: self._record_image(case, "image-blue", "left-blue-right-red", "blue"))
        self._add_case("image-red-repeat", self._record_red_repeat)

    def run_full(self):
        self.run_smoke()
        self._add_case("image-red-identical-repeat", lambda case: self._record_image(
            case, "image-red-identical-repeat", "left-red-right-blue", "red"))
        self._add_case("image-red-history-followup", lambda case: self._record_history(
            case, "left-red-right-blue", self._baseline_red, "blue", expected_cache='hit'))
        self._add_case("image-red-history-repeat", self._record_history_repeat)
        self._add_case("image-blue-history-followup", lambda case: self._record_history(
            case, "left-blue-right-red", self._baseline_blue, "red", expected_cache='miss'))
        self._add_case("image-red-sse-parity", self._record_stream_parity)
        self._add_case("active-image-stream-cancel", self._record_cancel_and_recovery)
        self._add_case("image-recovery-after-cancel",
                       lambda case: self._record_recovery_image(case, "image-recovery-after-cancel",
                           "left-blue-right-red", "blue", self._baseline_blue))
        self._add_case("invalid-image-corrupt-png", self._record_corrupt_rejection)
        self._add_case("image-recovery-after-corrupt",
                       lambda case: self._record_recovery_image(case, "image-recovery-after-corrupt",
                           "left-red-right-blue", "red", self._baseline_red))
        self._add_case("text-recovery-after-corrupt", self._record_text_recovery)

    def _record_recovery_image(self, case, request_name, image_key, expected, baseline):
        record = self._image_call(request_name, image_key)
        assert_color_answer(record, expected)
        self._same_content_usage(record, baseline)
        case["request"] = request_name
        case["compared_with"] = baseline['name']
        case["observed_visual_fact"] = record["visual_check"]

    def _record_text_recovery(self, case):
        body = self.body(self.text_messages(), max_tokens=8)
        record = self.call("text-recovery-after-corrupt", "/v1/chat/completions", body)
        self._chat_ok(record)
        assert "ready" in extract_words(record["content"]), record
        self._same_content_usage(record, self._text_baseline)
        case["request"] = "text-recovery-after-corrupt"
        case["compared_with"] = "text-baseline"


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    probe = VlmProbe(args)
    result = {
        "status": "FAIL",
        "phase": args.phase,
        "model": args.model,
        "base_url": args.base_url,
        "cache_mode_requested": args.cache,
        "decoder_requested": getattr(args, "codec", "flm"),
        "context_length": CONTEXT_LENGTH,
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "cancel_control_timeout_seconds": CONTROL_TIMEOUT_SECONDS,
        "first_content_timeout_seconds": FIRST_CONTENT_TIMEOUT_SECONDS,
        "cancel_wait_seconds": CANCEL_WAIT_SECONDS,
        "color_request_max_tokens": 8,
        "cancel_request_max_tokens": 256,
        "expected_invalid_image_http_status": 400,
        "invalid_image_http_status_used": args.invalid_image_http_status,
        "knownDefects": False,
        "known_defect_details": [],
        "cases": probe.cases,
        "records": probe.records,
    }
    try:
        assets, hashes = save_assets(output)
        probe._asset_data = assets
        probe._image_hashes = {key: value["sha256"] for key, value in hashes.items()}
        result["images"] = hashes
        result["case_names_expected"] = (
            ["text-baseline", "image-red", "image-blue", "image-red-repeat"]
            if args.phase == "smoke" else
            ["text-baseline", "image-red", "image-blue", "image-red-repeat",
             "image-red-identical-repeat", "image-red-history-followup", "image-red-history-repeat",
             "image-blue-history-followup", "image-red-sse-parity", "active-image-stream-cancel",
             "image-recovery-after-cancel", "invalid-image-corrupt-png", "image-recovery-after-corrupt",
             "text-recovery-after-corrupt"])
        if args.phase == "smoke":
            probe.run_smoke()
        else:
            probe.run_full()
        result["knownDefects"] = bool(probe._known_defects)
        result["known_defect_details"] = probe._known_defects
        result["status"] = "FAIL" if probe._known_defects else "PASS"
        if probe._known_defects:
            result["failure"] = "qualification contract failed because known defect(s) were observed"
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["knownDefects"] = bool(probe._known_defects)
        result["known_defect_details"] = probe._known_defects
        result["case_count"] = len(probe.cases)
        result["case_count_expected"] = 4 if args.phase == "smoke" else 14
        result["request_count"] = len(probe.records)
        result["cases"] = probe.cases
        result["records"] = probe.records
        for record in probe.records:
            record.pop("_first_content_event", None)
            record.pop("_connection", None)
        result_path = output / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result.get(key) for key in
                          ("status", "phase", "case_count", "request_count", "knownDefects", "failure")},
                         ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", choices=("off", "on"), required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--codec", choices=("oga", "flm", "compare", "external"), default="flm")
    parser.add_argument("--invalid-image-http-status", choices=(200, 400), type=int, default=400,
                        help="Use 200 only to characterize the known pre-fix corrupt-image HTTP status defect.")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
