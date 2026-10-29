#!/usr/bin/env python3
"""Qualify multimodal prefix caching across image order and message layout.

The server is owned by the caller. This script uses the shared VLM HTTP probe,
then writes complete request and response evidence to output/result.json.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback


FIRST_LEFT_QUESTION = "What color is the LEFT half of the FIRST image? Answer with one color word."
FIRST_RIGHT_QUESTION = "What color is the RIGHT half of the FIRST image? Answer with one color word."
CASE_NAMES = [
    "single-red-blue-image",
    "red-blue-plus-reversed-image",
    "swapped-two-image-order",
    "swapped-two-image-repeat",
    "first-image-history-extension",
    "images-across-messages",
]


def load_vlm_api(source_root):
    source_path = (source_root / "src/test/hip_oga_backend_static/test_hip_oga_vlm_api.py").resolve()
    if not source_path.is_file():
        raise FileNotFoundError("VLM API probe not found: %s" % source_path)
    spec = importlib.util.spec_from_file_location("hip_oga_vlm_api", source_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load VLM API probe from %s" % source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source_path


def write_assets(api, output):
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "left-red-right-blue": api.make_split_png((255, 0, 0), (0, 0, 255)),
        "left-blue-right-red": api.make_split_png((0, 0, 255), (255, 0, 0)),
    }
    metadata = {}
    for key, payload in payloads.items():
        path = image_dir / (key + ".png")
        path.write_bytes(payload)
        metadata[key] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return payloads, metadata


def image_message(probe, prompt, image_keys):
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": probe.image_url(key)}}
                   for key in image_keys)
    return {"role": "user", "content": content}


def request_layout(body, asset_by_hash):
    messages = body.get("messages", [])
    message_layout = []
    image_layout = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            part_types = [part.get("type", "unknown") if isinstance(part, dict) else "unknown"
                          for part in content]
            message_layout.append({
                "message_index": message_index,
                "role": message.get("role"),
                "content_kind": "multipart",
                "part_types": part_types,
            })
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                url = part.get("image_url", {}).get("url", "")
                image_entry = {
                    "image_order": len(image_layout),
                    "message_index": message_index,
                    "part_index": part_index,
                }
                if not url.startswith("data:image/png;base64,"):
                    image_entry["error"] = "image URL was not a PNG data URL"
                    image_layout.append(image_entry)
                    continue
                try:
                    payload = base64.b64decode(url.split(",", 1)[1], validate=True)
                    digest = hashlib.sha256(payload).hexdigest()
                    image_entry.update({
                        "asset_key": asset_by_hash.get(digest),
                        "sha256": digest,
                        "bytes": len(payload),
                    })
                except Exception as exc:
                    image_entry["error"] = repr(exc)
                image_layout.append(image_entry)
        else:
            message_layout.append({
                "message_index": message_index,
                "role": message.get("role"),
                "content_kind": "text" if isinstance(content, str) else "other",
                "part_types": [],
            })
    return {"messages": message_layout, "images": image_layout}


def assert_visual(api, record, expected, region):
    record["visual_expectation"] = {"expected_color": expected, "region": region}
    try:
        api.assert_color_answer(record, expected, region=region)
    except Exception as exc:
        words = api.extract_words(record.get("content", ""))
        record["visual_check"] = {
            "expected_color": expected,
            "region": region,
            "observed_color_words": [word for word in words if word in ("red", "blue")],
            "passed": False,
            "failure": repr(exc),
        }
        raise


def run_case(api, probe, name, body, expected_cache, expected_color, region, asset_by_hash):
    case = {
        "name": name,
        "status": "RUNNING",
        "request": name,
        "declared_expected_cache": expected_cache,
        "effective_expected_cache": "miss" if probe.args.cache == "off" else expected_cache,
        "expected_color": expected_color,
        "color_region": region,
        "payload_layout": request_layout(body, asset_by_hash),
    }
    probe.cases.append(case)
    record = None
    try:
        record = probe.call(name, "/v1/chat/completions", body)
        record["expected_cache"] = case["effective_expected_cache"]
        record["payload_layout"] = request_layout(body, asset_by_hash)
        record["has_image"] = True
        probe._chat_ok(record, expected_cache=expected_cache)
        assert_visual(api, record, expected_color, region)
        case["payload_layout"] = record["payload_layout"]
        case["usage"] = record.get("usage")
        case["visual_check"] = record.get("visual_check")
        case["status"] = "PASS"
        return record
    except Exception as exc:
        case.update(status="FAIL", failure=repr(exc), traceback=traceback.format_exc())
        if record is not None:
            case["usage"] = record.get("usage")
            case["payload_layout"] = record.get("payload_layout")
            case["visual_check"] = record.get("visual_check")
        raise


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "FAIL",
        "phase": "media-layout-api",
        "model": args.model,
        "base_url": args.base_url,
        "cache_mode_requested": args.cache,
        "case_names_expected": CASE_NAMES,
        "case_count_expected": len(CASE_NAMES),
        "case_count": 0,
        "request_count": 0,
        "cases": [],
        "records": [],
    }
    probe = None
    asset_by_hash = {}
    failure = None
    try:
        api, source_path = load_vlm_api(args.source_root.resolve())
        result["vlm_api_source"] = str(source_path)
        probe = api.VlmProbe(args)
        assets, image_metadata = write_assets(api, output)
        probe._asset_data = assets
        probe._image_hashes = {key: value["sha256"] for key, value in image_metadata.items()}
        result["images"] = image_metadata
        asset_by_hash = {item["sha256"]: key for key, item in image_metadata.items()}

        red_blue_message = image_message(probe, FIRST_LEFT_QUESTION, ["left-red-right-blue"])
        red_blue_body = probe.body([red_blue_message])
        run_case(api, probe, CASE_NAMES[0], red_blue_body, "miss", "red", "left-first-image", asset_by_hash)

        red_then_blue_message = image_message(
            probe, FIRST_LEFT_QUESTION, ["left-red-right-blue", "left-blue-right-red"])
        red_then_blue_body = probe.body([red_then_blue_message])
        run_case(api, probe, CASE_NAMES[1], red_then_blue_body, "miss", "red", "left-first-image", asset_by_hash)

        blue_then_red_message = image_message(
            probe, FIRST_LEFT_QUESTION, ["left-blue-right-red", "left-red-right-blue"])
        blue_then_red_body = probe.body([blue_then_red_message])
        swapped_record = run_case(
            api, probe, CASE_NAMES[2], blue_then_red_body, "miss", "blue", "left-first-image", asset_by_hash)

        # Reuse the exact body object, including both image data URLs and part order.
        run_case(api, probe, CASE_NAMES[3], blue_then_red_body, "miss", "blue", "left-first-image", asset_by_hash)

        extension_messages = list(blue_then_red_body["messages"])
        extension_messages.append({"role": "assistant", "content": swapped_record["content"]})
        extension_messages.append({"role": "user", "content": FIRST_RIGHT_QUESTION})
        history_body = probe.body(extension_messages)
        run_case(api, probe, CASE_NAMES[4], history_body, "hit", "red", "right-first-image", asset_by_hash)

        across_messages = [
            image_message(probe, "The first image follows.", ["left-blue-right-red"]),
            {"role": "assistant", "content": "The next image follows."},
            image_message(probe, FIRST_LEFT_QUESTION, ["left-red-right-blue"]),
        ]
        across_messages_body = probe.body(across_messages)
        run_case(api, probe, CASE_NAMES[5], across_messages_body,
                 "miss", "blue", "left-first-image", asset_by_hash)

        result["status"] = "PASS"
    except Exception as exc:
        failure = exc
        result["status"] = "FAIL"
        result["failure"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if probe is not None:
            for record in probe.records:
                body = record.get("request")
                if isinstance(body, dict):
                    record["payload_layout"] = request_layout(body, asset_by_hash)
                record.pop("_connection", None)
            result["cases"] = probe.cases
            result["records"] = probe.records
            result["case_count"] = len(probe.cases)
            result["request_count"] = len(probe.records)
            records_by_name = {record.get("name"): record for record in probe.records}
            for case in probe.cases:
                record = records_by_name.get(case.get("request"))
                if record is None:
                    continue
                case.setdefault("request_record_name", record.get("name"))
                case.setdefault("payload_layout", record.get("payload_layout"))
                case.setdefault("usage", record.get("usage"))
                case.setdefault("visual_check", record.get("visual_check"))
        result_path = output / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result.get(key) for key in
                          ("status", "phase", "case_count", "request_count", "failure")},
                         ensure_ascii=False))
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", choices=("off", "on"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
