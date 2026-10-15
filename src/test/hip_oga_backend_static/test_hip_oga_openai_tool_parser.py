#!/usr/bin/env python3
"""No-hardware fixture for HIP OGA OpenAI tool-call response shaping.

The production parser lives in HipOgaModel. This fixture mirrors the supported
qwen/gemma-style markup contract and asserts the route-visible OpenAI fields
that must appear once HIP generation emits valid tool-call markup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


THINK_START = "<think>"
THINK_END = "</think>"
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"


@dataclass
class ParsedNonStream:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class StreamEvent:
    type: str
    content: str = ""
    tool_name: str = ""
    tool_args: str = ""


def _decode_tool(raw: str) -> tuple[str, str] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or arguments is None:
        return None
    if isinstance(arguments, str):
        args = arguments
    else:
        args = json.dumps(arguments, separators=(",", ":"))
    return name, args


def parse_nonstream_markup(text: str) -> ParsedNonStream:
    parsed = ParsedNonStream()
    cursor = 0
    while cursor < len(text):
        think_pos = text.find(THINK_START, cursor)
        tool_pos = text.find(TOOL_START, cursor)
        if think_pos == -1 and tool_pos == -1:
            parsed.content += text[cursor:]
            break

        if think_pos != -1 and (tool_pos == -1 or think_pos < tool_pos):
            marker_pos = think_pos
            parsed.content += text[cursor:marker_pos]
            body_start = marker_pos + len(THINK_START)
            body_end = text.find(THINK_END, body_start)
            if body_end == -1:
                parsed.content += text[marker_pos:]
                break
            parsed.reasoning_content += text[body_start:body_end]
            cursor = body_end + len(THINK_END)
            continue

        marker_pos = tool_pos
        parsed.content += text[cursor:marker_pos]
        body_start = marker_pos + len(TOOL_START)
        body_end = text.find(TOOL_END, body_start)
        if body_end == -1:
            parsed.content += text[marker_pos:]
            break
        raw_tool = text[body_start:body_end]
        tool = _decode_tool(raw_tool)
        if tool is None:
            parsed.content += text[marker_pos:body_end + len(TOOL_END)]
        else:
            parsed.tool_calls.append(tool)
        cursor = body_end + len(TOOL_END)
    return parsed


def build_openai_nonstream_choice(parsed: ParsedNonStream) -> dict:
    message: dict = {"role": "assistant"}
    if parsed.reasoning_content:
        message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = [
            {
                "index": index,
                "id": f"call_fixture_{index}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            for index, (name, args) in enumerate(parsed.tool_calls)
        ]
        if parsed.content:
            message["content"] = parsed.content
    else:
        message["content"] = parsed.content
    return {"index": 0, "message": message, "finish_reason": "tool_calls"}


def stream_markup_events(chunks: list[str]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    buffer = ""
    in_tool = False
    tool_buffer = ""

    for chunk in chunks:
        buffer += chunk
        while buffer:
            if not in_tool:
                start = buffer.find(TOOL_START)
                if start == -1:
                    events.append(StreamEvent("content", content=buffer))
                    buffer = ""
                    break
                if start:
                    events.append(StreamEvent("content", content=buffer[:start]))
                    buffer = buffer[start:]
                    continue
                buffer = buffer[len(TOOL_START):]
                in_tool = True
                continue

            end = buffer.find(TOOL_END)
            if end == -1:
                tool_buffer += buffer
                buffer = ""
                break

            tool_buffer += buffer[:end]
            tool = _decode_tool(tool_buffer)
            if tool is None:
                events.append(StreamEvent("content", content="[Error parsing tool call]"))
            else:
                name, args = tool
                events.append(StreamEvent("tool_done", tool_name=name, tool_args=args))
            buffer = buffer[end + len(TOOL_END):]
            tool_buffer = ""
            in_tool = False

    return events


def build_openai_stream_chunk(event: StreamEvent, index: int) -> dict:
    delta: dict
    if event.type == "tool_done":
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_fixture_{index}",
                    "type": "function",
                    "function": {
                        "name": event.tool_name,
                        "arguments": event.tool_args,
                    },
                }
            ]
        }
    else:
        delta = {"content": event.content}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def main() -> None:
    tool_json = {
        "name": "get_current_weather",
        "arguments": {"location": "Paris, France", "unit": "celsius"},
    }
    markup = (
        "Checking weather. "
        "<think>Need live weather.</think>"
        f"{TOOL_START}{json.dumps(tool_json, separators=(',', ':'))}{TOOL_END}"
    )

    parsed = parse_nonstream_markup(markup)
    choice = build_openai_nonstream_choice(parsed)
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["role"] == "assistant"
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "get_current_weather"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Paris, France",
        "unit": "celsius",
    }
    assert message["content"] == "Checking weather. "
    assert message["reasoning_content"] == "Need live weather."

    stream_events = stream_markup_events([
        "Checking weather. ",
        TOOL_START,
        json.dumps(tool_json, separators=(",", ":")),
        TOOL_END,
    ])
    stream_chunks = [
        build_openai_stream_chunk(event, index)
        for index, event in enumerate(stream_events)
    ]
    tool_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"]
        for chunk in stream_chunks
        if "tool_calls" in chunk["choices"][0]["delta"]
    ]
    content_deltas = [
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in stream_chunks
        if "content" in chunk["choices"][0]["delta"]
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0][0]["function"]["name"] == "get_current_weather"
    assert json.loads(tool_deltas[0][0]["function"]["arguments"]) == {
        "location": "Paris, France",
        "unit": "celsius",
    }
    assert all(TOOL_START not in content for content in content_deltas)
    print("PASS: HIP OGA OpenAI tool-call parser fixture")


if __name__ == "__main__":
    main()
