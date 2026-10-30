#!/usr/bin/env python3
"""Check the HIP embedding error contract and recover with real HTTP calls."""
import argparse
import base64
import http.client
import json
from pathlib import Path
import uuid
from urllib.parse import urlsplit


TIMEOUT_SECONDS = 120


def require(condition, message):
    if not condition:
        raise ValueError(message)


def routes_for(base_path):
    """Preserve a caller's URL prefix while accepting a base URL ending in /v1."""
    prefix = base_path.rstrip("/")
    if prefix.endswith("/v1"):
        prefix = prefix[:-3]
    return {
        "/v1/models": prefix + "/v1/models",
        "/v1/embeddings": prefix + "/v1/embeddings",
        "/api/embeddings": prefix + "/api/embeddings",
        "/v1/chat/completions": prefix + "/v1/chat/completions",
    }


def case_definitions(model, request_id):
    return [
        ("embeddings-v1-string", "/v1/embeddings",
         {"model": model, "input": "A short embedding capability probe.", "request_id": request_id}),
        ("embeddings-v1-array", "/v1/embeddings",
         {"model": model, "input": ["First embedding probe.", "Second embedding probe."],
          "request_id": request_id}),
        ("embeddings-v1-missing-fields", "/v1/embeddings", {"request_id": request_id}),
        ("embeddings-api-string", "/api/embeddings",
         {"model": model, "input": "A short embedding capability probe.", "request_id": request_id}),
        ("embeddings-api-array", "/api/embeddings",
         {"model": model, "input": ["First embedding probe.", "Second embedding probe."],
          "request_id": request_id}),
        ("embeddings-api-missing-fields", "/api/embeddings", {"request_id": request_id}),
    ]


def request(args, routes, name, endpoint, body):
    record = {
        "name": name,
        "endpoint": routes[endpoint],
        "request": body,
        "http_status": None,
        "content_type": None,
        "response_body_utf8": "",
        "response_body_base64": "",
        "status": "FAIL",
    }
    encoded = None if body is None else json.dumps(body, ensure_ascii=False,
                                                   separators=(",", ":")).encode("utf-8")
    if encoded is not None:
        record["request_body_utf8"] = encoded.decode("utf-8")
    connection = None
    try:
        connection_type = http.client.HTTPSConnection if args.url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(args.url.hostname, args.url.port, timeout=TIMEOUT_SECONDS)
        headers = {"Accept": "application/json", "Connection": "close"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        connection.request("GET" if body is None else "POST", routes[endpoint],
                           body=encoded, headers=headers)
        response = connection.getresponse()
        record.update(
            http_status=response.status,
            content_type=response.getheader("Content-Type"),
        )
        try:
            raw = response.read()
        except http.client.IncompleteRead as error:
            raw = error.partial
            record["failure"] = f"IncompleteRead: {error}"
        record.update(
            response_body_utf8=raw.decode("utf-8", errors="replace"),
            response_body_base64=base64.b64encode(raw).decode("ascii"),
        )
        return record
    except Exception as error:
        record["failure"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        if connection is not None:
            connection.close()


def validate_embedding_rejection(record):
    require(record["http_status"] == 501,
            f"Expected HTTP 501, received {record['http_status']}")
    require((record.get("content_type") or "").split(";")[0].strip() == "application/json",
            "Expected application/json response")
    payload = json.loads(record["response_body_utf8"])
    require(isinstance(payload, dict), "Expected a JSON error object")
    error = payload.get("error")
    require(isinstance(error, dict), "Expected an error object")
    require(error.get("type") == "unsupported_feature", "Expected unsupported_feature error type")
    require(type(error.get("code")) is int and error["code"] == 501,
            "Expected integer error code 501")
    message = error.get("message")
    require(isinstance(message, str) and message.strip(), "Expected a nonempty error message")
    require("embedding" in message.lower(), "Error message must mention embedding")


def validate_models_recovery(record):
    require(record["http_status"] == 200,
            f"Expected HTTP 200, received {record['http_status']}")
    payload = json.loads(record["response_body_utf8"])
    require(isinstance(payload, dict) and "error" not in payload,
            "Expected a valid models response")
    require(isinstance(payload.get("data"), list), "Expected a models data array")


def validate_chat_recovery(record):
    require(record["http_status"] == 200,
            f"Expected HTTP 200, received {record['http_status']}")
    payload = json.loads(record["response_body_utf8"])
    require(isinstance(payload, dict) and "error" not in payload,
            "Chat returned an error object instead of a completion")
    choices = payload.get("choices")
    require(isinstance(choices, list) and choices,
            "Expected a nonempty chat choices array")
    message = choices[0].get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant",
            "Expected an assistant message")
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "Expected nonempty assistant content")
    usage = payload.get("usage")
    require(isinstance(usage, dict), "Expected completion usage")
    completion_tokens = usage.get("completion_tokens")
    require(type(completion_tokens) is int and completion_tokens > 0,
            "Expected positive completion token usage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.url = urlsplit(args.base_url)

    records, failures = [], []
    result = {
        "status": "FAIL",
        "model": args.model,
        "base_url": args.base_url,
        "records": records,
        "failures": failures,
        "request_count": 0,
    }
    try:
        require(args.url.scheme in ("http", "https") and args.url.hostname,
                "Expected an HTTP(S) base URL with a host")
        require(not args.url.query and not args.url.fragment,
                "Base URL must not contain a query or fragment")
        routes = routes_for(args.url.path)

        # Reuse one body ID across the six serial embedding calls. If the server
        # begins tracking these routes, accidental retained ownership is visible.
        request_id = "embedding-capability-" + uuid.uuid4().hex
        cases = case_definitions(args.model, request_id)
        validators = [validate_embedding_rejection] * len(cases)
        cases.extend([
            ("models-v1-recovery", "/v1/models", None),
            ("chat-v1-recovery", "/v1/chat/completions", {
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with exactly Hello."}],
                "temperature": 0,
                "max_tokens": 32,
            }),
        ])
        validators.extend([validate_models_recovery, validate_chat_recovery])

        for (name, endpoint, body), validator in zip(cases, validators):
            record = request(args, routes, name, endpoint, body)
            records.append(record)
            result["request_count"] += 1
            try:
                if "failure" in record:
                    raise RuntimeError(record["failure"])
                validator(record)
                record["status"] = "PASS"
            except Exception as error:
                record["failure"] = f"{type(error).__name__}: {error}"
                failures.append(name)
        result["status"] = "PASS" if not failures and len(records) == 8 else "FAIL"
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {result['request_count']} HTTP cases; failures={failures}; {output}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
