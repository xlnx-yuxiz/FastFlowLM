#!/usr/bin/env python3
"""Validate the no-hardware HIP OGA text-codec evidence contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "src/test/hip_oga_backend_static"
SCHEMA_PATH = TEST_ROOT / "hip_oga_text_codec_evidence.schema.json"
FIXTURE_PATH = TEST_ROOT / "fixtures/hip_oga_text_codec_deterministic_fixture.json"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            parsed = json.load(file)
    except FileNotFoundError:
        fail(f"missing required JSON file: {path.relative_to(ROOT)}")
    except json.JSONDecodeError as exc:
        fail(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return parsed


def require_keys(value: dict, keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(value))
    if missing:
        fail(f"{label} missing required keys: {', '.join(missing)}")


def validate_schema_shape(schema: dict) -> None:
    require_keys(
        schema,
        {
            "$schema",
            "title",
            "type",
            "required",
            "properties",
            "additionalProperties",
        },
        "evidence schema",
    )
    if schema["type"] != "object":
        fail("evidence schema root type must be object")
    if schema["additionalProperties"] is not False:
        fail("evidence schema must close the root object")
    required = set(schema["required"])
    expected = {
        "schema_version",
        "codec_selector",
        "default_provider",
        "fixture",
        "parity",
        "detokenizer",
    }
    if required != expected:
        fail(f"evidence schema required keys changed: {sorted(required)}")


def validate_fixture(fixture: dict) -> None:
    require_keys(
        fixture,
        {
            "schema_version",
            "codec_selector",
            "default_provider",
            "fixture",
            "parity",
            "detokenizer",
        },
        "deterministic fixture",
    )
    if fixture["schema_version"] != 1:
        fail("fixture schema_version must be 1")
    if fixture["codec_selector"] != "oga":
        fail("foundation fixture must keep codec_selector=oga")
    if fixture["default_provider"] != "oga":
        fail("default_provider must remain oga")

    fixture_contract = fixture["fixture"]
    require_keys(
        fixture_contract,
        {
            "id",
            "description",
            "prompt",
            "normalized_prompt",
            "expected_token_ids",
            "expected_decoded_text",
            "determinism",
        },
        "fixture contract",
    )
    if fixture_contract["prompt"] != fixture_contract["normalized_prompt"]:
        fail("foundation fixture must avoid hidden prompt normalization")
    if not fixture_contract["expected_token_ids"]:
        fail("fixture must contain deterministic token ids")
    if fixture_contract["expected_decoded_text"] != fixture_contract["prompt"]:
        fail("fixture decoded text must round-trip to the prompt")
    determinism = fixture_contract["determinism"]
    require_keys(determinism, {"temperature", "top_p", "max_new_tokens"}, "determinism")
    if determinism["temperature"] != 0 or determinism["top_p"] != 1:
        fail("deterministic fixture must pin temperature=0 and top_p=1")

    parity = fixture["parity"]
    require_keys(
        parity,
        {
            "status",
            "equivalence_class",
            "raw_byte_equal",
            "valid_utf8",
            "special_token_leakage",
            "first_mismatch",
            "normalization_applied",
            "oga_provider",
            "external_provider",
            "oga_token_ids",
            "external_token_ids",
            "oga_decoded_text",
            "external_decoded_text",
            "notes",
        },
        "parity evidence",
    )
    if parity["status"] == "not_run":
        fail("wired fixture must report real external codec parity, not not_run")
    if parity["equivalence_class"] not in {
        "byte_exact",
        "canonical_equivalent",
        "structured_equivalent",
        "incompatible",
    }:
        fail("parity evidence must use a stable decoder equivalence class")
    if parity["status"] == "match" and parity["equivalence_class"] == "incompatible":
        fail("matching parity evidence cannot classify decoder text as incompatible")
    if not isinstance(parity["raw_byte_equal"], bool):
        fail("raw_byte_equal must be a boolean diagnostic")
    if not isinstance(parity["valid_utf8"], bool):
        fail("valid_utf8 must be a boolean diagnostic")
    if not isinstance(parity["special_token_leakage"], bool):
        fail("special_token_leakage must be a boolean diagnostic")
    if parity["first_mismatch"] is not None:
        require_keys(
            parity["first_mismatch"],
            {"byte_index", "oga_text_window", "external_text_window"},
            "first mismatch evidence",
        )
    if not isinstance(parity["normalization_applied"], list):
        fail("normalization_applied must list canonical text rewrites")
    if parity["oga_provider"] != "oga" or parity["external_provider"] != "external":
        fail("parity providers must use stable provider enum names")
    if parity["oga_token_ids"] != fixture_contract["expected_token_ids"]:
        fail("OGA token evidence must mirror the deterministic fixture")
    if parity["external_token_ids"] != fixture_contract["expected_token_ids"]:
        fail("external token evidence must come from the wired external tokenizer callback")
    if parity["external_decoded_text"] != fixture_contract["expected_decoded_text"]:
        fail("external decoded evidence must come from the wired cumulative decoder callback")

    detokenizer = fixture["detokenizer"]
    require_keys(
        detokenizer,
        {
            "status",
            "mode",
            "input_token_ids",
            "chunk_emissions",
            "final_text",
            "expected_text",
            "notes",
        },
        "detokenizer evidence",
    )
    if detokenizer["status"] != "match":
        fail("detokenizer fixture must prove byte parity")
    if detokenizer["mode"] != "callback_cumulative":
        fail("detokenizer fixture must use the callback cumulative mode")
    if detokenizer["input_token_ids"] != fixture_contract["expected_token_ids"]:
        fail("detokenizer input ids must mirror the deterministic fixture")
    if detokenizer["final_text"] != detokenizer["expected_text"]:
        fail("detokenizer final text must match expected text")
    if detokenizer["final_text"] != fixture_contract["expected_decoded_text"]:
        fail("detokenizer final text must match fixture decoded text")
    chunk_text = "".join(chunk["emitted_text"] for chunk in detokenizer["chunk_emissions"])
    if chunk_text != detokenizer["final_text"]:
        fail("detokenizer chunk emissions must concatenate to final text")
    if not any(chunk.get("final") for chunk in detokenizer["chunk_emissions"]):
        fail("detokenizer evidence must include a final flush chunk")


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    fixture = load_json(FIXTURE_PATH)
    validate_schema_shape(schema)
    validate_fixture(fixture)
    print("PASS: HIP OGA text-codec evidence contract")


if __name__ == "__main__":
    main()
