#!/usr/bin/env python3
"""Validate and report HIP OGA external tokenizer parity fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "src/test/hip_oga_external_codec/fixtures/text_cases.json"
REQUIRED_MODEL_FAMILIES = {"qwen2.5", "gemma"}
REQUIRED_CATEGORIES = {
    "plain",
    "chat",
    "tool",
    "code",
    "whitespace",
    "cjk",
    "emoji",
    "added_tokens",
}
VALID_TEMPLATE_MODES = {"raw", "chat", "chat_tools"}


class FixtureError(ValueError):
    """Raised when a tokenizer parity fixture is malformed."""


def fail(message: str) -> None:
    raise FixtureError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            parsed = json.load(file)
    except FileNotFoundError:
        fail(f"missing JSON file: {path.relative_to(ROOT)}")
    except json.JSONDecodeError as exc:
        fail(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return parsed


def require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(value))
    if missing:
        fail(f"{label} missing required keys: {', '.join(missing)}")


def require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        fail(f"{label} contains duplicate values: {', '.join(duplicates)}")


def validate_token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        fail(f"{label} must be a list")
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int) or token_id < 0:
            fail(f"{label}[{index}] must be a non-negative integer")
        token_ids.append(token_id)
    return token_ids


def case_text(case: dict[str, Any]) -> str:
    if "text" in case:
        if not isinstance(case["text"], str):
            fail(f"case {case.get('name', '<unknown>')} text must be a string")
        return case["text"]
    if "messages" in case:
        return json.dumps(case["messages"], ensure_ascii=False, sort_keys=True)
    fail(f"case {case.get('name', '<unknown>')} must define text or messages")


def validate_case_document(document: dict[str, Any]) -> None:
    require_keys(document, {"schema_version", "model_families", "cases"}, "fixture")
    if document["schema_version"] != 1:
        fail("fixture schema_version must be 1")
    if not isinstance(document["model_families"], list) or not document["model_families"]:
        fail("fixture model_families must be a non-empty list")
    if not isinstance(document["cases"], list) or not document["cases"]:
        fail("fixture cases must be a non-empty list")

    family_ids: list[str] = []
    for family in document["model_families"]:
        if not isinstance(family, dict):
            fail("model_families entries must be objects")
        require_keys(family, {"id", "tokenizer_family", "capture_required"}, "model family")
        if not isinstance(family["id"], str) or not family["id"]:
            fail("model family id must be a non-empty string")
        if not isinstance(family["tokenizer_family"], str) or not family["tokenizer_family"]:
            fail(f"model family {family['id']} tokenizer_family must be non-empty")
        if family["capture_required"] is not True:
            fail(f"model family {family['id']} must mark capture_required=true")
        family_ids.append(family["id"])
    require_unique(family_ids, "model_families")
    missing_families = sorted(REQUIRED_MODEL_FAMILIES.difference(family_ids))
    if missing_families:
        fail(f"fixture missing model family coverage: {', '.join(missing_families)}")

    case_names: list[str] = []
    coverage: dict[str, set[str]] = {family_id: set() for family_id in family_ids}
    for case in document["cases"]:
        if not isinstance(case, dict):
            fail("cases entries must be objects")
        require_keys(
            case,
            {"name", "category", "template_mode", "model_families", "capture_required"},
            "case",
        )
        if not isinstance(case["name"], str) or not case["name"]:
            fail("case name must be a non-empty string")
        case_names.append(case["name"])
        if case["category"] not in REQUIRED_CATEGORIES:
            fail(f"case {case['name']} uses unknown category {case['category']}")
        if case["template_mode"] not in VALID_TEMPLATE_MODES:
            fail(f"case {case['name']} uses unknown template_mode {case['template_mode']}")
        if not isinstance(case["model_families"], list) or not case["model_families"]:
            fail(f"case {case['name']} model_families must be a non-empty list")
        for family_id in case["model_families"]:
            if family_id not in family_ids:
                fail(f"case {case['name']} references unknown model family {family_id}")
            coverage[family_id].add(case["category"])

        text = case_text(case)
        try:
            text.encode("utf-8").decode("utf-8")
        except UnicodeError as exc:
            fail(f"case {case['name']} is not valid UTF-8: {exc}")

        if case["template_mode"] == "raw" and "text" not in case:
            fail(f"case {case['name']} raw template_mode requires text")
        if case["template_mode"] in {"chat", "chat_tools"} and "messages" not in case:
            fail(f"case {case['name']} chat template_mode requires messages")
        if case["template_mode"] == "chat_tools" and "tools" not in case:
            fail(f"case {case['name']} chat_tools template_mode requires tools")

        has_expected_ids = "expected_oga_token_ids" in case
        if has_expected_ids:
            validate_token_ids(case["expected_oga_token_ids"], f"case {case['name']} expected_oga_token_ids")
        if "expected_external_token_ids" in case:
            validate_token_ids(
                case["expected_external_token_ids"],
                f"case {case['name']} expected_external_token_ids",
            )
        if not has_expected_ids and case["capture_required"] is not True:
            fail(
                f"case {case['name']} must provide expected_oga_token_ids or capture_required=true"
            )

    require_unique(case_names, "cases")
    for family_id in REQUIRED_MODEL_FAMILIES:
        missing_categories = sorted(REQUIRED_CATEGORIES.difference(coverage[family_id]))
        if missing_categories:
            fail(
                f"model family {family_id} missing categories: "
                f"{', '.join(missing_categories)}"
            )


def hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def first_token_mismatch(
    oga_token_ids: list[int],
    external_token_ids: list[int],
) -> dict[str, Any] | None:
    shared_count = min(len(oga_token_ids), len(external_token_ids))
    for index in range(shared_count):
        if oga_token_ids[index] != external_token_ids[index]:
            return {
                "kind": "token_id",
                "index": index,
                "oga_token_id": oga_token_ids[index],
                "external_token_id": external_token_ids[index],
            }
    if len(oga_token_ids) != len(external_token_ids):
        return {
            "kind": "length",
            "index": shared_count,
            "oga_token_id": oga_token_ids[shared_count]
            if shared_count < len(oga_token_ids)
            else None,
            "external_token_id": external_token_ids[shared_count]
            if shared_count < len(external_token_ids)
            else None,
        }
    return None


def make_evidence_row(
    model_family: str,
    case: dict[str, Any],
    oga_token_ids: list[int] | None,
    external_token_ids: list[int] | None,
) -> dict[str, Any]:
    if oga_token_ids is None or external_token_ids is None:
        return {
            "model_family": model_family,
            "case": case["name"],
            "category": case["category"],
            "template_mode": case["template_mode"],
            "result": "capture_required",
            "prompt_token_count": None,
            "external_token_count": None,
            "prompt_token_ids_sha256": None,
            "external_token_ids_sha256": None,
            "first_prompt_token_mismatch": None,
        }

    mismatch = first_token_mismatch(oga_token_ids, external_token_ids)
    return {
        "model_family": model_family,
        "case": case["name"],
        "category": case["category"],
        "template_mode": case["template_mode"],
        "result": "pass" if mismatch is None else "mismatch",
        "prompt_token_count": len(oga_token_ids),
        "external_token_count": len(external_token_ids),
        "prompt_token_ids_sha256": hash_token_ids(oga_token_ids),
        "external_token_ids_sha256": hash_token_ids(external_token_ids),
        "first_prompt_token_mismatch": mismatch,
    }


def build_callback_evidence(
    document: dict[str, Any],
    oga_tokenize: Callable[[str, dict[str, Any]], list[int]],
    external_tokenize: Callable[[str, dict[str, Any]], list[int]],
) -> dict[str, Any]:
    validate_case_document(document)
    rows = []
    for case in document["cases"]:
        for family_id in case["model_families"]:
            rows.append(
                make_evidence_row(
                    family_id,
                    case,
                    oga_tokenize(family_id, case),
                    external_tokenize(family_id, case),
                )
            )
    return {
        "schema_version": 1,
        "evidence_kind": "hip_oga_external_tokenizer_parity",
        "default_provider": "oga",
        "rows": rows,
    }


def load_capture_index(path: Path) -> dict[tuple[str, str], tuple[list[int], list[int]]]:
    document = load_json(path)
    require_keys(document, {"schema_version", "captures"}, "capture file")
    if document["schema_version"] != 1:
        fail("capture file schema_version must be 1")
    if not isinstance(document["captures"], list):
        fail("capture file captures must be a list")
    index: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    for capture in document["captures"]:
        if not isinstance(capture, dict):
            fail("capture entries must be objects")
        require_keys(
            capture,
            {"model_family", "case", "oga_token_ids", "external_token_ids"},
            "capture",
        )
        key = (capture["model_family"], capture["case"])
        if key in index:
            fail(f"duplicate capture for {capture['model_family']} / {capture['case']}")
        index[key] = (
            validate_token_ids(capture["oga_token_ids"], f"capture {key} oga_token_ids"),
            validate_token_ids(
                capture["external_token_ids"], f"capture {key} external_token_ids"
            ),
        )
    return index


def build_capture_evidence(
    document: dict[str, Any],
    capture_index: dict[tuple[str, str], tuple[list[int], list[int]]],
) -> dict[str, Any]:
    validate_case_document(document)
    rows = []
    for case in document["cases"]:
        for family_id in case["model_families"]:
            capture = capture_index.get((family_id, case["name"]))
            if capture is None and "expected_oga_token_ids" in case:
                capture = (
                    validate_token_ids(
                        case["expected_oga_token_ids"],
                        f"case {case['name']} expected_oga_token_ids",
                    ),
                    validate_token_ids(
                        case.get("expected_external_token_ids", case["expected_oga_token_ids"]),
                        f"case {case['name']} expected_external_token_ids",
                    ),
                )
            rows.append(
                make_evidence_row(
                    family_id,
                    case,
                    capture[0] if capture is not None else None,
                    capture[1] if capture is not None else None,
                )
            )
    return {
        "schema_version": 1,
        "evidence_kind": "hip_oga_external_tokenizer_parity",
        "default_provider": "oga",
        "rows": rows,
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def fake_tokenize(family_id: str, case: dict[str, Any]) -> list[int]:
    text = case_text(case)
    payload = f"{family_id}:{case['name']}:{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return [digest[0], digest[1], len(text.encode("utf-8"))]


def run_self_test() -> None:
    document = load_json(FIXTURE_PATH)
    validate_case_document(document)

    bad_duplicate = json.loads(json.dumps(document))
    bad_duplicate["cases"][1]["name"] = bad_duplicate["cases"][0]["name"]
    try:
        validate_case_document(bad_duplicate)
    except FixtureError as exc:
        if "duplicate" not in str(exc):
            raise
    else:
        fail("duplicate case names must be rejected")

    bad_uncaptured = json.loads(json.dumps(document))
    bad_uncaptured["cases"][0].pop("expected_oga_token_ids", None)
    bad_uncaptured["cases"][0]["capture_required"] = False
    try:
        validate_case_document(bad_uncaptured)
    except FixtureError as exc:
        if "capture_required" not in str(exc):
            raise
    else:
        fail("uncaptured cases without expected IDs must be rejected")

    evidence = build_callback_evidence(document, fake_tokenize, fake_tokenize)
    if not evidence["rows"]:
        fail("self-test evidence must contain rows")
    if {row["result"] for row in evidence["rows"]} != {"pass"}:
        fail("identical fake tokenizers must produce only pass rows")

    def mismatching_tokenize(family_id: str, case: dict[str, Any]) -> list[int]:
        token_ids = fake_tokenize(family_id, case)
        if family_id == "qwen2.5" and case["name"] == document["cases"][0]["name"]:
            token_ids[1] += 1
        return token_ids

    mismatch_evidence = build_callback_evidence(
        document,
        fake_tokenize,
        mismatching_tokenize,
    )
    mismatches = [
        row for row in mismatch_evidence["rows"] if row["result"] == "mismatch"
    ]
    if len(mismatches) != 1:
        fail("self-test must produce exactly one injected mismatch")
    if mismatches[0]["first_prompt_token_mismatch"]["kind"] != "token_id":
        fail("injected mismatch must be classified as token_id")

    with tempfile.TemporaryDirectory() as directory:
        evidence_path = Path(directory) / "tokenizer-parity-evidence.json"
        write_json(evidence_path, evidence)
        reloaded = load_json(evidence_path)
        if reloaded["evidence_kind"] != "hip_oga_external_tokenizer_parity":
            fail("written evidence kind changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cases", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--external-authoritative", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            run_self_test()
            print("PASS: HIP OGA external tokenizer parity self-test")
            return 0

        document = load_json(args.cases)
        validate_case_document(document)
        capture_index = load_capture_index(args.captures) if args.captures else {}
        evidence = build_capture_evidence(document, capture_index)
        if args.evidence_out:
            write_json(args.evidence_out, evidence)
        if args.external_authoritative:
            mismatches = [row for row in evidence["rows"] if row["result"] == "mismatch"]
            if mismatches:
                print(
                    f"FAIL: {len(mismatches)} tokenizer parity mismatches",
                    file=sys.stderr,
                )
                return 1
        capture_required = [
            row for row in evidence["rows"] if row["result"] == "capture_required"
        ]
        print(
            "PASS: HIP OGA external tokenizer fixtures "
            f"({len(evidence['rows'])} rows, {len(capture_required)} capture_required)"
        )
        return 0
    except FixtureError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
