#!/usr/bin/env python3
"""Black-box lifecycle checks for the HIP OGA model catalog CLI.

This script intentionally runs an existing flm binary. It never builds one and
uses only the tiny localhost Hugging Face fixture from
test_hip_oga_public_pull.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import quote

import test_hip_oga_public_pull as fixture


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_TAG = "hip-test:1b"
INSTALL_RELATIVE = Path("test-oga")
RECEIPT_NAME = "fastflowlm_hip_oga_install.json"
COMMAND_TIMEOUT_SECONDS = 60
CONTROLLED_ENV = (
    "FLM_CONFIG_PATH",
    "FLM_MODEL_PATH",
    "FLM_HIP_OGA_MODEL_ROOT",
    "FLM_HIP_OGA_HF_ENDPOINT",
)


def snapshot_tree(path: Path) -> dict[str, dict[str, Any]]:
    """Return a stable file snapshot, including relative paths and hashes."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            relative = item.relative_to(path).as_posix()
            data = item.read_bytes()
            result[relative] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return result


def assert_command(record: dict[str, Any], *, succeeds: bool) -> None:
    expected = 0 if succeeds else "nonzero"
    actual = record.get("exitCode")
    if record.get("timedOut"):
        raise AssertionError(f"{record['name']} timed out after {COMMAND_TIMEOUT_SECONDS}s")
    if succeeds and actual != 0:
        raise AssertionError(
            f"{record['name']} returned {actual}; see {record['stdoutFile']} and "
            f"{record['stderrFile']}"
        )
    if not succeeds and (actual is None or actual == 0):
        raise AssertionError(f"{record['name']} expected {expected} exit code, got {actual}")


class LifecycleCase:
    def __init__(self, name: str, case_root: Path, binary: Path) -> None:
        self.name = name
        self.root = case_root
        self.binary = binary
        self.public_root = case_root / "hf"
        self.config_dir = case_root / "config"
        self.install_root = case_root / "install"
        self.model_path = case_root / "model-path"
        self.config_path = self.config_dir / "model_list.json"
        self.entries: list[dict[str, object]] = []
        self.repo_id = "amd/test-oga"
        self.revision = fixture.REVISION
        self.commands: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}
        self.public_root.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.install_root.mkdir(parents=True, exist_ok=True)

    @property
    def install_path(self) -> Path:
        return self.install_root / INSTALL_RELATIVE

    @property
    def native_path(self) -> Path:
        return self.model_path / "models" / "HipTest-1B-OGA"

    def configure_package(self, *, config_path: str = "test-oga") -> None:
        self.repo_id, self.revision, self.entries = fixture.create_public_package(
            self.public_root
        )
        fixture.create_model_list(
            self.config_path,
            self.repo_id,
            self.revision,
            self.entries,
            config_path=config_path,
        )
        self.facts["configPath"] = config_path
        self.facts["files"] = [
            {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
            for item in self.entries
        ]

    def env(self, endpoint: str = "", *, model_root: Path | None | object = ...) -> dict[str, str]:
        result = os.environ.copy()
        result["FLM_CONFIG_PATH"] = str(self.config_path.resolve())
        result["FLM_MODEL_PATH"] = str(self.model_path.resolve())
        result["FLM_HIP_OGA_HF_ENDPOINT"] = endpoint
        if model_root is ...:
            result["FLM_HIP_OGA_MODEL_ROOT"] = str(self.install_root.resolve())
        elif model_root is None:
            result.pop("FLM_HIP_OGA_MODEL_ROOT", None)
        else:
            result["FLM_HIP_OGA_MODEL_ROOT"] = str(Path(model_root).resolve())
        return result

    def invoke(
        self,
        command_name: str,
        args: list[str],
        *,
        endpoint: str = "",
        model_root: Path | None | object = ...,
    ) -> dict[str, Any]:
        index = len(self.commands) + 1
        stdout_path = self.root / f"command-{index:02d}-{command_name}.stdout.log"
        stderr_path = self.root / f"command-{index:02d}-{command_name}.stderr.log"
        argv = [str(self.binary), *args]
        env = self.env(endpoint, model_root=model_root)
        record: dict[str, Any] = {
            "name": command_name,
            "argv": argv,
            "env": {key: env.get(key) for key in CONTROLLED_ENV},
            "exitCode": None,
            "timedOut": False,
            "stdoutFile": stdout_path.name,
            "stderrFile": stderr_path.name,
        }
        try:
            completed = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
            record["exitCode"] = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            record["timedOut"] = True
            record["exitCode"] = None
            stdout = _decode_output(error.stdout)
            stderr = _decode_output(error.stderr)
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        self.commands.append(record)
        self.write_evidence()
        return record

    def pull(self, server: fixture.ThreadedHttpFixture, *, force: bool = False) -> dict[str, Any]:
        args = ["pull", MODEL_TAG, "--backend", "hip"]
        if force:
            args.append("--force")
        return self.invoke("pull" + ("-force" if force else ""), args, endpoint=server.endpoint)

    def list_models(self, filter_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        record = self.invoke(
            "list-" + filter_name,
            ["list", "--filter", filter_name, "--backend", "hip", "--json"],
        )
        assert_command(record, succeeds=True)
        try:
            output = json.loads((self.root / record["stdoutFile"]).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise AssertionError(f"list {filter_name} did not return JSON: {error}") from error
        models = output.get("models")
        if not isinstance(models, list):
            raise AssertionError(f"list {filter_name} JSON has no models array")
        return record, models

    def write_evidence(self) -> None:
        evidence = {
            "case": self.name,
            "passed": None,
            "facts": self.facts,
            "commands": self.commands,
        }
        (self.root / "evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def assert_list_state(models: list[dict[str, Any]], *, installed: bool) -> None:
    matches = [model for model in models if model.get("name", model.get("model")) == MODEL_TAG]
    if len(matches) != 1:
        raise AssertionError(f"expected one {MODEL_TAG} entry, got {len(matches)}")
    if bool(matches[0].get("installed")) is not installed:
        raise AssertionError(
            f"{MODEL_TAG} installed={matches[0].get('installed')}, expected {installed}"
        )


def assert_no_transaction_dirs(install_root: Path) -> None:
    if not install_root.exists():
        return
    leftovers = [
        item
        for item in install_root.rglob("*")
        if item.is_dir() and (item.name.startswith(".tmp-") or item.name.startswith(".backup-"))
    ]
    if leftovers:
        raise AssertionError(f"transaction directories remain: {leftovers}")


def assert_installed_hashes(case: LifecycleCase) -> None:
    for entry in case.entries:
        path = case.install_path / str(entry["path"])
        if not path.is_file():
            raise AssertionError(f"installed file is missing: {path}")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != str(entry["sha256"]).lower() or len(data) != entry["size"]:
            raise AssertionError(f"installed file does not match catalog entry: {entry['path']}")


def installed_receipt(case: LifecycleCase) -> Path:
    path = case.install_path / RECEIPT_NAME
    if not path.is_file():
        raise AssertionError(f"managed install receipt is missing: {path}")
    return path


def case_normal_lifecycle(case: LifecycleCase) -> None:
    case.configure_package()
    case.native_path.mkdir(parents=True, exist_ok=True)
    sentinel = case.native_path / "native-sentinel.bin"
    sentinel.write_bytes(b"preserve native model sibling\n")
    case.facts["nativeSentinel"] = str(sentinel.relative_to(case.root))

    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    assert_installed_hashes(case)
    installed_receipt(case)
    assert_no_transaction_dirs(case.install_root)

    _, installed = case.list_models("installed")
    assert_list_state(installed, installed=True)
    assert_command(case.invoke("check", ["check", MODEL_TAG, "--backend", "hip"]), succeeds=True)
    assert_command(case.invoke("remove", ["remove", MODEL_TAG, "--backend", "hip"]), succeeds=True)
    if case.install_path.exists():
        raise AssertionError("remove left the installed HIP OGA package")
    if sentinel.read_bytes() != b"preserve native model sibling\n":
        raise AssertionError("remove changed the native model sibling sentinel")
    _, not_installed = case.list_models("not-installed")
    assert_list_state(not_installed, installed=False)
    assert_command(case.invoke("remove-again", ["remove", MODEL_TAG, "--backend", "hip"]), succeeds=True)
    if sentinel.read_bytes() != b"preserve native model sibling\n":
        raise AssertionError("idempotent remove changed the native model sibling sentinel")
    case.facts["nativeSentinelPreserved"] = True


def case_corrupt_check_preserves_file(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)

    text_file = case.install_path / "text.onnx"
    original = text_file.read_bytes()
    corrupt = b"X" * len(original)
    if len(corrupt) != len(original):
        raise AssertionError("corruption fixture changed text.onnx size")
    text_file.write_bytes(corrupt)
    case.facts["corruptFile"] = "text.onnx"
    case.facts["corruptSize"] = len(corrupt)

    record = case.invoke("check-corrupt", ["check", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if not text_file.is_file() or text_file.read_bytes() != corrupt:
        raise AssertionError("check deleted or changed the same-size corrupted text.onnx")
    case.facts["corruptionPreserved"] = True


def case_repeat_pull_repairs_corruption(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        target = case.install_path / "text.onnx"
        expected = target.read_bytes()
        target.write_bytes(b"Y" * len(expected))
        case.facts["corruptFile"] = "text.onnx"
        case.facts["corruptSize"] = len(expected)
        assert_command(case.pull(server), succeeds=True)

    assert_installed_hashes(case)
    assert_no_transaction_dirs(case.install_root)
    case.facts["hashesRepaired"] = True


def case_uppercase_hash_manifest_installs_and_checks(case: LifecycleCase) -> None:
    case.configure_package()
    case.entries = [
        {**entry, "sha256": str(entry["sha256"]).upper()}
        for entry in case.entries
    ]
    fixture.create_model_list(case.config_path, case.repo_id, case.revision, case.entries)
    case.facts["manifestHashesUppercase"] = True

    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        case.facts["downloadAttempts"] = dict(server.handler.attempts)

    assert_installed_hashes(case)
    receipt_data = json.loads(installed_receipt(case).read_text(encoding="utf-8"))
    receipt_hashes = {
        str(entry["path"]): str(entry["sha256"])
        for entry in receipt_data.get("files", [])
    }
    expected_hashes = {
        str(entry["path"]): str(entry["sha256"])
        for entry in case.entries
    }
    if receipt_hashes != expected_hashes:
        raise AssertionError("install receipt did not retain the manifest SHA256 casing")
    if not all(value == value.upper() for value in receipt_hashes.values()):
        raise AssertionError("receipt contains a non-uppercase SHA256 value")
    check = case.invoke("check-uppercase-hashes", ["check", MODEL_TAG, "--backend", "hip"])
    assert_command(check, succeeds=True)
    case.facts["receiptRetainedUppercaseHashes"] = True
    case.facts["uppercaseHashCheckPassed"] = True


def case_force_failure_preserves_old(case: LifecycleCase) -> None:
    case.configure_package()
    corrupt_path = quote(
        fixture.resolve_path(case.repo_id, case.revision, "text.onnx"), safe="/"
    )
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        before = snapshot_tree(case.install_path)

        source = case.public_root / case.repo_id / "resolve" / case.revision / fixture.SOURCE_SUBDIR / "text.onnx"
        source.write_bytes(b"Z" * source.stat().st_size)
        attempts_before = server.handler.attempts.get(corrupt_path, 0)
        case.facts["corruptSourcePath"] = corrupt_path
        case.facts["pinnedTextHash"] = next(
            str(entry["sha256"]) for entry in case.entries if entry["path"] == "text.onnx"
        )
        record = case.invoke(
            "pull-force-corrupt-source",
            ["pull", MODEL_TAG, "--backend", "hip", "--force"],
            endpoint=server.endpoint,
        )
        assert_command(record, succeeds=False)
        attempts_after = server.handler.attempts.get(corrupt_path, 0)

    after = snapshot_tree(case.install_path)
    if after != before:
        raise AssertionError("failed forced pull changed the previously installed package")
    if attempts_after - attempts_before < 3:
        raise AssertionError(
            "forced pull did not retry the corrupt text.onnx source at least three times: "
            f"before={attempts_before}, after={attempts_after}"
        )
    assert_no_transaction_dirs(case.install_root)
    case.facts["oldInstallPreserved"] = True
    case.facts["textOnnxRequestsBefore"] = attempts_before
    case.facts["textOnnxRequestsAfter"] = attempts_after


def case_force_without_receipt_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        installed_receipt(case).unlink()
        before = snapshot_tree(case.install_path)
        attempts_before = dict(server.handler.attempts)
        record = case.pull(server, force=True)
        attempts_after = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("forced pull replaced a valid payload whose receipt was missing")
    if attempts_after != attempts_before:
        raise AssertionError("forced pull contacted HTTP after discovering the missing receipt")
    assert_no_transaction_dirs(case.install_root)
    case.facts["receiptMissing"] = True
    case.facts["payloadUnchanged"] = True
    case.facts["httpAttemptsBefore"] = attempts_before
    case.facts["httpAttemptsAfter"] = attempts_after


def case_corrupt_without_receipt_pull_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        installed_receipt(case).unlink()
        payload = case.install_path / "text.onnx"
        payload.write_bytes(b"R" * payload.stat().st_size)
        before = snapshot_tree(case.install_path)
        attempts_before = dict(server.handler.attempts)
        record = case.pull(server)
        attempts_after = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("non-force pull replaced a corrupt payload without its receipt")
    if attempts_after != attempts_before:
        raise AssertionError("non-force pull contacted HTTP before refusing an unreceipted repair")
    assert_no_transaction_dirs(case.install_root)
    case.facts["receiptMissing"] = True
    case.facts["corruptPayloadPreserved"] = True
    case.facts["httpAttemptsBefore"] = attempts_before
    case.facts["httpAttemptsAfter"] = attempts_after


def case_unknown_file_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    user_file = case.install_path / "user-notes.txt"
    user_file.write_bytes(b"user-owned data\n")
    before = snapshot_tree(case.install_path)
    record = case.invoke("remove-with-user-file", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("refused remove changed a package containing a user file")
    case.facts["userFilePreserved"] = True


def case_no_receipt_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    receipt = installed_receipt(case)
    receipt.unlink()
    before = snapshot_tree(case.install_path)
    record = case.invoke("remove-without-receipt", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("refused remove changed an install without its receipt")
    case.facts["receiptMissingAndPayloadPreserved"] = True


def case_forged_receipt_revision_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    receipt = installed_receipt(case)
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["revision"] = "b" * len(case.revision)
    receipt.write_text(json.dumps(receipt_data, indent=2) + "\n", encoding="utf-8")
    before = snapshot_tree(case.install_path)
    case.facts["forgedReceiptRevision"] = receipt_data["revision"]
    record = case.invoke("remove-with-forged-receipt-revision", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("refused remove changed an install with a forged receipt revision")
    case.facts["forgedReceiptPreserved"] = True


def case_corrupt_payload_remove_allowed(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    receipt = installed_receipt(case)
    original_receipt = receipt.read_bytes()
    payload = case.install_path / "text.onnx"
    payload.write_bytes(b"Q" * payload.stat().st_size)
    case.facts["corruptPayload"] = "text.onnx"
    case.facts["receiptRemainedValid"] = receipt.read_bytes() == original_receipt
    if not case.facts["receiptRemainedValid"]:
        raise AssertionError("payload corruption unexpectedly changed the valid receipt")
    record = case.invoke("remove-corrupt-payload", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=True)
    if case.install_path.exists():
        raise AssertionError("remove did not clean a corrupt payload with a valid receipt")
    case.facts["corruptPayloadRemoved"] = True


def case_missing_root_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    before = snapshot_tree(case.install_path)
    record = case.invoke(
        "remove-without-explicit-root",
        ["remove", MODEL_TAG, "--backend", "hip"],
        model_root=None,
    )
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("remove without FLM_HIP_OGA_MODEL_ROOT touched the installed package")
    case.facts["installPreserved"] = True


def case_direct_path_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    before = snapshot_tree(case.install_path)
    record = case.invoke(
        "remove-direct-directory",
        ["remove", str(case.install_path), "--backend", "hip"],
    )
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("remove of a direct OGA directory changed the package")
    case.facts["directPackagePreserved"] = True


def case_leaf_payload_symlink_remove_refused(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    payload = case.install_path / "text.onnx"
    outside = case.root / "outside-text.onnx"
    original = payload.read_bytes()
    payload.rename(outside)
    payload.symlink_to(outside)
    before = snapshot_tree(case.install_path)
    case.facts["leafPayload"] = "text.onnx"
    case.facts["outsideTarget"] = outside.name
    record = case.invoke("remove-with-leaf-payload-symlink", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("refused remove changed package files around a linked payload")
    if not payload.is_symlink() or outside.read_bytes() != original:
        raise AssertionError("refused remove followed or removed the leaf payload symlink target")
    case.facts["outsidePayloadPreserved"] = True


def case_reserved_receipt_path_rejected_before_http(case: LifecycleCase) -> None:
    case.configure_package()
    reserved_bytes = b"must not replace the FastFlowLM install receipt"
    reserved_entry = {
        "path": RECEIPT_NAME,
        "sha256": fixture.sha256_bytes(reserved_bytes),
        "size": len(reserved_bytes),
    }
    case.entries.append(reserved_entry)
    case.entries.sort(key=lambda item: str(item["path"]))
    fixture.create_model_list(case.config_path, case.repo_id, case.revision, case.entries)
    case.facts["reservedPath"] = RECEIPT_NAME
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        record = case.invoke(
            "pull-reserved-receipt-path",
            ["pull", MODEL_TAG, "--backend", "hip"],
            endpoint=server.endpoint,
        )
        attempts = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if attempts:
        raise AssertionError(f"reserved install receipt path was rejected after HTTP: {attempts}")
    case.facts["httpAttempts"] = attempts


def case_huge_manifest_size_rejected_before_http(case: LifecycleCase) -> None:
    case.configure_package()
    huge_size = 2**62
    case.entries = [
        {**entry, "size": huge_size} if entry["path"] == "text.onnx" else entry
        for entry in case.entries
    ]
    fixture.create_model_list(case.config_path, case.repo_id, case.revision, case.entries)
    case.facts["textOnnxManifestSize"] = huge_size
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        record = case.invoke(
            "pull-insufficient-space-manifest",
            ["pull", MODEL_TAG, "--backend", "hip"],
            endpoint=server.endpoint,
        )
        attempts = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if attempts:
        raise AssertionError(f"huge manifest size was rejected after HTTP: {attempts}")
    case.facts["httpAttempts"] = attempts


def case_force_pull_with_user_file_preserves_it(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        user_file = case.install_path / "user-notes.txt"
        user_file.write_bytes(b"user-owned data must survive force pull\n")
        before = snapshot_tree(case.install_path)
        attempts_before = dict(server.handler.attempts)
        record = case.pull(server, force=True)
        attempts_after = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if snapshot_tree(case.install_path) != before:
        raise AssertionError("failed forced pull changed an install containing a user file")
    if attempts_after != attempts_before:
        raise AssertionError("force pull contacted HTTP before refusing an untracked user file")
    assert_no_transaction_dirs(case.install_root)
    case.facts["userFilePreserved"] = True
    case.facts["httpAttemptsBefore"] = attempts_before
    case.facts["httpAttemptsAfter"] = attempts_after


def case_vlm_unsupported_reason_rejected_before_http(case: LifecycleCase) -> None:
    case.configure_package()
    unsupported_reason = "fixture intentionally disables this VLM tag"
    fixture.create_model_list(
        case.config_path,
        case.repo_id,
        case.revision,
        case.entries,
        model_kind="vlm",
        unsupported_reason=unsupported_reason,
    )
    case.facts["modelKind"] = "vlm"
    case.facts["unsupportedReason"] = unsupported_reason
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        record = case.invoke(
            "pull-vlm-explicitly-unsupported",
            ["pull", MODEL_TAG, "--backend", "hip"],
            endpoint=server.endpoint,
        )
        attempts = dict(server.handler.attempts)
    assert_command(record, succeeds=False)
    if unsupported_reason not in (
        (case.root / record["stdoutFile"]).read_text(encoding="utf-8")
        + (case.root / record["stderrFile"]).read_text(encoding="utf-8")
    ):
        raise AssertionError("VLM pull did not report the tag's explicit unsupported_reason")
    if attempts:
        raise AssertionError(f"unsupported VLM tag triggered HTTP requests: {attempts}")
    case.facts["httpAttempts"] = attempts


def case_vlm_without_unsupported_reason_installs(case: LifecycleCase) -> None:
    case.configure_package()
    fixture.create_model_list(
        case.config_path,
        case.repo_id,
        case.revision,
        case.entries,
        model_kind="vlm",
    )
    case.facts["modelKind"] = "vlm"
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
        attempts = dict(server.handler.attempts)
    if not attempts:
        raise AssertionError("supported VLM catalog pull issued no localhost downloads")
    assert_installed_hashes(case)
    installed_receipt(case)
    assert_no_transaction_dirs(case.install_root)
    case.facts["httpAttempts"] = attempts
    case.facts["tinyPackageInstalled"] = True


def make_directory_link(link: Path, target: Path) -> str:
    """Create a symlink or, on Windows without symlink privilege, a junction."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except OSError:
        if os.name != "nt":
            raise
    completed = subprocess.run(
        f'mklink /J "{link}" "{target}"',
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        executable=os.environ.get("COMSPEC", "cmd.exe"),
    )
    if completed.returncode != 0 or not os.path.lexists(link):
        raise RuntimeError(
            "could not create a Windows junction for the escape case: "
            + (completed.stderr or completed.stdout).strip()
        )
    return "junction"


def case_linked_parent_remove_refused(case: LifecycleCase) -> None:
    case.configure_package(config_path="nested/test-oga")
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)

    linked_parent = case.install_root / "nested"
    outside_parent = case.root / "outside-parent"
    if not linked_parent.is_dir():
        raise AssertionError(f"expected installed parent directory: {linked_parent}")
    linked_parent.rename(outside_parent)
    link_kind = make_directory_link(linked_parent, outside_parent)
    outside_package = outside_parent / INSTALL_RELATIVE
    before = snapshot_tree(outside_package)
    case.facts["linkKind"] = link_kind
    case.facts["linkedParent"] = str(linked_parent.relative_to(case.root))
    record = case.invoke("remove-through-linked-parent", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(outside_package) != before:
        raise AssertionError("remove followed a symlink or junction and changed the outside package")
    if not os.path.lexists(linked_parent):
        raise AssertionError("refused remove removed the symlink or junction")
    case.facts["outsidePackagePreserved"] = True


def case_linked_parent_junction_remove_refused(case: LifecycleCase) -> None:
    if os.name != "nt":
        case.facts["skipReason"] = "Windows junctions require the Windows _winapi API"
        return
    import _winapi

    case.configure_package(config_path="nested/test-oga")
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)

    linked_parent = case.install_root / "nested"
    outside_parent = case.root / "outside-parent"
    if not linked_parent.is_dir():
        raise AssertionError(f"expected installed parent directory: {linked_parent}")
    linked_parent.rename(outside_parent)
    _winapi.CreateJunction(str(outside_parent), str(linked_parent))
    if not os.path.lexists(linked_parent):
        raise AssertionError("_winapi.CreateJunction did not create the parent junction")
    outside_package = outside_parent / INSTALL_RELATIVE
    before = snapshot_tree(outside_package)
    case.facts["linkKind"] = "junction"
    case.facts["linkedParent"] = str(linked_parent.relative_to(case.root))
    record = case.invoke("remove-through-parent-junction", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=False)
    if snapshot_tree(outside_package) != before:
        raise AssertionError("remove followed a Windows junction and changed the outside package")
    if not os.path.lexists(linked_parent):
        raise AssertionError("refused remove removed the Windows junction")
    case.facts["outsidePackagePreserved"] = True


def case_unsupported_tag_remove_allowed(case: LifecycleCase) -> None:
    case.configure_package()
    with fixture.ThreadedHttpFixture(case.public_root) as server:
        assert_command(case.pull(server), succeeds=True)
    before = snapshot_tree(case.install_path)
    data = json.loads(case.config_path.read_text(encoding="utf-8"))
    data["models"]["hip-test"]["1b"]["hip_oga"]["unsupported_reason"] = (
        "cataloged for a runtime not present on this host"
    )
    case.config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    case.facts["unsupportedReason"] = data["models"]["hip-test"]["1b"]["hip_oga"]["unsupported_reason"]
    record = case.invoke("remove-runtime-unsupported-tag", ["remove", MODEL_TAG, "--backend", "hip"])
    assert_command(record, succeeds=True)
    if case.install_path.exists():
        raise AssertionError("managed unsupported tag was not removed")
    case.facts["managedBytesBeforeRemove"] = before
    case.facts["removed"] = True


CASE_FUNCTIONS: tuple[tuple[str, Callable[[LifecycleCase], None]], ...] = (
    ("normal_lifecycle_native_sibling", case_normal_lifecycle),
    ("corrupt_check_preserves_file", case_corrupt_check_preserves_file),
    ("repeat_pull_repairs_corruption", case_repeat_pull_repairs_corruption),
    ("uppercase_hash_manifest_installs_and_checks", case_uppercase_hash_manifest_installs_and_checks),
    ("force_failure_preserves_old", case_force_failure_preserves_old),
    ("force_without_receipt_refused", case_force_without_receipt_refused),
    ("corrupt_without_receipt_pull_refused", case_corrupt_without_receipt_pull_refused),
    ("unknown_file_remove_refused", case_unknown_file_remove_refused),
    ("no_receipt_remove_refused", case_no_receipt_remove_refused),
    ("forged_receipt_revision_remove_refused", case_forged_receipt_revision_remove_refused),
    ("corrupt_payload_remove_allowed", case_corrupt_payload_remove_allowed),
    ("missing_root_remove_refused", case_missing_root_remove_refused),
    ("direct_path_remove_refused", case_direct_path_remove_refused),
    ("linked_parent_remove_refused", case_linked_parent_remove_refused),
    ("linked_parent_junction_remove_refused", case_linked_parent_junction_remove_refused),
    ("leaf_payload_symlink_remove_refused", case_leaf_payload_symlink_remove_refused),
    ("reserved_receipt_path_rejected_before_http", case_reserved_receipt_path_rejected_before_http),
    ("huge_manifest_size_rejected_before_http", case_huge_manifest_size_rejected_before_http),
    ("force_pull_user_file_preserved", case_force_pull_with_user_file_preserves_it),
    ("vlm_unsupported_reason_rejected_before_http", case_vlm_unsupported_reason_rejected_before_http),
    ("vlm_without_unsupported_reason_installs", case_vlm_without_unsupported_reason_installs),
    ("unsupported_tag_remove_allowed", case_unsupported_tag_remove_allowed),
)


def write_result(output: Path, cases: list[dict[str, Any]], *, complete: bool) -> None:
    summary = {
        "harness": "HIP OGA catalog lifecycle CLI regression",
        "binary": (output / "binary-path.txt").read_text(encoding="utf-8").strip(),
        "complete": complete,
        "passed": all(case["passed"] for case in cases) if complete else None,
        "caseCount": len(cases),
        "passCount": sum(bool(case["passed"]) and not case.get("skipped") for case in cases),
        "skipCount": sum(bool(case.get("skipped")) for case in cases),
        "failCount": sum(not bool(case["passed"]) for case in cases),
        "cases": cases,
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="existing flm.exe or flm binary")
    parser.add_argument("--output", required=True, type=Path, help="new directory for logs and case fixtures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = args.binary.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not binary.is_file():
        print(f"ERROR: binary does not exist: {binary}", file=sys.stderr)
        return 2
    if output.exists():
        print(f"ERROR: output directory must be new: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    (output / "binary-path.txt").write_text(str(binary) + "\n", encoding="utf-8")
    case_root = output / "cases"
    case_root.mkdir()
    results: list[dict[str, Any]] = []

    for name, run_case in CASE_FUNCTIONS:
        path = case_root / name
        path.mkdir()
        case = LifecycleCase(name, path, binary)
        started = time.time()
        entry: dict[str, Any] = {"name": name, "passed": False, "commands": case.commands}
        try:
            run_case(case)
            entry["passed"] = True
            entry["facts"] = case.facts
            if case.facts.get("skipReason"):
                entry["skipped"] = True
                print(f"SKIP {name}: {case.facts['skipReason']}")
            else:
                print(f"PASS {name}")
        except Exception as error:  # retain RED evidence and continue through all cases
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["facts"] = case.facts
            print(f"FAIL {name}: {entry['error']}", file=sys.stderr)
        entry["elapsedSeconds"] = round(time.time() - started, 3)
        case.write_evidence()
        evidence = json.loads((path / "evidence.json").read_text(encoding="utf-8"))
        evidence["passed"] = entry["passed"]
        evidence["skipped"] = entry.get("skipped", False)
        evidence["error"] = entry.get("error")
        (path / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        results.append(entry)
        write_result(output, results, complete=False)

    write_result(output, results, complete=True)
    failures = sum(not result["passed"] for result in results)
    skipped = sum(bool(result.get("skipped")) for result in results)
    print(
        f"Result: {len(results) - failures - skipped} passed, {skipped} skipped, "
        f"{failures} failed; evidence: {output / 'result.json'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
