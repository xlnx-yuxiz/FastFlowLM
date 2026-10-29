#!/usr/bin/env python3
"""Verify one-shot HIP OGA CLI failures clean their temporary model view.

The caller supplies an existing FLM binary and a known-failing local model.
The fixture creates a private linked copy, runs one real CLI request, records
all evidence under a new output directory, and removes only its owned models
tree after recording the result. It never builds FLM or modifies model assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any


VIEW_PREFIX = ".flm-hip-oga-config-"
OWNERSHIP_MARKER = ".hip-oga-cli-cleanup-fixture-owner"
TEMP_OWNERSHIP_MARKER = ".hip-oga-cli-cleanup-temp-owner"
PROMPT = "Say hello."
MAX_TOKENS = "8"
CONTEXT_LENGTH = "2048"


def is_symlink_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path) -> list[dict[str, Any]]:
    """Record stable metadata without rereading large model weight files."""
    result: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            directory = current_path / name
            if is_symlink_or_junction(directory):
                raise RuntimeError(f"model tree contains a symlink or junction directory: {directory}")
        for name in filenames:
            item = current_path / name
            if is_symlink_or_junction(item) or not item.is_file():
                raise RuntimeError(f"model tree contains a non-regular file: {item}")
            stat = item.stat()
            result.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtimeNs": stat.st_mtime_ns,
                    "mode": stat.st_mode,
                    "inode": stat.st_ino,
                }
            )
    result.sort(key=lambda entry: entry["path"])
    return result


def tree_contains_link_or_junction(root: Path) -> bool:
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if any(is_symlink_or_junction(current_path / name) for name in directories):
            return True
        if any(is_symlink_or_junction(current_path / name) for name in filenames):
            return True
    return False


def clone_linked_model(source: Path, destination: Path) -> None:
    """Hardlink model assets; copy the root config so OGA cannot alter source."""
    if destination.exists() or is_symlink_or_junction(destination):
        raise RuntimeError(f"fixture model destination is not fresh: {destination}")
    destination.mkdir(parents=True)
    config_seen = False
    for current, directories, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(source)
        target_dir = destination / relative_dir
        for name in list(directories):
            source_dir = current_path / name
            if is_symlink_or_junction(source_dir):
                raise RuntimeError(f"model tree contains a symlink or junction directory: {source_dir}")
            (target_dir / name).mkdir()
        for name in filenames:
            source_file = current_path / name
            if is_symlink_or_junction(source_file) or not source_file.is_file():
                raise RuntimeError(f"model tree contains a non-regular file: {source_file}")
            relative_file = source_file.relative_to(source)
            target_file = destination / relative_file
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if relative_file == Path("genai_config.json"):
                shutil.copy2(source_file, target_file)
                config_seen = True
            else:
                os.link(source_file, target_file)
    if not config_seen:
        raise RuntimeError(f"model has no root genai_config.json: {source}")


def verify_linked_model(source: Path, destination: Path) -> dict[str, Any]:
    source_files = {entry["path"]: entry for entry in file_inventory(source)}
    fixture_files = {entry["path"]: entry for entry in file_inventory(destination)}
    if source_files.keys() != fixture_files.keys():
        raise RuntimeError("fixture model file list does not match the source model")
    config_path = Path("genai_config.json").as_posix()
    linked_count = 0
    for relative, source_entry in source_files.items():
        fixture_entry = fixture_files[relative]
        if source_entry["size"] != fixture_entry["size"]:
            raise RuntimeError(f"fixture file size differs from source: {relative}")
        if relative == config_path:
            if source_entry["inode"] == fixture_entry["inode"]:
                raise RuntimeError("fixture genai_config.json must be an independent copy")
        else:
            if source_entry["inode"] != fixture_entry["inode"]:
                raise RuntimeError(f"fixture asset is not hardlinked to source: {relative}")
            linked_count += 1
    return {"hardlinkedAssetCount": linked_count, "rootConfigCopiedIndependently": True}


def view_directories(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(item.resolve())
        for item in root.iterdir()
        if item.name.startswith(VIEW_PREFIX)
        and (item.is_dir() or is_symlink_or_junction(item))
    )


def paths_under(root: Path) -> list[str]:
    if not root.exists():
        return []
    result: list[str] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            item = current_path / name
            result.append(item.relative_to(root).as_posix())
            if is_symlink_or_junction(item):
                directories.remove(name)
        for name in filenames:
            result.append((current_path / name).relative_to(root).as_posix())
    return sorted(result)


def decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    flm = args.flm.expanduser().resolve()
    source = args.model.expanduser().resolve()
    output = args.output.expanduser().absolute()
    if not flm.is_file():
        print(f"ERROR: FLM binary does not exist: {flm}", file=sys.stderr)
        return 2
    if not source.is_dir():
        print(f"ERROR: model directory does not exist: {source}", file=sys.stderr)
        return 2
    if output.exists() or is_symlink_or_junction(output):
        print(f"ERROR: output directory must be new: {output}", file=sys.stderr)
        return 2
    output_resolved = output.resolve()
    if output_resolved == source or source in output_resolved.parents:
        print("ERROR: output must be outside the shared source model tree", file=sys.stderr)
        return 2

    output.mkdir(parents=True)
    models_root = output / "models"
    fixture_model = models_root / "source"
    marker_value = secrets.token_hex(16)
    temp_marker_value = secrets.token_hex(16)
    receipt_path = output / "result.json"
    stdout_path = output / "flm.stdout.log"
    stderr_path = output / "flm.stderr.log"
    temp_root = output / "temp"
    receipt: dict[str, Any] = {
        "status": "FAIL",
        "binary": str(flm),
        "flmSha256": None,
        "model": str(source),
        "fixtureModel": str(fixture_model),
        "expectedError": args.expected_error,
        "timeoutSeconds": args.timeout,
        "argv": [],
        "exitCode": None,
        "timedOut": False,
        "stdoutFile": stdout_path.name,
        "stderrFile": stderr_path.name,
        "failures": [],
    }
    owned_tree_created = False
    try:
        receipt["flmSha256"] = sha256_file(flm)
        receipt["sourceFilesBefore"] = file_inventory(source)
        models_root.mkdir()
        (models_root / OWNERSHIP_MARKER).write_text(marker_value + "\n", encoding="ascii")
        owned_tree_created = True
        temp_root.mkdir()
        (temp_root / TEMP_OWNERSHIP_MARKER).write_text(
            temp_marker_value + "\n", encoding="ascii"
        )
        receipt["ownedTempDirectory"] = str(temp_root)
        clone_linked_model(source, fixture_model)
        receipt["fixtureIsolation"] = verify_linked_model(source, fixture_model)
        receipt["fixtureFilesBefore"] = file_inventory(fixture_model)
        source_views_before = view_directories(models_root)
        temp_views_before = view_directories(temp_root)
        receipt["modelSiblingViewsBefore"] = source_views_before
        receipt["controlledTempViewsBefore"] = temp_views_before
        if source_views_before:
            raise RuntimeError(f"fresh fixture already has HIP OGA views: {source_views_before}")

        argv = [
            str(flm),
            "run",
            str(fixture_model),
            "--backend",
            "hip",
            "--ctx-len",
            CONTEXT_LENGTH,
            "--hip-oga-text-codec",
            "oga",
            "--prompt",
            PROMPT,
            "--max-tokens",
            MAX_TOKENS,
        ]
        receipt["argv"] = argv
        child_env = os.environ.copy()
        for key in ("TEMP", "TMP", "TMPDIR"):
            child_env[key] = str(temp_root.resolve())
        receipt["temporaryEnvironment"] = {
            key: child_env[key] for key in ("TEMP", "TMP", "TMPDIR")
        }
        try:
            completed = subprocess.run(
                argv,
                env=child_env,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
            )
            receipt["exitCode"] = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            receipt["timedOut"] = True
            stdout = decode_output(error.stdout)
            stderr = decode_output(error.stderr)
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

        receipt["fixtureFilesAfter"] = file_inventory(fixture_model)
        receipt["sourceFilesAfter"] = file_inventory(source)
        receipt["modelSiblingViewsAfter"] = view_directories(models_root)
        receipt["controlledTempViewsAfter"] = view_directories(temp_root)
        receipt["modelsTreePathsAfterRun"] = paths_under(models_root)
        if receipt["timedOut"]:
            receipt["failures"].append(f"CLI timed out after {args.timeout} seconds")
        if receipt["exitCode"] != 1:
            receipt["failures"].append(
                f"CLI was expected to exit 1 normally, got {receipt['exitCode']}"
            )
        combined_output = stdout + "\n" + stderr
        if args.expected_error not in combined_output:
            receipt["failures"].append(
                f"CLI output did not contain expected error text: {args.expected_error!r}"
            )
        if receipt["modelSiblingViewsAfter"]:
            receipt["failures"].append(
                "HIP OGA sibling config directories remain after CLI exit: "
                + repr(receipt["modelSiblingViewsAfter"])
            )
        new_temp_views = sorted(
            set(receipt["controlledTempViewsAfter"]) - set(temp_views_before)
        )
        if new_temp_views:
            receipt["failures"].append(
                "new HIP OGA config directories remain in controlled temp: "
                + repr(new_temp_views)
            )
        if receipt["sourceFilesAfter"] != receipt["sourceFilesBefore"]:
            receipt["failures"].append("shared source model file metadata changed during CLI run")
        receipt["status"] = "PASS" if not receipt["failures"] else "FAIL"
    except Exception as error:
        receipt["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        # Persist diagnostic state before deleting the single newly created tree.
        write_receipt(receipt_path, receipt)
        receipt["outputPathsBeforeCleanup"] = paths_under(output)
        try:
            if owned_tree_created:
                marker = models_root / OWNERSHIP_MARKER
                if (
                    not is_symlink_or_junction(models_root)
                    and models_root.resolve() == (output / "models").resolve()
                    and not tree_contains_link_or_junction(models_root)
                    and marker.is_file()
                    and marker.read_text(encoding="ascii").strip() == marker_value
                ):
                    shutil.rmtree(models_root)
                    receipt["ownedModelsTreeCleaned"] = True
                else:
                    receipt["ownedModelsTreeCleaned"] = False
                    receipt["failures"].append(
                        "refused cleanup because the fresh fixture ownership marker changed"
                    )
            temp_marker = temp_root / TEMP_OWNERSHIP_MARKER
            if (
                temp_root.exists()
                and not is_symlink_or_junction(temp_root)
                and temp_root.resolve() == (output / "temp").resolve()
                and not tree_contains_link_or_junction(temp_root)
                and temp_marker.is_file()
                and temp_marker.read_text(encoding="ascii").strip() == temp_marker_value
            ):
                shutil.rmtree(temp_root)
                receipt["ownedTempDirectoryCleaned"] = True
            elif temp_root.exists():
                receipt["ownedTempDirectoryCleaned"] = False
                receipt["failures"].append(
                    "refused temp cleanup because the fresh fixture ownership marker changed"
                )
        except Exception as error:
            receipt["cleanupError"] = f"{type(error).__name__}: {error}"
            receipt["failures"].append(f"fixture cleanup failed: {type(error).__name__}: {error}")
        receipt["outputPathsAfterCleanup"] = paths_under(output)
        if receipt["failures"]:
            receipt["status"] = "FAIL"
        write_receipt(receipt_path, receipt)

    print(f"{receipt['status']}: HIP OGA CLI cleanup receipt: {receipt_path}")
    return 0 if receipt["status"] == "PASS" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flm", required=True, type=Path, help="existing FLM CLI binary")
    parser.add_argument("--model", required=True, type=Path, help="known-failing local model directory")
    parser.add_argument("--expected-error", required=True, help="error text expected in CLI output")
    parser.add_argument("--output", required=True, type=Path, help="new directory for logs and receipt")
    parser.add_argument("--timeout", required=True, type=float, help="CLI timeout in seconds")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
