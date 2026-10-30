#!/usr/bin/env python3
"""Local HTTP fixture for public AMD Hugging Face HIP OGA catalog installation."""

from __future__ import annotations

import hashlib
import http.server
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
from typing import Callable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[3]
TEST_CPP = ROOT / "src/test/hip_oga_backend_static/test_hip_oga_public_pull.cpp"
PUBLIC_DOWNLOADER_CPP = ROOT / "src/pull/hip_oga_public_downloader.cpp"
OVERLAY_CPP = ROOT / "src/test/hip_oga_backend_static/test_hip_oga_provider_overlay.cpp"

REVISION = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SOURCE_SUBDIR = "Upload Qwen2.5 ONNX files"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PublicHfHandler(http.server.SimpleHTTPRequestHandler):
    fail_once_paths: set[str] = set()
    corrupt_once_paths: set[str] = set()
    attempts: dict[str, int] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        self.attempts[path] = self.attempts.get(path, 0) + 1
        if path in self.fail_once_paths and self.attempts[path] == 1:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"retry")
            return
        if path in self.corrupt_once_paths and self.attempts[path] == 1:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(b"corrupt first response")
            return
        super().do_GET()


class ThreadedHttpFixture:
    def __init__(
        self,
        root: Path,
        fail_once_paths: set[str] | None = None,
        corrupt_once_paths: set[str] | None = None,
    ) -> None:
        self.root = root
        self.port = free_port()

        class Handler(PublicHfHandler):
            pass

        Handler.fail_once_paths = fail_once_paths or set()
        Handler.corrupt_once_paths = corrupt_once_paths or set()
        Handler.attempts = {}
        self.handler = Handler

        handler = lambda *args, **kwargs: Handler(*args, directory=str(root), **kwargs)
        self.httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self.port), handler
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ThreadedHttpFixture":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


DEFAULT_FILES: dict[str, bytes] = {
    "genai_config.json": b'{"model":{"decoder":{},"context_length":128,"vocab_size":16,"eos_token_id":2}}',
    "tokenizer.json": b"{}",
    "tokenizer_config.json": (
        b'{"eos_token":"</s>","eos_token_id":2,'
        b'"chat_template":"{{ messages[0][\'content\'] }}"}'
    ),
    "text.onnx": b"fake onnx payload",
}


def resolve_path(repo_id: str, revision: str, relative_path: str) -> str:
    return f"/{repo_id}/resolve/{revision}/{SOURCE_SUBDIR}/{relative_path}"


def create_public_package(
    root: Path,
    *,
    repo_id: str = "amd/test-oga",
    revision: str = REVISION,
    files: dict[str, bytes] | None = None,
) -> tuple[str, str, list[dict[str, object]]]:
    package_root = root / repo_id / "resolve" / revision / SOURCE_SUBDIR
    package_files = dict(DEFAULT_FILES if files is None else files)
    for relative_path, data in package_files.items():
        write_file(package_root / relative_path, data)

    entries = [
        {
            "path": relative_path,
            "sha256": sha256_bytes(data),
            "size": len(data),
        }
        for relative_path, data in sorted(package_files.items())
    ]
    return repo_id, revision, entries


def create_model_list(
    path: Path,
    repo_id: str,
    revision: str,
    entries: list[dict[str, object]],
    *,
    config_path: str = "test-oga",
    source_subdir: str | None = SOURCE_SUBDIR,
    license: str = "Apache-2.0",
    model_kind: str = "text",
    endpoint_env: str = "FLM_HIP_OGA_HF_ENDPOINT",
    include_public_hf: bool = True,
    unsupported_reason: str = "",
) -> None:
    hip_oga: dict[str, object] = {
        "provider": "AMDGPU",
        "profile": "hip",
        "config_path": config_path,
        "required_files": [entry["path"] for entry in entries],
        "provider_options": {},
    }
    if unsupported_reason:
        hip_oga["unsupported_reason"] = unsupported_reason
    if include_public_hf:
        public_hf: dict[str, object] = {
            "endpoint_env": endpoint_env,
            "repo_id": repo_id,
            "revision": revision,
            "license": license,
            "model_kind": model_kind,
            "files": entries,
        }
        if source_subdir is not None:
            public_hf["source_subdir"] = source_subdir
        hip_oga["public_hf"] = public_hf

    config = {
        "model_path": "models",
        "models": {
            "hip-test": {
                "1b": {
                    "name": "HipTest-1B-OGA",
                    "files": ["config.json", "model.q4nx"],
                    "default_context_length": 4096,
                    "max_prefill_len": 4096,
                    "details": {
                        "family": "llama3",
                        "parameter_size": "1B",
                        "quantization_level": "Q4_1",
                    },
                    "hip_oga": hip_oga,
                }
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def compile_test_binary(output: Path) -> tuple[Path, Path]:
    public_binary = output / "test_hip_oga_public_pull"
    cmd = [
        "g++",
        "-std=c++20",
        "-DCMAKE_INSTALL_PREFIX=\"/nonexistent\"",
        "-DCMAKE_XCLBIN_PREFIX=\"/nonexistent\"",
        "-D__FLM_VERSION__=\"0.9.24\"",
        "-D__NPU_VERSION__=\"0.0.0\"",
        "-I",
        str(ROOT / "src/include"),
        "-I",
        str(ROOT / "src/pull"),
        str(TEST_CPP),
        str(PUBLIC_DOWNLOADER_CPP),
        "-lcurl",
        "-pthread",
        "-o",
        str(public_binary),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    overlay_binary = output / "test_hip_oga_provider_overlay"
    cmd = [
        "g++",
        "-std=c++20",
        "-I",
        str(ROOT / "src/include"),
        str(OVERLAY_CPP),
        "-o",
        str(overlay_binary),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return public_binary, overlay_binary


def run_pull(
    binary: Path,
    model_list_path: Path,
    tmp: Path,
    endpoint: str,
    install_root: Path,
    expected: str,
    *,
    force: bool = False,
    default_root: bool = False,
    ensure: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(binary),
        str(model_list_path),
        str(tmp),
        endpoint,
        str(install_root),
        expected,
    ]
    if force:
        cmd.append("force")
    if default_root:
        cmd.append("default-root")
    if ensure:
        cmd.append("ensure")
    completed = subprocess.run(
        cmd, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def assert_no_temp_dirs(install_root: Path) -> None:
    if not install_root.exists():
        return
    leftovers = [
        path
        for path in install_root.iterdir()
        if path.name.startswith(".tmp-") or path.name.startswith(".backup-")
    ]
    assert not leftovers, f"unexpected temp dirs: {leftovers}"


def run_case(
    tmp: Path,
    name: str,
    binary: Path,
    setup: Callable[[Path, Path, Path], tuple[set[str], set[str], str, bool]],
) -> subprocess.CompletedProcess[str]:
    case_root = tmp / name
    public_root = case_root / "hf"
    config_dir = case_root / "config"
    default_root = name in {"default_model_root", "ensure_default_model_root"}
    install_root = case_root / ("models" if default_root else "install")
    fail_once, corrupt_once, expected, force = setup(
        public_root, config_dir, install_root
    )
    with ThreadedHttpFixture(public_root, fail_once, corrupt_once) as server:
        endpoint = server.endpoint
        if name == "endpoint_userinfo_rejected":
            endpoint = endpoint.replace("http://", "http://user:pass@")
        completed = run_pull(
            binary,
            config_dir / "model_list.json",
            case_root,
            endpoint,
            install_root,
            expected,
            force=force,
            default_root=default_root,
            ensure=name == "ensure_default_model_root",
        )
    if name == "endpoint_userinfo_rejected":
        output = completed.stdout + completed.stderr
        assert "user:pass" not in output
    assert_no_temp_dirs(install_root)
    return completed


def setup_success(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "success", False


def setup_retry_success(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return (
        {resolve_path(repo_id, revision, "genai_config.json")},
        {resolve_path(repo_id, revision, "text.onnx")},
        "success",
        False,
    )


def setup_checksum_mismatch(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    entries = [
        {**entry, "sha256": sha256_bytes(b"wrong")}
        if entry["path"] == "text.onnx"
        else entry
        for entry in entries
    ]
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_non_amd_repo(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(
        public_root, repo_id="FastFlowLM/test-oga"
    )
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_missing_revision(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root, revision="main")
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_unknown_license(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json", repo_id, revision, entries, license="unknown"
    )
    return set(), set(), "failure", False


def setup_credential_metadata_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    data = json.loads((config_dir / "model_list.json").read_text(encoding="utf-8"))
    data["models"]["hip-test"]["1b"]["hip_oga"]["public_hf"]["token_env"] = "HF_TOKEN"
    (config_dir / "model_list.json").write_text(json.dumps(data), encoding="utf-8")
    return set(), set(), "failure", False


def setup_credential_endpoint_env_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        endpoint_env="AWS_SECRET_ACCESS_KEY",
    )
    return set(), set(), "failure", False


def setup_prefixed_credential_endpoint_env_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        endpoint_env="FLM_HIP_OGA_SECRET_ENDPOINT",
    )
    return set(), set(), "failure", False


def setup_endpoint_userinfo_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_embedded_endpoint_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    data = json.loads((config_dir / "model_list.json").read_text(encoding="utf-8"))
    data["models"]["hip-test"]["1b"]["hip_oga"]["public_hf"]["endpoint"] = (
        "https://private.example.invalid"
    )
    (config_dir / "model_list.json").write_text(json.dumps(data), encoding="utf-8")
    return set(), set(), "failure", False


def setup_absolute_config_path(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        config_path=str(install_root.parent / "outside" / "abs-oga"),
    )
    return set(), set(), "failure", False


def setup_parent_traversal_file_path(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    entries = [dict(entry) for entry in entries]
    entries[0]["path"] = "../genai_config.json"
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_empty_path_segment(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    entries = [dict(entry) for entry in entries]
    entries[0]["path"] = "nested//genai_config.json"
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_case_collision(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    files = dict(DEFAULT_FILES)
    files["Text.onnx"] = DEFAULT_FILES["text.onnx"]
    repo_id, revision, entries = create_public_package(public_root, files=files)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", False


def setup_symlink_config_path_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    outside_root = install_root.parent / "outside"
    outside_root.mkdir(parents=True, exist_ok=True)
    install_root.mkdir(parents=True, exist_ok=True)
    (install_root / "escape").symlink_to(outside_root, target_is_directory=True)

    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        config_path="escape/test-oga",
    )
    return set(), set(), "failure", False


def setup_missing_public_metadata(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        include_public_hf=False,
    )
    return set(), set(), "failure", False


def setup_missing_required_checksum_entry(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    visible_entries = [entry for entry in entries if entry["path"] != "tokenizer.json"]
    create_model_list(config_dir / "model_list.json", repo_id, revision, visible_entries)
    data = json.loads((config_dir / "model_list.json").read_text(encoding="utf-8"))
    data["models"]["hip-test"]["1b"]["hip_oga"]["required_files"].append(
        "tokenizer.json"
    )
    (config_dir / "model_list.json").write_text(json.dumps(data), encoding="utf-8")
    return set(), set(), "failure", False


def setup_unsupported_vlm_rejected(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    repo_id, revision, entries = create_public_package(public_root)
    create_model_list(
        config_dir / "model_list.json",
        repo_id,
        revision,
        entries,
        model_kind="vlm",
        unsupported_reason="HIP OGA VLM runtime path is not implemented yet",
    )
    return set(), set(), "failure", False


def setup_force_failure_preserves_old(
    public_root: Path, config_dir: Path, install_root: Path
) -> tuple[set[str], set[str], str, bool]:
    old_model = install_root / "test-oga"
    for path, data in DEFAULT_FILES.items():
        write_file(old_model / path, data)
    write_file(old_model / "old-marker.txt", b"keep")

    files = dict(DEFAULT_FILES)
    del files["tokenizer_config.json"]
    repo_id, revision, entries = create_public_package(public_root, files=files)
    create_model_list(config_dir / "model_list.json", repo_id, revision, entries)
    return set(), set(), "failure", True


def assert_force_failure_preserved_old(tmp: Path) -> None:
    marker = (
        tmp
        / "force_failure_preserves_old"
        / "install"
        / "test-oga"
        / "old-marker.txt"
    )
    assert marker.read_bytes() == b"keep"


def assert_symlink_case_did_not_escape(tmp: Path) -> None:
    outside_install = (
        tmp
        / "symlink_config_path_rejected"
        / "outside"
        / "test-oga"
    )
    assert not outside_install.exists()


def main() -> None:
    scratch_parent = Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(
        prefix="flm-hip-oga-public-", dir=scratch_parent
    ) as tmp_name:
        tmp = Path(tmp_name)
        public_binary, overlay_binary = compile_test_binary(tmp)
        subprocess.run([str(overlay_binary)], cwd=ROOT, check=True)

        cases = [
            ("success", setup_success),
            ("default_model_root", setup_success),
            ("ensure_default_model_root", setup_success),
            ("retry_success", setup_retry_success),
            ("checksum_mismatch_cleanup", setup_checksum_mismatch),
            ("non_amd_repo", setup_non_amd_repo),
            ("missing_revision", setup_missing_revision),
            ("unknown_license", setup_unknown_license),
            ("credential_metadata_rejected", setup_credential_metadata_rejected),
            ("credential_endpoint_env_rejected", setup_credential_endpoint_env_rejected),
            (
                "prefixed_credential_endpoint_env_rejected",
                setup_prefixed_credential_endpoint_env_rejected,
            ),
            ("endpoint_userinfo_rejected", setup_endpoint_userinfo_rejected),
            ("embedded_endpoint_rejected", setup_embedded_endpoint_rejected),
            ("absolute_config_path", setup_absolute_config_path),
            ("parent_traversal_file_path", setup_parent_traversal_file_path),
            ("empty_path_segment", setup_empty_path_segment),
            ("case_collision", setup_case_collision),
            ("symlink_config_path_rejected", setup_symlink_config_path_rejected),
            ("missing_public_metadata", setup_missing_public_metadata),
            ("missing_required_checksum_entry", setup_missing_required_checksum_entry),
            ("unsupported_vlm_rejected", setup_unsupported_vlm_rejected),
            ("force_failure_preserves_old", setup_force_failure_preserves_old),
        ]
        for name, setup in cases:
            run_case(tmp, name, public_binary, setup)
        assert_force_failure_preserved_old(tmp)
        assert_symlink_case_did_not_escape(tmp)

    print("PASS: HIP OGA public Hugging Face pull fixture")


if __name__ == "__main__":
    main()
