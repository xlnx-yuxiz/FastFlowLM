#!/usr/bin/env python3
"""Check validation and model discovery using an existing FLM binary and tiny fixtures."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import test_hip_oga_public_pull as fixture


def invoke(binary, env, *args):
    return subprocess.run([str(binary), *args], env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=30)


def check_validation(binary, env, _root):
    result = invoke(binary, env, "validate", "--backend", "hip", "--json")
    status = json.loads(result.stdout)
    assert result.returncode == 2, (result.returncode, status)
    assert status["status"] == "not_checked" and status["ready"] is None, status


def check_native_validation(binary, env, _root):
    result = invoke(binary, env, "validate", "--backend", "flm", "--json")
    status = json.loads(result.stdout)
    assert isinstance(status["ready"], bool), status
    assert status.get("status") != "not_checked", status


def check_cache_options(binary, env, _root):
    for option, value in [("--prefix-slots", "2"), ("--prefix-cache-mb", "64"),
                          ("--prefix-min-gain", "128")]:
        result = invoke(binary, env, "validate", "--backend", "hip", option, value)
        assert result.returncode == 1 and option in result.stdout + result.stderr, (
            option, result.returncode, result.stdout, result.stderr)


def check_discovery(binary, env, root):
    public = root / "hf"
    repo, revision, entries = fixture.create_public_package(public)
    config = root / "catalog.json"
    fixture.create_model_list(config, repo, revision, entries, config_path="test-oga")
    doc = json.loads(config.read_text())
    native = copy.deepcopy(doc["models"]["hip-test"]["1b"])
    native.pop("hip_oga")
    native["name"] = "NativeOnly-1B"
    doc["models"]["native-only"] = {"1b": native}
    config.write_text(json.dumps(doc))
    env = dict(env, FLM_CONFIG_PATH=str(config), FLM_MODEL_PATH=str(root / "models"),
               FLM_HIP_OGA_MODEL_ROOT=str(root / "installed"))
    with fixture.ThreadedHttpFixture(public) as server:
        env["FLM_HIP_OGA_HF_ENDPOINT"] = server.endpoint
        pulled = invoke(binary, env, "pull", "hip-test:1b", "--backend", "hip")
        assert pulled.returncode == 0, pulled.stdout + pulled.stderr

    port = fixture.free_port()
    with (root / "server.log").open("wb") as log:
        process = subprocess.Popen([str(binary), "serve", "model-faker", "--backend", "hip",
                                    "--port", str(port)], env=env, stdout=log, stderr=log)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 20
            while True:
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                        models = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise AssertionError("Server did not become ready: " +
                                             (root / "server.log").read_text(errors="replace"))
                    time.sleep(0.1)
            assert {m["id"] for m in models["data"]} == {"hip-test:1b"}, models
            assert all(m["backend"] == "hip" for m in models["data"]), models
            with urllib.request.urlopen(base + "/api/tags", timeout=2) as response:
                tags = json.load(response)
            assert {m["model"] for m in tags["models"]} == {"hip-test:1b"}, tags
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    failed = []
    with tempfile.TemporaryDirectory(prefix="flm-cli-contract-") as tmp:
        root = Path(tmp)
        env = dict(os.environ)
        env.pop("FLM_BACKEND", None)
        for name, check in [("unchecked-validation", check_validation),
                            ("native-validation-preserved", check_native_validation),
                            ("hip-model-discovery", check_discovery),
                            ("unsupported-cache-options", check_cache_options)]:
            try:
                check(args.binary.resolve(), env, root)
                print("PASS:", name, flush=True)
            except Exception as error:
                failed.append(name)
                print("FAIL:", name, str(error), flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
