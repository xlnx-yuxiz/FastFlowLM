#!/usr/bin/env python3
"""Static checks for the prototype HIP OGA backend route.

This intentionally avoids loading OGA or touching AMD hardware. It verifies that
the opt-in backend is discoverable, that CLI/server routes can select it, and
that the HIP adapter source does not open the FastFlowLM NPU runtime path.
"""

import json
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def read_required(relative: str, message: str) -> str:
    path = ROOT / relative
    require(path.exists(), message)
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


cmake = read("src/CMakeLists.txt")
program_args = read("src/include/program_args.hpp")
vm_args = read("src/include/utils/vm_args.hpp")
all_models = read("src/include/AutoModel/all_models.hpp")
automodel_hpp = read("src/include/AutoModel/automodel.hpp")
model_list_hpp = read("src/include/model_list.hpp")
model_list_json = json.loads(read("src/model_list.json"))
runner = read("src/runner/runner.cpp")
rest_handler = read("src/server/rest_handler.cpp")
streaming_openai = read("src/server/streaming_ostream_openai.hpp")
main_cpp = read("src/src/main.cpp")
server = read("src/server/server.cpp")
model_downloader = read("src/pull/model_downloader.cpp")
adapter = read("src/common/AutoModel/hip_oga_model.cpp")
hip_tool_parser = read("src/include/AutoModel/hip_oga_tool_parser.hpp")
prefix_cache_hpp = read("src/include/AutoModel/hip_oga_prefix_cache.hpp")
hip_oga_docs = read("docs/docs/instructions/hip-oga.md")
model_list_test = read("src/test/hip_oga_backend_static/test_hip_oga_model_list.cpp")
public_pull_test = read("src/test/hip_oga_backend_static/test_hip_oga_public_pull.py")
tool_parser_fixture = read("src/test/hip_oga_backend_static/test_hip_oga_openai_tool_parser.py")
prefix_cache_test = read("src/test/hip_oga_backend_static/test_hip_oga_prefix_cache.cpp")
text_codec_header = read("src/include/AutoModel/hip_oga_text_codec.hpp")
text_codec_test = read("src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp")
text_codec_validator = read("src/test/hip_oga_backend_static/validate_hip_oga_text_codec_contract.py")
text_codec_schema = json.loads(read("src/test/hip_oga_backend_static/hip_oga_text_codec_evidence.schema.json"))
text_codec_fixture = json.loads(read("src/test/hip_oga_backend_static/fixtures/hip_oga_text_codec_deterministic_fixture.json"))
external_tokenizer_header = read("src/include/AutoModel/hip_oga_external_tokenizer.hpp")
external_tokenizer_source = read("src/common/AutoModel/hip_oga_external_tokenizer.cpp")
external_tokenizer_test = read("src/test/hip_oga_backend_static/test_hip_oga_external_tokenizer.cpp")
tokenizer_parity_tool = read("src/test/hip_oga_external_codec/tokenizer_parity.py")
tokenizer_text_cases = json.loads(read("src/test/hip_oga_external_codec/fixtures/text_cases.json"))
tool_parser_cpp = ROOT / "src/test/hip_oga_backend_static/test_hip_oga_tool_parser.cpp"
bootstrap_script = read_required(
    "scripts/bootstrap-fastflowlm-deps.sh",
    "missing bootstrap script for pinned third-party dependencies")
runtime_check = read_required(
    "scripts/check-hip-oga-runtime.ps1",
    "missing HIP OGA runtime provenance preflight script")

require("FLM_ENABLE_HIP_OGA" in cmake, "missing FLM_ENABLE_HIP_OGA CMake option")
require("ONNXRUNTIME_GENAI_HOME" in cmake, "missing OGA discovery through ONNXRUNTIME_GENAI_HOME")
require("OGA_HOME" in cmake, "missing OGA discovery through OGA_HOME")
require("ORT_GENAI_RUNTIME_DIRS" in cmake,
        "CMake does not track OGA runtime DLL directories")
for runtime_dll in ("onnxruntime-genai.dll", "onnxruntime.dll",
                    "onnxruntime_providers_shared.dll", "amdgpu-ep.dll",
                    "hip-backend.dll"):
    require(runtime_dll in cmake,
            f"CMake does not stage required HIP OGA runtime DLL: {runtime_dll}")
require("add_executable(flm ${SOURCES} ${HEADERS})" in cmake,
        "normal FastFlowLM target must build flm from the full source list")
require("add_executable(flm src/hip_oga_main.cpp)" not in cmake,
        "FLM_HIP_OGA_ONLY must not replace flm with the run-only HIP diagnostic")
require('message(FATAL_ERROR "FLM_HIP_OGA_ONLY was removed' in cmake,
        "CMake must fail old FLM_HIP_OGA_ONLY=ON builds instead of producing a run-only flm.exe")
require('list(FILTER SOURCES EXCLUDE REGEX ".*/hip_oga_main\\\\.cpp$")' in cmake,
        "normal flm target must exclude the HIP-only smoke entry point")
for source_name, source in (
    ("automodel.hpp", automodel_hpp),
    ("hip_oga_tool_parser.hpp", hip_tool_parser),
    ("hip_oga_model.cpp", adapter),
    ("streaming_ostream_openai.hpp", streaming_openai),
):
    require(
        not re.search(r"\bStreamEventType::ERROR\b", source)
        and not re.search(r"(^|\n)\s*ERROR,", source),
        f"{source_name} must not use ERROR as a StreamEventType name because Windows headers define ERROR as a macro")
require("ERROR_EVENT" in automodel_hpp and "ERROR_EVENT" in hip_tool_parser,
        "Stream error events must use the Windows-safe ERROR_EVENT enum name")
require("FLM_BOOST_PROGRAM_OPTIONS_LIBRARY" in cmake,
        "Windows non-vcpkg build must discover the installed Boost program_options import lib")
require("libboost_program_options-vc143-mt-x64-1_88" not in cmake,
        "Windows non-vcpkg build must not hard-code the old Boost 1.88/vc143 import lib")
require("std::string backend" in program_args, "program_args_t does not carry backend")
require("int max_tokens" in program_args, "program_args_t does not carry max_tokens")
require("(\"backend\"" in vm_args, "CLI parser does not expose --backend")
require("(\"max-tokens\"" in vm_args, "CLI parser does not expose --max-tokens")
require('parsed_args.command = "help";' in vm_args and
        'parsed_args.command = "version";' in vm_args,
        "CLI parser must preserve help/version as successful commands")
require('if (parsed_args.command == "help")' in main_cpp and
        'if (parsed_args.command == "version")' in main_cpp and
        'if (parsed_args.command == "port")' in main_cpp,
        "main must handle help/version/port as successful early exits")
require("FLM_BACKEND" in read("src/include/backend_options.hpp"), "FLM_BACKEND env override is not wired")
require("enum class TextCodecProvider" in text_codec_header,
        "missing shared text-codec provider enum")
require("TextCodecProvider::Oga" in text_codec_header and
        "TextCodecProvider::External" in text_codec_header,
        "text-codec provider enum must name OGA and external providers")
require("enum class ParityStatus" in text_codec_header and
        "struct TextCodecParityResult" in text_codec_header,
        "missing shared parity result types")
require("enum class DecoderEquivalenceClass" in text_codec_header and
        "ByteExact" in text_codec_header and
        "CanonicalEquivalent" in text_codec_header and
        "StructuredEquivalent" in text_codec_header and
        "Incompatible" in text_codec_header,
        "missing explicit decoder equivalence classes")
require("classify_decoder_text_equivalence" in text_codec_header and
        "gemma_standalone_metaspace_is_space" in text_codec_header,
        "missing model-aware decoder canonical-equivalence classifier")
require("decoder_policy_from_tokenizer_contract" in text_codec_header and
        "decoder_step_replaces_metaspace_with_space" in text_codec_header and
        '"Replace"' in text_codec_header and
        '"String"' in text_codec_header,
        "Gemma decoder policy must detect tokenizer.json Replace metaspace contracts")
require("parse_text_codec_selector" in text_codec_header,
        "missing text-codec selector parser")
require("default_text_codec_selector" in text_codec_header,
        "missing text-codec default selector")
require("ExternalCumulativeDetokenizer" in text_codec_header and
        "CumulativeDecodeCallback" in text_codec_header,
        "missing callback-based external cumulative detokenizer")
require("valid_utf8_prefix_length" in text_codec_header,
        "external detokenizer must guard UTF-8 prefix emission")
require('parse_text_codec_selector("")' in text_codec_test and
        "TextCodecProvider::Oga" in text_codec_test,
        "selector test must prove empty selector defaults to OGA")
require("tokenizers-cpp" in text_codec_test,
        "selector test must reject unsupported external selector names")
require("TextCodecParityResult result" in text_codec_test and
        "ParityStatus::NotRun" in text_codec_test,
        "selector test must cover parity result defaults")
require("test_external_cumulative_decoder_delays_until_prefix_is_stable" in text_codec_test and
        "test_external_cumulative_decoder_may_lag_immediate_oga_stream" in text_codec_test and
        "test_external_cumulative_decoder_holds_split_utf8_until_final_flush" in text_codec_test and
        "test_external_cumulative_decoder_fails_closed_when_emitted_prefix_is_revised" in text_codec_test,
        "detokenizer tests must cover stable-prefix lag, UTF-8, and revision behavior")
require("test_decoder_equivalence_accepts_gemma_standalone_metaspace_as_space" in text_codec_test and
        "test_gemma_policy_detects_replace_decoder_contract" in text_codec_test and
        "test_decoder_equivalence_does_not_globally_normalize_without_contract" in text_codec_test and
        "test_decoder_equivalence_keeps_structured_json_strict" in text_codec_test,
        "decoder-equivalence tests must cover Gemma canonicalization and strict structured semantics")
require(text_codec_schema["properties"]["default_provider"]["const"] == "oga",
        "text-codec evidence schema must keep default_provider=oga")
require("detokenizer" in text_codec_schema["required"] and
        text_codec_schema["properties"]["detokenizer"]["properties"]["mode"]["const"] == "callback_cumulative",
        "text-codec evidence schema must expose callback-cumulative detokenizer parity")
require(text_codec_fixture["default_provider"] == "oga" and
        text_codec_fixture["codec_selector"] == "oga",
        "deterministic fixture must keep OGA as the default codec selector")
require(text_codec_fixture["parity"]["status"] in {"match", "mismatch", "error"} and
        text_codec_fixture["parity"]["status"] != "not_run",
        "wired external text-codec fixture must report real parity, not not_run")
require(text_codec_fixture["parity"].get("equivalence_class") in {
            "byte_exact", "canonical_equivalent", "structured_equivalent", "incompatible"},
        "wired external text-codec fixture must record a decoder equivalence class")
require(text_codec_fixture["parity"]["external_token_ids"] ==
        text_codec_fixture["fixture"]["expected_token_ids"],
        "wired external tokenizer evidence must contain callback-produced token ids")
require(text_codec_fixture["parity"]["external_decoded_text"] ==
        text_codec_fixture["fixture"]["expected_decoded_text"],
        "wired external detokenizer evidence must contain callback-produced text")
require(text_codec_fixture["detokenizer"]["status"] == "match" and
        text_codec_fixture["detokenizer"]["final_text"] == text_codec_fixture["detokenizer"]["expected_text"],
        "detokenizer fixture must prove final byte parity")
require("".join(chunk["emitted_text"] for chunk in text_codec_fixture["detokenizer"]["chunk_emissions"]) ==
        text_codec_fixture["detokenizer"]["final_text"],
        "detokenizer fixture chunk emissions must concatenate to final text")
require("fixture schema_version must be 1" in text_codec_validator,
        "JSON contract validator must check fixture schema version")
require("class ExternalTokenizerProbe" in external_tokenizer_header,
        "missing callback-based external tokenizer probe")
require("using TokenizeCallback" in external_tokenizer_header and
        "std::function<std::vector<int>(std::string_view prompt)>" in external_tokenizer_header,
        "external tokenizer probe must be callback-based")
require("enum class TokenMismatchKind" in external_tokenizer_header and
        "struct TokenMismatch" in external_tokenizer_header,
        "external tokenizer probe must classify first mismatch kind")
require("find_first_mismatch" in external_tokenizer_source and
        "TokenMismatchKind::Length" in external_tokenizer_source,
        "external tokenizer probe must detect count mismatches after a shared prefix")
require("first mismatch must identify token index 1" in external_tokenizer_test and
        "callback errors must report parity status error" in external_tokenizer_test,
        "external tokenizer test must cover first mismatch and callback error paths")
for forbidden_token in ("OgaGenerator", "OgaTokenizerStream", "GenerateNextToken",
                        "GetSequenceData", "run_time_decoder"):
    require(forbidden_token not in external_tokenizer_header + external_tokenizer_source + external_tokenizer_test,
            f"tokenizer lane must not touch generation/decode token: {forbidden_token}")
require("REQUIRED_MODEL_FAMILIES = {\"qwen2.5\", \"gemma\"}" in tokenizer_parity_tool,
        "tokenizer parity tool must require Qwen and Gemma model-family coverage")
require("build_callback_evidence" in tokenizer_parity_tool and
        "first_token_mismatch" in tokenizer_parity_tool,
        "tokenizer parity tool must write evidence and classify first token mismatch")
case_categories = {case["category"] for case in tokenizer_text_cases["cases"]}
for required_category in ("plain", "chat", "tool", "code", "whitespace", "cjk", "emoji", "added_tokens"):
    require(required_category in case_categories,
            f"tokenizer text fixture missing category: {required_category}")
family_ids = {family["id"] for family in tokenizer_text_cases["model_families"]}
require({"qwen2.5", "gemma"}.issubset(family_ids),
        "tokenizer text fixture must cover Qwen and Gemma model families")
for case in tokenizer_text_cases["cases"]:
    require(case.get("capture_required") is True or "expected_oga_token_ids" in case,
            f"tokenizer text fixture case must require capture or provide token IDs: {case.get('name')}")
require("get_hip_oga_model_path" in model_list_hpp,
        "model_list cannot resolve HIP OGA package paths for tags")
require("has_hip_oga_model_path" in model_list_hpp,
        "model_list cannot validate HIP OGA tag metadata")
require("get_all_hip_oga_models" in model_list_hpp,
        "model_list cannot expose HIP OGA-capable catalog entries")
require("get_all_models_openai_hip_oga" in model_list_hpp,
        "model_list cannot expose HIP OGA /v1/models entries")
require("validate_hip_oga_model_dir" in model_list_hpp,
        "model_list cannot validate direct OGA directories for CLI check")
require("get_hip_oga_public_hf_config" in model_list_hpp,
        "model_list cannot expose public Hugging Face HIP metadata")
require("require_hip_oga_runtime_supported" in model_list_hpp,
        "model_list cannot reject unsupported HIP OGA catalog entries")
require("is_hip_oga_only_model" in model_list_hpp,
        "model_list cannot distinguish HIP-only catalog entries")
require("FLM_HIP_OGA_MODEL_ROOT" in model_list_hpp,
        "model_list does not document/env-resolve the HIP OGA model root")
hip_oga_policy_hpp = read("src/include/hip_oga_policy.hpp")
require("get_hip_oga_offload_policy" in model_list_hpp,
        "model_list cannot report HIP OGA strict/hybrid offload policy")
require("qualification" in model_list_hpp,
        "HIP OGA catalog output must expose policy qualification")
require("build_session_overlay" in hip_oga_policy_hpp,
        "HIP OGA policy helper must build provider/session overlays")
require("kStrictGpuPolicy" in hip_oga_policy_hpp and
        "kHybridGpuPolicy" in hip_oga_policy_hpp and
        "kUnqualifiedPolicy" in hip_oga_policy_hpp,
        "HIP OGA policy helper must expose strict_gpu, hybrid_gpu, and unqualified policies")
require('"utils/debug_utils.hpp"' in model_list_hpp and '"utils/utils.hpp"' not in model_list_hpp,
        "model_list static checks should not include NPU runtime headers")
llama32_1b = model_list_json["models"]["llama3.2"]["1b"]
require("hip_oga" in llama32_1b, "llama3.2:1b lacks representative HIP OGA metadata")
require(llama32_1b["hip_oga"].get("provider") == "AMDGPU",
        "llama3.2:1b HIP OGA metadata must select AMDGPU")
require(llama32_1b["hip_oga"].get("profile") == "hip",
        "llama3.2:1b HIP OGA metadata must select profile=hip")
require(llama32_1b["hip_oga"].get("config_path") == "llama3.2-1b-oga",
        "llama3.2:1b HIP OGA metadata must use a relative package path")
require(llama32_1b["hip_oga"].get("offload_policy") == "strict_gpu",
        "llama3.2:1b HIP OGA metadata must explicitly select strict_gpu")
hip_vlm_policies = []
for model_type, model_subset in model_list_json["models"].items():
    for size, model in model_subset.items():
        if "hip_oga" not in model:
            continue
        if "vision" not in model.get("label", []):
            continue
        hip_vlm_policies.append(model["hip_oga"].get("offload_policy", "unqualified"))
require("hybrid_gpu" in hip_vlm_policies,
        "HIP OGA catalog must include an explicitly labeled hybrid_gpu VLM package")
required_files = set(llama32_1b["hip_oga"].get("required_files", []))
for required_file in ("genai_config.json", "tokenizer.json", "tokenizer_config.json"):
    require(required_file in required_files,
            f"llama3.2:1b HIP OGA metadata must require {required_file}")

for forbidden_public_repo in (
    "amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-bf16-onnx-ryzen-strix",
    "amd/Llama-3.1-8B-awq-g128-int4-asym-bf16-onnx-ryzen-strix",
):
    require(
        forbidden_public_repo not in json.dumps(model_list_json),
        f"catalog must not ship rejected public HIP demo repo: {forbidden_public_repo}",
    )

expected_public_tags = {
    "hip-qwen2.5-coder:14b": {
        "repo_id": "amd/Qwen2.5-Coder-14B-Instruct_gpuep_1.0_gpu_128",
        "revision": "6de529bb30355eac01ad8daab89cda657ddf0418",
        "model_kind": "text",
        "unsupported": False,
        "size": 8853580693,
        "support_status": "strict_gpu",
        "labels": {"hip", "text", "strict_gpu", "default"},
        "source_subdir": "Upload Qwen2.5 ONNX files",
    },
    "hip-qwen2.5:14b": {
        "repo_id": "amd/Qwen2.5-14B-Instruct_gpuep_1.0_gpu_128",
        "revision": "b36841769dc5c3331a2d611c07e9c5f750377d5f",
        "model_kind": "text",
        "unsupported": False,
        "size": 8853580604,
        "support_status": "strict_gpu",
        "labels": {"hip", "text", "strict_gpu", "alternate"},
        "source_subdir": "Upload Qwen2.5 ONNX files",
    },
    "hip-gemma3:4b": {
        "repo_id": "amd/Gemma3-4B-it_gpuep_1.0_gpu_128",
        "revision": "72825e7565eaf95a69416a68a2274e367b6d3fcb",
        "model_kind": "vlm",
        "unsupported": True,
        "size": 4359351642,
        "support_status": "hybrid_gpu_experimental",
        "labels": {"hip", "vision", "hybrid_gpu", "experimental", "unqualified"},
        "source_subdir": "Upload gemma3-4b-it ONNX files",
    },
    "hip-qwen3.5-vl:9b": {
        "repo_id": "amd/Qwen3.5-9B_gpuep_1.0_gpu_128",
        "revision": "bad679999a75e4883384cdac008972614cfb0d00",
        "model_kind": "vlm",
        "unsupported": True,
        "size": 7491576178,
        "support_status": "hybrid_gpu_candidate",
        "labels": {"hip", "vision", "hybrid_gpu", "qualification", "unqualified"},
        "source_subdir": "",
    },
    "hip-qwen3.6-vl:35b-a3b": {
        "repo_id": "amd/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128",
        "revision": "c2fdd8966cfde2da5fa780004fe6fad4a1125bb8",
        "model_kind": "vlm",
        "unsupported": True,
        "size": 22090442291,
        "support_status": "hybrid_gpu_stress",
        "labels": {"hip", "vision", "hybrid_gpu", "stress", "optional", "unqualified"},
        "source_subdir": "",
    },
}
for tag, expected in expected_public_tags.items():
    model_type, size = tag.split(":", 1)
    entry = model_list_json["models"].get(model_type, {}).get(size)
    require(entry is not None, f"missing public HIP OGA catalog entry: {tag}")
    hip_oga = entry.get("hip_oga", {})
    public_hf = hip_oga.get("public_hf", {})
    require(hip_oga.get("hip_only") is True, f"{tag} must be marked HIP-only")
    require(public_hf.get("repo_id") == expected["repo_id"], f"{tag} has wrong public AMD repo_id")
    revision = expected["revision"]
    require(public_hf.get("revision") == revision and len(revision) == 40,
            f"{tag} must pin a full commit SHA")
    require(entry.get("size") == expected["size"],
            f"{tag} must expose the verified public byte size")
    require(hip_oga.get("support_status") == expected["support_status"],
            f"{tag} must expose the expected HIP OGA support status")
    require(expected["labels"].issubset(set(entry.get("label", []))),
            f"{tag} must include policy labels {sorted(expected['labels'])}")
    require(public_hf.get("source_subdir", "") == expected["source_subdir"],
            f"{tag} has wrong source_subdir")
    require(public_hf.get("endpoint_env") == "FLM_HIP_OGA_HF_ENDPOINT",
            f"{tag} must use the public HF endpoint env")
    require("endpoint" not in public_hf,
            f"{tag} must not embed a private/public endpoint URL in catalog metadata")
    require(public_hf.get("model_kind") == expected["model_kind"],
            f"{tag} has wrong public_hf.model_kind")
    require(public_hf.get("license") and public_hf.get("license") != "unknown",
            f"{tag} must declare a known license")
    file_entries = public_hf.get("files", [])
    require(file_entries, f"{tag} needs public_hf.files checksums")
    files_by_path = {item.get("path"): item for item in file_entries}
    for required_file in hip_oga.get("required_files", []):
        require(required_file in files_by_path,
                f"{tag} required file lacks public_hf checksum: {required_file}")
    for item in file_entries:
        require(len(item.get("sha256", "")) == 64,
                f"{tag} file entry has non-SHA256 checksum: {item}")
    require(all(not item.get("path", "").startswith("Upload Qwen2.5 ONNX files/")
                for item in file_entries),
            f"{tag} installer manifest must flatten source_subdir entries")
    if expected["unsupported"]:
        require("unsupported_reason" in hip_oga and hip_oga["unsupported_reason"],
                f"{tag} VLM entry must be clearly marked unsupported")
    else:
        require("unsupported_reason" not in hip_oga,
                f"{tag} text trial entry must not be marked unsupported")

require("validate_hip_oga_package" in model_list_hpp,
        "model_list does not validate the HIP OGA package contract for tag loads")
for token in ("tokenizer_config.json", "eos_token", "eos_token_id", "chat_template.jinja", "chat_template"):
    require(token in model_list_hpp,
            f"model_list HIP OGA package validation is missing {token}")
require("HipOgaModel" in all_models, "get_auto_model cannot construct HipOgaModel")
require("is_hip_oga_backend(resolved_backend)" in all_models, "get_auto_model does not branch on hip backend")
require("HipOgaModel::is_model_dir(model_tag)" in all_models,
        "get_auto_model does not accept direct OGA model directories for hip")
require("available_models.has_hip_oga_model_path(new_model_tag)" in all_models,
        "get_auto_model does not reject HIP tags missing hip_oga.config_path")
require("get_auto_model(this->tag" in runner and "this->backend" in runner, "Runner does not pass backend into model selection")
require("hip_oga_requested_max_tokens" in runner,
        "Runner must pass the requested max token budget before HIP OGA insert")
require("HipOgaModel::model_info_for_path(this->tag)" in runner,
        "Runner does not load direct OGA model directories through HipOgaModel")
require("this->supported_models.get_hip_oga_model_path(new_tag)" in runner,
        "Runner does not resolve HIP model-list tags to OGA package paths")
require("model_info[\"hip_oga\"][\"config_path\"] = model_path" in runner,
        "Runner does not pass the resolved HIP OGA package path into model metadata")
require("report_hip_oga_offload_policy" in runner,
        "Runner must report selected HIP OGA offload policy before load")
require("initial_prompt(args.input_file_name)" in runner and
        "this->generate_limit = args.max_tokens" in runner,
        "Runner does not wire --prompt/--max-tokens into one-shot generation")
require("generate_once(this->initial_prompt" in runner,
        "Runner does not execute --prompt non-interactively")
require("request.value(\"backend\"" in rest_handler, "REST handlers do not accept JSON backend")
require("current_backend != resolved_backend" in rest_handler, "REST handler does not reload when backend changes")
require("HipOgaModel::model_info_for_path(ensure_tag)" in rest_handler,
        "REST handler does not load direct OGA model directories through HipOgaModel")
require("supported_models.get_hip_oga_model_path(new_ensure_tag)" in rest_handler,
        "REST handler does not resolve HIP model-list tags to OGA package paths")
require("report_hip_oga_offload_policy" in rest_handler,
        "REST handler must report selected HIP OGA offload policy before load")
require("get_all_models_openai_hip_oga(direct_model_path)" in rest_handler,
        "REST /v1/models does not expose HIP-specific loadable model ids")
require("offload_policy" in rest_handler,
        "REST /v1/models must expose HIP OGA offload policy metadata")
require("static json normalize_template(json messages, bool fail_fast_invalid_images" in rest_handler,
        "OpenAI message normalizer must accept a HIP fail-fast image validation flag")
require("normalize_template(current_messages, flm::backend::is_hip_oga_backend(resolved_backend))" in rest_handler,
        "OpenAI chat completions must pass backend=hip into image normalization")
require("HIP OGA backend supports only data:image/png;base64 or data:image/jpeg;base64 image_url payloads" in rest_handler,
        "HIP OpenAI image normalization must reject unsupported data:image media types")
require("HIP OGA backend OpenAI image_url payloads must use data:image/png;base64 or data:image/jpeg;base64" in rest_handler,
        "HIP OpenAI image normalization must reject remote or otherwise non-data image_url payloads")
require("HIP OGA backend image_url payloads must be non-empty" in rest_handler,
        "HIP OpenAI image normalization must reject empty image_url payloads")
openai_chat_completion = rest_handler.split("void RestHandler::handle_openai_chat_completion", 1)[1].split(
    "void RestHandler::handle_openai_audio_transcriptions", 1)[0]
require('"type", "invalid_request_error"' in openai_chat_completion and
        '{"code", 400}' in openai_chat_completion,
        "HIP OpenAI image normalization failures must return an explicit request error")
require("request[\"backend\"]" in server and "is_hip_oga_backend" in server, "HTTP queue does not inspect backend")
benchmark = read("src/src/benchmarking.hpp")
require("std::string resolved_model_tag = model_tag" in benchmark,
        "benchmark HIP direct path must keep a CSV tag outside model-list lookup")
require("write_bench_csv(results, resolved_model_tag" in benchmark,
        "benchmark CSV writer must not use the model-list-only new_tag variable")
require("availble_models.get_hip_oga_model_path(new_tag)" in benchmark,
        "benchmark does not resolve HIP model-list tags to OGA package paths")
require("FLM_HIP_OGA_MODEL_ROOT" in model_list_test and
        "get_hip_oga_model_path(\"llama3.2:1b\")" in model_list_test,
        "runtime model-list check must prove env-root HIP tag resolution")
require("get_all_hip_oga_models()" in model_list_test and
        "get_all_models_openai_hip_oga" in model_list_test,
        "runtime model-list check must prove HIP list and OpenAI model-list behavior")
require("source_subdir" in public_pull_test and "Upload Qwen2.5 ONNX files" in public_pull_test,
        "public HIP pull fixture must prove source-subdir flattening")
for token in ("non_amd_repo", "missing_revision", "unknown_license",
              "credential_metadata_rejected", "endpoint_userinfo_rejected",
              "symlink_config_path_rejected", "unsupported_vlm_rejected"):
    require(token in public_pull_test,
            f"public HIP pull fixture is missing coverage for {token}")
require("hip-qwen2.5-coder:14b" in hip_oga_docs and
        "hip-qwen2.5:14b" in hip_oga_docs and
        "hip-gemma3:4b" in hip_oga_docs and
        "hip-qwen3.6-vl:35b-a3b" in hip_oga_docs,
        "HIP OGA docs must describe the corrected text default and optional VLM path")
require("hip-mistral:7b" not in hip_oga_docs and
        "hip-llama3.1:8b" not in hip_oga_docs,
        "HIP OGA docs must not keep rejected public demo rows")
public_surface = json.dumps(model_list_json) + "\n" + hip_oga_docs
for forbidden_public_token in (
    "internal-only-host",
    "HF_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "token_env",
    "private.example",
):
    require(forbidden_public_token not in public_surface,
            f"public HIP OGA catalog/docs leak forbidden token: {forbidden_public_token}")
require("model_list::validate_hip_oga_model_dir" in main_cpp,
        "CLI check --backend hip does not validate direct OGA directories")
require("get_all_hip_oga_models()" in main_cpp,
        "CLI list --backend hip does not use HIP OGA catalog semantics")
require("hip_oga_public_downloader.hpp" in main_cpp and "PublicHfDownloader" in main_cpp,
        "CLI pull --backend hip does not use the public Hugging Face HIP downloader")
require("is_hip_oga_only_model(model_tag)" in model_downloader and
        "requires --backend hip" in model_downloader,
        "default downloader must not silently pull HIP-only public catalog rows")
require("offload_policy" in main_cpp and "policy_warning(offload_policy)" in main_cpp,
        "CLI check/list must report policy and warn for hybrid_gpu")

for include_token in ("hip_oga_text_codec.hpp", "hip_oga_external_tokenizer.hpp"):
    require(include_token in adapter,
            f"HIP adapter must include external text-codec routing helper: {include_token}")
require("template_provider_selector_" not in adapter and
        "FLM_HIP_OGA_TEMPLATE_PROVIDER" not in adapter,
        "HIP adapter must not expose a misleading OGA template provider selector for the FastFlowLM/minja frontend")
for selector_member in ("tokenizer_provider_selector_", "decoder_provider_selector_"):
    require(selector_member in adapter,
            f"HIP adapter must keep an independent provider selector: {selector_member}")
codec_options = (ROOT / "src/include/AutoModel/hip_oga_codec_options.hpp").read_text()
for env_name in ("FLM_HIP_OGA_TEXT_CODEC", "FLM_HIP_OGA_TOKENIZER_PROVIDER",
                 "FLM_HIP_OGA_DECODER_PROVIDER", "FLM_HIP_OGA_TEXT_CODEC_COMPARE"):
    require(env_name in codec_options, f"codec configuration must support {env_name}")
require("text_codec_config_from_environment()" in adapter and
        "apply_text_codec_cli_options" in main_cpp,
        "CLI and adapter must share the codec configuration resolver")
require('"frontend=fastflowlm-minja"' in adapter and
        "provider_signature(tokenizer_provider_selector_, decoder_provider_selector_" in adapter,
        "codec identity must remain available for diagnostics")
cache_signature = re.search(
    r"const std::string signature =\s*hip_engine->prefix_cache_signature\((.*?)\);", adapter, re.S)
require(cache_signature is not None and
        cache_signature.group(1).strip() == "prefix_cache_request_signature_",
        "KV compatibility must use backend request state, not codec names or fingerprint")
require("OgaTokenizer::Create(*model_)" not in adapter.split("class HipOgaCausalLm", 1)[1].split(
        "void set_requested_max_tokens", 1)[0],
        "FLM-only mode must not eagerly construct OGA tokenizer")
require("decoder_execution.use_oga && this->is_normal_token" in adapter and
        "if (decoder_execution.compare)" in adapter,
        "native decoding and oracle comparison must be gated independently")
require("multimodal_turn_ = true" in adapter and
        "OgaTokenizerStream::Create(*processor_)" in adapter,
        "media decoding must preserve processor stream selection")
require("force_oga_text_providers_for_media" in adapter and
        "media request forces HIP OGA text providers" in adapter,
        "HIP adapter must force media requests through OGA text providers")
require("encode_prompt_with_selected_provider" in adapter and
        "decode_token_with_selected_provider" in adapter,
        "HIP adapter must route tokenizer/decoder through independent selectors")
for fail_closed_token in ("external tokenizer parity mismatch",
                          "external decoder parity mismatch"):
    require(fail_closed_token in adapter,
            f"HIP adapter must fail closed on parity mismatch: {fail_closed_token}")
require("ExternalTokenizerProbe" in adapter and
        "ExternalCumulativeDetokenizer" in adapter and
        "TextCodecParityResult" in adapter,
        "HIP adapter must thread tokenizer/detokenizer parity diagnostics into compare mode")
require("external text codec callbacks are not wired" not in adapter and
        "ParityStatus::NotRun" not in adapter,
        "HIP adapter compare mode must execute callbacks instead of reporting not_run")

tokenizers_dir = ROOT / "third_party/tokenizers-cpp"
expected_tokenizers_sha = git_output("rev-parse", "HEAD:third_party/tokenizers-cpp")
actual_tokenizers_sha = (
    git_output("-C", str(tokenizers_dir), "rev-parse", "HEAD")
    if (tokenizers_dir / ".git").exists()
    else ""
)
require((tokenizers_dir / "CMakeLists.txt").exists(),
        "tokenizers-cpp submodule is missing CMakeLists.txt; run scripts/bootstrap-fastflowlm-deps.sh")
for nested_submodule in ("msgpack", "sentencepiece"):
    require((tokenizers_dir / nested_submodule / "CMakeLists.txt").exists(),
            f"tokenizers-cpp nested submodule is missing CMakeLists.txt: {nested_submodule}")
require(actual_tokenizers_sha == expected_tokenizers_sha,
        "tokenizers-cpp submodule is not initialized at the pinned gitlink")
require("git submodule update --init --recursive third_party/tokenizers-cpp" in bootstrap_script,
        "bootstrap script must initialize the pinned tokenizers-cpp submodule")
require("git -C third_party/tokenizers-cpp submodule update --init --recursive" in bootstrap_script,
        "bootstrap fallback must initialize tokenizers-cpp nested submodules")
require("Get-Process -Id $PID" in runtime_check,
        "runtime preflight must enumerate the actual loaded DLL modules")
require("$loadedHandles" in runtime_check and "LoadLibraryEx($path" in runtime_check,
        "runtime preflight must load every required HIP OGA runtime DLL by path")
require("OrtGetApiBase" in runtime_check and "GetVersionString" in runtime_check,
        "runtime preflight must read the loaded onnxruntime.dll version through ORT")
require("1.27.0" in runtime_check,
        "runtime preflight must fail before an ORT 1.17.x/API-v27 mismatch")
require("onnxruntime-genai.dll" in runtime_check and "onnxruntime.dll" in runtime_check,
        "runtime preflight must validate OGA and ORT DLL provenance")


def branch_after(source: str, marker: str) -> str:
    require(marker in source, f"missing branch marker: {marker}")
    return source.split(marker, 1)[1].split("} else {", 1)[0]


runner_hip_tag_branch = branch_after(runner, "} else if (using_hip_oga) {")
rest_hip_tag_branch = branch_after(rest_handler, "} else if (using_hip_oga) {")
benchmark_hip_tag_branch = branch_after(benchmark, "} else if (using_hip_oga) {")
chat_handler = rest_handler.split("void RestHandler::handle_chat", 1)[1].split(
    "void RestHandler::handle_embeddings", 1)[0]
chat_stream_branch = chat_handler.split("if (stream) {", 1)[1].split(
    "} else {", 1)[0]

for name, branch in (
    ("Runner HIP tag branch", runner_hip_tag_branch),
    ("REST HIP tag branch", rest_hip_tag_branch),
    ("benchmark HIP tag branch", benchmark_hip_tag_branch),
):
    for token in ("ensure_npu_device", "downloader.is_model_downloaded",
                  "pull_model", "get_model_path("):
        require(token not in branch,
                f"{name} should not use NPU/download path token: {token}")

require(chat_stream_branch.count("auto_chat_engine->insert(meta_info, uniformed_input)") == 1,
        "/api/chat streaming should prefill exactly once")
require("auto_chat_engine->generate(meta_info, length_limit, ostream" in chat_stream_branch,
        "/api/chat streaming does not generate after prefill")

adapter_lower = adapter.lower()
for token in ("open_npu_device", "flm_rt::device(", "xrt::", "amdxdna", ".xclbin", "xclbin_manager("):
    require(token not in adapter_lower, f"HIP adapter should not reference NPU runtime token: {token}")
for token in ("TextCodecProvider", "text_codec_execution",
              "default_text_codec_selector"):
    require(token in adapter,
            f"HIP adapter must wire external text-codec selector path: {token}")
for env_name in ("FLM_HIP_OGA_TOKENIZER_PROVIDER",
                 "FLM_HIP_OGA_DECODER_PROVIDER", "FLM_HIP_OGA_TEXT_CODEC_COMPARE"):
    require(env_name in hip_oga_docs,
            f"HIP OGA docs must document external text-codec env var: {env_name}")
require("FLM_HIP_OGA_TEMPLATE_PROVIDER" not in hip_oga_docs and
        "FastFlowLM/minja" in hip_oga_docs,
        "HIP OGA docs must describe the fixed FastFlowLM/minja frontend instead of a template selector")

require("OgaConfig::Create" in adapter, "HIP adapter does not create OGA config")
require("static OgaHandle handle" in adapter,
        "HIP adapter does not register OgaHandle for deterministic OGA shutdown")
require("build_session_overlay" in adapter, "HIP adapter must build a provider/session overlay")
require("config_->Overlay" in adapter, "HIP adapter must apply provider/session overlay through OgaConfig::Overlay")
require("session.disable_cpu_ep_fallback" in adapter,
        "HIP adapter strict_gpu policy must disable ORT CPU EP fallback")
require("disable_cpu_ep_fallback" in adapter,
        "HIP adapter strict_gpu policy must set OGA disable_cpu_ep_fallback session option")
require('session_options["disable_cpu_ep_fallback"] = "1"' in hip_oga_policy_hpp,
        "OGA overlay must encode disable_cpu_ep_fallback as the string value expected by the corrected runtime")
require("FLM_HIP_OGA_DIAGNOSTICS" in adapter and "enable_profiling" in hip_oga_policy_hpp,
        "HIP adapter must expose diagnostics for per-section ORT profile evidence")
require("FLM_HIP_OGA_PREFIX_CACHE" in rest_handler,
        "REST handlers must gate HIP OGA prefix reuse behind FLM_HIP_OGA_PREFIX_CACHE")
require("FLM_HIP_OGA_PREFIX_CACHE=experimental" in hip_oga_docs,
        "HIP OGA docs must document the experimental prefix-cache opt-in")
require("prefix_cache_env_enabled" in prefix_cache_hpp and
        '"experimental"' in prefix_cache_hpp,
        "HIP OGA prefix cache must only enable on the explicit experimental env value")
require("PrefixCacheState" in prefix_cache_hpp and
        "prefix_diverged" in prefix_cache_hpp and
        "capacity_shortage" in prefix_cache_hpp,
        "HIP OGA prefix-cache policy helper is missing guarded fallback states")
require("prefix_cache_.evaluate" in adapter,
        "HIP adapter must evaluate the prefix-cache policy before OGA reuse")
require("restore_text_checkpoint" in adapter,
        "HIP adapter must restore the OGA generator before appending a cached suffix")
require("append_text_suffix" in adapter,
        "HIP adapter must append only the uncached suffix on reuse")
require("disable_for_session" in adapter,
        "HIP adapter must disable reuse for the current session after provider exceptions")
require("reset_decoder_stream" in adapter and "token_stream_ = OgaTokenizerStream::Create" in adapter,
        "HIP adapter must recreate OgaTokenizerStream for each generated response")
require("multimodal_forced_miss" in adapter and "prefix_cache_.reset_checkpoint" in adapter,
        "HIP adapter must force VLM/image/audio requests to miss and reset the text checkpoint")
require("hip_oga_request_signature" in rest_handler,
        "REST handlers must pass a request/search signature into the HIP OGA adapter")
require("test_vlm_request_forces_miss_and_clears_checkpoint" in prefix_cache_test and
        "test_provider_exception_disables_reuse_for_session" in prefix_cache_test,
        "HIP OGA prefix-cache unit fixture must cover VLM reset and provider-exception disable")
require("SnapshotState" not in adapter, "HIP adapter uses an OGA snapshot API absent from the staged runtime")
require("load_hip_oga_lm_config" in adapter,
        "HIP adapter does not populate shared LM config without the NPU loader")
require("from_pretrained(this->model_path)" not in adapter,
        "HIP adapter still calls the NPU-oriented LM config loader")
require("find_xclbin_path" not in adapter,
        "HIP adapter should not call or reference FastFlowLM xclbin discovery")
require("append_token_ids(this->eos_token_ids, genai_config[\"model\"][\"eos_token_id\"])" in adapter,
        "HIP adapter does not fall back to OGA genai_config.json eos_token_id")
require("effective_oga_max_length" in adapter,
        "HIP adapter must compute a request-bounded OGA max_length")
require("prompt_token_count + requested_max_tokens" in adapter,
        "HIP adapter max_length must be bounded by prompt tokens plus requested generated tokens")
require("kHipOgaSharedGenerateLookaheadTokens" in adapter and
        "requested_max_tokens + kHipOgaSharedGenerateLookaheadTokens" in adapter,
        "HIP adapter must reserve one OGA lookahead token for shared max_tokens=1 generation")
require("std::min<std::uint64_t>(context_length_" in adapter,
        "HIP adapter max_length must be capped by the model context length")
require("hip_oga_requested_max_tokens" in adapter and "configure_parameter" in adapter,
        "HIP adapter must accept the per-request max token budget before insert/prefill")
require("meta_info.generated_tokens >= length_limit" in adapter,
        "HIP adapter generation must count the first sampled token against --max-tokens")

encode_prompt_helper = re.search(
    r"std::vector<int>\s+encode_prompt_with_selected_provider\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
    adapter,
    re.S,
)
require(encode_prompt_helper is not None,
        "HIP adapter must define the selected-provider prompt encode helper")
encode_prompt_helper_body = encode_prompt_helper.group("body")
require("ExternalTokenizerProbe" in encode_prompt_helper_body and
        "hip_engine.encode(std::string(prompt))" in encode_prompt_helper_body and
        "fastflowlm_tokenizer.encode(std::string(prompt))" in encode_prompt_helper_body and
        "external_tokenizer_probe.compare(templated_text)" in encode_prompt_helper_body,
        "HIP adapter tokenizer helper must compare OGA and FastFlowLM callback tokenization")
require("describe_token_mismatch(tokenizer_parity)" in encode_prompt_helper_body and
        "external tokenizer parity mismatch" in encode_prompt_helper_body,
        "HIP adapter tokenizer helper must fail closed with first mismatch details")
require("return fastflowlm_tokenizer.encode(templated_text);" in encode_prompt_helper_body,
        "FLM tokenizer mode must bypass the OGA oracle")
require("return parity.external_token_ids;" in encode_prompt_helper_body,
        "HIP adapter direct external tokenizer mode must use external ids only after parity passes")
require("hip_engine->begin_turn(tokens)" in adapter,
        "HIP adapter must append the full prompt through the native OGA generator")
require("collect_hip_oga_image_inputs(input)" in adapter,
        "HIP adapter must collect CLI and server-normalized image inputs")
require("reject_hip_oga_audio_inputs(input)" in adapter,
        "HIP adapter must fail fast for audio inputs until OGA audio packages are verified")
require("hip_engine->begin_multimodal_turn(templated_text, images)" in adapter,
        "HIP adapter must route image prompts through the native OGA multimodal turn")
require("OgaMultiModalProcessor::Create" in adapter,
        "HIP adapter must create an OGA multimodal processor")
require("OgaImages::Load" in adapter,
        "HIP adapter must load encoded image buffers through OgaImages")
require("OgaNamedTensors" in adapter and
        ("inputs->Get(\"input_ids\")" in adapter or
         "inputs.Get(\"input_ids\")" in adapter),
        "HIP adapter must inspect OGA named tensor input_ids for prompt accounting")
require("generator_->SetInputs(*inputs)" in adapter,
        "HIP adapter must pass processor-produced named tensors to the OGA generator")
require("OgaTokenizerStream::Create(*processor_)" in adapter,
        "HIP adapter must use processor-aware decoding after multimodal prefill")
require("Native Image Inputs" in hip_oga_docs and
        "OgaMultiModalProcessor" in hip_oga_docs and
        "OgaGenerator::SetInputs" in hip_oga_docs,
        "HIP OGA docs must describe the native image processor path")
require("Audio inputs fail fast" in hip_oga_docs,
        "HIP OGA docs must document the audio fail-fast boundary")
require("strict_gpu" in hip_oga_docs and "hybrid_gpu" in hip_oga_docs and
        "FLM_HIP_OGA_DIAGNOSTICS" in hip_oga_docs,
        "HIP OGA docs must document strict/hybrid policy and diagnostics")
require("hip_engine->generate_next_token(sampled_token)" in adapter,
        "HIP adapter must consume OGA-generated token ids for user-visible text")

decode_token_helper = re.search(
    r"std::string\s+decode_token_with_selected_provider\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
    adapter,
    re.S,
)
require(decode_token_helper is not None,
        "HIP adapter must define the selected-provider generated-token decode helper")
decode_token_helper_body = decode_token_helper.group("body")
hip_generate = adapter.split(
    "std::string HipOgaModel::generate(chat_meta_info_t& meta_info", 1)[1].split(
        "std::string HipOgaModel::generate_with_prompt", 1)[0]
require("ExternalCumulativeDetokenizer" in hip_generate and
        "fastflowlm_tokenizer.decode(token_ids)" in hip_generate,
        "HIP adapter generate() must wire cumulative FastFlowLM decode callback")
require("return hip_engine.decode_token(token);" in decode_token_helper_body and
        "classify_decoder_text_equivalence" in hip_generate,
        "HIP adapter decoder compare mode must route final text through the canonical-equivalence classifier")
require("external cumulative decoded bytes diverged from OGA stream" not in hip_generate and
        "native_decoded_text.rfind(external_decoder->emitted_text(), 0)" not in hip_generate,
        "HIP adapter must not reject delayed-but-valid external cumulative decoder chunks")
require("external_decoder->emitted_text() != native_decoded_text" not in hip_generate and
        "final decoded bytes differ from OGA stream" not in hip_generate,
        "HIP adapter must not use raw byte equality as the final plain-text pass gate")
require("DecoderEquivalenceClass::Incompatible" in hip_generate and
        "decoder equivalence incompatible" in hip_generate,
        "HIP adapter must fail closed only after canonical decoder equivalence is incompatible")
require("decode_preserving_standalone_metaspace_tokens" not in adapter and
        "preserve_standalone_metaspace_tokens_for_oga" not in adapter,
        "HIP adapter must not merge the diagnostic literal-metaspace reproduction behavior")
require("return hip_engine.decode_token(token);" in decode_token_helper_body,
        "HIP adapter must keep OGA token-stream decoding available for native authoritative output")
require("sampler->sample" not in hip_generate,
        "HIP adapter generate() must not resample OGA logits through FastFlowLM")
require("lm_engine->forward(sampled_token)" not in hip_generate,
        "HIP adapter generate() must not use the logits/rewind path for text generation")
hip_generate_with_prompt = adapter.split("std::string HipOgaModel::generate_with_prompt", 1)[1].split("#else", 1)[0]
require("return this->generate(meta_info, length_limit, os);" in hip_generate_with_prompt,
        "HIP adapter generate_with_prompt must route through the HIP generation override")
require("return this->_shared_generate(meta_info, length_limit, os);" not in hip_generate_with_prompt,
        "HIP adapter generate_with_prompt must not bypass the HIP generation override")
require("NonStreamResult HipOgaModel::parse_nstream_content" in adapter,
        "HIP adapter must override non-stream OpenAI tool parsing")
require("StreamResult HipOgaModel::parse_stream_content" in adapter,
        "HIP adapter must override streaming OpenAI tool parsing")
require("flm::hip_oga::parse_stream_markup" in adapter,
        "HIP streaming parser must use the isolated HIP OGA parser helper")
require("flm::hip_oga::parse_tool_call_markup(response_text)" in adapter,
        "HIP non-stream parser must parse qwen/gemma <tool_call> markup")
require("npu_utils" not in hip_tool_parser and "ort_genai" not in hip_tool_parser,
        "HIP tool parser helper must stay hardware/runtime independent")
require("parse_stream_content_final" in read("src/include/AutoModel/hip_oga_model.hpp"),
        "HIP adapter must override final streaming parser cleanup")
require("call_\" + std::to_string(std::time(nullptr))" not in adapter,
        "HIP streaming tool-call IDs must not be timestamp-only; two tool calls in one second collide")
require("const std::string tool_call_id = result.tool_id.empty()" in streaming_openai and
        "std::to_string(tool_call_index)" in streaming_openai,
        "OpenAI stream assembler must assign a unique per-stream id for each emitted tool call")
require("model_output_error" in rest_handler,
        "OpenAI chat route must return explicit model-output errors for unsupported tool markup")
require("tool_choice=none" in rest_handler and "tools_for_template" in rest_handler,
        "OpenAI chat route must hide tools from the template when tool_choice=none")
require("int HipOgaModel::get_tool_start_token_id() const" in adapter,
        "HIP adapter must expose its tool-start masking decision")
require("this->tokenizer->encode(\"<tool_call>\")" in adapter and
        "tool_start_tokens.size() == 1" in adapter,
        "HIP tool_choice=none masking must only claim a single-token <tool_call> marker")
require("message[\"tool_calls\"]" in tool_parser_fixture and
        "\"finish_reason\": \"tool_calls\"" in tool_parser_fixture,
        "HIP OpenAI non-stream fixture must assert message.tool_calls")
require("\"tool_calls\"" in tool_parser_fixture and
        "build_openai_stream_chunk" in tool_parser_fixture,
        "HIP OpenAI stream fixture must assert delta.tool_calls")

fixture = subprocess.run(
    [sys.executable, str(ROOT / "src/test/hip_oga_backend_static/test_hip_oga_openai_tool_parser.py")],
    cwd=ROOT,
    text=True,
    capture_output=True,
)
if fixture.returncode != 0:
    sys.stderr.write(fixture.stdout)
    sys.stderr.write(fixture.stderr)
    raise SystemExit(fixture.returncode)

contract = subprocess.run(
    [sys.executable, str(ROOT / "src/test/hip_oga_backend_static/validate_hip_oga_text_codec_contract.py")],
    cwd=ROOT,
    text=True,
    capture_output=True,
)
if contract.returncode != 0:
    sys.stderr.write(contract.stdout)
    sys.stderr.write(contract.stderr)
    raise SystemExit(contract.returncode)

tokenizer_parity = subprocess.run(
    [sys.executable, str(ROOT / "src/test/hip_oga_external_codec/tokenizer_parity.py"), "--self-test"],
    cwd=ROOT,
    text=True,
    capture_output=True,
)
if tokenizer_parity.returncode != 0:
    sys.stderr.write(tokenizer_parity.stdout)
    sys.stderr.write(tokenizer_parity.stderr)
    raise SystemExit(tokenizer_parity.returncode)

with tempfile.TemporaryDirectory(prefix="flm-hip-oga-text-codec-") as tmpdir:
    text_codec_binary = Path(tmpdir) / (
        "hip_oga_text_codec_test.exe" if os.name == "nt" else "hip_oga_text_codec_test")
    compile_cmd = [
        os.environ.get("CXX", "g++"),
        "-std=c++20",
        "-I",
        str(ROOT / "src/include"),
        str(ROOT / "src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp"),
        "-o",
        str(text_codec_binary),
    ]
    compile_result = subprocess.run(
        compile_cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if compile_result.returncode != 0:
        sys.stderr.write(compile_result.stdout)
        sys.stderr.write(compile_result.stderr)
        raise SystemExit(compile_result.returncode)

    text_codec_result = subprocess.run(
        [str(text_codec_binary)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if text_codec_result.returncode != 0:
        sys.stderr.write(text_codec_result.stdout)
        sys.stderr.write(text_codec_result.stderr)
        raise SystemExit(text_codec_result.returncode)

with tempfile.TemporaryDirectory(prefix="flm-hip-oga-codec-options-") as tmpdir:
    binary = Path(tmpdir) / ("codec_options.exe" if os.name == "nt" else "codec_options")
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-I", str(ROOT / "src/include"),
                    str(ROOT / "src/test/hip_oga_backend_static/test_hip_oga_codec_options.cpp"),
                    "-o", str(binary)], cwd=ROOT, check=True)
    subprocess.run([str(binary)], cwd=ROOT, check=True)

with tempfile.TemporaryDirectory(prefix="flm-hip-oga-tool-parser-") as tmpdir:
    parser_test = Path(tmpdir) / ("hip_oga_tool_parser_test.exe" if os.name == "nt" else "hip_oga_tool_parser_test")
    compile_cmd = [
        os.environ.get("CXX", "g++"),
        "-std=c++20",
        "-I",
        str(ROOT / "src/include"),
        str(tool_parser_cpp),
        "-o",
        str(parser_test),
    ]
    compile_result = subprocess.run(
        compile_cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if compile_result.returncode != 0:
        sys.stderr.write(compile_result.stdout)
        sys.stderr.write(compile_result.stderr)
        raise SystemExit(compile_result.returncode)

    parser_result = subprocess.run(
        [str(parser_test)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if parser_result.returncode != 0:
        sys.stderr.write(parser_result.stdout)
        sys.stderr.write(parser_result.stderr)
        raise SystemExit(parser_result.returncode)

print("PASS: HIP OGA routing static checks")
