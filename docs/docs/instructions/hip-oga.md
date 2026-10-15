---
layout: docs
title: HIP OGA Backend
nav_order: 9
parent: Instructions
---

# HIP OGA Backend

The HIP OGA backend is an opt-in prototype path for ONNX Runtime GenAI models
that use the AMDGPU/HIP execution provider. It is separate from FastFlowLM's
NPU model execution path and does not consume FastFlowLM xclbin assets.

## Build

Configure FastFlowLM with ONNX Runtime GenAI visible through either
`ONNXRUNTIME_GENAI_HOME`, `OGA_HOME`, or `CONDA_PREFIX`:

```shell
scripts/bootstrap-fastflowlm-deps.sh
cmake -S src -B build-hip-oga \
  -DFLM_VERSION=0.0.0 \
  -DNPU_VERSION=0 \
  -DFLM_ENABLE_HIP_OGA=ON \
  -DONNXRUNTIME_GENAI_HOME=/path/to/onnxruntime-genai
cmake --build build-hip-oga --target flm -j
```

The build looks for `ort_genai.h` under `<root>/include` and the
`onnxruntime-genai` or `onnxruntime_genai` library under `<root>/lib`.
For Windows HIP OGA builds, the same root must also provide these runtime DLLs
under `<root>/bin` or `<root>/lib`: `onnxruntime-genai.dll`,
`onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `amdgpu-ep.dll`, and
`hip-backend.dll`. CMake stages those DLLs next to the built `flm.exe` and into
`src/out` so the app-local DLLs win over older system copies.

`FLM_HIP_OGA_ONLY=ON` is no longer a valid package build path. It previously
produced a run-only `flm`/`flm.exe` from `src/hip_oga_main.cpp`, which could not
run `check`, `list`, `serve`, or OpenAI-compatible API tests. Build the normal
FastFlowLM `flm` target with `-DFLM_ENABLE_HIP_OGA=ON` for every HIP OGA package
candidate. Diagnostic OGA binaries may be carried separately, but they must not
replace `bin/flm.exe` or be documented as the user entry point.

Before a Windows smoke run, validate the actual loader closure rather than PATH
ordering:

```powershell
.\scripts\check-hip-oga-runtime.ps1 `
  -BinaryDir .\build-hip-oga\Release `
  -ExpectedSha256File .\SHA256SUMS
```

The preflight loads `onnxruntime-genai.dll`, reports the process-loaded paths
and SHA256 values for the OGA/ORT/HIP EP DLLs, and fails if the loaded
`onnxruntime.dll` reports a version older than `1.27.0`.

## Select The Backend

Use the default FastFlowLM NPU path by omitting a backend or by setting
`--backend npu`.

Use the HIP OGA path by passing an OGA model directory that contains
`genai_config.json` and tokenizer files:

```shell
flm run /path/to/oga-model-dir --backend hip --prompt "Hello" --max-tokens 1
flm serve /path/to/oga-model-dir --backend hip
```

Direct OGA directories default to `strict_gpu`, which disables ONNX Runtime CPU
EP fallback for every discovered OGA session section. Override only when a
package has been explicitly qualified for hybrid execution:

```shell
export FLM_HIP_OGA_OFFLOAD_POLICY=hybrid_gpu
flm run /path/to/oga-model-dir --backend hip --prompt "Describe this" --max-tokens 8
```

or the environment:

```shell
export FLM_BACKEND=hip
flm run /path/to/oga-model-dir --prompt "Hello" --max-tokens 1
```

The server also accepts a per-request override on generation endpoints:

```json
{
  "model": "/path/to/oga-model-dir",
  "backend": "hip",
  "prompt": "Hello",
  "max_tokens": 1
}
```

## Native Image Inputs

When `flm serve` is running with `--backend hip`, OpenAI-compatible chat
completion requests can include `data:image/png;base64,...`,
`data:image/jpeg;base64,...`, or legacy per-message `images` arrays. The REST
layer normalizes those image entries onto the shared message shape; the HIP OGA
adapter decodes the image payloads back to encoded image bytes, passes them to
`OgaImages::Load`, calls `OgaMultiModalProcessor::ProcessImages`, injects the
returned `OgaNamedTensors` with `OgaGenerator::SetInputs`, and decodes generated
tokens through a processor-backed tokenizer stream.

The existing text-only path is unchanged for prompts without media: FastFlowLM
still applies the chat template, tokenizes with the native OGA tokenizer, and
appends tokens to the generator directly.

Audio inputs fail fast in the HIP OGA backend until an audio-capable package is
verified. Remote image URLs are also rejected; send inline base64 image data or
a local file path from an internal caller that fills `lm_uniform_input_t.images`
with `FILE_NAME` payload types.

FastFlowLM model-list tags can use `--backend hip` only when their metadata
includes a HIP OGA configuration directory. The normal NPU fields remain in the
entry for default backend runs; HIP runs read only the nested `hip_oga` contract,
skip FastFlowLM's NPU downloader, and pass the resolved OGA directory to
ONNX Runtime GenAI.

Every HIP OGA catalog entry has a visible `hip_oga.offload_policy`:

- `strict_gpu`: FastFlowLM overlays every present OGA pipeline section
  (`decoder`, `embedding`, `vision`, `speech`) with AMDGPU/HIP provider options
  and `disable_cpu_ep_fallback=true`, the OGA config equivalent of
  `session.disable_cpu_ep_fallback=1`. Model creation fails if ONNX Runtime
  cannot assign every node to the selected EP.
- `hybrid_gpu`: FastFlowLM overlays every present section with AMDGPU/HIP
  provider options but allows CPU EP fallback. This is only for packages whose
  heavy compute has been proven on AMDGPU/HIP and must be reported as hybrid.
  Release verification must reject candidates when MatMul, Conv, Attention,
  GQA, or vision-encoder main compute runs on CPU.
- `unqualified`: model-list entries without explicit strict or hybrid evidence
  remain unqualified in listing output. Direct OGA directories are reported as
  unqualified for support purposes but run under `strict_gpu` unless
  `FLM_HIP_OGA_OFFLOAD_POLICY=hybrid_gpu` is set.

Relative `hip_oga.config_path` values resolve below `FLM_HIP_OGA_MODEL_ROOT`
when it is set. Otherwise they resolve below the normal FastFlowLM `model_path`
root. Absolute paths are used as-is.

```shell
export FLM_HIP_OGA_MODEL_ROOT=/path/to/oga-models
flm run llama3.2:1b --backend hip --prompt "Hello" --max-tokens 1
```

## Pull A Public AMD HIP OGA Model

Public HIP OGA catalog entries use pinned `amd/*` Hugging Face revisions and
per-file SHA-256 checksums. `flm pull --backend hip` installs them below
`FLM_HIP_OGA_MODEL_ROOT`, verifies the temporary package, then promotes it into
place. The default endpoint is `https://huggingface.co`; set
`FLM_HIP_OGA_HF_ENDPOINT` only for a non-secret test endpoint.

Trial path from a release ZIP on Windows:

```powershell
$env:FLM_HIP_OGA_MODEL_ROOT = "$env:LOCALAPPDATA\FastFlowLM\hip-oga-models"
.\flm.exe pull hip-qwen2.5-coder:14b --backend hip
.\flm.exe run hip-qwen2.5-coder:14b --backend hip --ctx-len 4096 --prompt "Write a Python function that reverses a string." --max-tokens 64
```

The default text/tool demo is `hip-qwen2.5-coder:14b`. Use
`hip-qwen2.5:14b` when the demo should emphasize general instruction/chat
instead of code and tool-oriented prompts:

```powershell
.\flm.exe pull hip-qwen2.5:14b --backend hip
.\flm.exe run hip-qwen2.5:14b --backend hip --ctx-len 4096 --prompt "Summarize why hash-pinned model downloads matter." --max-tokens 64
```

A short `RUN-DEMO` wrapper should set `FLM_HIP_OGA_MODEL_ROOT`, select
`hip-qwen2.5-coder:14b` by default, run `flm pull --backend hip`, then run the
prompt smoke. Keep the normal release ZIP runtime-only; do not embed these
multi-GB weights by default. The public model installer downloads pinned
revisions, safely flattens any catalog `source_subdir`, verifies each listed
file by size and SHA-256, preserves the public original `genai_config.json`,
and relies on the runtime AMDGPU/HIP overlay instead of modifying that file.

The public VLM rows are explicit qualification candidates, not default runtime
claims. `hip-gemma3:4b` is `hybrid_gpu_experimental`,
`hip-qwen3.5-vl:9b` is `hybrid_gpu_candidate`, and
`hip-qwen3.6-vl:35b-a3b` is an optional `hybrid_gpu_stress` download candidate.
They remain marked unqualified and rejected by the prototype runtime until
strict CPU-fallback-off evidence shows zero CPU-assigned graph nodes.

Model metadata can override the OGA provider, profile, config directory, and
offload policy:

```json
{
  "name": "Llama-3.2-1B-NPU2",
  "hip_oga": {
    "provider": "AMDGPU",
    "profile": "hip",
    "config_path": "llama3.2-1b-oga",
    "offload_policy": "strict_gpu",
    "required_files": [
      "genai_config.json",
      "tokenizer.json",
      "tokenizer_config.json"
    ],
    "provider_options": {
      "cache_dir": "/optional/cache/path"
    }
  }
}
```

The default provider is `AMDGPU`, and the default provider option is
`profile=hip`. `config_path` must point at an OGA/ONNX package directory. Tag
routing validates `genai_config.json`, `tokenizer.json`, and
`tokenizer_config.json` before it enters OGA; it also requires
`tokenizer_config.json` to provide `eos_token`, and `eos_token_id` either there
or under `model.eos_token_id` in `genai_config.json`. The chat template can be a
neighboring `chat_template.jinja` file or a string `chat_template` field in
`tokenizer_config.json`. Raw Hugging Face or FastFlowLM NPU model directories
are not enough.

For AMDGPU with `profile=hip`, FastFlowLM rewrites `genai_config.json`
provider options through `OgaConfig::Overlay()` for every present OGA pipeline
section (`decoder`, `embedding`, `vision`, and `speech`) before creating the
model. It does not rely on `AppendProvider("AMDGPU")` for the HIP EP path.

## Experimental Text Prefix Cache

HIP OGA prompt KV reuse is off by default. Enable the experimental sequential
text-only path with:

```shell
export FLM_HIP_OGA_PREFIX_CACHE=experimental
```

The cache applies only to OpenAI-compatible chat completion requests served by
one loaded HIP OGA model instance. It reuses a saved pre-generation prompt
checkpoint only after the server `PromptCache` reports a reusable prefix and the
newly tokenized prompt independently begins with the saved token history. The
adapter then rewinds the OGA generator, appends only the uncached suffix, records
`prompt_tokens_details.cached_tokens`, and logs attempted/hit/fallback counts
with cached and newly-prefilled token totals.

Any image or audio request resets the text checkpoint and takes the normal
full-prefill OGA path. Cache miss, divergent tokens, model/context/signature
changes, insufficient generator capacity, cancellation, or provider exceptions
also rebuild a fresh generator. Rewind/append/provider exceptions disable reuse
for the current loaded model session and retry the full prompt once.

For text requests, the saved native OGA checkpoint capacity is measured only
against prompt tokens; the requested generation budget must not be added to that
capacity check. New text turns allocate their OGA generator to the configured
context window so a later conversation turn can reuse a shorter earlier prompt
checkpoint without tripping a synthetic `capacity_shortage`.

## List, Check, And Server Model IDs

`flm list --backend hip` uses HIP OGA catalog semantics. It lists only
model-list entries that include `hip_oga.config_path`, and the `installed`
status reflects whether the resolved OGA package passes the local HIP OGA
package checks. It does not look at FastFlowLM NPU downloads for HIP status.

```shell
export FLM_HIP_OGA_MODEL_ROOT=/path/to/oga-models
flm list --backend hip
flm list --backend hip --json
flm list --backend hip --filter installed
```

Direct OGA directories are not catalog entries, so they are not discovered by
`flm list`. Validate them explicitly:

```shell
flm check /path/to/oga-model-dir --backend hip
flm check llama3.2:1b --backend hip
```

For `--backend hip`, `flm check` validates the OGA package contract and reports
the resolved directory. It does not perform Hugging Face or ModelScope checksum
verification for OGA files.

When the server is started with `--backend hip`, `/v1/models` advertises model
ids that are loadable through the HIP route: the direct OGA directory used at
startup, if any, plus installed model-list tags that have valid
`hip_oga.config_path` metadata. HIP model listings include `offload_policy`,
`hip_oga.policy_warning`, and `hip_oga.qualification_warning` fields where
applicable, so strict, hybrid, and unqualified packages are visible to
operators. The default NPU catalog listing remains unchanged when the backend is
omitted or set to `npu`.

Enable per-section ORT profiling and verbose session log ids when collecting
CPU placement evidence:

```shell
export FLM_HIP_OGA_DIAGNOSTICS=1
export FLM_HIP_OGA_PROFILE_PREFIX=/tmp/flm-hip-oga-profile
flm run gemma3:4b --backend hip --prompt "Describe this image" --max-tokens 8
```

The profile prefix is suffixed with the OGA section name, for example
`/tmp/flm-hip-oga-profile-vision`. Hardware verification should record the CPU
assigned op types by section from those ORT profiles and logs.

The same direct OGA model directory form is used for normal FastFlowLM HIP
smoke runs:

```shell
flm run /path/to/oga-model-dir --backend hip --prompt "Hello" --max-tokens 1
```

The runtime route must resolve the OGA provider as `AMDGPU` plus `profile=hip`,
so `amdgpu-ep.dll`/`libamdgpu-ep.so`, `hip-backend`, `hipgpu`, `onnxruntime`,
and `onnxruntime-genai` must be ABI-compatible and discoverable in the runtime
path.

## Correctness And Support Scope

This backend is a prototype integration path, not a general model-quality
claim. Validation has covered HIP provider discovery, direct OGA model loading,
FastFlowLM CLI/server routing, simple text prompts, and no-hardware source
checks for the native OGA image path. It does not establish broad answer quality
across models, prompts, quantization formats, or packaged redistribution
channels.

The HIP path supports text prompts and image+text prompts for OGA VLM packages
that provide a working multimodal processor. NPU/AIE/XRT/xclbin execution, ASR,
embeddings, model download/conversion, broad catalog coverage, and production
OpenAI tool-calling behavior remain outside this backend's supported surface.

## OpenAI Tool Calling Status

HIP OGA supports the OpenAI chat-completions tool-call response shape when the
active OGA model package emits the FastFlowLM qwen/gemma-style markup contract:
`<think>...</think>` reasoning blocks and one or more
`<tool_call>{...}</tool_call>` tool blocks. Each tool block must contain a JSON
object with a string `name` and non-null `arguments`. Object/array arguments are
serialized as compact JSON strings; string arguments are passed through as-is.

Non-streaming `/v1/chat/completions` responses return `message.tool_calls` and
`finish_reason: "tool_calls"` for valid tool blocks. Streaming responses emit
`delta.tool_calls` chunks with stable per-stream indexes and unique per-stream
tool-call IDs, then finish with
`finish_reason: "tool_calls"`. Multiple generated tool blocks are preserved in
non-streaming responses and drained from the streaming parser before `[DONE]`.

`tool_choice=auto` is the default. `tool_choice=none` hides the tool schema from
the chat template for that request and reports a `model_output_error` if the HIP
OGA model still emits a tool call. OpenAI `tool_choice="required"` and
per-function object choices are not implemented; the server returns
`unsupported_request_error` instead of silently falling back to `auto`.

Malformed or unterminated generated tool blocks are treated as unsupported model
output and returned as `model_output_error` responses. This separates parser and
server behavior from model/package limitations: hardware smoke still needs a
known-good OGA package that actually emits the documented markup.
