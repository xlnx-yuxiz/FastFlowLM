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
`ONNXRUNTIME_GENAI_HOME`, `OGA_HOME`, or `CONDA_PREFIX`. This source branch
assumes an already obtained compatible AMDGPU/HIP EP SDK; it does not publish
or download the qualified SDK bundle. A generic ONNX Runtime GenAI installation
does not establish HIP EP compatibility:

```shell
git submodule update --init --recursive
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
  -BinaryDir .\build-hip-oga\Release
```

The preflight loads `onnxruntime-genai.dll`, reports the process-loaded paths
and SHA256 values for the OGA/ORT/HIP EP DLLs, and fails if the loaded
`onnxruntime.dll` reports a version older than `1.27.0`. To compare against an
existing trusted SDK checksum manifest, also pass `-ExpectedSha256File` with
that file's path. The manifest is not supplied by this source branch; without
it, the preflight reports hashes but does not attest package identity.

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

For prompts without media, FastFlowLM applies the chat template, tokenizes with
the selected text codec (OGA by default), and appends tokens to the generator
directly.

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

- `strict_gpu`: FastFlowLM configures every present OGA pipeline section
  (`decoder`, `embedding`, `vision`, `speech`) with AMDGPU/HIP provider options
  and the string session option `"session.disable_cpu_ep_fallback": "1"`.
  Model creation fails if ONNX Runtime
  cannot assign every node to the selected EP.
- `hybrid_gpu`: FastFlowLM configures every present section with AMDGPU/HIP
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

The two actively verified public families are Qwen2.5 Coder 14B and Gemma3 4B.
Their current public packages are recorded but **gated from pull/run by tag**:

| Public tag | Current runtime admission |
| --- | --- |
| `hip-qwen2.5-coder:14b` | The exact public decoder fails a HIP input-rank check. A separate graph with only the unused `inputs_embeds` input removed passes bounded OGA/FLM/compare text checks and strict GPU profiling; the public original is still gated. |
| `hip-qwen2.5:14b` | Shares the same decoder graph hash as the failing Coder export; its distinct weights have not been qualified here. |
| `hip-gemma3:4b` | Public image requests pass strict GPU checks, but text-only requests fail with `image_features` rank 2 versus 3. Adapted flat-feature exports require separate qualification. |

A matching model name or weight hash does not qualify a different graph/config.
Use a separately qualified local OGA directory with the direct-path commands
above while public graph/runtime compatibility is resolved. The catalog retains
public source hashes unchanged; it does not silently rewrite a downloaded graph.

For an admitted catalog entry, the installer safely flattens `source_subdir`,
verifies every file by size and SHA256, preserves `genai_config.json`, and supplies
runtime AMDGPU/HIP settings through a separate effective config. Keep the release
ZIP runtime-only; multi-GB model weights are downloaded separately.

Lifecycle command syntax for an admitted catalog tag (the selected public tags
above remain gated; managed removal also works after admission is withdrawn):

```powershell
.\flm.exe list --backend hip --filter installed --json
.\flm.exe check <qualified-tag> --backend hip
.\flm.exe remove hip-qwen2.5-coder:14b --backend hip
```

`check` reads every pinned payload and verifies its size and SHA256 without
modifying files. `list` remains a fast package-presence check. A valid manually
provided package may pass `check`; that does not give the installer ownership
of the directory. Direct-directory checks validate the local OGA structure and
do not imply a match to a public revision.

`remove` targets only a catalog package below the explicitly configured
`FLM_HIP_OGA_MODEL_ROOT`, with an install receipt matching that catalog revision.
It refuses direct directories, symlinks/junctions, missing or mismatched
receipts, and untracked files or directories. Corrupted managed payloads can
still be removed. Repeating removal of an absent package succeeds. Runtime
admission status does not prevent removal of an already managed package.

Repeated `pull` verifies the existing payload before reusing it. Replacement,
including `--force`, requires matching ownership metadata and preserves an old
installation if downloading or verification fails. Before changing a pinned
catalog revision, remove the managed old revision using its original catalog;
automatic migration between different install receipts is not implemented.
Downloads preflight the declared package size plus a 64 MiB filesystem margin.
An existing package stays on disk while its replacement is staged.

The public VLM rows are explicit qualification candidates, not default runtime
claims. `hip-gemma3:4b` is `hybrid_gpu_experimental`,
`hip-qwen3.5-vl:9b` is `hybrid_gpu_candidate`, and
`hip-qwen3.6-vl:35b-a3b` is an optional `hybrid_gpu_stress` download candidate.
They remain marked unqualified and rejected by the prototype runtime until
the exact package passes its text/image contract and strict CPU-fallback-off
profiling. GPU node placement alone does not prove successful generation.

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

For AMDGPU with `profile=hip`, FastFlowLM prepares the complete effective
`genai_config.json` before `OgaConfig::Create()` for every present OGA pipeline
section (`decoder`, `embedding`, `vision`, and `speech`). This also applies to
diagnostics: post-create `OgaConfig::Overlay()` can discard the provider lists
of optional multimodal sessions in the supported OGA runtime.

The effective config lives in a unique temporary model directory. All other
assets are hardlinked, preserving relative paths and avoiding weight copies.
The model directory's parent must be writable, or the OS temporary directory
must support hardlinks to the source volume. Nested directory links are
rejected. Original model files remain unchanged; normal model release removes
the temporary directory after all OGA objects have been destroyed.

## Text Codec Selection

HIP OGA defaults to OGA encoding and decoding. The normal full CLI supports
these process-wide options for both `run` and `serve`:

```shell
flm run MODEL --backend hip --hip-oga-text-codec oga
flm run MODEL --backend hip --hip-oga-text-codec flm
flm run MODEL --backend hip --hip-oga-text-codec compare
flm serve MODEL --backend hip --hip-oga-text-codec flm --hip-oga-decoder compare
```

| Mode | Encoding / decoding | Output authority | Runtime parity check |
| --- | --- | --- | --- |
| `oga` (default) | OGA | OGA | None |
| `flm` | FastFlowLM/tokenizers-cpp | FLM | None; skips OGA text codec calls |
| `compare` | Both | OGA | Exact prompt IDs and accepted final decoder equivalence |
| `external` (legacy) | Both | FLM | Same mandatory checks as the recovered experiment |

`--hip-oga-tokenizer` and `--hip-oga-decoder` override the combined option for
one layer. The chat template remains FastFlowLM/minja, and OGA generates all
tokens in every mode. Full `flm` text mode does not instantiate `OgaTokenizer`
or its stream; a mixed mode may need OGA for the selected layer. FLM uses the
current model's tokenizer bundle, which must be compatible with its embeddings.
Changing the codec does not convert token IDs between vocabularies.

Configuration priority is per-layer CLI > combined CLI > environment > OGA.
Environment-only configuration is also available:

```shell
export FLM_HIP_OGA_TEXT_CODEC=flm
# Optional per-layer overrides:
export FLM_HIP_OGA_TOKENIZER_PROVIDER=flm
export FLM_HIP_OGA_DECODER_PROVIDER=compare
```

Per-layer environment variables override `FLM_HIP_OGA_TEXT_CODEC`. Unset values,
`default`, `oga`, and `native-oga` select OGA. Unknown values fail with an error.
The legacy `FLM_HIP_OGA_TEXT_CODEC_COMPARE=1` overrides environment selectors
with `compare` for both layers, preserving OGA authority. Explicit CLI options
override this legacy flag for the selected layers. For example,
`--hip-oga-tokenizer flm` with that flag set uses FLM encoding and decoder
comparison. Options are resolved before server workers or models are created;
there is no per-request codec switch. Effective selectors are logged on load.

The legacy `external` value retains checked FLM output for existing scripts;
use `flm` for operation without OGA codec parity checks. In `compare` and
`external`, prompt-ID mismatch aborts generation. Decoder comparison runs on
the same OGA-generated IDs, using cumulative `Tokenizer::decode(vector<int>)`.
Streaming chunk boundaries may differ; final text determines equivalence.

Plain-text decoder parity uses four classes: `byte_exact`,
`canonical_equivalent`, `structured_equivalent`, and `incompatible`.
`byte_exact` is the expected result for Qwen-style runs. Gemma/SentencePiece
runs may be `canonical_equivalent` when the tokenizer/model contract explicitly
identifies Gemma plus a metaspace decoder and the only visible difference is
standalone `U+2581` rendered as a normal user-visible space. FastFlowLM records
the raw OGA stream, OGA full decode, FastFlowLM full decode, external cumulative
decode, token IDs, hashes, first mismatch, and normalization applied when
`FLM_HIP_OGA_DECODER_DISCRIMINATOR_JSON` is set and decoder comparison is active.
This diagnostic variable does not enable comparison in `flm` mode.

Structured JSON/tool/code-like outputs stay strict. Parsed JSON may classify as
`structured_equivalent` for formatting-only differences, but string values are
not text-normalized; tool-call markup and code blocks must be byte-exact or
they are incompatible. Image turns use the OGA multimodal processor for image
preprocessing and prompt token expansion. Output follows the selected decoder:
`oga` uses the processor's OGA stream, `flm` uses FLM cumulative decoding,
`compare` returns OGA output with FLM parity checking, and legacy `external`
returns FLM output with OGA parity checking. The tokenizer selector still applies
to text-only requests; it does not replace the image processor.

For example, serve a local VLM package with FLM image output decoding:

```shell
flm serve /path/to/oga-vlm-model --backend hip \
  --hip-oga-tokenizer oga --hip-oga-decoder flm
```

Set `FLM_HIP_OGA_DIAGNOSTICS=1` to record the effective input processor,
requested tokenizer, output decoder, and actual per-request decoder call counts.
Compare mode checks final text equivalence, including image requests; matching
stream chunk boundaries are not required.

Audio inputs remain unsupported. VLM coverage is bounded and model-specific;
see the [token-boundary evaluation](../../architecture/hip-oga-token-boundary-evaluation.md).

## Experimental Text Prefix Cache

HIP OGA prompt KV reuse is off by default. Enable the experimental sequential
text and qualified image-history path with:

```shell
export FLM_HIP_OGA_PREFIX_CACHE=experimental
```

The cache applies only to OpenAI-compatible chat completion requests served by
one loaded HIP OGA model instance. It reuses a saved pre-generation prompt
checkpoint after the server grants eligibility and the backend verifies that the
actual prompt tokens begin with the saved token history. For text, eligibility
comes from the server `PromptCache`; image requests are evaluated by the backend
using the actual processor inputs. The
adapter then rewinds the OGA generator, appends only the uncached suffix, records
`prompt_tokens_details.cached_tokens`, and logs attempted/hit/fallback counts
with cached and newly-prefilled token totals.

Codec names, comparison mode, and tokenizer file fingerprints are diagnostics,
not KV-cache compatibility keys. The cache manager checks actual token IDs and
backend execution state independently. Under the same loaded model/context,
identical token prefixes remain eligible regardless of which codec produced
them; divergent token prefixes miss automatically. A decoder-only change cannot
invalidate KV. Generator capacity and successful OGA checkpoint restoration
remain required.

Image and text checkpoints occupy separate domains. Image reuse is limited to
the tested Gemma3 processor contract with explicit `search.past_present_share_buffer=true`.
The manager compares ordered image bytes, effective processor configuration,
fixed `pixel_values`/`num_image_tokens` tensors, expanded `input_ids`, and the cached
prefix of `token_type_ids`. New suffix token types must be zero (text). Changed
media, image order or layout, unknown processor inputs, and other model types
take a fresh prefill. Filenames and app codec selectors cannot authorize reuse.

The image processor runs once per request before deciding reuse. A restored
generator avoids repeating the cached image-prefix inference; it does not avoid
image file decoding or processor work. Reuse failures rebuild once from the
same owned input snapshot. Cancelled or failed prefill never commits a media
checkpoint, and generation cancellation clears it. The single checkpoint keeps
encoded image bytes and fixed processor tensors in memory, and opted-in image
generators reserve the configured context capacity.

Audio requests are rejected. Cache miss, divergent tokens, model/context/signature
changes, insufficient generator capacity, cancellation, or provider exceptions
also rebuild a fresh generator. Rewind/append/provider exceptions disable reuse
for the current loaded model session and retry the full prompt once.

Identical prompts with an empty suffix deliberately miss. Rewind restoration is
checked against a fresh generator with identical image-prefix/text-suffix prefill
segmentation. On the tested HIP runtime, one-shot and segmented GPU prefills can
produce slightly different logits and longer greedy wording; arbitrary long
cache-on/off outputs are not promised to be bit-identical. Keep this numerical
boundary separate from restoration correctness and strict GPU placement.

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

Adapted Gemma3 packages whose vision output and embedding input both expose
rank-2 `[total_image_tokens, hidden_size]` features require the following
`flm_hip_oga.json` beside `genai_config.json`:

```json
{"schema_version": 1, "image_features": "gemma3_flat_tokens_v1"}
```

This package contract enables a checked conversion of the processor's per-image
256-token count to the total feature rows, including when prefix cache is off.
FLM checks the pixel tensor's image count and token-type counts before conversion.
For Gemma, it also checks that prompt IDs and token types have matching lengths,
that image placeholders match the image mask, and that the total agrees with
the processor's image count. Processor `input_ids` must be int32 for the pinned
OGA `SetInputs` interface. These checks run before changing live prefix KV.
The effective count and package contract enter the media cache identity. Use this
file only with that adapted graph interface: native rank-3 Gemma packages retain
the per-image count and must not declare the flat contract. The earlier rank-2
variant without this metadata fails on multiple images with the installed OGA
runtime. This repository does not provide a graph-conversion recipe for that
variant. The metadata declares an existing graph contract and does not convert
or qualify the model.

For native rank-3 Gemma text-only turns, the adapter supplies owned empty image
inputs with a retained batch dimension, alongside the exact int32 prompt IDs.
The empty pixel tensor uses float16; this repair targets the pinned Gemma
package with float16 pixel inputs. Native packages using other pixel types
require separate no-image qualification.
This avoids the pinned runtime's rank-2 empty-feature default. It does not modify
the original model graphs or weights; package and runtime qualification are
still required.

A missing sidecar, JSON `null`, or `{}` retains native behavior. A present file
must otherwise contain the supported contract. Malformed JSON, a zero-byte file,
non-object values, and unsupported fields or versions fail model loading.

The six-case image-order and message-layout regression is tracked at
`src/test/hip_oga_backend_static/test_hip_oga_media_layout_api.py`. Run it against
a fresh, separately managed HIP server whose cache mode matches `--cache`:

```shell
python src/test/hip_oga_backend_static/test_hip_oga_media_layout_api.py \
  --source-root . --base-url http://127.0.0.1:18579 \
  --model /path/to/qualified-oga-model --cache on --output /path/to/layout-results
```

The helper records all six request/response pairs, image hashes, visual checks
and cached-token counts. Cache-off must report zero reuse; cache-on requires a
history-extension hit. The caller owns server lifecycle and separate ORT
placement checks. Start a new model session after a corrupt-image test, since
that error intentionally disables cache for the existing session.

This backend is a prototype integration path, not a general model-quality
claim. Development validation covers retained text and image packages, strict
HIP execution, CLI/server routing, cache behavior and the feature-specific
checks described below. It does not establish broad answer quality
across models, prompts, quantization formats, or packaged redistribution
channels.

The HIP path supports text prompts and image+text prompts for OGA VLM packages
that provide a working multimodal processor. NPU/AIE/XRT/xclbin execution, audio/ASR,
sentence embeddings, graph conversion and broad catalog coverage remain outside
this backend's supported surface. Catalog installation is limited to admitted
packages. Experimental structured output and text tools have the model/format
restrictions described below.

For a server started with `--backend hip`, `POST /v1/embeddings` and
`POST /api/embeddings` return HTTP 501 with an `unsupported_feature` error.
An OGA VLM package's `embedding.onnx` supplies token vectors to its decoder;
it does not provide the sentence vectors expected by these endpoints.
The native FLM embedding interface remains `flm serve <chat-model> --embed 1`.

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

When `tool_routing` is omitted or set to `"template"`, tool selection retains
the existing template-based behavior; `tool_choice="auto"` is the default.
`tool_choice="none"` hides the tool schema from the chat template for that
request and reports a `model_output_error` if the HIP OGA model still emits a
tool call. OpenAI `tool_choice="required"` and
per-function object choices use request-scoped grammar constraints in this
experimental implementation. A required choice must produce at least one declared call; a named
choice allows exactly one call to that function. Explicit
`parallel_tool_calls: false` limits constrained tool generation to one call.
The model's detected chat format must define tool syntax. Requests that name an
undeclared function, omit required tools, or require unsupported tool syntax
return HTTP 400 before streaming starts.

Malformed or unterminated generated tool blocks are treated as unsupported model
output and returned as `model_output_error` responses. This separates parser and
server behavior from model/package limitations: hardware smoke still needs a
known-good OGA package that actually emits the documented markup.

## Structured Output (experimental)

Qualification is limited to the model packages, runtime and source revisions
identified in the regression receipts. Earlier text/VLM results do not
automatically qualify a changed implementation. The request interface is the
OpenAI `/v1/chat/completions` body, independent of the process-wide text codec:

```json
{
  "model": "YOUR_LOCAL_OGA_MODEL",
  "messages": [{"role": "user", "content": "Return a status object."}],
  "temperature": 0,
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "status",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": false
      }
    }
  }
}
```

Use `{"type":"json_object"}` for a JSON object without an application schema,
or `{"type":"text"}` for ordinary text. Malformed response formats and schemas
outside the supported subset fail with HTTP 400 before streaming. Combining a JSON
response format with required or named tool choice is currently rejected.
Explicit constraints on a backend without this integration are also rejected.

The initial supported schema subset is a single `type`, `const` or `enum`,
object `properties`/`required`/`additionalProperties`, array `items` and
`minItems`/`maxItems`, and string `minLength`/`maxLength`. Required properties
must be declared. Literal `const`/`enum` schemas may also specify a matching
`type`, but cannot combine other restrictions. Nested schemas follow the same
rules; property names and literal object values are ordinary data. Descriptive
annotations such as `title` and `description` do not constrain output.
References, composition, conditionals, uniqueness, patterns/formats and numeric
bounds/multiples are rejected because the pinned compiler can weaken some of
these combinations. Strict tool argument schemas use the same admission rules.
Constrained output is initially limited to content-only and Hermes chat formats;
other dialects return an explicit error. JSON strings preserve literal tool and
reasoning markers; an actual leading reasoning block remains a separate field.


The adapter applies a vocabulary mask before OGA selects each constrained token.
It creates a fresh matcher for every request, including cache hits. Changing a
schema does not invalidate compatible KV state: the cache manager still checks
the effective token prefix and generator state, while the new matcher enforces
the new response contract. A grammar/tokenizer mismatch aborts generation.

Streaming uses the same constraint. A response ending with `length` or `cancel`
can contain incomplete JSON; clients must check `finish_reason` before using a
completed structured result. The default template-auto path keeps its existing
behavior unless an explicit constraint selects grammar decoding.

## Request Sampling (experimental branch)

`flm run` and `flm serve` accept `--sampler backend|common`. The default,
`backend`, preserves the native FLM or OGA sampling implementation. `common`
selects FLM's shared CPU reference sampler independently of `--backend` and
text codec selection:

```sh
flm run /path/to/qualified-oga-model --backend hip --sampler common
flm serve /path/to/qualified-oga-model --backend hip --sampler common --port 8000
```

HTTP requests can set `"sampler":"common"` or `"sampler":"backend"` at the
top level or in `options`. Top-level values take precedence. An omitted
sampler returns to the server's startup selection on each request. Interactive
`/set sampler common` and `/set sampler backend` persist for subsequent turns
of the loaded model. Unknown values fail explicitly; there is no silent fallback.

Common sampling owns the numerical processors and final token selection. It
uses the current logical sequence (prompt plus committed continuation), a
request-scoped RNG, and the processing order documented below. Greedy ties
choose the lowest token ID and do not consume RNG. Fixed seeds reproduce
selection for identical logits, history and options; cross-device bitwise
model output equality is not promised. The first provider uses CPU logits;
this interface does not require future GPU providers to copy logits to CPU.

On HIP, OGA supplies logits and commits the selected token through a greedy
single-candidate step. Its sampling filters are neutralized. EOS is verified
through OGA's returned token and done state because this runtime does not
append EOS to its sequence. The existing conservative seeded-request cache
policy remains in effect. Provider changes are checked by the cache manager
as part of live generator compatibility.

Native common sampling is guarded off for Qwen3, Qwen3_TK, Qwen3.5VL,
Qwen3.6 MOE, Qwen3.5 Omni, LFM2.5_TK and Gemma4_12B while their custom logical
history paths await repair. Host-level native Sampler tests do not constitute
NPU model qualification. Real-model qualification for this change is tracked
separately for the retained Gemma/Qwen HIP packages. See the
[sampling tests](../../../src/test/common_sampling/README.md) for standalone
core/native test commands and the bounded HTTP fixture. Common-provider VLM
execution and long-duration resource behavior remain unqualified.

The following OGA-native selection details describe the default `backend`
provider; `common` shares the same request fields while owning final sampling.

The HIP OGA path resolves each HTTP request against the loaded OGA model's
search defaults. Sampling values inside `options` are accepted; a top-level
field overrides the same nested field. Omitted fields revert to the model
defaults instead of retaining values from the preceding request. This is
independent of `oga`, `flm`, or `compare` text codec selection.

The adapter forwards `temperature`, `top_k`, `top_p`, `repetition_penalty`, and
`seed` (OGA `random_seed`). `temperature=0` selects greedy generation. Explicit
positive temperature or a top-k/top-p setting enables sampling, unless the
request sets `do_sample=false`; temperature zero always remains greedy.
`seed` is an integer from -1 (random) through 2147483647. Top-k must be an
integer within the model vocabulary size. Numbers must be finite and fit
float32; invalid inputs return HTTP 400 before generation or stream startup.

The adapter also supports `min_p` in [0, 1] and signed
`presence_penalty`/`frequency_penalty`. These values must be finite float32
numbers; nonzero values that underflow to float32 zero are rejected. Penalties
are not restricted to [-2, 2]. Zero is a no-op and is omitted from the canonical
settings. Stochastic `top_k=0` with `0 < top_p < 1` uses the adapter's nucleus
filter, because the pinned raw OGA runtime does not honor that combination.

Requests needing these extensions use the existing logits hook. Processing
order is grammar/decoder and model EOS eligibility masks, repetition penalty, presence and frequency
penalties, temperature, top-k, top-p, then min-p. Top-p retains the candidate
that crosses the cumulative mass, after normalization over the top-k survivors.
Min-p retains candidates with probability at least `min_p` times the highest
probability. Equal scores are ordered by token ID for filtering. `top_k=0`,
`top_p=0` or `1`, and `min_p=0` disable the corresponding filters. Greedy decoding
applies penalties and ignores temperature and probability filters. Masked tokens
remain banned even with negative penalties. OGA still chooses the final token
and owns the random generator; its corresponding filters are neutralized to
avoid applying a processor twice. Ordinary supported requests retain the
original OGA sampling path.

Model `min_length` counts the complete current sequence, including the prompt.
Before that length, all configured EOS IDs are masked before top-k/top-p/min-p
filtering; at the boundary EOS becomes eligible again. This prevents early
termination when an EOS score would otherwise eliminate every non-EOS candidate.
The extension path requires model `no_repeat_ngram_size=0`: the pinned runtime
accepts a nonzero value but does not enforce it in the controlled reference test.
Such a model configuration is rejected for extension requests. Nonzero n-gram
blocking on the ordinary OGA path is also unqualified.

The penalty history is the complete current OGA sequence: prompt plus accepted
generated tokens, including the suffix after a cache restore. Repetition is
applied once per distinct token, presence once, and frequency once per
occurrence. This matches the existing HIP repetition scope. Native FLM uses a
generated-token history window (default 64); identical sampling across the two
backends is not promised.

For example, the existing chat API accepts these fields at the top level or
inside `options`:

```json
{
  "model": "/path/to/qualified-oga-model",
  "messages": [{"role": "user", "content": "Name a color."}],
  "temperature": 0.8,
  "top_k": 0,
  "top_p": 0.9,
  "min_p": 0.05,
  "presence_penalty": 0.2,
  "frequency_penalty": 0.1,
  "seed": 42,
  "max_tokens": 32
}
```

CLI `/set topk`, `/set topp`, `/set temp`, `/set rep-pen`, and `/set minp`
update the settings for subsequent turns. `/set seed 123` selects a fixed seed,
and `/set seed -1` restores random seeding. Invalid values leave the preceding
settings intact and keep the interactive CLI alive. CLI settings persist
until the model instance is replaced.

Sampling configuration is generator state. The cache manager checks effective
search settings along with token-prefix and backend compatibility. Fixed-seed
stochastic requests use a fresh generator because the supported OGA rewind API
does not reset RNG state. Greedy requests retain experimental prefix reuse.
`FLM_HIP_OGA_DIAGNOSTICS=1` prints application-effective settings and whether the
generator was reused. For extended requests a separate `sampling runtime=`
line shows the actual neutralized OGA search readback. The retained raw OGA
pure-top-p failure is a runtime boundary observation; adapter qualification
uses the numerical and controlled-runtime tests in
`src/test/hip_oga_sampling_runtime` and the sampling HTTP/CLI fixtures.

## Shared server engine and cancellation (experimental branch)

The server admits one text request at a time across `/api/generate`, `/api/chat`,
`/v1/completions`, and `/v1/chat/completions`. The request owns the engine through
model loading, configuration, prefill, generation and cleanup. An overlapping
HIP request receives HTTP 503 with `error.type="engine_busy"` and `Retry-After: 1`;
clients should retry after the active request completes. There is no HIP wait
queue or parallel inference. Backend overrides use the same engine guard.

Give each text request a nonempty string `request_id` when cancellation is
needed. A duplicate active ID returns HTTP 409 without replacing its owner.
POST `/api/cancel` with `{"request_id":"your-request-id"}` to signal cancellation.
The response's `cancelled` field indicates whether a live text request was
found; model loading and an in-flight device operation may finish before the
handler observes that signal. The final generation response uses finish reason
`cancel`. Both streaming and nonstreaming requests honor cancellation, and
disconnecting an active client also signals it. Partial context and prompt-cache
metadata are cleared before another request can use the engine.

`/api/version`, `/v1/models`, and `/api/cancel` remain available during generation.
`/api/ps` reads mutable loaded-model state and returns 503 while the engine is
busy. The existing native NPU wait queue is retained; cancelling requests still
waiting in that native queue, ASR cancellation, and embedding cancellation are
outside this text-path contract. Long-session cache recovery and VLM require
separate qualification.


One-shot `run --prompt` failures return a nonzero status through stack unwinding,
so the loaded model and its temporary linked config directory are released.
Forced process termination can still bypass destructors; test controllers must
clean only their own verified run directories after stopping their processes.

## Client-managed tool loops (experimental branch)

The OpenAI chat endpoint returns tool calls; the client executes each tool and
sends its result back in a `role: "tool"` message with the matching `tool_call_id`.
Assistant `content` may be null or omitted. HIP requests reject orphan, unknown,
duplicate or unresolved tool results with HTTP 400 before generation or SSE
startup. Completed result batches are ordered by their call IDs before rendering
model templates that omit those IDs. This admission check does not execute tools.

The pinned Qwen2.5 Coder package has an unresolved limitation in the existing
template-auto route: first-turn prompts can produce function descriptions as
ordinary text. A direct OGA probe reproduced this behavior with the same
rendered prompt. Applications must not execute that ordinary text as a tool
call. This finding does not qualify the separate structured-auto route below.

The agent-loop fixture therefore has an explicit `tool_only` client policy. It
sends `tool_choice: "required"` and `parallel_tool_calls: false`, advertises its
ordinary fixture tools plus an application-defined `finish_task` function, and
lets the model choose each call, its arguments and when to finish. `finish_task`
is a client convention, not a built-in FLM tool or server configuration. Its
arguments contain the final answer; the client acknowledges the call with a
matching tool result before sending a follow-up user message.

Consume the complete stream and check its finish reason before executing a call.
A cancelled tool stream may contain incomplete arguments and ends with `cancel`;
discard that partial call. Cancellation does not turn an incomplete block into
a model-output error. An error already detected before cancellation remains an
error. Retry using the last complete valid conversation history.

The bounded test client is `src/test/hip_oga_backend_static/test_hip_oga_agent_api.py`:

```powershell
& <sdk-python> <source>/src/test/hip_oga_backend_static/test_hip_oga_agent_api.py `
  --base-url http://127.0.0.1:18579 --model <local-model-directory> `
  --output <fresh-output-directory> --phase full --cache on `
  --client openai --tool-policy tool_only
```

Start the server separately with the matching cache setting and desired
`--hip-oga-text-codec oga|flm|compare`. The fixture runs seven scenarios: single
read, streamed read, two-step and three-step workflows, recovery from a missing
file, rereading a changed file, and live cancellation followed by a fresh read.
It executes only allowlisted tools against temporary fixture files. Repeat with
fresh servers for OGA/FLM/compare codecs and cache off/on, recording each server
command and the source, binary, runtime and model hashes. A successful
`tool_only` matrix qualifies the
required-call fixture only; it does not qualify structured-auto routing,
arbitrary models, parallel tools or long-running agents.

## Structured auto tool routing (opt-in)

The explicit `/v1/chat/completions` opt-in is `tool_choice: "auto"` together
with `tool_routing: "structured"` and a nonempty `tools` array. Omitting
`tool_routing` or setting it to `"template"` preserves the existing behavior
described above. For example, this request may return a greeting directly or
select `add_numbers` if the user asks for arithmetic:

```json
{
  "model": "YOUR_LOCAL_OGA_MODEL",
  "messages": [{"role": "user", "content": "Say hello briefly."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "add_numbers",
      "description": "Add two integers.",
      "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": false
      }
    }
  }],
  "tool_choice": "auto",
  "tool_routing": "structured",
  "parallel_tool_calls": false
}
```

Structured auto constrains each response to either an ordinary answer or one
declared function call. An ordinary answer is returned as `message.content` with
`finish_reason: "stop"`; it does not create or require a `finish_task` call. A
tool decision is returned through the standard `message.tool_calls` shape with
`finish_reason: "tool_calls"`. FLM returns the selected call and does not execute arbitrary functions; clients must provide an executor
for custom functions. A definition by itself describes the interface only.

For `stream: true`, V1 buffers the complete structured response before emitting
SSE content. It does not stream partial routing JSON or incomplete tool
arguments; the client receives the resolved answer or complete tool call after
generation finishes.

The pinned Qwen2.5 Coder 14B HIP package passed six codec/cache combinations
with eight scenarios each (126 model requests and 72 actual tool executions),
21 API controls, and four native CLI checks. This is separate from
the historical required-mode qualification. Structured auto supports HIP text
with a non-forced Hermes format; other models need their own qualification.
Constrained decoding remains unavailable on the native NPU path.

## Native CLI tool definitions and execution

The development CLI accepts a tool definition file directly. The default
`--tool-choice required` is unchanged; `--tool-choice auto` explicitly selects
the new structured-auto route when used with `--tools`:

```powershell
flm run <local-oga-model> --backend hip --ctx-len 2048 --tools examples/tool-calling/calculator.json --tool-executor demo --prompt "Use add_numbers to add 17 and 28, then report the result."
```

Omit `--prompt` to enter tasks interactively. `--tools` with the default required
choice emits a selected function call as assistant JSON. Add
`--tool-executor demo` to execute the registered arithmetic and record-reading
tools and display `[call]`, `[result]`, and `[answer]` events. In required mode,
the native demo executor adds `finish_task`; structured auto answers directly
when finished and does not add that function. The demo executor implements only
the registered example signatures; custom JSON definitions require an
implementation in the caller. See `examples/tool-calling/README.md` for the
required and auto CLI examples and limits. Structured auto remains limited to
HIP text with non-forced Hermes formats.

String `minLength` and `maxLength` constraints count decoded Unicode code points
in both user-supplied tool parameters and `response_format`, including nested
strings. Legal JSON escapes are supported: `\n` and `\\` each count as one,
`\u4e2d` counts as one, and a valid surrogate pair such as `\uD83D\uDE00`
counts as one. Literal UTF-8 uses the same counting rule; combining sequences
remain separate code points. Bounded strings reject unpaired surrogate escapes
and unescaped U+0000–U+001F controls. Existing schema admission limits remain.

CMake applies a pinned source patch automatically when configuring the shared
`flm_xgrammar` target. It writes the corrected converter into the build tree,
keeps the submodule unchanged, and rejects unexpected upstream source contents.
Normal build commands need no new option or external patch utility. See
`src/common/chat/xgrammar-patches.md` for dependency-update instructions and
`src/test/hip_oga_backend_static/test_hip_oga_string_bounds_api.py` for the real
JSON/SSE, codec and cache regression client.
