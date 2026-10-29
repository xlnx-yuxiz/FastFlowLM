# FastFlowLM HIP OGA experimental integration

This branch extends the earlier HIP OGA and selectable-codec demos with cache
recovery, image handling, structured output and tools, and selectable common
sampling. It is an experimental source branch. Use the
[HIP OGA guide](docs/docs/instructions/hip-oga.md) for the complete interface
and package restrictions. The upstream NPU installers linked from the main
README do not contain this branch's HIP OGA changes.

## Build and runtime prerequisites

Use the normal FastFlowLM build dependencies, CMake 3.22 or newer, and an
ABI-matched ONNX Runtime GenAI/ONNX Runtime/AMDGPU HIP EP SDK. A Windows build
uses an x64 Visual Studio developer prompt. Initialize the pinned dependencies
before configuring:

```powershell
git submodule update --init --recursive
cmake -S src -B build -A x64 `
  -DFLM_VERSION=0.1.0 `
  -DNPU_VERSION=0.1.0 `
  -DFLM_ENABLE_HIP_OGA=ON `
  -DFLM_USE_HRX=OFF `
  -DONNXRUNTIME_GENAI_HOME=C:\path\to\oga-sdk
cmake --build build --config Release --target flm -- /m:8
```

CMake applies the pinned xgrammar string-bound correction into the build
directory automatically; it does not modify the submodule checkout. See
[patch maintenance](src/common/chat/xgrammar-patches.md).

Stage the complete matching runtime beside `flm.exe`, including its HIP,
rocBLAS and hipBLASLt kernel/data resources. The DLLs staged automatically by
CMake are only part of that closure. Use a separately qualified local OGA
model directory containing `genai_config.json`, the model graphs/weights, and
the matching tokenizer/processor files. Runtime packages and model weights are
not provided by this branch. See the guide's
[build checks](docs/docs/instructions/hip-oga.md#build) and
[public package admission status](docs/docs/instructions/hip-oga.md#pull-a-public-amd-hip-oga-model).

## Text codecs and sampling

From a prepared runtime directory, set `$Model` to your qualified model:

```powershell
$Model = 'C:\path\to\qualified-oga-model'
.\flm.exe run $Model --backend hip --hip-oga-text-codec oga --prompt 'Hello'
.\flm.exe run $Model --backend hip --hip-oga-text-codec flm --prompt 'Hello'
.\flm.exe run $Model --backend hip --hip-oga-text-codec compare --prompt 'Hello'
.\flm.exe run $Model --backend hip --sampler common --prompt 'Name three colors.'
.\flm.exe serve $Model --backend hip --sampler common --port 8000
```

The default codec is `oga`; `flm` uses FastFlowLM/tokenizers-cpp for text;
`compare` runs both codecs, checks parity and uses OGA output. The tokenizer and
decoder can also be selected independently. Image requests retain the OGA
multimodal processor. Codec selection and sampler selection are independent.

`--sampler backend` is the default and preserves the backend's existing
sampling implementation. `--sampler common` uses shared numerical processing,
request history/RNG semantics, and the first common token-selection provider,
which runs on CPU. OGA still executes the model and commits the selected token.
The native FLM sampler also ran on CPU before this change.

For HTTP requests, set `"sampler": "common"` at the top level or in `options`;
top-level values take precedence. Omitting it resets to the server's startup
selection for that request. Interactive `/set sampler common`,
`/set sampler backend`, and `/set seed 42` are available. See
[sampling semantics](docs/docs/instructions/hip-oga.md#request-sampling-experimental-branch).

## Other cumulative features

| Feature | Current behavior |
| --- | --- |
| Prefix cache | Opt-in via `FLM_HIP_OGA_PREFIX_CACHE=experimental`; checks actual token prefixes, media identity and live generator compatibility; includes recovery and conservative no-suffix handling. |
| Image requests | OGA multimodal processing, checked image-feature contracts and bounded media-prefix reuse. Graph/runtime combinations require their own strict GPU qualification. |
| Structured output | Experimental JSON/schema constraints, including Unicode-aware string bounds. |
| Tools | Required/named calls and opt-in structured auto routing for supported HIP text formats; definitions can be supplied directly to `flm run --tools`. |
| Catalog | HIP list/validation/install lifecycle and public metadata; the tracked public Gemma/Qwen exports remain gated where graph/runtime compatibility is unresolved. |
| Embeddings/audio | HIP sentence-embedding and audio execution remain unsupported. |

For a tool demo using a compatible Qwen2.5 Coder package, run from the source
checkout with the prepared `flm` executable on PATH:

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools examples/tool-calling/calculator.json --tool-choice auto --tool-executor demo --prompt 'Use add_numbers to add 17 and 28, then report the result.'
```

See [tool definitions and execution](examples/tool-calling/README.md) for
required mode, call/result/answer events, and custom executor limits.

## Sampling validation and limits

Recorded development validation covered retained Gemma3 4B and Qwen2.5 Coder
14B HIP text packages across 30 lanes: 256 common-sampling HTTP requests,
236 compatibility requests, 23 recovery-probe records, CLI, EOS/rewind, cache,
seed replay, cancellation and disconnect handling. Actual constrained tools
were exercised on Qwen; Gemma exercised unsupported-tool rejection and recovery.
These counts describe the tested package/runtime combinations, not every model
with the same family name or public download tag.

The native `Sampler::sample(buffer<bf16>&)` adapter passed 7 common-provider
cases and 4 legacy cases using synthetic logits. AutoModel boundary tests used
a test model. **Real NPU model generation with common sampling has not been
qualified.** Models with incomplete logical-history paths reject common
sampling explicitly; the guide lists those models.

Common-provider VLM execution and long-duration resource behavior have not
been qualified. There is no common GPU sampler provider yet. In bounded
64-token decode measurements (median of three measured runs, cache and
diagnostics off), common sampling increased decode time per token by roughly
23–32% for Gemma and 5–6% for Qwen. The implementation does not promise equal
performance or bitwise model outputs across devices/providers.

The standalone [common sampling tests](src/test/common_sampling/README.md)
describe how to check the numerical core, native adapter and HTTP boundary.
Validation of a new package must include its actual runtime and model files.

This branch is based on the public selectable-codec demo at
`ce17c7c74047bf27fddc36bf0842fc129a76ec61`, whose original FastFlowLM base was
`06513091bf84044a8f9d98ebccd86f04fe5b4ac2`. It is not rebased onto current
upstream main and is not a production release.
