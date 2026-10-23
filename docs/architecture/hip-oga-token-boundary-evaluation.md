# HIP OGA Text Codec Evaluation

Assessment date: 2026-10-23. Status: experimental text boundary is viable for the
tested Qwen2.5 and Gemma3 packages; production qualification remains open.

## Decision

Keep the default OGA codec and the opt-in FastFlowLM codec implementation.
The evidence supports separating text encoding/decoding from token generation:

```text
FastFlowLM/minja template
  -> OGA or FastFlowLM tokenizer
  -> OGA AppendTokens / GenerateNextToken
  -> OGA or FastFlowLM cumulative decoder
  -> text
```

OGA remains responsible for model execution, HIP EP, sampling, generated IDs,
and generator/KV state. FastFlowLM can supply model-compatible prompt IDs and
decode the resulting IDs. Promote this ownership boundary into a future
`PromptFrontend`, `TextTokenizer`, `GenerationBackend`, and `TextDecoder`
design, while retaining the current adapter hooks for this experiment. This
report does not introduce those public interfaces or an arbitrary-bundle API.

A tokenizer bundle must preserve the model's vocabulary-to-ID mapping, special
tokens, normalization, chat template, and decoding semantics. Supporting a
different implementation does not permit swapping in an unrelated vocabulary.

## Recovered Source

The accepted implementation is `fix/hip-oga-canonical-decoder-equivalence` at
`a2155f6c8c731f1f59f6cbfb447bb68844c304a7`. Its codec delta starts after
`07fefdbc699eebff68672737e6bd52f7a6e50b83` and includes the tokenizer probe,
cumulative decoder, selector integration, canonical equivalence policy,
discriminator capture, and regression fixtures.

The continuation applies only that delta to GitHub demo base
`2eb85e7bbea04552ea471668bceafe9b09c2c577`. It preserves the demo's instructions,
provenance files, public-host checks, and existing prefix-cache integration.
Runtime codec source matches the accepted implementation; this import does not
add a new tokenizer algorithm or generation path. Existing build directories
must rerun CMake to discover the added `hip_oga_external_tokenizer.cpp`.

## Delivered Selection And Ownership

| Mode | Prompt IDs | Output text | OGA codec calls |
| --- | --- | --- | --- |
| Default / `oga` | OGA | OGA | Required |
| `external` tokenizer, compare unset | FastFlowLM after exact parity | Selected decoder | Required as oracle |
| `external` decoder, compare unset | Selected tokenizer | FastFlowLM cumulative decode | Required as oracle |
| Compare enabled | OGA | OGA | Required; FLM results compared |

Both provider variables accept `oga`, `native-oga`, `default`, or `external`.
Chat templating is fixed to FastFlowLM/minja. There is no delivered template
selector and no `flm`/`fflm` provider value.

Direct external mode is not a no-oracle mode. `OgaTokenizer` and
`OgaTokenizerStream` are still constructed and called. Exact input-ID parity
remains mandatory; generation aborts on a mismatch. Final plain-text decode may
be `byte_exact` or `canonical_equivalent`; structured JSON/tool/code handling
has stricter rules. Streaming chunk boundaries need not match.

The Gemma policy recognizes direct `Metaspace` and
`Sequence -> Replace("▁", " ")` decoder contracts. It classifies the observed
standalone metaspace/space representation difference without forcing literal
`▁` into FastFlowLM user output. Raw strings and mismatch diagnostics remain
available in the discriminator capture.

## Retained Hardware Evidence

The accepted normal, serve-capable Windows `flm.exe` has SHA-256
`2ffc1da52b142e0d831a195f829cf5f6827b5c776a4ea4dd4e923f0699fd40eb`.
The binary and primary result hashes were read again on 2026-10-23. These
retained results were generated on 2026-10-19/20; reading them again is not a
new execution of those long-running tests.

| Layer / test | Qwen2.5-Coder 14B | Gemma3 4B | Scope |
| --- | --- | --- | --- |
| Prompt encoding | Exact OGA/FLM IDs | Exact OGA/FLM IDs | Tested rendered prompts |
| Long generation | 872 IDs, normal stop | 1024 IDs | Final accepted matrix uses OGA-authoritative compare mode |
| Final decode | `byte_exact` | `canonical_equivalent` | Gemma standalone metaspace normalization |
| UTF-8 / special-token leakage | Valid / none | Valid / none | Captured outputs |
| 50-loop execution | 50/50 `ready`, zero errors | Not established by this run | 50 fresh processes, one generated token each |
| CLI / API / stream | Each returned `ready` | Not established by this run | Bounded functional smoke |
| Image request | Not run | Correctly identified red image | Separate hybrid VLM package; CPU fallback permitted |

The text matrix records strict HIP execution with CPU fallback disabled. The
VLM fixture is `hybrid_gpu_experimental` and must not inherit that strict claim.
It enables external selectors plus compare mode, uses OGA media input and
output authority, and also logs an external decoder parity comparison.

Primary SHA-256 receipts:

| Artifact | SHA-256 |
| --- | --- |
| Accepted stage report | `a428199732ab1046978d8ccd614c897b921d5be39b588c55c4e87e7ddff69d67` |
| Qwen discriminator | `62ca3af748c57cdc4b69e2a92e623fe28442108343dac5fab2cfd698e87c76d6` |
| Gemma discriminator | `90e5690de278a27d06c0dee479f83cd86efc18a6b592396bd3b605575a85e758` |
| VLM result | `39c3a7de56b7def18e8a1b3d7027b9a05ea677f9e3c74b94a20109dd724582e8` |

Some historical summaries disagree with the raw evidence. The long-matrix
summary says failure even though both result JSON files have exit 0 and the
Gemma discriminator has the accepted canonical result. The 50-loop aggregate
says failure because an optional `mlir-backend` log line is absent, while all
50 records have exit 0 and strict/fallback-disabled evidence. The API supervisor
timed out after successful outputs; a later cleanup JSON confirms the server
was stopped and its port closed. The VLM result's `listener_cleared=false`
disagrees with its closeout comment; a fresh process/port check on 2026-10-23
found neither `flm` nor listeners on the two test ports. Preserve these original
receipts and their limitations instead of rewriting historical verdicts.

## Fresh Direct-External Verification

On 2026-10-23, the retained accepted Windows binary ran four new cases with
`--ctx-len 2048 --max-tokens 256`. Each model ran once with OGA-authoritative
compare mode and once with both providers set to `external` and compare unset.
The prompt was: "Explain why spaces and token boundaries matter in language
models. Use plain prose without code or markup."

| Model | Mode | Generated IDs | Decoder result | Exit |
| --- | --- | --- | --- | --- |
| Qwen2.5-Coder 14B | OGA + compare | 256 | `byte_exact` | 0 |
| Qwen2.5-Coder 14B | Direct FLM input/output | 251, normal stop | `byte_exact` | 0 |
| Gemma3 4B | OGA + compare | 256 | `canonical_equivalent` | 0 |
| Gemma3 4B | Direct FLM input/output | 256 | `canonical_equivalent` | 0 |

All four recorded exact prompt-token parity, valid UTF-8, no detected special
token leakage, strict HIP policy, and disabled CPU fallback. The runner removed
its lock and recorded zero remaining `flm` processes. The launch SSH channel
timed out at 50 seconds; the durable runner completed all four cases in about
68 seconds, and its completion/result/discriminator files were retrieved after
completion. No duplicate run was launched.

The [machine-readable receipt](../provenance/hip-oga-text-codec-20261023.json)
records exact source/binary/config hashes, selector settings, counts, and
discriminator hashes. Package sampling remained enabled. These are per-run
codec parity checks, not a cross-run deterministic generated-ID comparison or
a performance comparison. This verifies direct FLM operation on the retained
accepted binary; it is not a new Windows build of the continuation branch.

Fresh local verification also passed the complete HIP OGA routing check
(including C++ decoder/tool-parser fixtures), the separately compiled C++
tokenizer-probe test, the evidence schema validator, tokenizer parity self-test,
and whitespace checks. A focused source review found no import regression.

## Open Qualification Work

- Direct-external 1024-token generation, a complete isolated-mode matrix, and
  deterministic generation comparisons must be distinguished from compare-mode
  evidence. The retained packages enable sampling, so independently generated
  outputs are not expected to have identical IDs without a deterministic setup.
- Fresh-process one-token repetition does not establish memory stability in a
  long-lived server, multi-turn correctness, or codec-plus-prefix-cache behavior.
- The shared cache key includes codec metadata/selector state, but this is not
  an offline, content-addressed compatibility guarantee for arbitrary bundles.
- Image selectors force OGA authority, but global compare mode still invokes
  the external decoder and can fail the request. The inherited generation path
  also resets the processor-created stream using the standalone tokenizer.
  One passing Gemma VLM fixture does not qualify every processor-dependent VLM.
- Cumulative full-sequence decode prioritizes correctness and is not a measured
  tokenizer/streaming performance optimization.
- No-oracle text operation needs a separate implementation and verification:
  avoid constructing OGA codec objects, validate model-compatible bundles
  offline, and test structured output, multi-turn operation, and cache identity.

Keep external mode opt-in and scope support to tested packages until those
gates are addressed. The existing OGA demo package and production/default
settings are not promoted by this architecture assessment.
