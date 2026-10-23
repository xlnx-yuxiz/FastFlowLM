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
Recovery commit `3a738599` imported the accepted codec implementation. The
continuation adds selectable `flm`/`compare` execution and lazy native codecs,
without changing the tokenizer algorithm or OGA token generation. Existing build directories
must rerun CMake to discover the added `hip_oga_external_tokenizer.cpp`.

## Delivered Selection And Ownership

| Mode | Prompt IDs | Output text | OGA codec calls |
| --- | --- | --- | --- |
| Default / `oga` | OGA | OGA | Required |
| `flm` | FLM | FLM cumulative decode | None for text |
| `compare` | OGA | OGA | Both codecs; parity required |
| Legacy `external` | FLM after exact parity | FLM after final equivalence | Required as oracle |

The normal CLI supports `--hip-oga-text-codec` plus independent
`--hip-oga-tokenizer` and `--hip-oga-decoder` overrides for `run` and `serve`.
See [configuration and precedence](../docs/instructions/hip-oga.md#text-codec-selection).
Chat templating is fixed to FastFlowLM/minja. There is no template selector.

Native tokenizer/stream objects are created lazily for the layers that need
them. Media forces OGA codecs and preserves the processor-backed decoder stream.
Codec selection and file fingerprints no longer enter the KV compatibility
signature: actual token prefixes, backend request state, capacity, and successful
checkpoint restoration determine reuse.

Historical receipts below were produced by the recovered implementation,
including its mandatory runtime oracle checks. They do not verify the new
`flm` path. New-source verification is recorded separately.

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

## New Codec-Mode Verification (2026-10-23)

Built the normal Windows `flm` target from the continuation source using Visual
Studio 18 2026, x64 Release, and `FLM_ENABLE_HIP_OGA=ON`. Configure and build both
returned zero. The transferred source snapshot contained 1,938 files, verified
by SHA-256 before building; the receipt records the changed runtime file hashes.

| Fresh binary smoke check | Result |
| --- | --- |
| Qwen 2.5 Coder 14B: `oga`, `flm`, `compare` | All passed |
| Gemma 3 4B text: `oga`, `flm`, `compare` | All passed |
| Both `compare` runs | Exact prompt-ID parity and byte-exact decoder parity |
| Both `flm` runs | Generated text without native OGA tokenizer creation or codec comparison |
| Qwen `serve`, FLM codecs, two turns | Cached prompt tokens: 0 then 38; remembered the requested word |

CLI runs used a 32-token upper bound and the packages' existing sampling
settings. All six reported strict GPU execution with CPU EP fallback disabled.
The harness deliberately enabled the legacy compare environment flag while
passing explicit CLI modes, verifying that CLI selection overrides that flag.
The FLM server also ran without native tokenizer creation. All owned model
processes were cleaned up after the checks.

Local checks passed for configuration precedence and legacy compatibility,
cumulative decoding/canonical equivalence, tokenizer parity, routing, prefix
policy, and checkpoint capacity. A focused review found no blocking issue.
The [new-source receipt](../provenance/hip-oga-codec-modes-20261023.json)
records this build and its smoke results separately from the historical receipts.
This is bounded text verification; it does not establish performance, hot codec
switching, long-lived server stability, or new VLM hardware coverage.

## Open Qualification Work

- Direct-external 1024-token generation, a complete isolated-mode matrix, and
  deterministic generation comparisons must be distinguished from compare-mode
  evidence. The retained packages enable sampling, so independently generated
  outputs are not expected to have identical IDs without a deterministic setup.
- Fresh-process one-token repetition does not establish memory stability in a
  long-lived server, multi-turn correctness, or codec-plus-prefix-cache behavior.
- The new codec-independent cache boundary has policy and integration checks;
  extended cross-mode multi-turn hardware qualification remains open.
- The media stream and global-compare coupling have been corrected in this
  continuation. One historical Gemma VLM fixture does not qualify every
  processor-dependent VLM or establish coverage of the new binary.
- Cumulative full-sequence decode prioritizes correctness and is not a measured
  tokenizer/streaming performance optimization.
- FLM-only operation is implemented. Broader package qualification, offline
  bundle compatibility checks, structured output and long-lived server tests
  remain separate from bounded smoke verification.

Keep external mode opt-in and scope support to tested packages until those
gates are addressed. The existing OGA demo package and production/default
settings are not promoted by this architecture assessment.
