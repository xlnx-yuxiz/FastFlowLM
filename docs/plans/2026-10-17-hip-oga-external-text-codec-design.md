# HIP OGA External Text Codec Experiment

Status: implemented experiment; architecture assessed on 2026-10-23.

The [evaluation report](../architecture/hip-oga-token-boundary-evaluation.md)
records the delivered behavior, recovered evidence, and remaining qualification
gaps. Work-lane descriptions below retain the original development plan and are
not instructions to repeat completed work.

Date: 2026-10-17

Base branch: `feature/hip-oga-prefix-cache-c41`

Coordination branch: `feature/hip-oga-external-text-codec`

## Objective

Determine whether FastFlowLM can provide text chat templating, tokenization,
and detokenization while OGA continues to own model execution, search, and
generated token selection for the HIP EP backend.

The experiment must isolate text codec behavior from OGA generation behavior.
It is intended to establish a reusable token-level boundary for future
FastFlowLM backends, not to replace the qualified OGA path immediately.

## Existing Evidence

OGA already exposes the required token-level API:

- `OgaGenerator::AppendTokens` accepts caller-owned `int32_t` token IDs.
- `OgaGenerator::GetSequenceData` exposes generated token IDs.
- `OgaGenerator::GenerateNextToken` can remain the only sampling and generation
  authority.
- `OgaTokenizer` and `OgaTokenizerStream` are separate objects, not required to
  construct the generator for text-only input.

FastFlowLM already constructs `Tokenizer` from the model's `tokenizer.json` in
the HIP adapter, but the current user-visible text path still uses OGA for chat
templating, prompt encoding, and streaming decode.

The earlier prototype at `ee93796b064cd278f162eec56e714c12f549ec00`
did not isolate external tokenization. It changed all of these at once:

- FastFlowLM chat templating and prompt encoding;
- OGA generation followed by logits copy and generator rewind;
- FastFlowLM sampling instead of OGA-selected generated token IDs;
- FastFlowLM per-token decoding.

That combined path produced empty or malformed output in `WOR-256`. The later
native OGA repair changed templating, encoding, generation, sampling, and
decoding together and restored correctness. The historical failure therefore
does not prove that an external tokenizer or detokenizer is incompatible.

## Scope

The experiment covers text-only requests and keeps native OGA generation
fixed. It evaluates three independently selectable frontend layers:

1. Chat template renderer: fixed FastFlowLM/minja in the delivered experiment.
2. Prompt tokenizer: OGA or FastFlowLM/tokenizers-cpp.
3. Generated-token decoder: OGA stream or FastFlowLM/tokenizers-cpp.

The experiment does not replace OGA model execution, OGA search, HIP EP,
provider overlays, generator state, prefix-cache ownership, or generated-token
selection.

Multimodal requests remain on `OgaMultiModalProcessor` and `SetInputs`.
External text tokenization must not be selected for image or audio requests.

## Layered Architecture

The target data flow is:

```text
messages
  -> PromptFrontend.render()
  -> TextTokenizer.encode()
  -> GenerationBackend.append(token_ids)
  -> GenerationBackend.next_token_id()
  -> TextDecoder.push(token_id)
  -> text chunks and final text
```

For HIP OGA:

- `GenerationBackend` is always OGA during this experiment.
- OGA remains responsible for generator parameters, sampling, KV state, and
  generated token IDs.
- `PromptFrontend`, `TextTokenizer`, and `TextDecoder` may use either OGA or
  FastFlowLM implementations.

For multimodal input, a separate backend-specific boundary remains:

```text
messages + media
  -> OgaMultiModalProcessor
  -> OgaNamedTensors
  -> OgaGenerator::SetInputs
```

## Experimental Selection

All external behavior is default-off. The native OGA path remains the default.

Laboratory selectors:

```text
FLM_HIP_OGA_TOKENIZER_PROVIDER=oga|external
FLM_HIP_OGA_DECODER_PROVIDER=oga|external
FLM_HIP_OGA_TEXT_CODEC_COMPARE=1
```

Compare mode runs both implementations, records parity evidence, and uses OGA
as the authoritative result regardless of the provider selectors. Unset compare
mode to make an `external` selection authoritative. Direct external mode still
uses OGA as a parity oracle. There is no template selector; `fflm` and `compare`
are not provider values. `default` and `native-oga` alias `oga`.

The selector values are experimental diagnostics, not a public compatibility
contract.

## Parallel Work Lanes

### Foundation

A short serial foundation defines provider enums, comparison result types, the
JSON evidence schema, deterministic prompt fixtures, and default-off selector
parsing. It must not change production routing.

### Lane A: External Tokenizer

Branch: `feature/hip-oga-external-tokenizer`

Responsibilities:

- Compare OGA and FastFlowLM rendered chat text independently.
- Compare prompt token IDs, counts, hashes, and first mismatch.
- Cover plain text, chat, tools, code, whitespace, CJK, emoji, BOS/EOS, and
  added tokens.
- Allow FastFlowLM IDs to be passed to native OGA generation only after exact
  parity is established for the request.
- Leave generated-token decoding on OGA.

### Lane B: External Detokenizer

Branch: `feature/hip-oga-external-detokenizer`

Responsibilities:

- Consume token IDs selected by native OGA generation.
- Compare final FastFlowLM decode with OGA decode byte-for-byte.
- Add a stateful incremental decoder; do not use `run_time_decoder(int)` as the
  compatibility proof.
- Detect non-prefix revisions, incomplete UTF-8, replacement characters, and
  special-token policy differences explicitly.
- Leave prompt templating and tokenization on OGA.

The lanes add separate source and test files. They must not independently edit
the central HIP generation loop. A later integration step wires both providers
into `HipOgaModel`, minimizing merge conflicts.

### Integration And Hardware Gate

After both lane reviews pass, merge them into the coordination branch and run
the combined matrix. Halo hardware use is serialized under the existing host
lock. Parallel work is limited to source, fixtures, and no-hardware tests.

## Decoder Design

FastFlowLM's current `run_time_decoder(int)` maps one vocabulary item at a time.
It is not sufficient for the proof because byte-level BPE and byte-fallback
tokenizers may span UTF-8 boundaries or revise whitespace across tokens.

The first external decoder proof performs full-sequence decode over the exact
generated token IDs and compares final visible text with OGA. Raw byte equality
is preserved as a diagnostic classification, not the only plain-text pass gate.

The streaming proof maintains a decoder session with:

- all generated token IDs;
- the previous cumulative decoded text;
- the stable emitted prefix;
- pending bytes or text that are not yet stable UTF-8.

Each update decodes the cumulative sequence and emits only a verified append.
If previously emitted bytes would change, the decoder reports an incompatibility
instead of silently rewriting output.

Final decoder acceptance uses explicit classes:

- `byte_exact`: raw OGA and external bytes match.
- `canonical_equivalent`: plain text is valid UTF-8, has no special-token
  leakage, and matches after an explicit model/tokenizer canonicalization.
- `structured_equivalent`: parsed structured values match without normalizing
  meaningful string content.
- `incompatible`: all other cases.

Gemma/SentencePiece standalone metaspace is the only current plain-text
canonicalization. It is enabled only when tokenizer metadata identifies a Gemma
tokenizer and tokenizer.json exposes a metaspace decoder plus standalone
metaspace vocabulary. The canonical form maps standalone `U+2581` to a normal
user-visible space. Code, JSON, and tool-call surfaces do not receive this
normalization.

This cumulative implementation is acceptable for the experiment even if it is
not the final performance design. Performance optimization follows only after
correctness parity.

## Evidence Schema

Each parity case records:

```json
{
  "model_fingerprint": "...",
  "case": "...",
  "template_provider": "...",
  "tokenizer_provider": "...",
  "decoder_provider": "...",
  "rendered_text_sha256": "...",
  "prompt_token_count": 0,
  "prompt_token_ids_sha256": "...",
  "first_prompt_token_mismatch": null,
  "generated_token_ids_sha256": "...",
  "oga_text_sha256": "...",
  "external_text_sha256": "...",
  "raw_byte_equal": false,
  "decoder_equivalence_class": "canonical_equivalent",
  "first_text_mismatch": null,
  "utf8_valid": true,
  "special_token_leakage": false,
  "normalization_applied": ["gemma_standalone_metaspace_to_space"],
  "result": "pass"
}
```

Raw prompts and text may be retained only in internal test evidence. Public
reports should prefer hashes and small non-sensitive fixtures.

## Error Handling

- Unknown provider selectors fail at startup.
- External tokenization requires exact model/tokenizer provenance.
- Token-ID mismatch in compare or external mode aborts generation.
- External mode fails closed on unsupported tokenizer models or special-token
  disagreement.
- External streaming decode fails closed on unstable prefix or invalid UTF-8.
- Media requests force the OGA processor path and report the effective mode.
- The qualified OGA default path remains unchanged when selectors are unset.

## Verification Matrix

No-hardware verification:

- selector parsing and default-off behavior;
- tokenizer parity over deterministic fixtures;
- final decode parity over recorded token-ID sequences;
- incremental chunk reconstruction equals final decode;
- UTF-8 and special-token edge cases;
- static guard that native OGA generation remains authoritative;
- no change to multimodal routing or prefix-cache default behavior.

Hardware verification on a Strix Halo host:

- Qwen2.5 text model for byte-level BPE and tool/chat prompts;
- Gemma text model for a second tokenizer family;
- baseline OGA/OGA/OGA;
- external tokenizer only;
- external decoder only;
- fully external text codec after both isolated modes pass;
- deterministic generated-token ID comparison;
- short correctness, long generation, 50-loop stability, and memory checks;
- strict HIP execution evidence and CPU-fallback policy checks;
- one VLM regression confirming the OGA multimodal processor remains active.

## Acceptance Criteria

- Exact rendered-text parity when comparing equivalent template providers.
- Exact prompt token-ID parity before external IDs may drive OGA.
- Identical generated token IDs when prompt IDs and generation settings match.
- Byte-exact or explicitly model-aware canonical-equivalent plain-text decoding;
  structured content retains its strict comparison policy.
- Streaming reconstruction equals final external decoding and valid UTF-8.
- No native OGA path regression when selectors are unset.
- OGA remains authoritative for multimodal requests. Global compare mode still
  invokes the external decoder comparison; it is not a complete media bypass.
- Both tokenizer families pass the isolated and combined matrices.
- Any incompatibility is classified by layer and preserved as evidence.

## Delivery Boundary

The first deliverable is an experimental branch and evidence report, not a new
demo package. The qualified demo branch and package remain unchanged.

If parity succeeds, a follow-up design may promote the four interfaces
`PromptFrontend`, `TextTokenizer`, `GenerationBackend`, and `TextDecoder` into a
formal backend seam. If parity fails, the report must identify the exact layer,
model family, token index, or decoded byte where compatibility ends.
