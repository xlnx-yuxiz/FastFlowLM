# HIP OGA External Text Codec Implementation Plan

Historical implementation plan, recovered 2026-10-23. Source tasks 1-7 were
completed on the external codec and canonical-equivalence branches. Task 8 has
bounded hardware evidence, with its remaining gaps recorded in the
[Task 9 evaluation report](../architecture/hip-oga-token-boundary-evaluation.md).
The original task snippets below include proposed selector names that differ
from the delivered interface; use the current HIP OGA instructions. Do not
repeat completed tasks or treat this plan as a claim that every hardware gate
has passed.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove independently that FastFlowLM can tokenize prompts and detokenize generated token IDs around an otherwise unchanged native OGA generation session.

**Architecture:** Add a default-off text-codec experiment around the existing HIP OGA adapter. A short shared foundation defines provider selection, parity results, and fixtures; tokenizer and detokenizer implementations then proceed on separate branches and files. Only the integration phase edits `hip_oga_model.cpp`, and OGA remains the sole generation and sampling authority throughout.

**Tech Stack:** C++20, FastFlowLM `Tokenizer`/tokenizers-cpp, minja, ONNX Runtime GenAI C++ API, Python 3 fixture tooling, CMake, PowerShell, HIP EP/AMDGPU on Windows Strix Halo.

---

## Execution Graph

```text
Task 1: shared contracts and fixtures
  |-- Task 2-3: external tokenizer lane -----------------|
  |-- Task 4-5: external detokenizer lane ---------------|-> Task 6 integration
                                                            -> Task 7 review
                                                            -> Task 8 Halo gate
                                                            -> Task 9 architecture report
```

Tasks 2-3 and 4-5 run in parallel. They use sibling worktrees and must not edit
`src/common/AutoModel/hip_oga_model.cpp`. Task 6 is the only integration owner
for that file. Halo hardware work begins only after Task 7 review.

### Task 1: Shared Selection And Parity Contracts

**Branch:** `feature/hip-oga-external-text-codec`

**Files:**
- Create: `src/include/AutoModel/hip_oga_text_codec.hpp`
- Create: `src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp`
- Create: `src/test/hip_oga_external_codec/fixtures/schema.json`
- Modify: `src/test/hip_oga_backend_static/check_hip_oga_routing.py`

**Step 1: Write the failing provider-selection test**

Cover unset values, `oga`, `fflm`, `compare`, and invalid values for template,
tokenizer, and decoder selectors. Require unset selectors to resolve to OGA.

```cpp
using flm::hip_oga::TextProvider;
require(parse_text_provider(nullptr, "tokenizer") == TextProvider::Oga,
        "unset tokenizer provider must preserve OGA");
require(parse_text_provider("compare", "tokenizer") == TextProvider::Compare,
        "compare tokenizer provider must be accepted");
require_throws([] { parse_text_provider("unknown", "tokenizer"); });
```

**Step 2: Run the test to verify it fails**

```bash
g++ -std=c++20 -Isrc/include \
  src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp \
  -o /tmp/test_hip_oga_text_codec
/tmp/test_hip_oga_text_codec
```

Expected: compile failure because `hip_oga_text_codec.hpp` does not exist.

**Step 3: Implement the minimal header-only contract**

Define:

```cpp
enum class TextProvider { Oga, FastFlowLm, Compare };

struct TokenParity {
    bool equal;
    std::size_t oga_count;
    std::size_t external_count;
    std::optional<std::size_t> first_mismatch;
};

struct TextParity {
    bool equal;
    bool utf8_valid;
    std::optional<std::size_t> first_mismatch;
};

TextProvider parse_text_provider(const char* value, std::string_view layer);
TokenParity compare_token_ids(std::span<const int> oga,
                              std::span<const int> external);
TextParity compare_text(std::string_view oga, std::string_view external);
```

Keep it independent of OGA headers and FastFlowLM runtime libraries.

**Step 4: Add the evidence schema**

Define required fields from the approved design: model fingerprint, case name,
three providers, prompt/generated token counts and SHA-256 values, first token
or text mismatch, UTF-8 validity, and result.

**Step 5: Strengthen the static guard**

Require the selectors to default to OGA and reject product routing controlled
only by external mode. Continue rejecting FastFlowLM sampling and authoritative
rewind in the user-visible HIP generation loop.

**Step 6: Run foundation checks**

```bash
g++ -std=c++20 -Isrc/include \
  src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp \
  -o /tmp/test_hip_oga_text_codec
/tmp/test_hip_oga_text_codec
python3 src/test/hip_oga_backend_static/check_hip_oga_routing.py
python3 -m json.tool src/test/hip_oga_external_codec/fixtures/schema.json >/dev/null
git diff --check
```

Expected: all pass.

**Step 7: Commit and create sibling branches**

```bash
git add src/include/AutoModel/hip_oga_text_codec.hpp \
  src/test/hip_oga_backend_static/test_hip_oga_text_codec.cpp \
  src/test/hip_oga_external_codec/fixtures/schema.json \
  src/test/hip_oga_backend_static/check_hip_oga_routing.py
git commit -m "test: add HIP OGA text codec experiment contracts"
git branch feature/hip-oga-external-tokenizer
git branch feature/hip-oga-external-detokenizer
```

### Task 2: Tokenizer Parity Library

**Branch:** `feature/hip-oga-external-tokenizer`

**Files:**
- Create: `src/include/AutoModel/hip_oga_external_tokenizer.hpp`
- Create: `src/common/AutoModel/hip_oga_external_tokenizer.cpp`
- Create: `src/test/hip_oga_backend_static/test_hip_oga_external_tokenizer.cpp`

**Step 1: Write failing token-parity tests**

Use callback-based encoders so the test does not require OGA or model weights.
Cover exact parity, count mismatch, first-ID mismatch, BOS/EOS mismatch, and
empty input.

```cpp
ExternalTokenizerProbe probe(
    [](std::string_view) { return std::vector<int>{1, 2, 3}; },
    [](std::string_view) { return std::vector<int>{1, 9, 3}; });
auto result = probe.compare("prompt");
require(!result.equal && result.first_mismatch == 1,
        "first mismatch must identify token index 1");
```

**Step 2: Run the test and verify failure**

Expected: missing `ExternalTokenizerProbe`.

**Step 3: Implement the minimal probe**

The probe returns both token vectors and `TokenParity`. It does not select the
authoritative vector; integration owns that policy.

**Step 4: Run and commit**

```bash
g++ -std=c++20 -Isrc/include \
  src/common/AutoModel/hip_oga_external_tokenizer.cpp \
  src/test/hip_oga_backend_static/test_hip_oga_external_tokenizer.cpp \
  -o /tmp/test_hip_oga_external_tokenizer
/tmp/test_hip_oga_external_tokenizer
git diff --check
git add src/include/AutoModel/hip_oga_external_tokenizer.hpp \
  src/common/AutoModel/hip_oga_external_tokenizer.cpp \
  src/test/hip_oga_backend_static/test_hip_oga_external_tokenizer.cpp
git commit -m "test: add HIP OGA tokenizer parity probe"
```

### Task 3: Tokenizer Fixture And Model-Family Coverage

**Branch:** `feature/hip-oga-external-tokenizer`

**Files:**
- Create: `src/test/hip_oga_external_codec/tokenizer_parity.py`
- Create: `src/test/hip_oga_external_codec/fixtures/text_cases.json`
- Create: `src/test/hip_oga_external_codec/fixtures/README.md`
- Modify: `src/test/hip_oga_backend_static/check_hip_oga_routing.py`

**Step 1: Add deterministic text cases**

Include plain ASCII, repeated spaces, leading/trailing newlines, code, JSON,
CJK, emoji, combining characters, tool calls, added tokens, and chat messages.

**Step 2: Write a failing fixture validator**

Require unique case names, valid UTF-8, explicit template mode, and expected
OGA token IDs or an explicit `capture_required` marker.

**Step 3: Implement the validator and evidence writer**

The script consumes captured OGA IDs and external IDs, writes the approved JSON
schema, and exits nonzero on mismatches in external-authoritative mode.

**Step 4: Run and commit**

```bash
python3 src/test/hip_oga_external_codec/tokenizer_parity.py --self-test
python3 -m py_compile src/test/hip_oga_external_codec/tokenizer_parity.py
python3 src/test/hip_oga_backend_static/check_hip_oga_routing.py
git diff --check
git add src/test/hip_oga_external_codec \
  src/test/hip_oga_backend_static/check_hip_oga_routing.py
git commit -m "test: cover external tokenizer parity cases"
```

### Task 4: Stateful External Decoder

**Branch:** `feature/hip-oga-external-detokenizer`

**Files:**
- Create: `src/include/AutoModel/hip_oga_incremental_decoder.hpp`
- Create: `src/common/AutoModel/hip_oga_incremental_decoder.cpp`
- Create: `src/test/hip_oga_backend_static/test_hip_oga_incremental_decoder.cpp`

**Step 1: Write failing decoder-session tests**

Inject a cumulative decode callback. Cover normal append, one-token-delayed
stability, UTF-8 split boundaries, whitespace revision, non-prefix revision,
replacement characters, and final flush.

```cpp
CumulativeTextDecoder decoder(fake_decode);
require(decoder.push(10).empty(), "first token remains pending");
require(decoder.push(11) == "hello", "stable prefix is emitted");
require(decoder.finish() == " world", "final pending suffix is flushed");
```

**Step 2: Run the test and verify failure**

Expected: missing `CumulativeTextDecoder`.

**Step 3: Implement fail-closed cumulative decode**

Maintain token IDs, previous cumulative text, and emitted prefix. On each push:

1. Decode all generated IDs.
2. Require the new text to begin with the already emitted prefix.
3. Find the UTF-8-safe common prefix of previous and current decoded text.
4. Emit only newly stable bytes.
5. Throw a typed compatibility error if emitted bytes would change.

`finish()` validates UTF-8 and emits the remaining suffix.

**Step 4: Run and commit**

```bash
g++ -std=c++20 -Isrc/include \
  src/common/AutoModel/hip_oga_incremental_decoder.cpp \
  src/test/hip_oga_backend_static/test_hip_oga_incremental_decoder.cpp \
  -o /tmp/test_hip_oga_incremental_decoder
/tmp/test_hip_oga_incremental_decoder
git diff --check
git add src/include/AutoModel/hip_oga_incremental_decoder.hpp \
  src/common/AutoModel/hip_oga_incremental_decoder.cpp \
  src/test/hip_oga_backend_static/test_hip_oga_incremental_decoder.cpp
git commit -m "test: add stateful external token decoder"
```

### Task 5: Detokenizer Fixture And Byte-Parity Coverage

**Branch:** `feature/hip-oga-external-detokenizer`

**Files:**
- Create: `src/test/hip_oga_external_codec/detokenizer_parity.py`
- Create: `src/test/hip_oga_external_codec/fixtures/generated_token_cases.json`
- Modify: `src/test/hip_oga_backend_static/check_hip_oga_routing.py`

**Step 1: Write failing fixture tests**

Cover ASCII, multibyte UTF-8, emoji, byte fallback, added/special tokens,
leading-space behavior, newline behavior, and malformed fixture rejection.

**Step 2: Implement final and chunk parity reporting**

Compare OGA final text, external final text, concatenated OGA chunks, and
concatenated external chunks. Record first byte mismatch and UTF-8 validity.

**Step 3: Run and commit**

```bash
python3 src/test/hip_oga_external_codec/detokenizer_parity.py --self-test
python3 -m py_compile src/test/hip_oga_external_codec/detokenizer_parity.py
python3 src/test/hip_oga_backend_static/check_hip_oga_routing.py
git diff --check
git add src/test/hip_oga_external_codec \
  src/test/hip_oga_backend_static/check_hip_oga_routing.py
git commit -m "test: cover external detokenizer byte parity"
```

### Task 6: Integrate Both Providers Without Changing OGA Generation

**Branch:** `feature/hip-oga-external-text-codec`

**Files:**
- Modify: `src/common/AutoModel/hip_oga_model.cpp`
- Modify: `src/include/AutoModel/hip_oga_model.hpp`
- Modify: `src/include/AutoModel/hip_oga_prefix_cache.hpp`
- Modify: `src/test/hip_oga_backend_static/check_hip_oga_routing.py`
- Create: `src/test/hip_oga_backend_static/test_hip_oga_text_codec_routing.cpp`
- Modify: `docs/docs/instructions/hip-oga.md`

**Step 1: Merge the two reviewed lane heads**

Resolve only shared static-check fixture additions. Do not hand-merge product
logic from the lanes because product wiring has not yet been added.

**Step 2: Write failing routing tests**

Require:

- unset selectors use OGA template, tokenizer, and decoder;
- compare mode runs both but uses OGA values;
- external tokenizer IDs may drive OGA only after parity passes;
- external decoder consumes OGA-generated token IDs;
- `GenerateNextToken` and `GetSequenceData` remain authoritative;
- no `sampler->sample`, `_shared_generate`, or user-text rewind path;
- media requests force OGA processor and decoder;
- prefix-cache signatures include effective text providers and tokenizer
  fingerprint.

**Step 3: Add selector state to `HipOgaModel`**

Read environment values once at model load. Log effective providers only when
an experimental selector is set. Keep default construction behavior unchanged.

**Step 4: Wire template and tokenizer selection**

Create both rendered strings or token vectors only in compare mode. Report the
first mismatch. External mode fails closed on mismatch when an OGA oracle is
available; compare mode continues with OGA IDs.

Pass the selected token IDs to the existing `begin_turn()` and
`OgaGenerator::AppendTokens()` path. Do not alter OGA generator parameters or
sampling.

**Step 5: Wire decoder selection**

Feed every OGA-selected generated token ID to both decoders in compare mode.
Use OGA chunks as output in compare mode. External mode uses only verified
stable chunks and performs a final byte-parity assertion when configured for a
test run.

**Step 6: Protect multimodal and prefix-cache paths**

Force OGA processor-backed decode for media. Add provider and tokenizer
fingerprint fields to the prefix-cache compatibility signature so checkpoints
cannot cross codec modes.

**Step 7: Run all no-hardware tests**

```bash
python3 src/test/hip_oga_backend_static/check_hip_oga_routing.py
python3 src/test/hip_oga_backend_static/test_hip_oga_openai_tool_parser.py
python3 src/test/hip_oga_external_codec/tokenizer_parity.py --self-test
python3 src/test/hip_oga_external_codec/detokenizer_parity.py --self-test
python3 -m py_compile \
  src/test/hip_oga_backend_static/check_hip_oga_routing.py \
  src/test/hip_oga_external_codec/tokenizer_parity.py \
  src/test/hip_oga_external_codec/detokenizer_parity.py
g++ -std=c++20 -Isrc/include \
  src/common/AutoModel/hip_oga_external_tokenizer.cpp \
  src/common/AutoModel/hip_oga_incremental_decoder.cpp \
  src/test/hip_oga_backend_static/test_hip_oga_text_codec_routing.cpp \
  -o /tmp/test_hip_oga_text_codec_routing
/tmp/test_hip_oga_text_codec_routing
git diff --check
```

Expected: all pass.

**Step 8: Commit**

```bash
git add src/common/AutoModel src/include/AutoModel \
  src/test/hip_oga_backend_static src/test/hip_oga_external_codec \
  docs/docs/instructions/hip-oga.md
git commit -m "feat: add experimental external HIP OGA text codec"
```

### Task 7: Independent Source Review

**Owner:** HIP EP FLM Reviewer

**Files:**
- Review the complete diff from the Task 1 foundation to the Task 6 head.
- Create: `docs/reviews/hip-oga-external-text-codec-review.md`

**Step 1: Review invariants**

Confirm OGA remains the only generation/sampling authority, default behavior is
unchanged, media stays on the OGA processor, prefix-cache modes cannot mix, and
all external incompatibilities fail closed.

**Step 2: Re-run no-hardware gates from a fresh worktree**

Expected: all Task 6 commands pass without untracked source dependencies.

**Step 3: Record GO or REQUEST CHANGES and commit the review**

Do not start Halo execution on REQUEST CHANGES.

### Task 8: Serialized Strix Halo A/B Matrix

**Owner:** Strix Halo Verification Engineer

**Files:**
- Create: `scripts/hip_oga_external_codec_matrix.ps1`
- Create: `docs/evidence/hip-oga-external-text-codec-halo.md`

**Step 1: Acquire the host lock and verify a quiet GPU window**

Use SSH only. Use neutral remote paths and names. Do not modify shared model
directories or the qualified demo package.

**Step 2: Build a side-by-side experimental `flm.exe`**

Build the normal serve-capable target with `FLM_ENABLE_HIP_OGA=ON` and the exact
qualified OGA/HIP runtime. Record source head, compiler, CMake flags, and binary
hashes.

**Step 3: Capture native OGA baselines**

For Qwen2.5 and Gemma, capture rendered text hash, prompt token IDs, generated
token IDs, OGA decode chunks/final text, runtime modules, and strict HIP policy.
Decoder evidence must preserve raw OGA stream, OGA full decode, FastFlowLM full
decode, external cumulative text, token IDs, hashes, first mismatch, UTF-8 and
special-token diagnostics, equivalence class, and any canonicalization applied.

**Step 4: Run isolated modes**

Run:

```text
OGA template + external tokenizer + OGA decoder
OGA template + OGA tokenizer + external decoder
external template + external tokenizer + OGA decoder
fully external text codec
```

Do not run a later mode if its prerequisite parity gate failed.

**Step 5: Run behavior and stability gates**

Cover short correctness, code/CJK/emoji/tool prompts, 1024-token generation,
50 repeated cycles, memory growth, CLI/API/stream equality, strict HIP module
evidence, and one unchanged VLM request through the OGA processor.

Plain-text cases pass on `byte_exact` or `canonical_equivalent` final text.
Streaming chunk boundaries and token timing do not have to match; only the
concatenated final canonical text is compared. JSON/tool/code cases stay
strict: parsed JSON may be `structured_equivalent` for formatting-only
differences, but string values, tool-call markup, and code blocks must not be
Gemma-normalized.

**Step 6: Clean up and record evidence**

Require no listener, process, lock, or temporary model mutation remains. Record
hashes and a per-layer PASS/FAIL/INCOMPATIBLE verdict. Qwen 1024 combined
generation should remain `byte_exact`; strict-Gemma 1024 combined generation is
expected to classify standalone metaspace differences as
`canonical_equivalent`, not as a raw-byte failure.

### Task 9: Layered Architecture Decision Report

**Files:**
- Create: `docs/architecture/hip-oga-token-boundary-evaluation.md`
- Modify: `docs/plans/2026-10-17-hip-oga-external-text-codec-design.md`

**Step 1: Summarize parity by model family and layer**

Separate template, tokenization, final decode, streaming decode, generation,
multimodal, and stability verdicts.

**Step 2: Decide the formal seam**

Recommend one of:

- promote `PromptFrontend`, `TextTokenizer`, `GenerationBackend`, and
  `TextDecoder` interfaces;
- promote only tokenizer/detokenizer comparison hooks;
- retain OGA-owned text codec for specified model families.

**Step 3: Update the design status and commit**

Do not change the qualified demo branch or package as part of this decision
report.

## Paperclip Work Breakdown

Create one parent issue and these dependency-ordered children after this plan is
committed:

1. Shared contracts and fixtures, ROCm Integration Engineer.
2. External tokenizer parity lane, ROCm Research Analyst or an isolated coding
   agent, blocked only by child 1.
3. External detokenizer parity lane, ROCm Integration Engineer or a second
   isolated coding agent, blocked only by child 1.
4. Integration branch, blocked by children 2 and 3.
5. Independent source review, blocked by child 4.
6. Serialized Halo matrix, blocked by child 5.
7. Architecture decision report, blocked by children 5 and 6.

Children 2 and 3 must run concurrently. No agent owns both during the parallel
phase. Hardware work remains single-owner and serialized.
