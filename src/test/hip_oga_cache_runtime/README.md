# HIP OGA cache recovery probe

This test compiles the production text-prefill transaction from
`src/common/AutoModel/hip_oga_model.cpp`, including recovery, usage accounting and
checkpoint commit. `extract_prefill.py` records the source hash and line span;
the fragment hash uses UTF-8 text with LF newlines. The generated include may
have host-native line endings.

The test adapter calls real OGA rewind, append, generator creation and generation.
Only exception injection, timing scaffolding and fixture state are test-owned.
There is no fault option in the shipping server. Cases include faults before and
after rewind, partial and full append, failed rebuilding, subsequent session
disable, identical-prompt fresh rebuilding, and reuse after generator capacity
exhaustion. Each generation call must append exactly one token.

Run compilation and model tests on an available AMD GPU machine with the
matching OGA SDK and runtime. Windows example using Visual Studio 2022:

```powershell
$RuntimeRoot = '<qualified OGA SDK root>'
cmake -S src/test/hip_oga_cache_runtime -B cache-runtime-build -G 'Visual Studio 17 2022' -A x64 "-DONNXRUNTIME_GENAI_HOME=$RuntimeRoot"
cmake --build cache-runtime-build --config Release
ctest --test-dir cache-runtime-build -C Release --output-on-failure
# Stage and verify the complete qualified runtime, including kernel/data files.
# Use a qualified model directory and serialize access to the test GPU.
$ModelDirectory = 'C:\path\to\qualified-oga-model'
& cache-runtime-build/Release/test_cache_runtime.exe $ModelDirectory result.json
```

For staged tests, `-DFLM_SOURCE_ROOT=<qualified-source-root>` selects the source
excerpt and headers explicitly. Put the qualified DLLs beside the executable;
PATH alone can allow an older system ORT to load first. Serialize this probe with
other model tests. Verify the full runtime and model file inventory, including
rocBLAS/hipBLASLt kernel/data resources. DLL-only staging is insufficient:
missing kernel resources previously corrupted even fresh-control output.
Require exit code zero, result status PASS, and no fatal GPU loading/attention
diagnostics. Record the exact source, binary, runtime and model hashes alongside
the result; the required files depend on the selected package.

The runtime regression exposed an OGA boundary: rewind to a fully cached prompt
followed by generation without an append can advance the sequence by two tokens,
while the FLM adapter returns only the last token. The prefix policy therefore
uses a fresh prefill with reason `empty_suffix` and cached count zero for an
identical prompt. Nonempty suffixes still reuse; this miss does not disable the
session. Replaying the last prompt token is a possible future optimization.

The test adapter counts accepted state through OGA sequence length and releases
the old generator before recreating it. The shipping provider tracks accepted
tokens separately and creates the replacement before releasing the old generator.
The probe therefore qualifies the extracted transaction and its adapter, rather
than every shipping provider bookkeeping or lifetime detail.

Pair this probe with
`../hip_oga_backend_static/test_hip_oga_cache_reliability.py` for the real HTTP
path. The neighboring `check_cache_matrix.py` validates historical receipts with
fixed fixture tags and a 1024-token context; it is not a general model runner. The runtime probe does not test
REST cancellation, transport failures, non-throwing provider corruption, physical
device failure, VLM, or long-duration endurance. A passing result proves only the
named model, source excerpt, runtime, fixtures and injected failure cases.
