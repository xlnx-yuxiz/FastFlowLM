# xgrammar string-bounds source patch

`chat_sources.cmake` includes `xgrammar_string_bounds.cmake` when creating the
`flm_xgrammar` target, for the application and standalone tests alike. Normal
CMake configure/build commands need no extra option or patch executable.

The input is `third_party/xgrammar/cpp/json_schema_converter.cc` from pinned
submodule commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`. CMake validates its
SHA-256 after CRLF-to-LF normalization, replaces the bounded-string character
atom, and writes `flm_xgrammar/json_schema_converter.cc` under the build directory.
Only that generated converter is compiled. Original internal headers come from
the submodule. No source, shared dependency cache or submodule revision is mutated.
An unchanged configure preserves the generated file timestamp. CMake watches the
original file and patch; a changed upstream input fails configuration for review.

The fix counts decoded Unicode code points for `minLength`/`maxLength`: raw UTF-8
scalars, simple JSON escapes, non-surrogate BMP escapes and valid surrogate pairs
each occupy one repetition. Unpaired surrogate escapes and all raw U+0000–U+001F
controls are rejected in this bounded path. Combining sequences remain separate
code points. Existing pattern/format/exclusion handling and schema admission are
unchanged. This patch does not claim broader JSON Schema support.

When upgrading xgrammar, check upstream behavior and run `chat_constraint` using
an existing tokenizer. Remove the overlay if upstream implements the same
semantics; otherwise review the new source and update the patch and checksum
together. Do not merely refresh the checksum to bypass a mismatch.
