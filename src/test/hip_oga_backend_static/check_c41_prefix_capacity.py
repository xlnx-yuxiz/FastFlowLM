#!/usr/bin/env python3
"""Static guard for the c41 HIP OGA PrefixCache capacity bug."""

from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)


if len(sys.argv) == 2:
    source_path = Path(sys.argv[1])
else:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "common"
        / "AutoModel"
        / "hip_oga_model.cpp"
    )

source = source_path.read_text(encoding="utf-8")

native = source.split(
    "std::uint32_t native_text_required_max_length(std::size_t prompt_token_count) const",
    1,
)[1].split("std::uint32_t generator_capacity() const", 1)[0]
begin_turn = source.split("void begin_turn(const std::vector<int>& tokens)", 1)[1].split(
    "std::size_t begin_multimodal_turn", 1
)[0]

require(
    "effective_oga_max_length(prompt_token_count)" not in native,
    "native text prefix-cache capacity must not include requested generation budget",
)
require(
    "std::numeric_limits<std::uint32_t>::max()" in native,
    "native text capacity must be bounded by uint32_t without adding max_tokens",
)
require(
    "reset_generator(context_length_)" in begin_turn,
    "text turns must allocate the OGA generator to the full context window",
)
require(
    "reset_generator(effective_oga_max_length(converted.size()))" not in begin_turn,
    "text turns must not size the generator from the current prompt length",
)

print("PASS: c41 HIP OGA prefix-cache text capacity")
