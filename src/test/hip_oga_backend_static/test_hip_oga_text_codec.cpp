/// \file test_hip_oga_text_codec.cpp
/// \brief No-hardware tests for the HIP OGA external text-codec contract.
#include "AutoModel/hip_oga_text_codec.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using flm::hip_oga::ParityStatus;
using flm::hip_oga::TextCodecProvider;
using flm::hip_oga::TextCodecSelector;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void require_provider(const TextCodecSelector& selector,
                      TextCodecProvider expected,
                      const std::string& label) {
    require(selector.provider == expected,
            label + " selected the wrong text-codec provider");
}

void test_default_selector_remains_oga() {
    require_provider(flm::hip_oga::parse_text_codec_selector(""),
                     TextCodecProvider::Oga,
                     "empty selector");
    require_provider(flm::hip_oga::parse_text_codec_selector("   "),
                     TextCodecProvider::Oga,
                     "blank selector");
    require_provider(flm::hip_oga::parse_text_codec_selector("default"),
                     TextCodecProvider::Oga,
                     "default selector");
    require_provider(flm::hip_oga::default_text_codec_selector(),
                     TextCodecProvider::Oga,
                     "programmatic default selector");
}

void test_selector_parses_supported_providers() {
    const TextCodecSelector oga =
        flm::hip_oga::parse_text_codec_selector("oga");
    require_provider(oga, TextCodecProvider::Oga, "oga");
    require(oga.normalized_name == "oga", "OGA selector name must normalize to oga");

    const TextCodecSelector native_oga =
        flm::hip_oga::parse_text_codec_selector("native-oga");
    require_provider(native_oga, TextCodecProvider::Oga, "native-oga");
    require(native_oga.normalized_name == "oga",
            "native-oga selector name must normalize to oga");

    const TextCodecSelector external =
        flm::hip_oga::parse_text_codec_selector(" external ");
    require_provider(external, TextCodecProvider::External, "external");
    require(external.normalized_name == "external",
            "external selector name must normalize to external");
}

void test_selector_rejects_unknown_values() {
    bool rejected = false;
    try {
        (void)flm::hip_oga::parse_text_codec_selector("tokenizers-cpp");
    } catch (const std::invalid_argument& e) {
        rejected = std::string(e.what()).find("text codec") != std::string::npos;
    }
    require(rejected, "unknown text-codec selector must throw invalid_argument");
}

void test_parity_result_types_have_stable_defaults() {
    flm::hip_oga::TextCodecParityResult result;
    require(result.status == ParityStatus::NotRun,
            "parity result must default to not_run");
    require(result.prompt.empty(), "parity result prompt should default empty");
    require(result.oga_token_ids.empty(),
            "OGA token evidence should default empty");
    require(result.external_token_ids.empty(),
            "external token evidence should default empty");
    require(!result.matches(), "not_run parity must not report a match");

    result.status = ParityStatus::Match;
    result.prompt = "Hello";
    result.oga_token_ids = {9906};
    result.external_token_ids = {9906};
    result.oga_decoded_text = "Hello";
    result.external_decoded_text = "Hello";
    require(result.matches(), "matching parity result should report a match");

    result.external_token_ids = {15496};
    result.status = ParityStatus::Mismatch;
    require(!result.matches(), "mismatched token evidence must not report a match");
}

void test_fixture_contract_type_names_are_stable() {
    require(flm::hip_oga::to_string(TextCodecProvider::Oga) == "oga",
            "OGA provider string is part of the fixture contract");
    require(flm::hip_oga::to_string(TextCodecProvider::External) == "external",
            "external provider string is part of the fixture contract");
    require(flm::hip_oga::to_string(ParityStatus::NotRun) == "not_run",
            "not_run status string should remain stable for internal defaults");
    require(flm::hip_oga::to_string(ParityStatus::Match) == "match",
            "match status string is part of the evidence contract");
    require(flm::hip_oga::to_string(ParityStatus::Mismatch) == "mismatch",
            "mismatch status string is part of the evidence contract");
    require(flm::hip_oga::to_string(ParityStatus::Error) == "error",
            "error status string is part of the evidence contract");
    require(flm::hip_oga::to_string(
                flm::hip_oga::DecoderEquivalenceClass::ByteExact) ==
                "byte_exact",
            "byte_exact is part of the decoder evidence contract");
    require(flm::hip_oga::to_string(
                flm::hip_oga::DecoderEquivalenceClass::CanonicalEquivalent) ==
                "canonical_equivalent",
            "canonical_equivalent is part of the decoder evidence contract");
    require(flm::hip_oga::to_string(
                flm::hip_oga::DecoderEquivalenceClass::StructuredEquivalent) ==
                "structured_equivalent",
            "structured_equivalent is part of the decoder evidence contract");
    require(flm::hip_oga::to_string(
                flm::hip_oga::DecoderEquivalenceClass::Incompatible) ==
                "incompatible",
            "incompatible is part of the decoder evidence contract");
}

void test_decoder_equivalence_classifies_raw_byte_exact_text() {
    const auto result = flm::hip_oga::classify_decoder_text_equivalence(
        "Hello world", "Hello world", flm::hip_oga::DecoderCanonicalizationPolicy{});

    require(result.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::ByteExact,
            "identical decoder bytes must classify as byte_exact");
    require(result.raw_byte_equal, "byte_exact result must record raw byte equality");
    require(result.valid_utf8, "byte_exact result must record valid UTF-8");
    require(result.accepted(), "byte_exact plain text must pass");
}

void test_decoder_equivalence_accepts_gemma_standalone_metaspace_as_space() {
    flm::hip_oga::DecoderCanonicalizationPolicy policy;
    policy.contract_name = "gemma_standalone_metaspace";
    policy.gemma_standalone_metaspace_is_space = true;

    const auto result = flm::hip_oga::classify_decoder_text_equivalence(
        std::string("alpha \xE2\x96\x81 beta"),
        "alpha   beta",
        policy);

    require(result.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::CanonicalEquivalent,
            "Gemma standalone metaspace must classify as canonical_equivalent");
    require(!result.raw_byte_equal,
            "canonical_equivalent Gemma metaspace case must preserve raw-byte diagnostic");
    require(result.normalization_applied.size() == 1 &&
                result.normalization_applied[0] ==
                    "gemma_standalone_metaspace_to_space",
            "Gemma canonicalization must record the applied normalization");
    require(result.first_mismatch.has_value() &&
                result.first_mismatch->byte_index == 6,
            "Gemma canonicalization must preserve first raw mismatch evidence");
    require(result.accepted(), "canonical_equivalent plain text must pass");
}

void test_gemma_policy_detects_replace_decoder_contract() {
    const nlohmann::json tokenizer_config = {
        {"tokenizer_class", "GemmaTokenizer"},
        {"processor_class", "Gemma3Processor"},
    };
    const nlohmann::json tokenizer_json = {
        {"decoder", {
            {"type", "Sequence"},
            {"decoders", nlohmann::json::array({
                {
                    {"type", "Replace"},
                    {"pattern", {{"String", "\xE2\x96\x81"}}},
                    {"content", " "},
                },
                {{"type", "ByteFallback"}},
                {{"type", "Fuse"}},
            })},
        }},
        {"model", {
            {"type", "BPE"},
            {"vocab", {
                {"\xE2\x96\x81", 1},
                {"\xE2\x96\x81\xE2\x96\x81", 2},
            }},
        }},
    };

    const auto policy =
        flm::hip_oga::decoder_policy_from_tokenizer_contract(
            tokenizer_config, tokenizer_json);
    require(policy.gemma_standalone_metaspace_is_space,
            "Gemma policy must detect tokenizer.json Replace metaspace decoder contracts");
    require(policy.contract_name == "gemma_standalone_metaspace",
            "Gemma Replace decoder contract must select the Gemma metaspace policy");
}

void test_decoder_equivalence_does_not_globally_normalize_without_contract() {
    const auto result = flm::hip_oga::classify_decoder_text_equivalence(
        std::string("alpha \xE2\x96\x81 beta"),
        "alpha   beta",
        flm::hip_oga::DecoderCanonicalizationPolicy{});

    require(result.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::Incompatible,
            "metaspace bytes must remain incompatible without an explicit model contract");
    require(result.normalization_applied.empty(),
            "no normalization should be recorded without a model contract");
}

void test_decoder_equivalence_keeps_structured_json_strict() {
    flm::hip_oga::DecoderCanonicalizationPolicy policy;
    policy.mode = flm::hip_oga::DecoderEquivalenceMode::StructuredJson;
    policy.contract_name = "structured_json";
    policy.gemma_standalone_metaspace_is_space = true;

    const auto whitespace_only =
        flm::hip_oga::classify_decoder_text_equivalence(
            R"json({"name":"calc","arguments":{"x":1}})json",
            "{\n  \"arguments\": {\"x\": 1},\n  \"name\": \"calc\"\n}",
            policy);
    require(whitespace_only.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::StructuredEquivalent,
            "parsed JSON structure and string values should classify as structured_equivalent");

    const auto meaningful_string_change =
        flm::hip_oga::classify_decoder_text_equivalence(
            R"json({"text":"alpha ▁ beta"})json",
            R"json({"text":"alpha   beta"})json",
            policy);
    require(meaningful_string_change.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::Incompatible,
            "structured JSON string values must not receive Gemma text normalization");
}

void test_decoder_equivalence_rejects_invalid_utf8_and_special_leakage() {
    const auto invalid_utf8 = flm::hip_oga::classify_decoder_text_equivalence(
        std::string("\xF0\x9F", 2),
        std::string("\xF0\x9F", 2),
        flm::hip_oga::DecoderCanonicalizationPolicy{});
    require(invalid_utf8.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::Incompatible,
            "invalid UTF-8 must classify as incompatible");
    require(!invalid_utf8.valid_utf8,
            "invalid UTF-8 diagnostic must be recorded");

    const auto special = flm::hip_oga::classify_decoder_text_equivalence(
        "hello <|endoftext|>",
        "hello <|endoftext|>",
        flm::hip_oga::DecoderCanonicalizationPolicy{});
    require(special.equivalence_class ==
                flm::hip_oga::DecoderEquivalenceClass::Incompatible,
            "special-token leakage must classify as incompatible even when bytes match");
    require(special.special_token_leakage,
            "special-token leakage diagnostic must be recorded");
}

void test_external_cumulative_decoder_delays_until_prefix_is_stable() {
    const std::unordered_map<std::string, std::string> decoded_by_tokens = {
        {"1", "Hel"},
        {"1,2", "Hello"},
        {"1,2,3", "Hello!"},
    };
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [&](const std::vector<int>& token_ids) {
            return flm::hip_oga::CumulativeDecodeResult::match(
                decoded_by_tokens.at(flm::hip_oga::join_token_ids_for_evidence(token_ids)));
        });

    auto first = decoder.append_token(1);
    require(first.status == ParityStatus::Match, "first decode should succeed");
    require(first.emitted_text.empty(),
            "first cumulative decode must be withheld until a later decode stabilizes it");

    auto second = decoder.append_token(2);
    require(second.emitted_text == "Hel",
            "second decode should emit the stable prefix from the first decode");

    auto third = decoder.append_token(3);
    require(third.emitted_text == "lo",
            "third decode should emit only the newly stable prefix bytes");

    auto final = decoder.finish();
    require(final.emitted_text == "!",
            "final flush should emit the remaining stable suffix");
    require(decoder.emitted_text() == "Hello!",
            "cumulative decoder should preserve final byte parity");
}

void test_external_cumulative_decoder_may_lag_immediate_oga_stream() {
    const std::unordered_map<std::string, std::string> decoded_by_tokens = {
        {"1", "Hel"},
        {"1,2", "Hello"},
        {"1,2,3", "Hello!"},
    };
    const std::unordered_map<int, std::string> native_oga_chunks = {
        {1, "Hel"},
        {2, "lo"},
        {3, "!"},
    };
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [&](const std::vector<int>& token_ids) {
            return flm::hip_oga::CumulativeDecodeResult::match(
                decoded_by_tokens.at(flm::hip_oga::join_token_ids_for_evidence(token_ids)));
        });

    std::string native_oga_stream;
    for (int token_id : {1, 2, 3}) {
        native_oga_stream += native_oga_chunks.at(token_id);
        auto chunk = decoder.append_token(token_id);
        require(chunk.status == ParityStatus::Match,
                "delayed cumulative decode chunk should remain a parity candidate");
        require(native_oga_stream.rfind(decoder.emitted_text(), 0) == 0,
                "external stable prefix may lag but must stay inside the OGA stream prefix");
    }

    require(native_oga_stream == "Hello!",
            "fixture must model immediate OGA token-stream output");
    require(decoder.emitted_text() == "Hello",
            "external cumulative output should lag before final flush");

    auto final = decoder.finish();
    require(final.status == ParityStatus::Match,
            "delayed cumulative decode should finish cleanly");
    require(decoder.emitted_text() == native_oga_stream,
            "final cumulative decoded bytes must equal the OGA stream");
}

void test_external_cumulative_decoder_holds_split_utf8_until_final_flush() {
    const std::string partial_smile = std::string("\xF0\x9F", 2);
    const std::string complete_smile = std::string("\xF0\x9F\x99\x82", 4);
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [&](const std::vector<int>& token_ids) {
            if (token_ids.size() == 1) {
                return flm::hip_oga::CumulativeDecodeResult::match(partial_smile);
            }
            return flm::hip_oga::CumulativeDecodeResult::match(complete_smile);
        });

    require(decoder.append_token(7).emitted_text.empty(),
            "partial UTF-8 bytes must not be emitted");
    require(decoder.append_token(8).emitted_text.empty(),
            "bytes that were previously an incomplete UTF-8 prefix stay withheld");

    auto final = decoder.finish();
    require(final.status == ParityStatus::Match,
            "complete UTF-8 text should succeed on final flush");
    require(final.emitted_text == complete_smile,
            "final flush should emit the complete UTF-8 sequence");
}

void test_external_cumulative_decoder_fails_closed_on_final_invalid_utf8() {
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [](const std::vector<int>&) {
            return flm::hip_oga::CumulativeDecodeResult::match(
                std::string("\xF0\x9F", 2));
        });

    require(decoder.append_token(9).emitted_text.empty(),
            "invalid in-flight UTF-8 must not be emitted");
    auto final = decoder.finish();
    require(final.status == ParityStatus::Error,
            "final invalid UTF-8 must fail closed");
    require(final.emitted_text.empty(),
            "invalid final UTF-8 must not leak bytes");
    require(decoder.emitted_text().empty(),
            "failed closed decoder must not expose partial output");
}

void test_external_cumulative_decoder_fails_closed_when_emitted_prefix_is_revised() {
    const std::unordered_map<std::string, std::string> decoded_by_tokens = {
        {"1", "ab"},
        {"1,2", "abc"},
        {"1,2,3", "axc"},
    };
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [&](const std::vector<int>& token_ids) {
            return flm::hip_oga::CumulativeDecodeResult::match(
                decoded_by_tokens.at(flm::hip_oga::join_token_ids_for_evidence(token_ids)));
        });

    require(decoder.append_token(1).emitted_text.empty(),
            "first decode should be withheld");
    require(decoder.append_token(2).emitted_text == "ab",
            "second decode should stabilize the first prefix");
    auto revised = decoder.append_token(3);
    require(revised.status == ParityStatus::Error,
            "revision of already-emitted bytes must fail closed");
    require(revised.emitted_text.empty(),
            "revision failure must not emit replacement bytes");
}

void test_external_cumulative_decoder_allows_callback_to_hide_special_tokens() {
    const std::unordered_map<std::string, std::string> decoded_by_tokens = {
        {"101", "Hello"},
        {"101,0", "Hello"},
        {"101,0,102", "Hello world"},
    };
    flm::hip_oga::ExternalCumulativeDetokenizer decoder(
        [&](const std::vector<int>& token_ids) {
            return flm::hip_oga::CumulativeDecodeResult::match(
                decoded_by_tokens.at(flm::hip_oga::join_token_ids_for_evidence(token_ids)));
        });

    require(decoder.append_token(101).emitted_text.empty(),
            "visible token should be delayed until stable");
    require(decoder.append_token(0).emitted_text == "Hello",
            "special-token no-op decode should stabilize previous visible text");
    require(decoder.append_token(102).emitted_text.empty(),
            "new suffix should wait for a final decode or flush");
    require(decoder.finish().emitted_text == " world",
            "final flush should emit suffix after hidden special token");
}

}  // namespace

int main() {
    test_default_selector_remains_oga();
    test_selector_parses_supported_providers();
    test_selector_rejects_unknown_values();
    test_parity_result_types_have_stable_defaults();
    test_fixture_contract_type_names_are_stable();
    test_decoder_equivalence_classifies_raw_byte_exact_text();
    test_decoder_equivalence_accepts_gemma_standalone_metaspace_as_space();
    test_gemma_policy_detects_replace_decoder_contract();
    test_decoder_equivalence_does_not_globally_normalize_without_contract();
    test_decoder_equivalence_keeps_structured_json_strict();
    test_decoder_equivalence_rejects_invalid_utf8_and_special_leakage();
    test_external_cumulative_decoder_delays_until_prefix_is_stable();
    test_external_cumulative_decoder_may_lag_immediate_oga_stream();
    test_external_cumulative_decoder_holds_split_utf8_until_final_flush();
    test_external_cumulative_decoder_fails_closed_on_final_invalid_utf8();
    test_external_cumulative_decoder_fails_closed_when_emitted_prefix_is_revised();
    test_external_cumulative_decoder_allows_callback_to_hide_special_tokens();

    std::cout << "PASS: HIP OGA external text-codec contract" << std::endl;
    return 0;
}
