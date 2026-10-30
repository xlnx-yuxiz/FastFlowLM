/// \file hip_oga_text_codec.hpp
/// \brief Shared no-hardware contract for HIP OGA text-codec selection evidence.
#pragma once

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace flm::hip_oga {

enum class TextCodecProvider {
    Oga,
    Flm,
    Compare,
    // Legacy FLM output with mandatory OGA parity checks.
    External,
};

enum class ParityStatus {
    NotRun,
    Match,
    Mismatch,
    Error,
};

enum class DecoderEquivalenceClass {
    ByteExact,
    CanonicalEquivalent,
    StructuredEquivalent,
    Incompatible,
};

enum class DecoderEquivalenceMode {
    PlainText,
    StructuredJson,
};

struct TextCodecSelector {
    TextCodecProvider provider = TextCodecProvider::Oga;
    std::string requested_name;
    std::string normalized_name = "oga";
    bool explicit_request = false;
};

struct DeterministicTextCodecFixture {
    std::string id;
    std::string prompt;
    std::string normalized_prompt;
    std::vector<int> expected_token_ids;
    std::string expected_decoded_text;
};

struct TextCodecParityResult {
    ParityStatus status = ParityStatus::NotRun;
    std::string prompt;
    std::vector<int> oga_token_ids;
    std::vector<int> external_token_ids;
    std::string oga_decoded_text;
    std::string external_decoded_text;
    std::string notes;

    bool matches() const {
        return status == ParityStatus::Match &&
            oga_token_ids == external_token_ids &&
            oga_decoded_text == external_decoded_text;
    }
};

struct DecoderCanonicalizationPolicy {
    DecoderEquivalenceMode mode = DecoderEquivalenceMode::PlainText;
    bool gemma_standalone_metaspace_is_space = false;
    std::string contract_name = "plain_text";
};

struct TextMismatchEvidence {
    std::size_t byte_index = 0;
    std::string oga_text_window;
    std::string external_text_window;
};

struct DecoderEquivalenceResult {
    DecoderEquivalenceClass equivalence_class =
        DecoderEquivalenceClass::Incompatible;
    bool raw_byte_equal = false;
    bool valid_utf8 = false;
    bool special_token_leakage = false;
    std::string oga_canonical_text;
    std::string external_canonical_text;
    std::optional<TextMismatchEvidence> first_mismatch;
    std::vector<std::string> normalization_applied;
    std::string notes;

    bool accepted() const {
        return equivalence_class == DecoderEquivalenceClass::ByteExact ||
            equivalence_class == DecoderEquivalenceClass::CanonicalEquivalent ||
            equivalence_class == DecoderEquivalenceClass::StructuredEquivalent;
    }
};

struct CumulativeDecodeResult {
    ParityStatus status = ParityStatus::NotRun;
    std::string text;
    std::string error_message;

    static CumulativeDecodeResult match(std::string decoded_text) {
        CumulativeDecodeResult result;
        result.status = ParityStatus::Match;
        result.text = std::move(decoded_text);
        return result;
    }

    static CumulativeDecodeResult error(std::string message) {
        CumulativeDecodeResult result;
        result.status = ParityStatus::Error;
        result.error_message = std::move(message);
        return result;
    }
};

struct DetokenizerChunkEmission {
    ParityStatus status = ParityStatus::NotRun;
    std::vector<int> token_ids;
    std::string cumulative_text;
    std::string emitted_text;
    std::string notes;
    bool final = false;
};

using CumulativeDecodeCallback =
    std::function<CumulativeDecodeResult(const std::vector<int>& token_ids)>;

inline std::string to_string(TextCodecProvider provider) {
    switch (provider) {
        case TextCodecProvider::Oga:
            return "oga";
        case TextCodecProvider::Flm:
            return "flm";
        case TextCodecProvider::Compare:
            return "compare";
        case TextCodecProvider::External:
            return "external";
    }
    throw std::invalid_argument("unknown HIP OGA text codec provider");
}

inline std::string to_string(ParityStatus status) {
    switch (status) {
        case ParityStatus::NotRun:
            return "not_run";
        case ParityStatus::Match:
            return "match";
        case ParityStatus::Mismatch:
            return "mismatch";
        case ParityStatus::Error:
            return "error";
    }
    throw std::invalid_argument("unknown HIP OGA text codec parity status");
}

inline std::string to_string(DecoderEquivalenceClass equivalence_class) {
    switch (equivalence_class) {
        case DecoderEquivalenceClass::ByteExact:
            return "byte_exact";
        case DecoderEquivalenceClass::CanonicalEquivalent:
            return "canonical_equivalent";
        case DecoderEquivalenceClass::StructuredEquivalent:
            return "structured_equivalent";
        case DecoderEquivalenceClass::Incompatible:
            return "incompatible";
    }
    throw std::invalid_argument("unknown HIP OGA decoder equivalence class");
}

inline std::string to_string(DecoderEquivalenceMode mode) {
    switch (mode) {
        case DecoderEquivalenceMode::PlainText:
            return "plain_text";
        case DecoderEquivalenceMode::StructuredJson:
            return "structured_json";
    }
    throw std::invalid_argument("unknown HIP OGA decoder equivalence mode");
}

inline std::string normalize_selector_name(const std::string& selector) {
    const auto begin = std::find_if_not(
        selector.begin(), selector.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        });
    const auto end = std::find_if_not(
        selector.rbegin(), selector.rend(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }).base();
    if (begin >= end) return {};

    std::string normalized(begin, end);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) {
                       if (ch == '_') return '-';
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

inline TextCodecSelector default_text_codec_selector() {
    return TextCodecSelector{};
}

inline TextCodecSelector parse_text_codec_selector(const std::string& selector) {
    const std::string normalized = normalize_selector_name(selector);
    if (normalized.empty() || normalized == "default" ||
        normalized == "oga" || normalized == "native-oga") {
        return TextCodecSelector{
            TextCodecProvider::Oga,
            selector,
            "oga",
            !normalized.empty() && normalized != "default"};
    }
    if (normalized == "external" || normalized == "flm" || normalized == "compare") {
        return TextCodecSelector{
            normalized == "flm" ? TextCodecProvider::Flm :
                normalized == "compare" ? TextCodecProvider::Compare : TextCodecProvider::External,
            selector,
            normalized,
            true};
    }

    throw std::invalid_argument(
        "Unknown HIP OGA text codec selector '" + selector +
        "'. Supported text codec selectors: oga, flm, compare, external (legacy checked FLM)");
}

struct TextCodecExecution {
    bool use_oga;
    bool use_flm;
    bool compare;
    bool output_flm;
};

inline TextCodecExecution text_codec_execution(const TextCodecSelector& selector) {
    switch (selector.provider) {
        case TextCodecProvider::Oga: return {true, false, false, false};
        case TextCodecProvider::Flm: return {false, true, false, true};
        case TextCodecProvider::Compare: return {true, true, true, false};
        case TextCodecProvider::External: return {true, true, true, true};
    }
    throw std::invalid_argument("unknown HIP OGA text codec provider");
}

inline std::string join_token_ids_for_evidence(const std::vector<int>& token_ids) {
    std::string joined;
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) joined += ",";
        joined += std::to_string(token_ids[index]);
    }
    return joined;
}

inline std::size_t valid_utf8_prefix_length(const std::string& text,
                                            std::size_t limit) {
    limit = std::min(limit, text.size());
    std::size_t index = 0;
    while (index < limit) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t width = 0;
        if (lead <= 0x7F) {
            width = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            width = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            width = 3;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            width = 4;
        } else {
            return index;
        }
        if (index + width > limit) return index;

        auto continuation = [&](std::size_t offset) {
            const auto ch = static_cast<unsigned char>(text[index + offset]);
            return (ch & 0xC0) == 0x80;
        };
        if ((width >= 2 && !continuation(1)) ||
            (width >= 3 && !continuation(2)) ||
            (width >= 4 && !continuation(3))) {
            return index;
        }

        const auto second =
            width >= 2 ? static_cast<unsigned char>(text[index + 1]) : 0;
        if ((lead == 0xE0 && second < 0xA0) ||
            (lead == 0xED && second > 0x9F) ||
            (lead == 0xF0 && second < 0x90) ||
            (lead == 0xF4 && second > 0x8F)) {
            return index;
        }
        index += width;
    }
    return index;
}

inline bool is_valid_utf8(const std::string& text) {
    return valid_utf8_prefix_length(text, text.size()) == text.size();
}

inline bool contains_special_token_marker(const std::string& text) {
    return text.find("<|") != std::string::npos ||
        text.find("<s>") != std::string::npos ||
        text.find("</s>") != std::string::npos ||
        text.find("<bos>") != std::string::npos ||
        text.find("<eos>") != std::string::npos ||
        text.find("<pad>") != std::string::npos;
}

inline TextMismatchEvidence first_text_mismatch(const std::string& oga_text,
                                                const std::string& external_text) {
    const std::size_t limit = std::min(oga_text.size(), external_text.size());
    std::size_t index = 0;
    while (index < limit && oga_text[index] == external_text[index]) ++index;

    const std::size_t window_start = index > 16 ? index - 16 : 0;
    const std::size_t oga_window_len =
        std::min<std::size_t>(32, oga_text.size() - window_start);
    const std::size_t external_window_len =
        std::min<std::size_t>(32, external_text.size() - window_start);
    return TextMismatchEvidence{
        index,
        oga_text.substr(window_start, oga_window_len),
        external_text.substr(window_start, external_window_len)};
}

inline std::string canonicalize_plain_text(
    const std::string& text,
    const DecoderCanonicalizationPolicy& policy,
    std::vector<std::string>& normalization_applied) {
    if (!policy.gemma_standalone_metaspace_is_space) {
        return text;
    }

    constexpr std::string_view metaspace = "\xE2\x96\x81";
    std::string canonical;
    canonical.reserve(text.size());
    bool rewrote_metaspace = false;
    for (std::size_t offset = 0; offset < text.size();) {
        if (text.compare(offset, metaspace.size(), metaspace) == 0) {
            canonical.push_back(' ');
            offset += metaspace.size();
            rewrote_metaspace = true;
            continue;
        }
        canonical.push_back(text[offset]);
        ++offset;
    }
    if (rewrote_metaspace &&
        std::find(normalization_applied.begin(), normalization_applied.end(),
                  "gemma_standalone_metaspace_to_space") ==
            normalization_applied.end()) {
        normalization_applied.push_back("gemma_standalone_metaspace_to_space");
    }
    return canonical;
}

inline bool parsed_json_values_equal(const std::string& left,
                                     const std::string& right) {
    auto left_json = nlohmann::json::parse(left, nullptr, false);
    auto right_json = nlohmann::json::parse(right, nullptr, false);
    return !left_json.is_discarded() && !right_json.is_discarded() &&
        left_json == right_json;
}

inline std::string lowercase_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

inline bool json_string_field_contains(const nlohmann::json& object,
                                       const char* key,
                                       std::string_view needle) {
    if (!object.is_object() || !object.contains(key) ||
        !object[key].is_string()) {
        return false;
    }
    return lowercase_ascii(object[key].get<std::string>()).find(needle) !=
        std::string::npos;
}

inline bool tokenizer_config_declares_gemma(const nlohmann::json& config) {
    return json_string_field_contains(config, "tokenizer_class", "gemma") ||
        json_string_field_contains(config, "model_type", "gemma") ||
        json_string_field_contains(config, "processor_class", "gemma");
}

inline bool json_object_contains_key(const nlohmann::json& object,
                                     std::string_view key) {
    return object.is_object() &&
        object.find(std::string(key)) != object.end();
}

inline bool token_is_only_metaspace(std::string_view token) {
    constexpr std::string_view metaspace = "\xE2\x96\x81";
    if (token.empty() || token.size() % metaspace.size() != 0) return false;
    for (std::size_t offset = 0; offset < token.size();
         offset += metaspace.size()) {
        if (token.substr(offset, metaspace.size()) != metaspace) return false;
    }
    return true;
}

inline bool tokenizer_json_has_standalone_metaspace_vocab(
    const nlohmann::json& tokenizer_json) {
    const auto& model = tokenizer_json.value("model", nlohmann::json::object());
    const auto& vocab = model.value("vocab", nlohmann::json::object());
    if (!vocab.is_object()) return false;
    constexpr std::string_view metaspace = "\xE2\x96\x81";
    if (json_object_contains_key(vocab, metaspace)) return true;
    for (auto item = vocab.begin(); item != vocab.end(); ++item) {
        if (token_is_only_metaspace(item.key())) return true;
    }
    return false;
}

inline bool decoder_step_replaces_metaspace_with_space(
    const nlohmann::json& decoder_step) {
    if (!decoder_step.is_object() ||
        decoder_step.value("type", "") != "Replace" ||
        decoder_step.value("content", "") != " ") {
        return false;
    }
    const auto& pattern = decoder_step.value("pattern", nlohmann::json::object());
    return pattern.is_object() &&
        pattern.value("String", "") == "\xE2\x96\x81";
}

inline bool tokenizer_json_has_metaspace_decoder_contract(
    const nlohmann::json& tokenizer_json) {
    const auto& decoder = tokenizer_json.value("decoder", nlohmann::json::object());
    if (!decoder.is_object()) return false;
    if (decoder.value("type", "") == "Metaspace" ||
        decoder_step_replaces_metaspace_with_space(decoder)) {
        return true;
    }
    if (decoder.value("type", "") == "Sequence" &&
        decoder.contains("decoders") && decoder["decoders"].is_array()) {
        return std::any_of(
            decoder["decoders"].begin(), decoder["decoders"].end(),
            [](const nlohmann::json& step) {
                return step.is_object() &&
                    (step.value("type", "") == "Metaspace" ||
                     decoder_step_replaces_metaspace_with_space(step));
            });
    }
    return false;
}

inline DecoderCanonicalizationPolicy decoder_policy_from_tokenizer_contract(
    const nlohmann::json& tokenizer_config,
    const nlohmann::json& tokenizer_json) {
    DecoderCanonicalizationPolicy policy;
    if (tokenizer_config_declares_gemma(tokenizer_config) &&
        tokenizer_json_has_metaspace_decoder_contract(tokenizer_json) &&
        tokenizer_json_has_standalone_metaspace_vocab(tokenizer_json)) {
        policy.gemma_standalone_metaspace_is_space = true;
        policy.contract_name = "gemma_standalone_metaspace";
    }
    return policy;
}

// XGrammar follows the tokenizer's metaspace-to-space contract. Some OGA
// decoders preserve a standalone multi-metaspace token literally instead.
// Filter only proven incompatible IDs; do not rewrite generated JSON strings.
inline std::vector<int> incompatible_metaspace_tokens(
    const nlohmann::json& tokenizer_json,
    const std::function<bool(int, const std::string&)>& decoder_matches) {
    if (!tokenizer_json_has_metaspace_decoder_contract(tokenizer_json)) return {};
    std::map<int, std::string> candidates;
    const auto record = [&](const nlohmann::json& id, const std::string& token) {
        if (!id.is_number_integer() || id.get<double>() < 0 ||
            id.get<double>() > std::numeric_limits<int>::max()) return;
        const int index = id.get<int>();
        if (token_is_only_metaspace(token)) candidates[index] = std::string(token.size() / 3, ' ');
        else candidates.erase(index);
    };
    const auto& model = tokenizer_json.value("model", nlohmann::json::object());
    const auto& vocab = model.value("vocab", nlohmann::json::object());
    if (vocab.is_object()) {
        for (auto item = vocab.begin(); item != vocab.end(); ++item) record(item.value(), item.key());
    }
    const auto& added = tokenizer_json.value("added_tokens", nlohmann::json::array());
    if (added.is_array()) {
        for (const auto& entry : added) {
            if (entry.is_object() && entry.contains("id") && entry.contains("content") &&
                entry["content"].is_string()) record(entry["id"], entry["content"].get<std::string>());
        }
    }
    std::vector<int> excluded;
    for (const auto& [id, expected] : candidates)
        if (!decoder_matches(id, expected)) excluded.push_back(id);
    return excluded;
}

inline DecoderEquivalenceResult classify_decoder_text_equivalence(
    const std::string& oga_text,
    const std::string& external_text,
    const DecoderCanonicalizationPolicy& policy) {
    DecoderEquivalenceResult result;
    result.raw_byte_equal = oga_text == external_text;
    result.valid_utf8 = is_valid_utf8(oga_text) && is_valid_utf8(external_text);
    result.special_token_leakage =
        contains_special_token_marker(oga_text) ||
        contains_special_token_marker(external_text);
    result.oga_canonical_text = oga_text;
    result.external_canonical_text = external_text;

    if (!result.raw_byte_equal) {
        result.first_mismatch = first_text_mismatch(oga_text, external_text);
    }
    if (!result.valid_utf8) {
        result.notes = "decoder output is not valid UTF-8";
        return result;
    }
    if (result.special_token_leakage) {
        result.notes = "decoder output leaked a special-token marker";
        return result;
    }
    if (result.raw_byte_equal) {
        result.equivalence_class = DecoderEquivalenceClass::ByteExact;
        return result;
    }

    if (policy.mode == DecoderEquivalenceMode::StructuredJson) {
        if (parsed_json_values_equal(oga_text, external_text)) {
            result.equivalence_class =
                DecoderEquivalenceClass::StructuredEquivalent;
        } else {
            result.notes =
                "structured decoder outputs are not parsed-value equivalent";
        }
        return result;
    }

    result.oga_canonical_text =
        canonicalize_plain_text(oga_text, policy, result.normalization_applied);
    result.external_canonical_text =
        canonicalize_plain_text(external_text, policy, result.normalization_applied);
    if (result.oga_canonical_text == result.external_canonical_text) {
        result.equivalence_class =
            DecoderEquivalenceClass::CanonicalEquivalent;
    } else {
        result.notes = "canonical visible text differs";
    }
    return result;
}

class ExternalCumulativeDetokenizer {
public:
    explicit ExternalCumulativeDetokenizer(CumulativeDecodeCallback decode)
        : decode_(std::move(decode)) {
        if (!decode_) {
            throw std::invalid_argument(
                "external cumulative detokenizer requires a decode callback");
        }
    }

    DetokenizerChunkEmission append_token(int token_id) {
        if (failed_) return failed_chunk(false);
        token_ids_.push_back(token_id);
        const CumulativeDecodeResult decoded = decode_(token_ids_);
        return emit_from_cumulative(decoded, false);
    }

    DetokenizerChunkEmission finish() {
        if (failed_) return failed_chunk(true);
        if (previous_cumulative_.empty()) {
            return DetokenizerChunkEmission{
                ParityStatus::Match, token_ids_, {}, {}, {}, true};
        }
        if (!is_valid_utf8(previous_cumulative_)) {
            return fail("final cumulative decode is not valid UTF-8", true);
        }
        return emit_stable_prefix(previous_cumulative_, previous_cumulative_.size(),
                                  true);
    }

    const std::string& emitted_text() const {
        return emitted_text_;
    }

private:
    DetokenizerChunkEmission emit_from_cumulative(
        const CumulativeDecodeResult& decoded, bool final) {
        if (decoded.status != ParityStatus::Match) {
            return fail(decoded.error_message.empty()
                            ? "external cumulative decode failed"
                            : decoded.error_message,
                        final);
        }
        if (decoded.text.rfind(emitted_text_, 0) != 0) {
            return fail("external cumulative decode revised emitted prefix", final);
        }

        std::size_t stable_prefix =
            final ? decoded.text.size()
                  : common_prefix_length(previous_cumulative_, decoded.text);
        if (!final) {
            // ByteFallback can return one U+FFFD per incomplete byte for several
            // tokens before resolving them into a single Unicode scalar. Even
            // repeated replacement characters at the end are not stable yet.
            constexpr std::string_view replacement = "\xEF\xBF\xBD";
            std::size_t resolved_end = decoded.text.size();
            while (resolved_end >= replacement.size() &&
                   decoded.text.compare(resolved_end - replacement.size(), replacement.size(), replacement) == 0)
                resolved_end -= replacement.size();
            stable_prefix = std::min(stable_prefix, resolved_end);
        }
        auto chunk = emit_stable_prefix(decoded.text, stable_prefix, final);
        previous_cumulative_ = decoded.text;
        return chunk;
    }

    static std::size_t common_prefix_length(const std::string& left,
                                            const std::string& right) {
        const std::size_t limit = std::min(left.size(), right.size());
        std::size_t index = 0;
        while (index < limit && left[index] == right[index]) ++index;
        return index;
    }

    DetokenizerChunkEmission emit_stable_prefix(const std::string& cumulative,
                                                std::size_t stable_prefix,
                                                bool final) {
        stable_prefix = valid_utf8_prefix_length(cumulative, stable_prefix);
        if (stable_prefix < emitted_text_.size()) {
            return fail("external cumulative decode invalidated emitted bytes",
                        final);
        }

        DetokenizerChunkEmission chunk;
        chunk.status = ParityStatus::Match;
        chunk.token_ids = token_ids_;
        chunk.cumulative_text = cumulative;
        chunk.final = final;
        chunk.emitted_text =
            cumulative.substr(emitted_text_.size(),
                              stable_prefix - emitted_text_.size());
        emitted_text_ += chunk.emitted_text;
        return chunk;
    }

    DetokenizerChunkEmission fail(std::string message, bool final) {
        failed_ = true;
        error_message_ = std::move(message);
        return failed_chunk(final);
    }

    DetokenizerChunkEmission failed_chunk(bool final) const {
        DetokenizerChunkEmission chunk;
        chunk.status = ParityStatus::Error;
        chunk.token_ids = token_ids_;
        chunk.cumulative_text = previous_cumulative_;
        chunk.notes = error_message_;
        chunk.final = final;
        return chunk;
    }

    CumulativeDecodeCallback decode_;
    std::vector<int> token_ids_;
    std::string previous_cumulative_;
    std::string emitted_text_;
    std::string error_message_;
    bool failed_ = false;
};

}  // namespace flm::hip_oga
