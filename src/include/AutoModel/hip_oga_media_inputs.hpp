/// Checked input contract for explicitly packaged Gemma flat-feature models.
#pragma once

#include "nlohmann/json.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flm::hip_oga {

inline void validate_gemma3_token_layout(const std::vector<int>& input_ids,
                                       const std::vector<int>& token_types,
                                       int image_token_id, std::int64_t expected_image_tokens) {
    if (input_ids.empty() || input_ids.size() != token_types.size() ||
        image_token_id < 0 || expected_image_tokens < 0) {
        throw std::runtime_error("Invalid Gemma processor token layout dimensions");
    }
    std::size_t image_tokens = 0;
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        if (input_ids[i] < 0 || (token_types[i] != 0 && token_types[i] != 1) ||
            ((input_ids[i] == image_token_id) != (token_types[i] == 1))) {
            throw std::runtime_error("Gemma processor image placeholders and token types disagree");
        }
        image_tokens += token_types[i] == 1;
    }
    if (image_tokens != static_cast<std::uint64_t>(expected_image_tokens)) {
        throw std::runtime_error("Gemma processor image placeholders and feature rows disagree");
    }
}

inline nlohmann::json read_media_compatibility(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return nlohmann::json::object();
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("flm_hip_oga.json must be a regular file");
    }
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read flm_hip_oga.json");
    }
    auto parsed = nlohmann::json::parse(file, nullptr, false);
    if (file.bad() || parsed.is_discarded()) {
        throw std::runtime_error("Invalid JSON in flm_hip_oga.json");
    }
    // Preserve non-object values so contract validation can reject them.
    return parsed;
}

inline bool uses_flat_gemma3_features(const nlohmann::json& compatibility,
                                     const nlohmann::json& genai_config) {
    if (compatibility.is_null() || (compatibility.is_object() && compatibility.empty())) {
        return false;
    }
    if (!compatibility.is_object() || compatibility.size() != 2 ||
        !compatibility.contains("schema_version") ||
        !compatibility.at("schema_version").is_number_integer() ||
        compatibility.at("schema_version") != 1 ||
        !compatibility.contains("image_features") ||
        compatibility.at("image_features") != "gemma3_flat_tokens_v1") {
        throw std::runtime_error("Unsupported flm_hip_oga.json image feature contract");
    }
    if (!genai_config.contains("model") || !genai_config.at("model").is_object() ||
        !genai_config.at("model").contains("type") ||
        genai_config.at("model").at("type") != "gemma3") {
        throw std::runtime_error("Flat Gemma image features require a gemma3 model");
    }
    return true;
}

inline std::int64_t gemma3_total_image_tokens(
    std::size_t image_count, const std::vector<std::int64_t>& pixel_shape,
    std::int64_t per_image_tokens, const std::vector<int>& token_types) {
    constexpr std::int64_t tokens_per_image = 256;
    if (image_count == 0 || image_count > std::numeric_limits<std::size_t>::max() / tokens_per_image ||
        image_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / tokens_per_image)) {
        throw std::runtime_error("Invalid flat Gemma image count");
    }
    if (pixel_shape.size() != 4 || pixel_shape[0] != static_cast<std::int64_t>(image_count) ||
        pixel_shape[1] != 3 || pixel_shape[2] <= 0 || pixel_shape[3] <= 0 ||
        per_image_tokens != tokens_per_image) {
        throw std::runtime_error("Flat Gemma processor shape or per-image token count mismatch");
    }
    const auto total = static_cast<std::int64_t>(image_count) * tokens_per_image;
    std::size_t observed = 0;
    for (int type : token_types) {
        if (type != 0 && type != 1) {
            throw std::runtime_error("Flat Gemma processor returned an unknown token type");
        }
        observed += type == 1;
    }
    if (observed != static_cast<std::size_t>(total)) {
        throw std::runtime_error("Flat Gemma image token types disagree with the image count");
    }
    return total;
}

}  // namespace flm::hip_oga
