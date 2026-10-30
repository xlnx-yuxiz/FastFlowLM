#include "AutoModel/hip_oga_media_inputs.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace flm::hip_oga;

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (...) {
        threw = true;
    }
    require(threw, message);
}

nlohmann::json gemma3_compatibility() {
    return {{"schema_version", 1}, {"image_features", "gemma3_flat_tokens_v1"}};
}

nlohmann::json gemma3_config() {
    return {{"model", {{"type", "gemma3"}}}};
}

std::vector<int> image_token_types(std::size_t image_count) {
    std::vector<int> token_types(11, 0);  // Prompt tokens may be zero.
    token_types.insert(token_types.end(), image_count * 256, 1);
    return token_types;
}

void test_gemma3_feature_compatibility() {
    const auto config = gemma3_config();
    require(!uses_flat_gemma3_features({}, config), "empty compatibility metadata must disable flat features");
    require(uses_flat_gemma3_features(gemma3_compatibility(), config),
            "the exact compatibility contract must enable Gemma 3 flat features");

    auto rejects = [&](const nlohmann::json& compatibility,
                       const nlohmann::json& model_config,
                       const char* message) {
        require_throws([&] { (void)uses_flat_gemma3_features(compatibility, model_config); }, message);
    };

    rejects("gemma3_flat_tokens_v1", config, "compatibility metadata must be an object");
    rejects(nlohmann::json::array({1, "gemma3_flat_tokens_v1"}), config,
            "compatibility metadata arrays must be rejected");
    rejects({{"image_features", "gemma3_flat_tokens_v1"}}, config,
            "missing schema_version must be rejected");
    rejects({{"schema_version", 1}}, config, "missing image_features must be rejected");
    rejects({{"schema_version", "1"}, {"image_features", "gemma3_flat_tokens_v1"}}, config,
            "string schema_version must be rejected");
    rejects({{"schema_version", 1.0}, {"image_features", "gemma3_flat_tokens_v1"}}, config,
            "floating-point schema_version must be rejected");
    rejects({{"schema_version", true}, {"image_features", "gemma3_flat_tokens_v1"}}, config,
            "boolean schema_version must be rejected");
    rejects({{"schema_version", 0}, {"image_features", "gemma3_flat_tokens_v1"}}, config,
            "unsupported schema version must be rejected");
    rejects({{"schema_version", 2}, {"image_features", "gemma3_flat_tokens_v1"}}, config,
            "future schema version must be rejected");
    rejects({{"schema_version", 1}, {"image_features", 7}}, config,
            "non-string image_features must be rejected");
    rejects({{"schema_version", 1}, {"image_features", "other_features_v1"}}, config,
            "unknown image feature value must be rejected");
    rejects({{"schema_version", 1}, {"image_features", "gemma3_flat_tokens_v1"}, {"extra", true}}, config,
            "unknown compatibility fields must be rejected");

    rejects(gemma3_compatibility(), nlohmann::json::object(), "missing model metadata must be rejected");
    rejects(gemma3_compatibility(), {{"model", nlohmann::json::array()}},
            "model metadata with the wrong type must be rejected");
    rejects(gemma3_compatibility(), {{"model", nlohmann::json::object()}},
            "missing model.type must be rejected");
    rejects(gemma3_compatibility(), {{"model", {{"type", 3}}}},
            "non-string model.type must be rejected");
    rejects(gemma3_compatibility(), {{"model", {{"type", "gemma2"}}}},
            "flat Gemma 3 features must reject non-Gemma 3 models");
}

void test_total_image_token_count() {
    for (const std::size_t image_count : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        const auto tokens = image_token_types(image_count);
        const std::vector<std::int64_t> pixel_shape{
            static_cast<std::int64_t>(image_count), 3, 896, 896
        };
        const auto total = gemma3_total_image_tokens(image_count, pixel_shape, 256, tokens);
        require(total == static_cast<std::int64_t>(image_count * 256),
                "Gemma 3 total image tokens must equal 256 per image");
    }

    const auto one_image_tokens = image_token_types(1);
    require_throws([&] {
        (void)gemma3_total_image_tokens(0, {0, 3, 896, 896}, 256, {});
    }, "zero images must be rejected");
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896}, 256, one_image_tokens);
    }, "pixel_values must have rank four");
    require_throws([&] {
        (void)gemma3_total_image_tokens(2, {1, 3, 896, 896}, 256, image_token_types(2));
    }, "pixel_values image dimension must match image_count");
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 1, 896, 896}, 256, one_image_tokens);
    }, "pixel_values must have three channels");
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 0, 896}, 256, one_image_tokens);
    }, "zero pixel height must be rejected");
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896, -1}, 256, one_image_tokens);
    }, "non-positive pixel width must be rejected");

    for (const std::int64_t invalid_per_image : {std::int64_t{0}, std::int64_t{128}, std::int64_t{512}}) {
        require_throws([&] {
            (void)gemma3_total_image_tokens(1, {1, 3, 896, 896}, invalid_per_image, one_image_tokens);
        }, "per-image token count must be exactly 256, not absent, partial, or already totaled");
    }

    auto too_few_image_tokens = one_image_tokens;
    too_few_image_tokens.pop_back();
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896, 896}, 256, too_few_image_tokens);
    }, "fewer than 256 image token types must be rejected");
    auto too_many_image_tokens = one_image_tokens;
    too_many_image_tokens.push_back(1);
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896, 896}, 256, too_many_image_tokens);
    }, "more than 256 image token types must be rejected");
    auto negative_token_type = one_image_tokens;
    negative_token_type[0] = -1;
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896, 896}, 256, negative_token_type);
    }, "negative token types must be rejected");
    auto unknown_token_type = one_image_tokens;
    unknown_token_type[0] = 2;
    require_throws([&] {
        (void)gemma3_total_image_tokens(1, {1, 3, 896, 896}, 256, unknown_token_type);
    }, "token types other than zero or one must be rejected");

    // On the supported 64-bit target, unchecked multiplication wraps to zero.
    // An empty token vector then cannot conceal a missing overflow guard.
    const std::size_t overflowing_count = std::numeric_limits<std::size_t>::max() / 256 + 1;
    require_throws([&] {
        (void)gemma3_total_image_tokens(
            overflowing_count,
            {static_cast<std::int64_t>(overflowing_count), 3, 1, 1},
            256,
            {});
    }, "image count multiplied by 256 must reject int64 overflow");
}

void test_gemma3_token_layout() {
    // Literal miniature layout keeps token/mask alignment independent of the
    // 256-row tensor helper; actual processor rows are covered by hardware.
    validate_gemma3_token_layout({2, 262144, 262144, 107}, {0, 1, 1, 0}, 262144, 2);
    validate_gemma3_token_layout({2, 107}, {0, 0}, 262144, 0);
    validate_gemma3_token_layout({2, 262144, 107, 262144}, {0, 1, 0, 1}, 262144, 2);
    require_throws([] { validate_gemma3_token_layout({2, 262144}, {0}, 262144, 1); },
                   "token-type length must match input IDs");
    require_throws([] { validate_gemma3_token_layout({2, 107}, {0, 1}, 262144, 1); },
                   "image mask must name an image placeholder");
    require_throws([] { validate_gemma3_token_layout({2, 262144}, {0, 0}, 262144, 0); },
                   "image placeholder must carry image token type");
    require_throws([] { validate_gemma3_token_layout({2, 262144}, {0, 1}, 262144, 2); },
                   "image token span total must match processor rows");
    require_throws([] { validate_gemma3_token_layout({-1, 262144}, {0, 1}, 262144, 1); },
                   "Gemma input IDs must not be negative");
    require_throws([] { validate_gemma3_token_layout({2, 262144}, {0, 2}, 262144, 1); },
                   "unknown Gemma token type must be rejected");
    require_throws([] { validate_gemma3_token_layout({2}, {0}, -1, 0); },
                   "invalid image token ID must be rejected");
    require_throws([] { validate_gemma3_token_layout({2}, {0}, 262144, -1); },
                   "negative image feature count must be rejected");
    require_throws([] { validate_gemma3_token_layout({}, {}, 262144, 0); },
                   "empty processor prompt must be rejected");
}

int main() {
    try {
        test_gemma3_feature_compatibility();
        test_total_image_token_count();
        test_gemma3_token_layout();
        std::cout << "PASS: HIP OGA Gemma 3 media input contract\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
