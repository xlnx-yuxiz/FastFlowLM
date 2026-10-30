#include "AutoModel/hip_oga_constraint.hpp"
#include "AutoModel/hip_oga_text_codec.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
template<class F> void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unsafe mask/logits were accepted");
}
int main() {
    // A missing mask or the wrong bit order would let the high-scoring forbidden token win.
    std::vector<float> scores{100.0f, 2.0f, 3.0f};
    flm::hip_oga::mask_constraint_logits(scores, {0b110u});
    require(std::isinf(scores[0]) && scores[0] < 0, "forbidden score was not masked");
    require(scores[1] == 2.0f && scores[2] == 3.0f, "allowed scores changed");
    rejects([] { std::vector<float> x(33, 1); flm::hip_oga::mask_constraint_logits(x, {1u}); });
    rejects([] { std::vector<float> x(3, 1); flm::hip_oga::mask_constraint_logits(x, {8u}); });
    rejects([] { std::vector<float> x{1, 2}; flm::hip_oga::mask_constraint_logits(x, {0u}); });
    rejects([] { std::vector<float> x{std::numeric_limits<float>::quiet_NaN()};
                 flm::hip_oga::mask_constraint_logits(x, {1u}); });
    rejects([] { std::vector<float> x{std::numeric_limits<float>::infinity()};
                 flm::hip_oga::mask_constraint_logits(x, {1u}); });
    require(flm::hip_oga::constraint_token_allowed({2u}, 1, 2), "valid token rejected");
    require(!flm::hip_oga::constraint_token_allowed({2u}, 0, 2), "forbidden token accepted");
    require(!flm::hip_oga::constraint_token_allowed({0xffffffffu}, 2, 2), "padded token accepted");
    require(!flm::hip_oga::constraint_token_allowed({0xffffffffu}, -1, 2), "negative token accepted");
    // Decoder exclusions must intersect the grammar, never re-enable a token.
    std::vector<std::uint32_t> mask{0b110u};
    flm::hip_oga::exclude_constraint_tokens(mask, 3, {2});
    scores = {100, 2, 50};
    flm::hip_oga::mask_constraint_logits(scores, mask);
    require(std::isinf(scores[0]) && scores[1] == 2 && std::isinf(scores[2]),
            "decoder filter changed the grammar intersection");
    rejects([&] { flm::hip_oga::exclude_constraint_tokens(mask, 3, {3}); });
    rejects([&] { flm::hip_oga::exclude_constraint_tokens(mask, 3, {1});
                 flm::hip_oga::mask_constraint_logits(scores, mask); });
    const std::string meta = "\xE2\x96\x81";
    nlohmann::json tokenizer = {
        {"model", {{"vocab", {{meta + meta, 1}, {meta, 2}, {"<0x20>", 3}, {meta + "a", 4}}}}},
        {"decoder", {{"type", "Replace"}, {"pattern", {{"String", meta}}}, {"content", " "}}}};
    const auto compatible = [](int token, const std::string& expected) {
        return token == 2 && expected == " ";
    };
    require(flm::hip_oga::incompatible_metaspace_tokens(tokenizer, compatible) == std::vector<int>{1},
            "decoder compatibility did not preserve the valid single-space token");
    tokenizer["added_tokens"] = {{{"id", 1}, {"content", "literal"}}};
    require(flm::hip_oga::incompatible_metaspace_tokens(tokenizer, compatible).empty(),
            "added token override was treated as a metaspace token");
    tokenizer.erase("decoder");
    require(flm::hip_oga::incompatible_metaspace_tokens(tokenizer, compatible).empty(),
            "token filter guessed a missing decoder contract");
    std::cout << "constraint mask boundary checks PASS\n";
}
