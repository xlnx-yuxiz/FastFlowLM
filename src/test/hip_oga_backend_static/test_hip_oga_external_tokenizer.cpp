/// \file test_hip_oga_external_tokenizer.cpp
/// \brief No-hardware tests for HIP OGA external tokenizer parity probing.
#include "AutoModel/hip_oga_external_tokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using flm::hip_oga::TokenMismatchKind;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void test_exact_parity_reports_match() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view prompt) {
            require(prompt == "plain prompt", "OGA callback should receive the prompt");
            return std::vector<int>{10, 20, 30};
        },
        [](std::string_view prompt) {
            require(prompt == "plain prompt", "external callback should receive the prompt");
            return std::vector<int>{10, 20, 30};
        });

    const auto result = probe.compare("plain prompt");
    require(result.matches(), "identical token IDs must report a match");
    require(!result.first_mismatch.has_value(),
            "matching token IDs must not report a first mismatch");
}

void test_first_id_mismatch_is_classified() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view) { return std::vector<int>{1, 2, 3}; },
        [](std::string_view) { return std::vector<int>{1, 9, 3}; });

    const auto result = probe.compare("prompt");
    require(!result.matches(), "different token IDs must not report a match");
    require(result.first_mismatch.has_value(),
            "mismatch result must identify the first mismatch");
    require(result.first_mismatch->kind == TokenMismatchKind::TokenId,
            "different token IDs must be classified as token-id mismatch");
    require(result.first_mismatch->index == 1,
            "first mismatch must identify token index 1");
    require(result.first_mismatch->oga_token_id == 2,
            "first mismatch must capture the OGA token ID");
    require(result.first_mismatch->external_token_id == 9,
            "first mismatch must capture the external token ID");
}

void test_count_mismatch_is_classified_after_common_prefix() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view) { return std::vector<int>{5, 6, 7}; },
        [](std::string_view) { return std::vector<int>{5, 6}; });

    const auto result = probe.compare("prompt");
    require(!result.matches(), "different token counts must not report a match");
    require(result.first_mismatch.has_value(),
            "count mismatch must identify the first missing token index");
    require(result.first_mismatch->kind == TokenMismatchKind::Length,
            "different token counts must be classified as length mismatch");
    require(result.first_mismatch->index == 2,
            "count mismatch index must follow the shared prefix");
    require(result.first_mismatch->oga_token_id == 7,
            "count mismatch must capture the present OGA token ID");
    require(result.first_mismatch->external_token_id == -1,
            "count mismatch must use -1 for a missing external token");
}

void test_bos_eos_boundary_mismatches_remain_token_mismatches() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view) { return std::vector<int>{1, 100, 2}; },
        [](std::string_view) { return std::vector<int>{100, 2}; });

    const auto result = probe.compare("chat prompt");
    require(result.first_mismatch.has_value(),
            "BOS disagreement must report a first mismatch");
    require(result.first_mismatch->kind == TokenMismatchKind::TokenId,
            "BOS disagreement must remain a token-id mismatch");
    require(result.first_mismatch->index == 0,
            "BOS disagreement should be visible at index 0");
}

void test_empty_input_can_match_empty_tokens() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view prompt) {
            require(prompt.empty(), "OGA callback should receive empty input");
            return std::vector<int>{};
        },
        [](std::string_view prompt) {
            require(prompt.empty(), "external callback should receive empty input");
            return std::vector<int>{};
        });

    const auto result = probe.compare("");
    require(result.matches(), "empty token vectors may be a parity match");
}

void test_callback_errors_fail_closed() {
    flm::hip_oga::ExternalTokenizerProbe probe(
        [](std::string_view) -> std::vector<int> {
            throw std::runtime_error("OGA unavailable");
        },
        [](std::string_view) { return std::vector<int>{1}; });

    const auto result = probe.compare("prompt");
    require(!result.matches(), "callback errors must not report a match");
    require(result.status == flm::hip_oga::ParityStatus::Error,
            "callback errors must report parity status error");
    require(result.notes.find("OGA unavailable") != std::string::npos,
            "callback errors must preserve diagnostic text");
}

}  // namespace

int main() {
    test_exact_parity_reports_match();
    test_first_id_mismatch_is_classified();
    test_count_mismatch_is_classified_after_common_prefix();
    test_bos_eos_boundary_mismatches_remain_token_mismatches();
    test_empty_input_can_match_empty_tokens();
    test_callback_errors_fail_closed();

    std::cout << "PASS: HIP OGA external tokenizer parity probe" << std::endl;
    return 0;
}
