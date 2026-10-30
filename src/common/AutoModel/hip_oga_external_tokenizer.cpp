/// \file hip_oga_external_tokenizer.cpp
/// \brief Callback-based tokenizer parity probe for HIP OGA source checks.
#include "AutoModel/hip_oga_external_tokenizer.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace flm::hip_oga {

namespace {

std::optional<TokenMismatch> find_first_mismatch(
    const std::vector<int>& oga_token_ids,
    const std::vector<int>& external_token_ids) {
    const std::size_t shared_count =
        std::min(oga_token_ids.size(), external_token_ids.size());
    for (std::size_t index = 0; index < shared_count; ++index) {
        if (oga_token_ids[index] != external_token_ids[index]) {
            return TokenMismatch{
                TokenMismatchKind::TokenId,
                index,
                oga_token_ids[index],
                external_token_ids[index]};
        }
    }

    if (oga_token_ids.size() != external_token_ids.size()) {
        return TokenMismatch{
            TokenMismatchKind::Length,
            shared_count,
            shared_count < oga_token_ids.size() ? oga_token_ids[shared_count] : -1,
            shared_count < external_token_ids.size()
                ? external_token_ids[shared_count]
                : -1};
    }

    return std::nullopt;
}

}  // namespace

ExternalTokenizerProbe::ExternalTokenizerProbe(
    TokenizeCallback oga_tokenize,
    TokenizeCallback external_tokenize)
    : oga_tokenize_(std::move(oga_tokenize)),
      external_tokenize_(std::move(external_tokenize)) {
    if (!oga_tokenize_ || !external_tokenize_) {
        throw std::invalid_argument(
            "ExternalTokenizerProbe requires both OGA and external callbacks");
    }
}

ExternalTokenizerProbeResult ExternalTokenizerProbe::compare(
    std::string_view prompt) const {
    ExternalTokenizerProbeResult result;
    result.prompt = std::string(prompt);

    try {
        result.oga_token_ids = oga_tokenize_(prompt);
        result.external_token_ids = external_tokenize_(prompt);
    } catch (const std::exception& e) {
        result.status = ParityStatus::Error;
        result.notes = e.what();
        return result;
    } catch (...) {
        result.status = ParityStatus::Error;
        result.notes = "unknown tokenizer callback error";
        return result;
    }

    result.first_mismatch =
        find_first_mismatch(result.oga_token_ids, result.external_token_ids);
    result.status =
        result.first_mismatch.has_value() ? ParityStatus::Mismatch
                                          : ParityStatus::Match;
    if (result.first_mismatch.has_value()) {
        result.notes = "first_mismatch=" +
            to_string(result.first_mismatch->kind);
    }
    return result;
}

std::string to_string(TokenMismatchKind kind) {
    switch (kind) {
        case TokenMismatchKind::None:
            return "none";
        case TokenMismatchKind::TokenId:
            return "token_id";
        case TokenMismatchKind::Length:
            return "length";
    }
    throw std::invalid_argument("unknown token mismatch kind");
}

}  // namespace flm::hip_oga
