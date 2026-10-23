/// \file hip_oga_external_tokenizer.hpp
/// \brief Callback-based no-hardware tokenizer parity probe for HIP OGA.
#pragma once

#include "AutoModel/hip_oga_text_codec.hpp"

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace flm::hip_oga {

enum class TokenMismatchKind {
    None,
    TokenId,
    Length,
};

struct TokenMismatch {
    TokenMismatchKind kind = TokenMismatchKind::None;
    std::size_t index = 0;
    int oga_token_id = -1;
    int external_token_id = -1;
};

struct ExternalTokenizerProbeResult {
    ParityStatus status = ParityStatus::NotRun;
    std::string prompt;
    std::vector<int> oga_token_ids;
    std::vector<int> external_token_ids;
    std::optional<TokenMismatch> first_mismatch;
    std::string notes;

    bool matches() const {
        return status == ParityStatus::Match &&
            oga_token_ids == external_token_ids &&
            !first_mismatch.has_value();
    }
};

using TokenizeCallback =
    std::function<std::vector<int>(std::string_view prompt)>;

class ExternalTokenizerProbe {
public:
    ExternalTokenizerProbe(TokenizeCallback oga_tokenize,
                           TokenizeCallback external_tokenize);

    ExternalTokenizerProbeResult compare(std::string_view prompt) const;

private:
    TokenizeCallback oga_tokenize_;
    TokenizeCallback external_tokenize_;
};

std::string to_string(TokenMismatchKind kind);

}  // namespace flm::hip_oga
