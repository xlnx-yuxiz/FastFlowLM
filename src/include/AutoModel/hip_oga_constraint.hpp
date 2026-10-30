#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flm::hip_oga {

inline bool constraint_token_allowed(const std::vector<std::uint32_t>& mask,
                                     int token, std::size_t vocab_size) {
    return token >= 0 && static_cast<std::size_t>(token) < vocab_size &&
        static_cast<std::size_t>(token) / 32 < mask.size() &&
        (mask[static_cast<std::size_t>(token) / 32] & (1u << (token % 32))) != 0;
}

inline void exclude_constraint_tokens(std::vector<std::uint32_t>& mask,
                                      std::size_t vocab_size,
                                      const std::vector<int>& excluded) {
    if (mask.size() != (vocab_size + 31) / 32)
        throw std::runtime_error("HIP OGA decoder filter vocabulary mismatch");
    for (int token : excluded) {
        if (token < 0 || static_cast<std::size_t>(token) >= vocab_size)
            throw std::runtime_error("HIP OGA decoder filter token is outside the vocabulary");
        mask[token / 32] &= ~(1u << (token % 32));
    }
}

inline void mask_constraint_logits(std::vector<float>& logits,
                                   const std::vector<std::uint32_t>& mask) {
    if (logits.empty() || mask.size() != (logits.size() + 31) / 32)
        throw std::runtime_error("HIP OGA constraint mask does not match the logits vocabulary");
    bool allowed = false;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if ((mask[i / 32] & (1u << (i % 32))) != 0) {
            if (!std::isfinite(logits[i]))
                throw std::runtime_error("HIP OGA constrained logits contain a non-finite allowed score");
            allowed = true;
        } else {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
    if (!allowed) throw std::runtime_error("HIP OGA constraint allows no model token");
}

} // namespace flm::hip_oga
