// CPU reference provider. Device adapters own logits transfer and token commit.
#pragma once

#include "sampling/logits.hpp"
#include <random>

namespace flm::sampling {

// A half-open uniform in [0, 1) selects from processed logits in token-ID order.
// Explicit bans and underflowed zero-mass candidates can never be selected,
// including when uniform is exactly zero.
inline int32_t choose_token(const std::vector<float>& logits, double uniform) {
    if (!std::isfinite(uniform) || uniform < 0 || uniform >= 1)
        throw std::invalid_argument("Sampling uniform must be in [0, 1)");
    if (logits.empty() || logits.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Invalid sampling vocabulary size");
    const float banned = -std::numeric_limits<float>::infinity();
    double maximum = banned;
    for (float score : logits) {
        if (score == banned) continue;
        if (!std::isfinite(score)) throw std::invalid_argument("Non-finite sampling logit");
        maximum = std::max(maximum, static_cast<double>(score));
    }
    if (maximum == banned) throw std::invalid_argument("Sampling allows no model token");
    double total = 0;
    for (float score : logits) {
        if (score != banned) total += std::exp(static_cast<double>(score) - maximum);
    }
    const double target = uniform * total;
    double cumulative = 0;
    int32_t last = -1;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == banned) continue;
        const double mass = std::exp(static_cast<double>(logits[i]) - maximum);
        if (mass == 0) continue;
        last = static_cast<int32_t>(i);
        cumulative += mass;
        if (target < cumulative) return last;
    }
    // Floating-point multiplication may round a near-one target up to total.
    return last;
}

class CpuSampler {
public:
    // The owner resets the RNG at each request. Unseeded request policy belongs
    // to that owner; the reference provider itself has no clock/global state.
    void reset(uint64_t seed) { rng_.seed(seed); }

    int32_t sample(std::vector<float> logits, const int32_t* history,
                   std::size_t history_count, const LogitsOptions& options) {
        process_logits(logits, history, history_count, options);
        if (!options.do_sample) {
            // process_logits has validated the row and preserved explicit bans.
            // max_element's first match fixes equal-score selection by token ID.
            return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        // Specify conversion instead of depending on uniform_real_distribution,
        // whose mapping can differ between standard library implementations.
        const double uniform = static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0);
        return choose_token(logits, uniform);
    }

private:
    std::mt19937_64 rng_{0};
};

} // namespace flm::sampling
