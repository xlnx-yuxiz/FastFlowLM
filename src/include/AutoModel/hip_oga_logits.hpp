// Compatibility names for the existing HIP sampling adapter and fixtures.
#pragma once
#include "sampling/logits.hpp"

namespace flm::hip_oga {
using SamplingLogitsOptions = flm::sampling::LogitsOptions;

inline void process_sampling_logits(std::vector<float>& logits, const int32_t* history,
                                    std::size_t history_count, const SamplingLogitsOptions& options) {
    flm::sampling::process_logits(logits, history, history_count, options);
}
} // namespace flm::hip_oga
