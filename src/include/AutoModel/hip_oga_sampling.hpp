// OGA-specific sampling adapter; public request semantics live in sampling/.
#pragma once

#include "sampling/request.hpp"
#include "AutoModel/hip_oga_logits.hpp"

namespace flm::hip_oga {

using flm::sampling::SamplingRequestError;
using flm::sampling::Provider;
using flm::sampling::is_sampling_field;
using flm::sampling::parse_sampling_integer;
using flm::sampling::parse_sampling_seed;
using flm::sampling::parse_sampling_float;
using flm::sampling::normalize_sampling_request;
using flm::sampling::sampling_number;
using flm::sampling::resolve_sampling_options;
using flm::sampling::SamplingState;
using flm::sampling::sampling_logits_options;

inline bool requires_logits_processing(const nlohmann::json& effective) {
    for (const char* name : {"min_p", "presence_penalty", "frequency_penalty"})
        if (effective.value(name, 0.0) != 0) return true;
    const double top_p = effective.at("top_p").get<double>();
    return effective.at("do_sample").get<bool>() && effective.at("top_k").get<int>() == 0 &&
           top_p > 0 && top_p < 1;
}

// Model eligibility must be composed before probability truncation.
template<class Params>
void mask_oga_sampling_logits(std::vector<float>& logits, std::size_t history_count,
                              const Params& params, const std::vector<int>& eos_ids) {
    const double minimum = params.GetSearchNumber("min_length");
    if (!std::isfinite(minimum) || minimum < 0)
        throw SamplingRequestError("Invalid model min_length");
    if (history_count < minimum) {
        if (eos_ids.empty()) throw SamplingRequestError("Model min_length requires EOS token IDs");
        for (int eos : eos_ids) {
            if (eos < 0 || static_cast<std::size_t>(eos) >= logits.size())
                throw SamplingRequestError("Model EOS token is outside the vocabulary");
            logits[eos] = -std::numeric_limits<float>::infinity();
        }
    }
}

template<class Params>
void process_oga_sampling_logits(std::vector<float>& logits, const int32_t* history,
                                 std::size_t history_count, const nlohmann::json& effective,
                                 const Params& params, const std::vector<int>& eos_ids) {
    mask_oga_sampling_logits(logits, history_count, params, eos_ids);
    process_sampling_logits(logits, history, history_count, sampling_logits_options(effective));
}

template<class Params>
nlohmann::json read_oga_sampling_options(const Params& params) {
    nlohmann::json result = {{"do_sample", params.GetSearchBool("do_sample")}};
    for (const char* name : {"temperature", "top_p", "repetition_penalty"})
        result[name] = params.GetSearchNumber(name);
    for (const char* name : {"top_k", "random_seed"})
        result[name] = static_cast<int>(params.GetSearchNumber(name));
    return result;
}

template<class Params>
void apply_oga_sampling_options(Params& params, const nlohmann::json& effective,
                                Provider provider = Provider::Backend) {
    const bool common = provider == Provider::Common;
    const bool processed = common || requires_logits_processing(effective);
    // The pinned runtime accepts this option but does not enforce it. Do not
    // silently advertise it as compatible with application-side truncation.
    if (processed && params.GetSearchNumber("no_repeat_ngram_size") != 0)
        throw SamplingRequestError("HIP OGA sampling extensions require model no_repeat_ngram_size=0");
    params.SetSearchOptionBool("do_sample", !common && effective.at("do_sample").get<bool>());
    // Common selection supplies a single finite candidate. The compatibility
    // extension path leaves final selection/RNG in OGA. Neither repeats filters.
    for (const char* name : {"temperature", "top_p", "repetition_penalty"})
        params.SetSearchOption(name, processed ? 1.0 : effective.at(name).get<double>());
    params.SetSearchOption("top_k", processed ? 0.0 : effective.at("top_k").get<double>());
    params.SetSearchOption("random_seed", effective.at("random_seed").get<double>());
}

} // namespace flm::hip_oga
