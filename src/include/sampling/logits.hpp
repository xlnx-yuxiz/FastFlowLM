// Backend-independent numerical sampling contract for host float32 logits.
// Caller supplies the current logical history; this function retains no state.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flm::sampling {

struct LogitsOptions {
    bool do_sample = true;
    double temperature = 1;
    double top_p = 1;
    double min_p = 0;
    double repetition_penalty = 1;
    double presence_penalty = 0;
    double frequency_penalty = 0;
    int top_k = 0;
};

inline void process_logits(std::vector<float>& logits, const int32_t* history,
                                    std::size_t history_count, const LogitsOptions& options) {
    // These are float32 API options. Bounding them here also guarantees that
    // penalties, counts and division by a tiny temperature fit in double.
    const auto float32_option = [](double value) {
        return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max() &&
               (value == 0 || static_cast<float>(value) != 0);
    };
    for (double value : {options.temperature, options.top_p, options.min_p,
                         options.repetition_penalty, options.presence_penalty, options.frequency_penalty})
        if (!float32_option(value)) throw std::invalid_argument("Invalid float32 sampling option");
    if (options.temperature <= 0 || options.repetition_penalty <= 0 ||
        options.top_p < 0 || options.top_p > 1 || options.min_p < 0 || options.min_p > 1 ||
        options.top_k < 0 || static_cast<std::size_t>(options.top_k) > logits.size())
        throw std::invalid_argument("Invalid sampling option range");
    if (logits.empty() || (history_count && !history))
        throw std::invalid_argument("Empty logits or missing sampling history");

    std::vector<std::size_t> counts(logits.size(), 0);
    for (std::size_t i = 0; i < history_count; ++i) {
        if (history[i] < 0 || static_cast<std::size_t>(history[i]) >= logits.size())
            throw std::invalid_argument("Sampling history token is outside the vocabulary");
        ++counts[static_cast<std::size_t>(history[i])];
    }

    const double banned = -std::numeric_limits<double>::infinity();
    std::vector<double> scores(logits.size(), banned);
    std::vector<std::size_t> candidates;
    candidates.reserve(logits.size());
    double maximum = banned;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        double score = logits[i];
        if (score == banned) continue; // A grammar/decoder ban cannot be revived.
        if (!std::isfinite(score)) throw std::invalid_argument("Non-finite sampling logit");
        if (counts[i]) {
            score = score > 0 ? score / options.repetition_penalty : score * options.repetition_penalty;
            score -= options.presence_penalty;
            score -= options.frequency_penalty * static_cast<double>(counts[i]);
        }
        scores[i] = score;
        maximum = std::max(maximum, score);
        candidates.push_back(i);
    }
    if (candidates.empty()) throw std::invalid_argument("Sampling allows no model token");
    const double temperature = options.do_sample ? options.temperature : 1;
    for (auto id : candidates) scores[id] = (scores[id] - maximum) / temperature;

    if (options.do_sample) {
        const auto higher_score = [&](std::size_t left, std::size_t right) {
            return scores[left] != scores[right] ? scores[left] > scores[right] : left < right;
        };
        const bool nucleus = options.top_p > 0 && options.top_p < 1;
        if (options.top_k > 0 && static_cast<std::size_t>(options.top_k) < candidates.size()) {
            const auto end = candidates.begin() + options.top_k;
            std::partial_sort(candidates.begin(), end, candidates.end(), higher_score);
            for (auto it = end; it != candidates.end(); ++it) scores[*it] = banned;
            candidates.resize(static_cast<std::size_t>(options.top_k));
        } else if (nucleus) {
            std::sort(candidates.begin(), candidates.end(), higher_score);
        }
        if (nucleus) {
            // Normalize over the candidates left by top-k and retain the token
            // that crosses the requested mass, including at least one maximum.
            double total = 0;
            for (auto id : candidates) total += std::exp(scores[id]);
            const double target = options.top_p * total;
            double cumulative = 0;
            std::size_t kept = 0;
            do { cumulative += std::exp(scores[candidates[kept++]]); }
            while (kept < candidates.size() && cumulative < target);
            for (std::size_t i = kept; i < candidates.size(); ++i) scores[candidates[i]] = banned;
            candidates.resize(kept);
        }
        if (options.min_p > 0) {
            const double threshold = std::log(options.min_p);
            for (auto id : candidates) if (scores[id] < threshold) scores[id] = banned;
        }
    }
    // Relative logits preserve the categorical distribution and greedy order.
    // Clamp extreme differences, never the maximum or an explicit ban.
    const double lowest = -static_cast<double>(std::numeric_limits<float>::max());
    for (std::size_t i = 0; i < logits.size(); ++i)
        logits[i] = scores[i] == banned ? -std::numeric_limits<float>::infinity()
                                       : static_cast<float>(std::max(lowest, scores[i]));
}

} // namespace flm::sampling
