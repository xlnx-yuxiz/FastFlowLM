// Literal numerical contract for the HIP OGA sampling-logits processor.
#include "AutoModel/hip_oga_logits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using flm::hip_oga::SamplingLogitsOptions;
using flm::hip_oga::process_sampling_logits;

namespace {

struct Result {
    std::string name;
    bool passed;
    std::string error;
};

std::string escape_json(const std::string& value) {
    std::string escaped;
    for (char ch : value) {
        if (ch == '"' || ch == '\\') escaped.push_back('\\');
        if (ch == '\n') escaped += "\\n";
        else if (ch == '\r') escaped += "\\r";
        else if (ch == '\t') escaped += "\\t";
        else escaped.push_back(ch);
    }
    return escaped;
}

void record(std::vector<Result>& results, const std::string& name,
            const std::function<void()>& check) {
    try {
        check();
        results.push_back({name, true, {}});
    } catch (const std::exception& error) {
        results.push_back({name, false, error.what()});
    } catch (...) {
        results.push_back({name, false, "unknown exception"});
    }
}

void expect_logits(const std::vector<float>& actual,
                   const std::vector<float>& expected,
                   const char* context) {
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string(context) + ": output size changed");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::isinf(expected[i]) && expected[i] < 0) {
            if (!(std::isinf(actual[i]) && actual[i] < 0))
                throw std::runtime_error(std::string(context) + ": expected a banned -inf candidate");
        } else if (!std::isfinite(actual[i]) ||
                   std::abs(actual[i] - expected[i]) > 1e-6f) {
            throw std::runtime_error(std::string(context) + ": unexpected relative logit");
        }
    }
}

void transform(const char* context, std::vector<float> logits,
               const std::vector<int32_t>& history,
               const SamplingLogitsOptions& options,
               const std::vector<float>& expected) {
    process_sampling_logits(logits, history.data(), history.size(), options);
    expect_logits(logits, expected, context);
}

void rejects(const char* context, const std::vector<float>& logits,
             const int32_t* history, std::size_t history_count,
             const SamplingLogitsOptions& options) {
    auto row = logits;
    try {
        process_sampling_logits(row, history, history_count, options);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(context) + ": invalid input was accepted");
}

std::string report_json(const std::vector<Result>& results) {
    const auto failures = std::count_if(results.begin(), results.end(),
                                        [](const Result& result) { return !result.passed; });
    std::ostringstream out;
    out << "{\"status\":\"" << (failures == 0 ? "PASS" : "FAIL")
        << "\",\"case_count\":" << results.size() << ",\"cases\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) out << ',';
        out << "{\"name\":\"" << escape_json(results[i].name)
            << "\",\"status\":\"" << (results[i].passed ? "PASS" : "FAIL") << '\"';
        if (!results[i].passed)
            out << ",\"error\":\"" << escape_json(results[i].error) << '\"';
        out << '}';
    }
    out << "]}";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_sampling_logits [report.json]\n";
        return 2;
    }

    std::vector<Result> results;
    SamplingLogitsOptions options;

    record(results, "neutral_max_relative", [&] {
        transform("neutral_max_relative", {4.0f, 3.0f}, {}, options, {0.0f, -1.0f});
    });
    record(results, "presence_positive_once", [&] {
        options = SamplingLogitsOptions{};
        options.presence_penalty = 2.0;
        transform("presence_positive_once", {4.0f, 3.0f}, {0}, options, {-1.0f, 0.0f});
    });
    record(results, "presence_negative_once", [&] {
        options = SamplingLogitsOptions{};
        options.presence_penalty = -2.0;
        transform("presence_negative_once", {4.0f, 3.0f}, {1}, options, {-1.0f, 0.0f});
    });
    record(results, "frequency_positive_counts", [&] {
        options = SamplingLogitsOptions{};
        options.frequency_penalty = 0.75;
        transform("frequency_positive_counts", {4.0f, 3.0f}, {0, 0}, options, {-0.5f, 0.0f});
    });
    record(results, "frequency_negative_counts", [&] {
        options = SamplingLogitsOptions{};
        options.frequency_penalty = -0.75;
        transform("frequency_negative_counts", {4.0f, 3.0f}, {1, 1}, options, {-0.5f, 0.0f});
    });
    record(results, "large_finite_negative_frequency_penalty", [&] {
        options = SamplingLogitsOptions{};
        options.frequency_penalty = -static_cast<double>(std::numeric_limits<float>::max());
        transform("large_finite_negative_frequency_penalty", {4.0f, 3.0f}, {0, 0}, options,
                  {0.0f, -std::numeric_limits<float>::max()});
    });
    record(results, "repetition_sign_and_distinct_history", [&] {
        options = SamplingLogitsOptions{};
        options.repetition_penalty = 2.0;
        transform("repetition_sign_and_distinct_history", {4.0f, -2.0f, 3.0f, -3.0f},
                  {0, 0, 1, 1}, options, {-1.0f, -7.0f, 0.0f, -6.0f});
    });
    record(results, "repetition_before_presence", [&] {
        options = SamplingLogitsOptions{};
        options.repetition_penalty = 2.0;
        options.presence_penalty = 1.6;
        transform("repetition_before_presence", {4.0f, 3.0f}, {0}, options, {-2.6f, 0.0f});
    });
    record(results, "min_p_half_bans_below_relative_threshold", [&] {
        options = SamplingLogitsOptions{};
        options.min_p = 0.5;
        transform("min_p_half_bans_below_relative_threshold", {4.0f, 3.0f}, {}, options,
                  {0.0f, -std::numeric_limits<float>::infinity()});
    });
    record(results, "top_p_temperature_one", [&] {
        options = SamplingLogitsOptions{};
        options.top_p = 0.65;
        transform("top_p_temperature_one", {4.0f, 3.0f}, {}, options,
                  {0.0f, -std::numeric_limits<float>::infinity()});
    });
    record(results, "temperature_precedes_top_p", [&] {
        options = SamplingLogitsOptions{};
        options.temperature = 2.0;
        options.top_p = 0.65;
        transform("temperature_precedes_top_p", {4.0f, 3.0f}, {}, options, {0.0f, -0.5f});
    });
    record(results, "top_p_keeps_crossing_token", [&] {
        options = SamplingLogitsOptions{};
        options.top_p = 0.8;
        transform("top_p_keeps_crossing_token", {4.0f, 3.0f, 2.0f}, {}, options,
                  {0.0f, -1.0f, -std::numeric_limits<float>::infinity()});
    });
    record(results, "top_k_then_top_p_tie_order", [&] {
        options = SamplingLogitsOptions{};
        options.top_k = 2;
        options.top_p = 0.4;
        transform("top_k_then_top_p_tie_order", {4.0f, 4.0f, 4.0f, 4.0f}, {}, options,
                  {0.0f, -std::numeric_limits<float>::infinity(),
                   -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
    });
    record(results, "top_k_equal_scores_use_token_order", [&] {
        options = SamplingLogitsOptions{};
        options.top_k = 2;
        transform("top_k_equal_scores_use_token_order", {1.0f, 2.0f, 2.0f, 2.0f}, {}, options,
                  {-std::numeric_limits<float>::infinity(), 0.0f, 0.0f,
                   -std::numeric_limits<float>::infinity()});
    });
    record(results, "min_p_one_keeps_maximum_ties", [&] {
        options = SamplingLogitsOptions{};
        options.min_p = 1.0;
        transform("min_p_one_keeps_maximum_ties", {4.0f, 4.0f, 3.0f}, {}, options,
                  {0.0f, 0.0f, -std::numeric_limits<float>::infinity()});
    });
    record(results, "top_p_zero_disables_nucleus", [&] {
        options = SamplingLogitsOptions{};
        options.top_p = 0.0;
        transform("top_p_zero_disables_nucleus", {4.0f, 3.0f}, {}, options, {0.0f, -1.0f});
    });
    record(results, "top_p_one_disables_nucleus", [&] {
        options = SamplingLogitsOptions{};
        options.top_p = 1.0;
        transform("top_p_one_disables_nucleus", {4.0f, 3.0f}, {}, options, {0.0f, -1.0f});
    });
    record(results, "top_k_zero_disables_top_k", [&] {
        options = SamplingLogitsOptions{};
        options.top_k = 0;
        transform("top_k_zero_disables_top_k", {4.0f, 3.0f, 2.0f}, {}, options,
                  {0.0f, -1.0f, -2.0f});
    });
    record(results, "min_p_zero_disables_minimum_probability", [&] {
        options = SamplingLogitsOptions{};
        options.min_p = 0.0;
        transform("min_p_zero_disables_minimum_probability", {4.0f, 3.0f}, {}, options,
                  {0.0f, -1.0f});
    });
    record(results, "greedy_skips_sampling_filters_but_keeps_penalties", [&] {
        options = SamplingLogitsOptions{};
        options.do_sample = false;
        options.temperature = 2.0;
        options.top_k = 1;
        options.top_p = 0.01;
        options.min_p = 1.0;
        options.presence_penalty = 1.6;
        transform("greedy_skips_sampling_filters_but_keeps_penalties", {4.0f, 3.0f}, {0},
                  options, {-0.6f, 0.0f});
    });
    record(results, "negative_penalties_do_not_revive_mask", [&] {
        options = SamplingLogitsOptions{};
        options.do_sample = false;
        options.presence_penalty = -2.0;
        transform("negative_penalties_do_not_revive_mask",
                  {-std::numeric_limits<float>::infinity(), 3.0f, 2.0f}, {0}, options,
                  {-std::numeric_limits<float>::infinity(), 0.0f, -1.0f});
    });
    record(results, "mask_precedes_top_p_selection", [&] {
        options = SamplingLogitsOptions{};
        options.top_p = 0.65;
        transform("mask_precedes_top_p_selection",
                  {-std::numeric_limits<float>::infinity(), 4.0f, 3.0f}, {}, options,
                  {-std::numeric_limits<float>::infinity(), 0.0f,
                   -std::numeric_limits<float>::infinity()});
    });
    record(results, "extreme_finite_scores_and_small_temperature", [&] {
        options = SamplingLogitsOptions{};
        options.temperature = 1e-38;
        const float max = std::numeric_limits<float>::max();
        transform("equal extreme maxima remain tied", {max, max}, {}, options, {0.0f, 0.0f});
        transform("extreme difference clamps finite", {max, 0.0f}, {}, options,
                  {0.0f, -std::numeric_limits<float>::max()});
    });
    record(results, "sequential_calls_use_only_current_history", [&] {
        options = SamplingLogitsOptions{};
        options.presence_penalty = 1.0;
        transform("history token zero", {4.0f, 3.0f}, {0}, options, {0.0f, 0.0f});
        transform("history token one", {4.0f, 3.0f}, {1}, options, {0.0f, -2.0f});
        transform("empty history after penalty calls", {4.0f, 3.0f}, {}, options, {0.0f, -1.0f});
    });

    const std::vector<float> valid_row{2.0f, 1.0f};
    const int32_t valid_history[] = {0};
    record(results, "rejects_empty_row", [&] {
        rejects("empty_row", {}, nullptr, 0, SamplingLogitsOptions{});
    });
    record(results, "rejects_all_banned_row", [&] {
        rejects("all_banned_row", {-std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()},
                nullptr, 0, SamplingLogitsOptions{});
    });
    record(results, "rejects_nan_logit", [&] {
        rejects("nan_logit", {std::numeric_limits<float>::quiet_NaN(), 1.0f},
                nullptr, 0, SamplingLogitsOptions{});
    });
    record(results, "rejects_positive_infinite_logit", [&] {
        rejects("positive_infinite_logit", {std::numeric_limits<float>::infinity(), 1.0f},
                nullptr, 0, SamplingLogitsOptions{});
    });
    record(results, "rejects_negative_history_id", [&] {
        const int32_t history[] = {-1};
        rejects("negative_history_id", valid_row, history, 1, SamplingLogitsOptions{});
    });
    record(results, "rejects_out_of_vocabulary_history_id", [&] {
        const int32_t history[] = {2};
        rejects("out_of_vocabulary_history_id", valid_row, history, 1, SamplingLogitsOptions{});
    });
    record(results, "rejects_null_nonempty_history", [&] {
        rejects("null_nonempty_history", valid_row, nullptr, 1, SamplingLogitsOptions{});
    });

    auto invalid = [&](const char* name, const SamplingLogitsOptions& bad) {
        record(results, name, [&] { rejects(name, valid_row, valid_history, 1, bad); });
    };
    options = SamplingLogitsOptions{}; options.temperature = 0.0; invalid("rejects_zero_temperature", options);
    options = SamplingLogitsOptions{}; options.temperature = -1.0; invalid("rejects_negative_temperature", options);
    options = SamplingLogitsOptions{}; options.temperature = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_temperature", options);
    options = SamplingLogitsOptions{}; options.temperature = std::numeric_limits<double>::quiet_NaN(); invalid("rejects_nan_temperature", options);
    options = SamplingLogitsOptions{}; options.top_p = -0.1; invalid("rejects_negative_top_p", options);
    options = SamplingLogitsOptions{}; options.top_p = 1.1; invalid("rejects_top_p_above_one", options);
    options = SamplingLogitsOptions{}; options.top_p = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_top_p", options);
    options = SamplingLogitsOptions{}; options.top_p = std::numeric_limits<double>::quiet_NaN(); invalid("rejects_nan_top_p", options);
    options = SamplingLogitsOptions{}; options.min_p = -0.1; invalid("rejects_negative_min_p", options);
    options = SamplingLogitsOptions{}; options.min_p = 1.1; invalid("rejects_min_p_above_one", options);
    options = SamplingLogitsOptions{}; options.min_p = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_min_p", options);
    options = SamplingLogitsOptions{}; options.min_p = std::numeric_limits<double>::quiet_NaN(); invalid("rejects_nan_min_p", options);
    options = SamplingLogitsOptions{}; options.top_k = -1; invalid("rejects_negative_top_k", options);
    options = SamplingLogitsOptions{}; options.top_k = 3; invalid("rejects_top_k_above_vocabulary", options);
    options = SamplingLogitsOptions{}; options.repetition_penalty = 0.0; invalid("rejects_zero_repetition_penalty", options);
    options = SamplingLogitsOptions{}; options.repetition_penalty = -1.0; invalid("rejects_negative_repetition_penalty", options);
    options = SamplingLogitsOptions{}; options.repetition_penalty = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_repetition_penalty", options);
    options = SamplingLogitsOptions{}; options.presence_penalty = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_presence_penalty", options);
    options = SamplingLogitsOptions{}; options.presence_penalty = std::numeric_limits<double>::quiet_NaN(); invalid("rejects_nan_presence_penalty", options);
    options = SamplingLogitsOptions{}; options.frequency_penalty = std::numeric_limits<double>::infinity(); invalid("rejects_infinite_frequency_penalty", options);
    options = SamplingLogitsOptions{}; options.frequency_penalty = std::numeric_limits<double>::quiet_NaN(); invalid("rejects_nan_frequency_penalty", options);

    const auto output = report_json(results);
    std::cout << output << '\n';
    if (argc == 2) {
        std::ofstream report(argv[1]);
        if (!report) {
            std::cerr << "could not open report path\n";
            return 2;
        }
        report << output << '\n';
    }
    return std::any_of(results.begin(), results.end(),
                       [](const Result& result) { return !result.passed; }) ? 1 : 0;
}
