#include "sampling/cpu_sampler.hpp"
#include "sampling/logits.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using flm::sampling::CpuSampler;
using flm::sampling::LogitsOptions;
using flm::sampling::choose_token;
using flm::sampling::process_logits;

struct TestCase {
    std::string name;
    bool passed;
    std::string error;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(float actual, float expected, float tolerance, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + " did not match its hand-derived value");
    }
}

void require_negative_infinity(float value, const char* label) {
    if (!(std::isinf(value) && value < 0)) {
        throw std::runtime_error(std::string(label) + " should remain an explicit ban");
    }
}

template <typename Callable>
void require_throws(Callable&& action, const char* label) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void record(std::vector<TestCase>& results, const char* name,
            const std::function<void()>& test) {
    try {
        test();
        results.push_back({name, true, {}});
    } catch (const std::exception& error) {
        results.push_back({name, false, error.what()});
    } catch (...) {
        results.push_back({name, false, "unknown exception"});
    }
}

std::string json_escape(const std::string& input) {
    std::ostringstream escaped;
    for (unsigned char ch : input) {
        switch (ch) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (ch < 0x20) escaped << "?";
                else escaped << static_cast<char>(ch);
        }
    }
    return escaped.str();
}

bool write_receipt(const std::string& path, const std::vector<TestCase>& results) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    std::size_t failures = 0;
    for (const auto& result : results) failures += result.passed ? 0 : 1;
    out << "{\"status\":\"" << (failures == 0 ? "PASS" : "FAIL")
        << "\",\"case_count\":" << results.size() << ",\"cases\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) out << ',';
        out << "{\"name\":\"" << json_escape(results[i].name)
            << "\",\"status\":\"" << (results[i].passed ? "PASS" : "FAIL") << '"';
        if (!results[i].passed)
            out << ",\"error\":\"" << json_escape(results[i].error) << '"';
        out << '}';
    }
    out << "]}\n";
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_common_sampling [receipt.json]\n";
        return 2;
    }

    std::vector<TestCase> results;
    record(results, "temperature_scaled_nucleus_keeps_mass_crossing_token", [] {
        LogitsOptions options;
        options.temperature = 2;
        options.top_p = 0.65;
        std::vector<float> logits{4, 3};
        process_logits(logits, nullptr, 0, options);
        // At T=2 the two probabilities are about .622 and .378, so .65
        // crosses on token 1; relative logits are 0 and -0.5.
        require(logits.size() == 2, "vocabulary width changed");
        require_near(logits[0], 0, 1e-6f, "maximum logit");
        require_near(logits[1], -0.5f, 1e-6f, "temperature-scaled logit");
    });

    record(results, "top_p_zero_and_one_leave_the_full_support", [] {
        for (double top_p : {0.0, 1.0}) {
            LogitsOptions options;
            options.top_p = top_p;
            std::vector<float> logits{3, 2, 1};
            process_logits(logits, nullptr, 0, options);
            require_near(logits[0], 0, 1e-6f, "first relative logit");
            require_near(logits[1], -1, 1e-6f, "second relative logit");
            require_near(logits[2], -2, 1e-6f, "third relative logit");
        }
    });

    record(results, "top_k_zero_disables_top_k_filtering", [] {
        LogitsOptions options;
        options.top_k = 0;
        std::vector<float> logits{3, 2, 1};
        process_logits(logits, nullptr, 0, options);
        require_near(logits[0], 0, 1e-6f, "first relative logit");
        require_near(logits[1], -1, 1e-6f, "second relative logit");
        require_near(logits[2], -2, 1e-6f, "third relative logit");
    });

    record(results, "nucleus_filter_retains_only_the_crossing_prefix", [] {
        LogitsOptions options;
        options.top_p = 0.6;
        std::vector<float> logits{4, 3};
        process_logits(logits, nullptr, 0, options);
        require_near(logits[0], 0, 1e-6f, "maximum logit");
        require_negative_infinity(logits[1], "below-threshold token");
    });

    record(results, "top_k_keeps_highest_scores_and_bans_the_rest", [] {
        LogitsOptions options;
        options.top_k = 2;
        std::vector<float> logits{3, 2, 1};
        process_logits(logits, nullptr, 0, options);
        require_near(logits[0], 0, 1e-6f, "maximum logit");
        require_near(logits[1], -1, 1e-6f, "second logit");
        require_negative_infinity(logits[2], "outside top-k token");
    });

    record(results, "min_p_uses_the_relative_logit_threshold", [] {
        LogitsOptions options;
        options.min_p = 0.2;
        std::vector<float> logits{0, -1, -3};
        process_logits(logits, nullptr, 0, options);
        require_near(logits[0], 0, 1e-6f, "maximum logit");
        require_near(logits[1], -1, 1e-6f, "above-threshold token");
        require_negative_infinity(logits[2], "below min-p threshold token");
    });

    record(results, "prompt_repetition_presence_and_frequency_change_the_winner", [] {
        const int32_t prompt_history[] = {0, 0};
        LogitsOptions options;
        options.do_sample = false;
        options.repetition_penalty = 2;
        options.presence_penalty = 0.5;
        options.frequency_penalty = 0.25;
        std::vector<float> logits{4, 3};
        process_logits(logits, prompt_history, 2, options);
        // Token 0 becomes 4/2 - .5 - 2*.25 = 1; token 1 stays 3.
        require_near(logits[0], -2, 1e-6f, "penalized prompt token");
        require_near(logits[1], 0, 1e-6f, "winning unobserved token");
    });

    record(results, "negative_penalties_reward_repeated_tokens", [] {
        const int32_t history[] = {0, 0};
        LogitsOptions options;
        options.repetition_penalty = 1;
        options.presence_penalty = -1;
        options.frequency_penalty = -0.5;
        std::vector<float> logits{0, 0};
        process_logits(logits, history, 2, options);
        require_near(logits[0], 0, 1e-6f, "rewarded repeated token");
        require_near(logits[1], -2, 1e-6f, "unobserved token");
    });

    record(results, "negative_penalties_do_not_revive_a_model_ban", [] {
        const int32_t history[] = {0, 0};
        LogitsOptions options;
        options.repetition_penalty = 0.5;
        options.presence_penalty = -100;
        options.frequency_penalty = -100;
        std::vector<float> logits{-std::numeric_limits<float>::infinity(), 0};
        process_logits(logits, history, 2, options);
        require_negative_infinity(logits[0], "explicitly banned token");
        require_near(logits[1], 0, 1e-6f, "allowed token");
    });

    record(results, "invalid_logits_options_and_history_are_rejected", [] {
        require_throws([] {
            std::vector<float> logits;
            process_logits(logits, nullptr, 0, LogitsOptions{});
        }, "empty logits");
        require_throws([] {
            std::vector<float> logits{std::numeric_limits<float>::quiet_NaN()};
            process_logits(logits, nullptr, 0, LogitsOptions{});
        }, "NaN logit");
        require_throws([] {
            std::vector<float> logits{std::numeric_limits<float>::infinity()};
            process_logits(logits, nullptr, 0, LogitsOptions{});
        }, "positive-infinity logit");
        require_throws([] {
            std::vector<float> logits{-std::numeric_limits<float>::infinity()};
            process_logits(logits, nullptr, 0, LogitsOptions{});
        }, "all-banned row");
        require_throws([] {
            std::vector<float> logits{1, 0};
            const int32_t bad[] = {-1};
            process_logits(logits, bad, 1, LogitsOptions{});
        }, "negative history token");
        require_throws([] {
            std::vector<float> logits{1, 0};
            const int32_t bad[] = {2};
            process_logits(logits, bad, 1, LogitsOptions{});
        }, "out-of-vocabulary history token");
        require_throws([] {
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 1, LogitsOptions{});
        }, "missing nonempty history");
        for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
            require_throws([value] {
                LogitsOptions options;
                options.temperature = value;
                std::vector<float> logits{1, 0};
                process_logits(logits, nullptr, 0, options);
            }, "invalid temperature");
        }
        require_throws([] {
            LogitsOptions options;
            options.top_p = 1.01;
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "top-p above one");
        require_throws([] {
            LogitsOptions options;
            options.min_p = -0.1;
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "negative min-p");
        require_throws([] {
            LogitsOptions options;
            options.repetition_penalty = 0;
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "zero repetition penalty");
        require_throws([] {
            LogitsOptions options;
            options.top_k = -1;
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "negative top-k");
        require_throws([] {
            LogitsOptions options;
            options.top_k = 3;
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "top-k larger than vocabulary");
        require_throws([] {
            LogitsOptions options;
            options.frequency_penalty = std::numeric_limits<double>::max();
            std::vector<float> logits{1, 0};
            process_logits(logits, nullptr, 0, options);
        }, "non-float32 frequency penalty");
    });

    record(results, "categorical_selection_skips_banned_tokens_at_uniform_boundaries", [] {
        const std::vector<float> processed{
            -std::numeric_limits<float>::infinity(), 0,
            -std::numeric_limits<float>::infinity(), -std::log(3.0f)};
        require(choose_token(processed, 0.0) == 1,
                "uniform zero selected a zero-mass banned token");
        require(choose_token(processed, std::nextafter(1.0, 0.0)) == 3,
                "uniform near one missed the final supported token");
        require(choose_token(processed, 0.70) == 1 && choose_token(processed, 0.80) == 3,
                "categorical selection did not respect the 3:1 probability split");
    });

    record(results, "categorical_selection_rejects_invalid_inputs", [] {
        const std::vector<float> logits{0, -1};
        for (double uniform : {-0.01, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
            require_throws([&] { (void)choose_token(logits, uniform); },
                           "out-of-range categorical uniform");
        }
        require_throws([] {
            (void)choose_token({}, 0.5);
        }, "empty categorical row");
        require_throws([] {
            const std::vector<float> banned(2, -std::numeric_limits<float>::infinity());
            (void)choose_token(banned, 0.5);
        }, "all-banned categorical row");
        require_throws([] {
            const std::vector<float> invalid{0, std::numeric_limits<float>::quiet_NaN()};
            (void)choose_token(invalid, 0.5);
        }, "NaN categorical score");
    });

    record(results, "greedy_ties_choose_lowest_id_and_do_not_advance_rng", [] {
        constexpr uint64_t seed = 42;
        CpuSampler sampler;
        sampler.reset(seed);

        LogitsOptions greedy;
        greedy.do_sample = false;
        const std::vector<float> tied_logits{2, 2, 1};
        require(sampler.sample(tied_logits, nullptr, 0, greedy) == 0,
                "greedy tied maximum did not choose the lowest ID");

        const std::vector<float> random_logits{0, 0, 0, 0};
        LogitsOptions stochastic;
        // mt19937_64(42)'s first value is 13930160852258120406:
        // the specified 53-bit mapping is .7551555..., in the fourth equal bin.
        require(sampler.sample(random_logits, nullptr, 0, stochastic) == 3,
                "greedy selection consumed the request RNG");
    });

    record(results, "seeded_reset_replays_choices_across_interleaved_requests", [] {
        constexpr uint64_t seed = 20261028;
        const std::vector<float> logits{1.5f, 0.9f, 0.2f};
        const int32_t request_history[] = {2, 1, 2};
        LogitsOptions options;
        options.repetition_penalty = 1.1;

        CpuSampler sampler;
        sampler.reset(seed);
        std::vector<int32_t> first_run;
        for (int i = 0; i < 12; ++i)
            first_run.push_back(sampler.sample(logits, request_history, 3, options));

        sampler.reset(7);
        (void)sampler.sample(std::vector<float>{0, 1, 2}, nullptr, 0, LogitsOptions{});
        sampler.reset(seed);
        std::vector<int32_t> replay;
        for (int i = 0; i < 12; ++i)
            replay.push_back(sampler.sample(logits, request_history, 3, options));
        require(first_run == replay,
                "request reset did not reproduce the same choices after another request");
    });

    record(results, "caller_history_rewind_discards_tokens_absent_from_new_history", [] {
        CpuSampler sampler;
        sampler.reset(1);
        LogitsOptions greedy;
        greedy.do_sample = false;
        greedy.repetition_penalty = 2;
        const std::vector<float> logits{4, 3};
        const int32_t old_history[] = {0};
        require(sampler.sample(logits, old_history, 1, greedy) == 1,
                "repetition in supplied history did not affect the choice");
        require(sampler.sample(logits, nullptr, 0, greedy) == 0,
                "a token omitted from rewound history remained penalized");
    });

    std::size_t failures = 0;
    for (const auto& result : results) {
        if (!result.passed) {
            ++failures;
            std::cerr << "FAIL " << result.name << ": " << result.error << '\n';
        }
    }
    if (argc == 2 && !write_receipt(argv[1], results)) {
        std::cerr << "could not write receipt\n";
        return 2;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " common sampling: " << results.size() << " cases, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
