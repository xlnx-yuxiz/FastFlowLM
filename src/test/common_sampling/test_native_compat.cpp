#include "modules/sampler.hpp"

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

namespace {

struct TestCase {
    std::string name;
    bool passed;
    std::string error;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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

buffer<bf16> bf16_row(std::vector<bf16>& values) {
    return buffer<bf16>(values);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_native_compat [receipt.json]\n";
        return 2;
    }

    std::vector<TestCase> results;

    record(results, "default_sampler_config_keeps_existing_policy_and_api", [] {
        sampler_config config;
        Sampler sampler(64, config);
        require(sampler.top_k == 40, "default top-k changed");
        require(std::abs(sampler.top_p - 0.9f) < 1e-6f, "default top-p changed");
        require(std::abs(sampler.min_p - 0.1f) < 1e-6f, "default min-p changed");
        require(std::abs(sampler.temperature - 0.8f) < 1e-6f, "default temperature changed");
        require(std::abs(sampler.rep_penalty - 1.0f) < 1e-6f, "default repetition penalty changed");
        require(std::abs(sampler.freq_penalty) < 1e-6f, "default frequency penalty changed");
        require(sampler.use_optimized_sampling, "default optimized sampler path changed");

        std::vector<bf16> scores;
        scores.reserve(64);
        for (int id = 0; id < 64; ++id) scores.emplace_back(static_cast<float>(64 - id));
        auto row = bf16_row(scores);
        const int32_t selected = sampler.sample(row);
        require(selected >= 0 && selected < 40,
                "existing buffer-based sample API escaped the configured top-k");
        require(sampler.total_tokens == 1 && sampler.token_history.size() == 1 &&
                    sampler.token_history.front() == selected,
                "native sample did not record its selected continuation token");
    });

    record(results, "native_sampler_applies_top_p_before_temperature", [] {
        sampler_config config;
        config.top_k = 3;
        config.top_p = 0.8f;
        config.min_p = 0;
        config.temperature = 0.5f;
        config.rep_penalty = 1;
        config.freq_penalty = 0;
        config.pre_penalty = 0;
        config.repeat_last_n = 0;
        config.use_optimized_sampling = true;
        config.has_rng_seed = true;
        config.rng_seed = 11;
        Sampler sampler(3, config);

        // Before temperature, p(token 0) for [4,3,-20] is about .731,
        // below .8, so top-p retains token 1. Applying T=.5 first would make
        // p(token 0) about .881 and incorrectly remove token 1.
        std::vector<bf16> scores{bf16(4.0f), bf16(3.0f), bf16(-20.0f)};
        auto row = bf16_row(scores);
        (void)sampler.sample(row);
        require(sampler.top_k_logits.size() == 2,
                "top-p support changed when temperature was applied");
        require(sampler.top_k_logits[0].token_id == 0 && sampler.top_k_logits[1].token_id == 1,
                "top-p did not retain the expected ordered support");
    });

    record(results, "native_sampler_explicit_seed_repeats_the_same_sequence", [] {
        sampler_config config;
        config.top_k = 4;
        config.top_p = 1;
        config.min_p = 0;
        config.temperature = 1;
        config.rep_penalty = 1;
        config.freq_penalty = 0;
        config.pre_penalty = 0;
        config.repeat_last_n = 0;
        config.has_rng_seed = true;
        config.rng_seed = 7;

        Sampler first(4, config);
        Sampler second(4, config);
        first.set_seed(0x5eed);
        second.set_seed(0x5eed);
        std::vector<bf16> scores{bf16(1.5f), bf16(1.0f), bf16(0.5f), bf16(0.0f)};
        auto row = bf16_row(scores);
        std::vector<int> first_sequence;
        std::vector<int> second_sequence;
        for (int i = 0; i < 16; ++i) {
            first_sequence.push_back(first.sample(row));
            second_sequence.push_back(second.sample(row));
        }
        require(first_sequence == second_sequence,
                "native sampler seed did not reproduce a sequence");
    });

    record(results, "native_penalty_ring_tracks_generated_tokens_and_window", [] {
        sampler_config config;
        config.top_k = 1;
        config.top_p = 1;
        config.min_p = 0;
        config.temperature = 1;
        config.rep_penalty = 2;
        config.freq_penalty = 0;
        config.pre_penalty = 0;
        config.repeat_last_n = 2;
        config.rep_penalty_window = 2;
        config.freq_penalty_window = 2;
        config.has_rng_seed = true;
        config.rng_seed = 1;
        Sampler sampler(2, config);
        std::vector<bf16> scores{bf16(10.0f), bf16(9.0f)};
        auto row = bf16_row(scores);

        require(sampler.sample(row) == 0, "first generated token changed");
        require(sampler.token_history.size() == 1 && sampler.token_history.front() == 0,
                "first selected token was not recorded in native penalty history");
        require(sampler.sample(row) == 1,
                "the repeated generated token did not change the next greedy choice");
        require(sampler.token_history.size() == 2 && sampler.token_history[0] == 0 &&
                    sampler.token_history[1] == 1,
                "native penalty ring did not retain the last two generated tokens");
        require(sampler.sample(row) == 0, "expected choice after both tokens were penalized changed");
        require(sampler.token_history.size() == 2 && sampler.token_history[0] == 1 &&
                    sampler.token_history[1] == 0,
                "native penalty ring did not evict the oldest generated token");
        require(sampler.total_tokens == 3, "native generated-token counter changed");

        sampler.reset_penalties();
        require(sampler.token_history.empty() && sampler.total_tokens == 0,
                "reset_penalties did not clear the native generated-token ring");
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
              << " native sampler compatibility: " << results.size() << " cases, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
