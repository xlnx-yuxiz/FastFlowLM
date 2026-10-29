#include "modules/sampler.hpp"
#include "sampling/logits.hpp"

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
                if (ch < 0x20) escaped << '?';
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

sampler_config legacy_config(int top_k = 1) {
    sampler_config config;
    config.top_k = top_k;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    config.freq_penalty = 0.0f;
    config.pre_penalty = 0.0f;
    config.rep_penalty_window = 16;
    config.freq_penalty_window = 16;
    config.repeat_last_n = 16;
    config.use_optimized_sampling = true;
    config.has_rng_seed = true;
    config.rng_seed = 17;
    return config;
}

flm::sampling::LogitsOptions greedy_options() {
    flm::sampling::LogitsOptions options;
    options.do_sample = false;
    options.temperature = 1.0;
    options.top_p = 1.0;
    options.min_p = 0.0;
    options.top_k = 0;
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_native_common [receipt.json]\n";
        return 2;
    }

    std::vector<TestCase> results;

    record(results, "caller_prompt_and_continuation_history_changes_greedy_winner", [] {
        auto config = legacy_config();
        Sampler sampler(2, config);
        auto options = greedy_options();
        options.repetition_penalty = 2.0;
        options.presence_penalty = 0.5;
        options.frequency_penalty = 0.25;
        std::vector<int> logical_history;
        std::vector<bf16> scores{bf16(4.0f), bf16(3.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 5);
        require(sampler.sample(row) == 0,
                "empty caller history changed the unpenalized greedy winner");

        logical_history = {0, 0};
        require(sampler.sample(row) == 1,
                "prompt and continuation tokens did not penalize token 0 enough to change the winner");
    });

    record(results, "replacing_caller_history_discards_old_penalties", [] {
        auto config = legacy_config();
        Sampler sampler(2, config);
        auto options = greedy_options();
        options.repetition_penalty = 2.0;
        std::vector<int> logical_history{0};
        std::vector<bf16> scores{bf16(6.0f), bf16(7.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 5);
        require(sampler.sample(row) == 1,
                "first supplied history did not penalize token 0");

        logical_history = {1};
        require(sampler.sample(row) == 0,
                "replacing the logical history retained a discarded token penalty");
    });

    record(results, "common_greedy_ties_choose_lowest_token_id", [] {
        auto config = legacy_config();
        Sampler sampler(3, config);
        const auto options = greedy_options();
        const std::vector<int> logical_history;
        std::vector<bf16> scores{bf16(4.0f), bf16(4.0f), bf16(1.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 0);
        require(sampler.sample(row) == 0,
                "a tied greedy maximum did not select the lowest token ID");
    });

    record(results, "negative_penalties_do_not_revive_explicitly_banned_token", [] {
        auto config = legacy_config();
        Sampler sampler(2, config);
        auto options = greedy_options();
        options.presence_penalty = -100.0;
        options.frequency_penalty = -100.0;
        const std::vector<int> logical_history{0, 0};
        std::vector<bf16> scores{
            bf16(-std::numeric_limits<float>::infinity()), bf16(0.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 0);
        require(sampler.sample(row) == 1,
                "negative penalties revived a token banned with negative infinity");
    });

    record(results, "seed_zero_replays_after_an_intervening_common_request", [] {
        auto config = legacy_config(3);
        Sampler sampler(3, config);
        flm::sampling::LogitsOptions options;
        options.do_sample = true;
        options.temperature = 1.0;
        options.top_p = 1.0;
        options.min_p = 0.0;
        options.top_k = 0;
        const std::vector<int> logical_history;
        std::vector<bf16> scores{bf16(2.0f), bf16(1.0f), bf16(0.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 0);
        std::vector<int> first_sequence;
        for (int i = 0; i < 32; ++i) first_sequence.push_back(sampler.sample(row));

        sampler.configure_common(options, &logical_history, 1234);
        for (int i = 0; i < 9; ++i) (void)sampler.sample(row);
        sampler.reset_common_seed(0);

        std::vector<int> replay;
        for (int i = 0; i < 32; ++i) replay.push_back(sampler.sample(row));
        require(first_sequence == replay,
                "seed zero did not reproduce the same sequence after another request and reseed");
    });

    record(results, "common_sampling_leaves_legacy_configuration_and_history_untouched", [] {
        auto config = legacy_config();
        config.rep_penalty = 1.75f;
        config.freq_penalty = 0.125f;
        config.pre_penalty = 0.25f;
        Sampler sampler(2, config);
        std::vector<bf16> scores{bf16(10.0f), bf16(9.0f)};
        auto row = bf16_row(scores);

        require(sampler.sample(row) == 0,
                "legacy setup sample did not establish token 0 as the winner");
        const auto legacy_history = sampler.token_history;
        const auto legacy_counters = sampler.counters;
        const auto legacy_positions = sampler.token_positions;
        const auto legacy_sparse_counts = sampler.token_counts_sparse;
        const int legacy_total_tokens = sampler.total_tokens;
        const int legacy_top_k = sampler.top_k;
        const float legacy_top_p = sampler.top_p;
        const float legacy_min_p = sampler.min_p;
        const float legacy_temperature = sampler.temperature;
        const float legacy_rep_penalty = sampler.rep_penalty;
        const float legacy_freq_penalty = sampler.freq_penalty;
        const float legacy_pre_penalty = sampler.pre_penalty;
        const auto legacy_rep_window = sampler.rep_penalty_window;
        const auto legacy_freq_window = sampler.freq_penalty_window;
        const auto legacy_repeat_window = sampler.repeat_last_n;
        const bool legacy_optimized = sampler.use_optimized_sampling;

        auto options = greedy_options();
        options.repetition_penalty = 2.0;
        const std::vector<int> logical_history{0};
        sampler.configure_common(options, &logical_history, 7);
        require(sampler.sample(row) == 1,
                "common options were not used independently of the legacy greedy configuration");

        require(sampler.token_history == legacy_history &&
                    sampler.counters == legacy_counters &&
                    sampler.token_positions == legacy_positions &&
                    sampler.token_counts_sparse == legacy_sparse_counts &&
                    sampler.total_tokens == legacy_total_tokens,
                "common sampling changed the legacy continuation history");
        require(sampler.top_k == legacy_top_k && sampler.top_p == legacy_top_p &&
                    sampler.min_p == legacy_min_p && sampler.temperature == legacy_temperature &&
                    sampler.rep_penalty == legacy_rep_penalty &&
                    sampler.freq_penalty == legacy_freq_penalty &&
                    sampler.pre_penalty == legacy_pre_penalty &&
                    sampler.rep_penalty_window == legacy_rep_window &&
                    sampler.freq_penalty_window == legacy_freq_window &&
                    sampler.repeat_last_n == legacy_repeat_window &&
                    sampler.use_optimized_sampling == legacy_optimized,
                "common sampling changed the legacy sampler configuration");
    });

    record(results, "disabling_common_restores_legacy_sampling", [] {
        auto config = legacy_config();
        Sampler sampler(2, config);
        auto options = greedy_options();
        options.repetition_penalty = 2.0;
        const std::vector<int> logical_history{0};
        std::vector<bf16> scores{bf16(10.0f), bf16(9.0f)};
        auto row = bf16_row(scores);

        sampler.configure_common(options, &logical_history, 0);
        require(sampler.sample(row) == 1,
                "common sampling did not apply the supplied history penalty");
        sampler.disable_common();
        require(sampler.sample(row) == 0,
                "legacy sampling was not restored after disabling common mode");
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
              << " native common sampler: " << results.size() << " cases, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
