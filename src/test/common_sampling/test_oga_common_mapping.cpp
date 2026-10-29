// Pure adapter tests for common-vs-backend OGA sampling configuration.
#include "AutoModel/hip_oga_sampling.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using flm::sampling::Provider;
using flm::hip_oga::SamplingRequestError;
using json = nlohmann::json;

struct CapturedParams {
    bool do_sample = false;
    double min_length = 0;
    double no_repeat_ngram_size = 0;
    std::map<std::string, double> numbers;

    void SetSearchOptionBool(const char*, bool value) { do_sample = value; }
    void SetSearchOption(const char* name, double value) { numbers[name] = value; }
    bool GetSearchBool(const char*) const { return do_sample; }
    double GetSearchNumber(const char* name) const {
        const std::string key(name);
        if (key == "min_length") return min_length;
        if (key == "no_repeat_ngram_size") return no_repeat_ngram_size;
        return numbers.at(key);
    }
};

struct TestCase {
    std::string name;
    bool passed;
    std::string error;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_sampling_error(Callable&& action, const char* label) {
    try {
        action();
    } catch (const SamplingRequestError&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void require_negative_infinity(float value, const char* label) {
    if (!(std::isinf(value) && value < 0)) {
        throw std::runtime_error(std::string(label) + " should be masked to negative infinity");
    }
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

bool write_receipt(const std::string& path, const std::vector<TestCase>& results) {
    std::size_t failures = 0;
    json cases = json::array();
    for (const auto& result : results) {
        failures += result.passed ? 0 : 1;
        json item = {{"name", result.name},
                     {"status", result.passed ? "PASS" : "FAIL"}};
        if (!result.passed) item["error"] = result.error;
        cases.push_back(std::move(item));
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << json({{"status", failures == 0 ? "PASS" : "FAIL"},
                 {"case_count", results.size()}, {"cases", std::move(cases)}}).dump(2)
        << '\n';
    return static_cast<bool>(out);
}

json ordinary_effective_options() {
    return {{"do_sample", true}, {"temperature", 1.25}, {"top_k", 4},
            {"top_p", 0.75}, {"repetition_penalty", 1.25}, {"random_seed", 23}};
}

json extension_effective_options() {
    return {{"do_sample", true}, {"temperature", 1.25}, {"top_k", 3},
            {"top_p", 0.5}, {"repetition_penalty", 1.25}, {"random_seed", 0},
            {"min_p", 0.2}, {"presence_penalty", 0.5}};
}

void test_backend_default_and_explicit_mapping_preserve_native_options() {
    const auto effective = ordinary_effective_options();
    CapturedParams default_provider;
    CapturedParams explicit_backend;

    flm::hip_oga::apply_oga_sampling_options(default_provider, effective);
    flm::hip_oga::apply_oga_sampling_options(explicit_backend, effective, Provider::Backend);

    const std::map<std::string, double> expected{
        {"temperature", 1.25}, {"top_k", 4}, {"top_p", 0.75},
        {"repetition_penalty", 1.25}, {"random_seed", 23}};
    require(default_provider.do_sample && explicit_backend.do_sample,
            "backend provider changed stochastic mode");
    require(default_provider.numbers == expected && explicit_backend.numbers == expected,
            "default and explicit backend providers must preserve ordinary OGA options");
}

void test_backend_extension_mapping_keeps_neutral_engine_filters_and_seed() {
    const auto effective = extension_effective_options();
    CapturedParams default_provider;
    CapturedParams explicit_backend;

    flm::hip_oga::apply_oga_sampling_options(default_provider, effective);
    flm::hip_oga::apply_oga_sampling_options(explicit_backend, effective, Provider::Backend);

    const std::map<std::string, double> expected{
        {"temperature", 1.0}, {"top_k", 0}, {"top_p", 1.0},
        {"repetition_penalty", 1.0}, {"random_seed", 0}};
    require(default_provider.do_sample && explicit_backend.do_sample,
            "backend extension mapping changed stochastic mode");
    require(default_provider.numbers == expected && explicit_backend.numbers == expected,
            "default and explicit backend extension paths must neutralize OGA filters and retain the seed");
}

void test_common_mapping_forces_greedy_and_retains_configured_seed() {
    for (const auto& effective : {ordinary_effective_options(), extension_effective_options()}) {
    CapturedParams params;
    flm::hip_oga::apply_oga_sampling_options(params, effective, Provider::Common);

    require(!params.do_sample, "common OGA mapping must force greedy sampling");
    require(params.numbers == std::map<std::string, double>{
                {"temperature", 1.0}, {"top_k", 0}, {"top_p", 1.0},
                {"repetition_penalty", 1.0}, {"random_seed", effective.at("random_seed").get<double>()}},
            "common OGA mapping must neutralize OGA filters while retaining the configured seed");
    }
}

void test_common_mapping_rejects_nonzero_model_ngram_filter_atomically() {
    for (const auto& effective : {ordinary_effective_options(), extension_effective_options()}) {
    auto params = CapturedParams{};
    params.no_repeat_ngram_size = 2;
    params.do_sample = true;
    params.numbers = {{"temperature", 0.8}, {"top_k", 5}};
    const auto before_numbers = params.numbers;
    const bool before_do_sample = params.do_sample;

    require_sampling_error([&] {
        flm::hip_oga::apply_oga_sampling_options(
            params, effective, Provider::Common);
    }, "nonzero no_repeat_ngram_size under common sampling");
    require(params.numbers == before_numbers && params.do_sample == before_do_sample,
            "rejected common mapping changed the captured model options");
    }
}

void test_eos_mask_obeys_min_length_and_preserves_eos_afterward() {
    CapturedParams params;
    params.min_length = 2;
    std::vector<float> logits{5.0f, 4.0f, 3.0f};
    const std::vector<int> eos_ids{0, 2};

    flm::hip_oga::mask_oga_sampling_logits(logits, 1, params, eos_ids);
    require_negative_infinity(logits[0], "first EOS token below min_length");
    require(logits[1] == 4.0f, "non-EOS logit changed below min_length");
    require_negative_infinity(logits[2], "second EOS token below min_length");

    std::vector<float> at_minimum{5.0f, 4.0f, 3.0f};
    flm::hip_oga::mask_oga_sampling_logits(at_minimum, 2, params, eos_ids);
    require(at_minimum == std::vector<float>({5.0f, 4.0f, 3.0f}),
            "EOS tokens were masked after the minimum length had been reached");
}

void test_eos_mask_rejects_out_of_range_and_negative_ids() {
    CapturedParams params;
    params.min_length = 1;
    for (const std::vector<int> eos_ids : {std::vector<int>{-1}, std::vector<int>{3}}) {
        require_sampling_error([&] {
            std::vector<float> logits{2.0f, 1.0f, 0.0f};
            flm::hip_oga::mask_oga_sampling_logits(logits, 0, params, eos_ids);
        }, "invalid EOS token ID");
    }
}

void test_oga_logits_processing_applies_model_eos_mask_before_common_processing() {
    CapturedParams params;
    params.min_length = 1;
    const json effective{{"do_sample", false}, {"temperature", 1.0}, {"top_k", 0},
                         {"top_p", 1.0}, {"repetition_penalty", 1.0}};
    std::vector<float> logits{10.0f, 9.0f};
    flm::hip_oga::process_oga_sampling_logits(logits, nullptr, 0, effective, params, {0});

    require_negative_infinity(logits[0], "model EOS after OGA sampling processing");
    require(logits[1] == 0.0f,
            "allowed common-sampler logit should be max-relative after processing");
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_oga_common_mapping [receipt.json]\n";
        return 2;
    }

    std::vector<TestCase> results;
    record(results, "backend_default_and_explicit_options", test_backend_default_and_explicit_mapping_preserve_native_options);
    record(results, "backend_extension_neutrality_and_seed", test_backend_extension_mapping_keeps_neutral_engine_filters_and_seed);
    record(results, "common_provider_forces_greedy_and_retains_seed", test_common_mapping_forces_greedy_and_retains_configured_seed);
    record(results, "common_rejects_model_ngram_filter_atomically", test_common_mapping_rejects_nonzero_model_ngram_filter_atomically);
    record(results, "eos_mask_tracks_min_length", test_eos_mask_obeys_min_length_and_preserves_eos_afterward);
    record(results, "eos_mask_rejects_invalid_ids", test_eos_mask_rejects_out_of_range_and_negative_ids);
    record(results, "oga_processing_applies_eos_mask", test_oga_logits_processing_applies_model_eos_mask_before_common_processing);

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
              << " OGA common sampling mapping: " << results.size() << " cases, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
