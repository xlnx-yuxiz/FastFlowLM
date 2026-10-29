#include "sampling/request.hpp"

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using flm::sampling::Provider;
using flm::sampling::SamplingRequestError;
using flm::sampling::SamplingState;
using flm::sampling::normalize_sampling_request;
using flm::sampling::parse_provider;
using flm::sampling::provider_name;
using flm::sampling::resolve_sampling_options;
using json = nlohmann::json;

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
    throw std::runtime_error(std::string(label) + " did not raise SamplingRequestError");
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

json model_defaults() {
    return {
        {"do_sample", true},
        {"temperature", 0.75},
        {"top_k", 20},
        {"top_p", 0.875},
        {"repetition_penalty", 1.0},
        {"random_seed", -1},
    };
}

bool write_receipt(const std::string& path, const std::vector<TestCase>& results) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    std::size_t failures = 0;
    json cases = json::array();
    for (const auto& result : results) {
        failures += result.passed ? 0 : 1;
        json item = {{"name", result.name},
                     {"status", result.passed ? "PASS" : "FAIL"}};
        if (!result.passed) item["error"] = result.error;
        cases.push_back(std::move(item));
    }
    const json receipt = {{"status", failures == 0 ? "PASS" : "FAIL"},
                          {"case_count", results.size()},
                          {"cases", std::move(cases)}};
    out << receipt.dump(2) << '\n';
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: test_sampling_request [receipt.json]\n";
        return 2;
    }

    std::vector<TestCase> results;

    record(results, "provider_names_are_exact_and_round_trip", [] {
        require(parse_provider("backend") == Provider::Backend,
                "backend provider parsed incorrectly");
        require(parse_provider("common") == Provider::Common,
                "common provider parsed incorrectly");
        require(provider_name(Provider::Backend) == "backend",
                "backend provider name changed");
        require(provider_name(Provider::Common) == "common",
                "common provider name changed");
    });

    record(results, "provider_parser_rejects_unknown_values", [] {
        for (const std::string value : {"", "auto", "Common", "backend "}) {
            require_sampling_error([&] { (void)parse_provider(value); },
                                   "unsupported provider value");
        }
    });

    record(results, "normalization_merges_sampler_with_request_precedence", [] {
        const json normalized = normalize_sampling_request(
            {{"temperature", 1.25}, {"top_k", 3}, {"sampler", "common"},
             {"unrelated", "ignored"}},
            {{"temperature", 0.5}, {"sampler", "backend"}, {"model", "fixture"}});
        require(normalized == json({{"temperature", 0.5}, {"top_k", 3},
                                    {"sampler", "backend"}}),
                "request values did not override options or unrelated fields leaked");
    });

    record(results, "normalization_without_sampler_preserves_legacy_shape", [] {
        const json normalized = normalize_sampling_request(
            {{"temperature", 1.5}, {"top_k", 4}, {"min_p", 0.25},
             {"unrelated", "ignored"}},
            {{"top_p", 0.5}, {"model", "fixture"}});
        require(normalized == json({{"temperature", 1.5}, {"top_k", 4},
                                    {"min_p", 0.25}, {"top_p", 0.5}}),
                "legacy request shape changed or gained a sampler default");
        require(!normalized.contains("sampler"),
                "normalization inserted an unrequested sampler key");
    });

    record(results, "valid_top_level_sampler_overrides_invalid_nested_string", [] {
        const auto value = normalize_sampling_request(
            {{"sampler", "bogus"}}, {{"sampler", "common"}});
        require(value == json({{"sampler", "common"}}), "effective sampler did not override invalid nested string");
    });

    record(results, "valid_top_level_sampler_overrides_invalid_nested_type", [] {
        const auto value = normalize_sampling_request(
            {{"sampler", 7}}, {{"sampler", "backend"}});
        require(value == json({{"sampler", "backend"}}), "effective sampler did not override invalid nested type");
    });

    record(results, "normalization_rejects_unknown_or_non_string_sampler", [] {
        const std::vector<json> invalid_values{
            json{{"sampler", 1}}, json{{"sampler", true}}, json{{"sampler", nullptr}},
            json{{"sampler", "automatic"}}, json{{"sampler", "Common"}},
        };
        for (const auto& invalid : invalid_values) {
            require_sampling_error([&] {
                (void)normalize_sampling_request(invalid, json::object());
            }, "invalid options sampler");
            require_sampling_error([&] {
                (void)normalize_sampling_request(json::object(), invalid);
            }, "invalid request sampler");
        }
    });

    record(results, "state_starts_with_exact_model_defaults_and_backend_provider", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        require(state.effective() == defaults,
                "initial effective options changed the model defaults");
        require(!state.effective().contains("sampler"),
                "initial effective options gained a sampler key");
        require(state.provider() == Provider::Backend,
                "default provider is not backend");
    });

    record(results, "explicit_sampler_selects_provider_and_resolves_options", [] {
        const json defaults = model_defaults();
        const json overrides = {{"sampler", "common"}, {"temperature", 0.5}, {"top_k", 2}};
        json expected = defaults;
        expected["sampler"] = "common";
        expected["temperature"] = 0.5;
        expected["top_k"] = 2;
        require(resolve_sampling_options(defaults, overrides, 64) == expected,
                "explicit provider or sampling overrides were not resolved");

        SamplingState state(defaults, 64);
        state.reset_request(overrides);
        require(state.provider() == Provider::Common,
                "explicit common sampler did not select the common provider");
        require(state.effective() == expected,
                "state effective options diverged from resolved options");
    });

    record(results, "new_request_restores_defaults_and_provider", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"sampler", "common"}, {"top_k", 3}, {"temperature", 0.25}});
        require(state.provider() == Provider::Common, "first request provider was not common");

        state.reset_request({{"temperature", 0.5}});
        json expected = defaults;
        expected["temperature"] = 0.5;
        require(state.provider() == Provider::Backend,
                "omitted provider leaked from the previous request");
        require(state.effective() == expected,
                "omitted sampling options leaked from the previous request");
        require(!state.effective().contains("sampler"),
                "new request gained a sampler key without an override");
    });

    record(results, "updates_accumulate_until_request_reset", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"sampler", "common"}, {"top_k", 3}});
        state.update({{"temperature", 0.5}});
        require(state.provider() == Provider::Common,
                "update discarded the current request provider");
        require(state.effective().at("top_k") == 3 &&
                    state.effective().at("temperature") == 0.5,
                "update did not preserve earlier overrides and apply the new field");

        state.reset_request({{"frequency_penalty", 0.25}});
        json expected = defaults;
        expected["frequency_penalty"] = 0.25f;
        require(state.provider() == Provider::Backend,
                "request reset did not restore the default provider");
        require(state.effective() == expected,
                "request reset retained an earlier request override");
    });

    record(results, "startup_provider_resets_overrides_and_sets_request_default", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"sampler", "backend"}, {"top_k", 3}});
        state.set_default_provider("common");
        require(state.provider() == Provider::Common,
                "startup provider did not become the active default");
        require(state.effective() == defaults,
                "setting the startup provider did not clear request overrides");
        require(!state.effective().contains("sampler"),
                "startup provider inserted a gratuitous sampler key");

        state.reset_request(json::object());
        require(state.provider() == Provider::Common,
                "empty request did not inherit the configured startup provider");
        require(state.effective() == defaults,
                "empty request changed the model-default JSON shape");
    });

    record(results, "invalid_update_and_reset_leave_effective_state_atomic", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"sampler", "common"}, {"top_k", 3}});
        const json before_update = state.effective();
        const Provider provider_before_update = state.provider();
        require_sampling_error([&] {
            state.update({{"temperature", 0.5}, {"sampler", "future"}});
        }, "invalid update");
        require(state.effective() == before_update && state.provider() == provider_before_update,
                "failed update partially changed the effective request");

        state.update({{"temperature", 0.5}});
        require(state.effective().at("top_k") == 3 && state.provider() == Provider::Common,
                "failed update changed the stored request overrides");

        const json before_reset = state.effective();
        const Provider provider_before_reset = state.provider();
        require_sampling_error([&] {
            state.reset_request({{"sampler", 42}, {"top_k", 1}});
        }, "invalid reset");
        require(state.effective() == before_reset && state.provider() == provider_before_reset,
                "failed reset partially changed the effective request");

        state.update({{"frequency_penalty", 0.25}});
        require(state.effective().at("top_k") == 3 &&
                    state.effective().at("frequency_penalty") == 0.25f &&
                    state.provider() == Provider::Common,
                "failed reset changed the stored request overrides");

        require_sampling_error([&] { state.set_default_provider("future"); },
                               "invalid startup provider");
        json after_invalid_reset = before_reset;
        after_invalid_reset["frequency_penalty"] = 0.25f;
        require(state.effective() == after_invalid_reset && state.provider() == provider_before_reset,
                "failed startup provider change mutated request state");
    });

    record(results, "zero_temperature_selects_greedy_internal_temperature", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"temperature", 0}, {"seed", 0}});
        require(state.effective().at("temperature") == 1.0,
                "zero temperature was not normalized for the backend runtime");
        require(state.effective().at("do_sample") == false,
                "zero temperature did not select greedy mode");
        require(state.effective().at("random_seed") == 0,
                "seed zero was treated as unspecified");
        require(!state.requires_fresh_generator(),
                "greedy mode requested a stochastic generator reset");
    });

    record(results, "seed_zero_is_fixed_and_minus_one_is_unspecified", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"seed", 0}});
        require(state.effective().at("random_seed") == 0,
                "seed zero was not retained explicitly");
        require(state.requires_fresh_generator(),
                "fixed seed zero did not request a fresh stochastic generator");

        state.reset_request({{"seed", -1}});
        require(state.effective().at("random_seed") == -1,
                "seed -1 was not retained as the unspecified sentinel");
        require(!state.requires_fresh_generator(),
                "unspecified seed requested a fresh generator");
    });

    record(results, "int32_seed_boundary_is_accepted_and_invalid_seeds_reject", [] {
        const json defaults = model_defaults();
        SamplingState state(defaults, 64);
        state.reset_request({{"seed", std::numeric_limits<int32_t>::max()}});
        require(state.effective().at("random_seed") == std::numeric_limits<int32_t>::max(),
                "maximum int32 seed did not round-trip");

        for (const json value : {json(2147483648LL), json(-2), json(1.5)}) {
            require_sampling_error([&] {
                (void)resolve_sampling_options(defaults, {{"seed", value}}, 64);
            }, "invalid seed value");
        }
    });

    record(results, "top_k_is_bounded_by_model_vocabulary", [] {
        const json defaults = model_defaults();
        require_sampling_error([&] {
            (void)resolve_sampling_options(defaults, {{"top_k", 65}}, 64);
        }, "top-k above vocabulary size");
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
              << " sampling request: " << results.size() << " cases, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
