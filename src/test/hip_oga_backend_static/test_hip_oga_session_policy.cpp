#include "hip_oga_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const nlohmann::json config = {{"model", {
            {"decoder", {{"filename", "text.onnx"}}},
            {"embedding", {{"filename", "embedding.onnx"}}},
            {"vision", {{"filename", "vision.onnx"}}}
        }}};
        for (const auto& policy : {"strict_gpu", "hybrid_gpu"}) {
            const auto overlay = flm::hip_oga::build_session_overlay(
                config, {{"offload_policy", policy}}, true, "trace");
            for (const auto& section : {"decoder", "embedding", "vision"}) {
                const auto& options = overlay.at("model").at(section).at("session_options");
                const std::string expected = std::string(policy) == "strict_gpu" ? "1" : "0";
                if (!options.contains("session.disable_cpu_ep_fallback") ||
                    options.at("session.disable_cpu_ep_fallback") != expected) {
                    throw std::runtime_error(std::string(policy) + " must set the effective ORT fallback key for " + section);
                }
                if (options.contains("disable_cpu_ep_fallback")) {
                    throw std::runtime_error("Obsolete short key must not be emitted");
                }
                if (options.at("provider_options") != nlohmann::json::array({
                        {{"AMDGPU", {{"profile", "hip"}}}}})) {
                    throw std::runtime_error("Every session must select only the requested GPU provider");
                }
                if (options.at("enable_profiling") != std::string("trace-") + section) {
                    throw std::runtime_error("Every session must receive its own profile path");
                }
            }
        }

        const auto make_section = [](const char* filename) {
            return nlohmann::json{
                {"filename", filename},
                {"session_options", {
                    {"provider_options", nlohmann::json::array({
                        {{"CPUExecutionProvider", {{"device_id", "0"}}}},
                        {{"DmlExecutionProvider", nlohmann::json::object()}}
                    })},
                    {"disable_cpu_ep_fallback", "legacy-short-value"},
                    {"session.disable_cpu_ep_fallback", "legacy-full-value"},
                    {"enable_profiling", "existing-profile.json"},
                    {"custom_session_option", "preserve-me"}
                }}
            };
        };
        nlohmann::json effective_input = {
            {"model", {
                {"model_type", "policy-fixture"},
                {"vocab_size", 1234},
                {"search", {{"top_k", 17}, {"temperature", 0.25}}},
                {"decoder", make_section("text.onnx")},
                {"embedding", make_section("embedding.onnx")},
                {"vision", make_section("vision.onnx")},
                {"speech", make_section("speech.onnx")}
            }}
        };
        const nlohmann::json original_effective_input = effective_input;
        const nlohmann::json strict_options = {
            {"provider", "AMDGPU"},
            {"profile", "hip"},
            {"offload_policy", "strict_gpu"}
        };
        const nlohmann::json hybrid_options = {
            {"provider", "AMDGPU"},
            {"profile", "hip"},
            {"offload_policy", "hybrid_gpu"}
        };
        const auto assert_effective_config = [&](const nlohmann::json& effective,
                                                 const std::string& expected_fallback) {
            const auto& model = effective.at("model");
            if (model.at("model_type") != "policy-fixture" ||
                model.at("vocab_size") != 1234 ||
                model.at("search") != original_effective_input.at("model").at("search")) {
                throw std::runtime_error("Effective HIP OGA config must preserve unrelated model/search fields");
            }
            const nlohmann::json expected_providers = nlohmann::json::array({
                {{"AMDGPU", {{"profile", "hip"}}}}
            });
            for (const auto& section : {"decoder", "embedding", "vision", "speech"}) {
                const auto& pipeline = model.at(section);
                const auto& options = pipeline.at("session_options");
                if (pipeline.at("filename") != original_effective_input.at("model").at(section).at("filename") ||
                    options.at("custom_session_option") != "preserve-me" ||
                    options.at("enable_profiling") != "existing-profile.json") {
                    throw std::runtime_error(std::string("Effective HIP OGA config must preserve unrelated fields in ") + section);
                }
                if (options.at("provider_options") != expected_providers) {
                    throw std::runtime_error(std::string("Effective HIP OGA config must replace the provider array in ") + section);
                }
                if (options.at("session.disable_cpu_ep_fallback") != expected_fallback ||
                    options.contains("disable_cpu_ep_fallback")) {
                    throw std::runtime_error(std::string("Effective HIP OGA config must write only the full fallback key in ") + section);
                }
            }
        };

        const auto strict_effective = flm::hip_oga::build_effective_config(
            effective_input, strict_options);
        assert_effective_config(strict_effective, "1");
        const auto hybrid_effective = flm::hip_oga::build_effective_config(
            strict_effective, hybrid_options);
        assert_effective_config(hybrid_effective, "0");
        const auto restored_strict_effective = flm::hip_oga::build_effective_config(
            hybrid_effective, strict_options);
        assert_effective_config(restored_strict_effective, "1");
        if (effective_input != original_effective_input) {
            throw std::runtime_error("Building effective HIP OGA configs must leave the base JSON unchanged");
        }

        std::cout << "PASS: strict and hybrid emit effective ORT session policy\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
