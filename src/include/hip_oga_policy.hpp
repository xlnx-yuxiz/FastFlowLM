/// \file hip_oga_policy.hpp
/// \brief Shared HIP OGA offload policy helpers.
#pragma once

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flm::hip_oga {

inline constexpr const char* kStrictGpuPolicy = "strict_gpu";
inline constexpr const char* kHybridGpuPolicy = "hybrid_gpu";
inline constexpr const char* kUnqualifiedPolicy = "unqualified";
inline constexpr const char* kDefaultProvider = "AMDGPU";
inline constexpr const char* kDefaultProfile = "hip";

inline bool is_known_offload_policy(std::string_view policy) {
    return policy == kStrictGpuPolicy ||
           policy == kHybridGpuPolicy ||
           policy == kUnqualifiedPolicy;
}

inline bool is_runtime_offload_policy(std::string_view policy) {
    return policy == kStrictGpuPolicy || policy == kHybridGpuPolicy;
}

inline std::string getenv_string(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value != nullptr) {
        std::string result(value);
        free(value);
        return result;
    }
    return {};
#else
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string();
#endif
}

inline bool getenv_flag_enabled(const char* name) {
    std::string value = getenv_string(name);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

inline std::string require_known_offload_policy(const std::string& policy,
                                                const std::string& context) {
    if (!is_known_offload_policy(policy)) {
        throw std::runtime_error(
            context + " must be one of strict_gpu, hybrid_gpu, or unqualified");
    }
    return policy;
}

inline std::string metadata_offload_policy(const nlohmann::json& hip_oga_options) {
    if (!hip_oga_options.is_object()) return kUnqualifiedPolicy;
    const auto policy = hip_oga_options.find("offload_policy");
    if (policy == hip_oga_options.end() || policy->is_null()) {
        return kUnqualifiedPolicy;
    }
    if (!policy->is_string()) {
        throw std::runtime_error("hip_oga.offload_policy must be a string");
    }
    return require_known_offload_policy(
        policy->get<std::string>(),
        "hip_oga.offload_policy");
}

inline std::string direct_model_offload_policy() {
    const std::string requested = getenv_string("FLM_HIP_OGA_OFFLOAD_POLICY");
    if (requested.empty()) return kStrictGpuPolicy;
    const std::string policy =
        require_known_offload_policy(requested, "FLM_HIP_OGA_OFFLOAD_POLICY");
    if (!is_runtime_offload_policy(policy)) {
        throw std::runtime_error(
            "FLM_HIP_OGA_OFFLOAD_POLICY must be strict_gpu or hybrid_gpu");
    }
    return policy;
}

inline std::string runtime_offload_policy(const nlohmann::json& model_info) {
    if (!model_info.is_object()) return kStrictGpuPolicy;
    const auto hip_oga = model_info.find("hip_oga");
    if (hip_oga == model_info.end()) return kStrictGpuPolicy;
    const std::string policy = metadata_offload_policy(*hip_oga);
    return policy == kUnqualifiedPolicy ? kStrictGpuPolicy : policy;
}

inline std::string policy_warning(std::string_view policy) {
    if (policy == kHybridGpuPolicy) {
        return "hybrid_gpu permits CPU EP fallback; verify ORT placement "
               "evidence and reject MatMul/Conv/Attention/GQA or vision-encoder "
               "main compute on CPU.";
    }
    if (policy == kUnqualifiedPolicy) {
        return "unqualified HIP OGA package; it is not release-qualified until "
               "hardware evidence proves strict_gpu or hybrid_gpu behavior.";
    }
    return {};
}

inline std::vector<std::string> pipeline_sections(const nlohmann::json& genai_config) {
    std::vector<std::string> sections;
    const auto model = genai_config.find("model");
    if (model != genai_config.end() && model->is_object()) {
        for (const char* name : {"decoder", "embedding", "vision", "speech"}) {
            const auto section = model->find(name);
            if (section != model->end() && section->is_object()) {
                sections.emplace_back(name);
            }
        }
    }
    if (sections.empty()) {
        sections.emplace_back("decoder");
    }
    return sections;
}

inline nlohmann::json build_session_overlay(const nlohmann::json& genai_config,
                                            const nlohmann::json& hip_oga_options,
                                            bool diagnostics_enabled = false,
                                            const std::string& diagnostics_prefix = "") {
    const std::string policy = runtime_offload_policy(
        nlohmann::json{{"hip_oga", hip_oga_options}});
    if (!is_runtime_offload_policy(policy)) {
        throw std::runtime_error("HIP OGA runtime policy must be strict_gpu or hybrid_gpu");
    }

    const std::string provider = hip_oga_options.value("provider", kDefaultProvider);
    const std::string profile = hip_oga_options.value("profile", kDefaultProfile);
    if (provider.empty()) {
        throw std::runtime_error("hip_oga.provider must not be empty");
    }

    nlohmann::json provider_config = nlohmann::json::object();
    if (!profile.empty()) {
        provider_config["profile"] = profile;
    }
    const auto extra_provider_options = hip_oga_options.find("provider_options");
    if (extra_provider_options != hip_oga_options.end() &&
        extra_provider_options->is_object()) {
        for (const auto& [key, value] : extra_provider_options->items()) {
            provider_config[key] = value.is_string() ? value : nlohmann::json(value.dump());
        }
    }

    nlohmann::json overlay_model = nlohmann::json::object();
    const std::string profile_root =
        diagnostics_prefix.empty() ? "flm-hip-oga-profile" : diagnostics_prefix;
    for (const auto& section : pipeline_sections(genai_config)) {
        nlohmann::json session_options = {
            {"provider_options",
             nlohmann::json::array({nlohmann::json::object({{provider, provider_config}})})}
        };
        if (policy == kStrictGpuPolicy) {
            session_options["disable_cpu_ep_fallback"] = "1";
        }
        if (diagnostics_enabled) {
            session_options["enable_profiling"] = profile_root + "-" + section;
            session_options["log_id"] = "flm-hip-oga-" + section + "-" + policy;
            session_options["log_severity_level"] = 0;
        }
        overlay_model[section] = {{"session_options", std::move(session_options)}};
    }
    return {{"model", std::move(overlay_model)}};
}

}  // namespace flm::hip_oga
