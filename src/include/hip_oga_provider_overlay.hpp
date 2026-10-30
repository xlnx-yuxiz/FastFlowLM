/// \file hip_oga_provider_overlay.hpp
/// \brief Helpers for ONNX Runtime GenAI provider overlays
#pragma once

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

namespace flm::hip_oga {

inline std::vector<std::string> detect_pipeline_sections(
    const nlohmann::json& genai_config) {
    static const std::vector<std::string> kKnownSections{
        "decoder",
        "embedding",
        "vision",
        "speech",
    };

    std::vector<std::string> sections;
    const auto model = genai_config.find("model");
    if (model != genai_config.end() && model->is_object()) {
        for (const std::string& section : kKnownSections) {
            const auto value = model->find(section);
            if (value != model->end() && value->is_object()) {
                sections.push_back(section);
            }
        }
    }
    if (sections.empty()) {
        sections.push_back("decoder");
    }
    return sections;
}

inline nlohmann::json build_provider_overlay(
    const nlohmann::json& genai_config,
    const std::string& provider,
    const std::string& profile,
    const nlohmann::json& extra_options) {
    nlohmann::json options = nlohmann::json::object();
    if (!profile.empty()) {
        options["profile"] = profile;
    }
    if (extra_options.is_object()) {
        for (const auto& [key, value] : extra_options.items()) {
            options[key] = value.is_string() ? value : nlohmann::json(value.dump());
        }
    }

    nlohmann::json sections = nlohmann::json::object();
    for (const std::string& section : detect_pipeline_sections(genai_config)) {
        sections[section] = {
            {"session_options", {
                {"provider_options", nlohmann::json::array({
                    nlohmann::json{{provider, options}}
                })}
            }}
        };
    }
    return nlohmann::json{{"model", std::move(sections)}};
}

}  // namespace flm::hip_oga
