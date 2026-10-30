/// \file test_hip_oga_provider_overlay.cpp
/// \brief No-hardware check for HIP OGA provider overlay generation
#include "hip_oga_provider_overlay.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        nlohmann::json genai_config = {
            {"model", {
                {"decoder", {
                    {"session_options", {
                        {"provider_options", nlohmann::json::array({
                            {{"VitisAI", {{"config_file", "old.json"}}}}
                        })}
                    }}
                }},
                {"embedding", nlohmann::json::object()},
                {"vision", {
                    {"session_options", {
                        {"provider_options", nlohmann::json::array({
                            {{"dml", nlohmann::json::object()}}
                        })}
                    }}
                }},
                {"speech", nlohmann::json::object()},
            }}
        };
        nlohmann::json extra = {
            {"cache_dir", "/tmp/flm-cache"},
            {"device_id", 0}
        };

        const nlohmann::json overlay =
            flm::hip_oga::build_provider_overlay(
                genai_config, "AMDGPU", "hip", extra);

        for (const std::string& section :
             {"decoder", "embedding", "vision", "speech"}) {
            const auto& provider_options =
                overlay["model"][section]["session_options"]["provider_options"];
            require(provider_options.is_array() && provider_options.size() == 1,
                    "provider_options must be a single-entry array for " + section);
            require(provider_options[0].contains("AMDGPU"),
                    "provider_options must select AMDGPU for " + section);
            const auto& amd_options = provider_options[0]["AMDGPU"];
            require(amd_options["profile"] == "hip",
                    "AMDGPU profile must be hip for " + section);
            require(amd_options["cache_dir"] == "/tmp/flm-cache",
                    "string provider option was not preserved for " + section);
            require(amd_options["device_id"] == "0",
                    "non-string provider option was not stringified for " + section);
        }
        require(!overlay["model"].contains("encoder"),
                "overlay must only include detected pipeline sections");

        const nlohmann::json fallback =
            flm::hip_oga::build_provider_overlay(
                nlohmann::json::object(), "AMDGPU", "hip",
                nlohmann::json::object());
        require(fallback["model"].contains("decoder"),
                "overlay must fall back to decoder when no sections are detected");

        std::cout << "PASS: HIP OGA provider overlay covers detected pipeline sections\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
