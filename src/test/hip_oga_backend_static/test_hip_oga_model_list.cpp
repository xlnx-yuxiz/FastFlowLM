/// \file test_hip_oga_model_list.cpp
/// \brief Runtime check for HIP OGA model-list tag path resolution
#include "hip_oga_policy.hpp"
#include "model_list.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open " + path.string());
    }
    out << content;
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(name, value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable");
    }
#else
    if (setenv(name, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable");
    }
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path root =
            argc > 1 ? std::filesystem::path(argv[1])
                     : std::filesystem::temp_directory_path() /
                           "flm-hip-oga-model-list-test";
        std::filesystem::remove_all(root);

        const std::filesystem::path config_dir = root / "config";
        const std::filesystem::path npu_model_dir =
            root / "models" / "Llama-3.2-1B-NPU2";
        const std::filesystem::path hip_root = root / "oga-models";
        const std::filesystem::path hip_model_dir =
            hip_root / "llama3.2-1b-oga";
        const std::filesystem::path hip_hybrid_model_dir =
            hip_root / "gemma3-4b-oga";
        const std::filesystem::path hip_unqualified_model_dir =
            hip_root / "qwen3vl-4b-oga";

        std::filesystem::create_directories(npu_model_dir);
        write_file(hip_model_dir / "genai_config.json",
                   R"json({"model":{"context_length":4096,"vocab_size":128000,"decoder":{},"embedding":{},"vision":{},"speech":{}}})json");
        write_file(hip_model_dir / "tokenizer.json", R"json({})json");
        write_file(hip_model_dir / "tokenizer_config.json",
                   R"json({"eos_token":"<|end_of_text|>","eos_token_id":[128001],"chat_template":"{{ messages[0]['content'] }}"})json");
        for (const auto& path : {hip_hybrid_model_dir, hip_unqualified_model_dir}) {
            write_file(path / "genai_config.json",
                       R"json({"model":{"context_length":4096,"vocab_size":256000,"decoder":{},"embedding":{},"vision":{}}})json");
            write_file(path / "tokenizer.json", R"json({})json");
            write_file(path / "tokenizer_config.json",
                       R"json({"eos_token":"<|end_of_text|>","eos_token_id":[1],"chat_template":"{{ messages[0]['content'] }}"})json");
        }
        write_file(config_dir / "model_list.json",
                   R"json({
  "model_path": "models",
    "models": {
    "hip-test": {
      "1b": {
        "name": "Hip-Test-1B-OGA",
        "files": [],
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {
          "family": "llama3",
          "parameter_size": "1B",
          "quantization_level": "OGA"
        },
        "hip_oga": {
          "hip_only": true,
          "provider": "AMDGPU",
          "profile": "hip",
          "config_path": "hip-test-1b-oga",
          "required_files": [
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json"
          ],
          "provider_options": {}
        }
      }
    },
    "hip-alias": {
      "1b": {
        "name": "Hip-Alias-1B-OGA",
        "files": [],
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {"family": "llama3", "parameter_size": "1B", "quantization_level": "OGA"},
        "hip_oga": {
          "hip_only": true,
          "legacy_alias_for": "llama3.2:1b",
          "provider": "AMDGPU",
          "profile": "hip",
          "config_path": "llama3.2-1b-oga",
          "offload_policy": "strict_gpu",
          "required_files": ["genai_config.json", "tokenizer.json", "tokenizer_config.json"],
          "provider_options": {}
        }
      }
    },
    "llama3.2": {
      "1b": {
        "name": "Llama-3.2-1B-NPU2",
        "files": ["config.json", "model.q4nx"],
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {
          "family": "llama3",
          "parameter_size": "1B",
          "quantization_level": "Q4_1"
        },
        "hip_oga": {
          "provider": "AMDGPU",
          "profile": "hip",
          "config_path": "llama3.2-1b-oga",
          "offload_policy": "strict_gpu",
          "required_files": [
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json"
          ],
          "provider_options": {}
        }
      }
    },
    "gemma3": {
      "4b": {
        "name": "Gemma3-4B-NPU2",
        "files": ["config.json", "model.q4nx"],
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {
          "family": "gemma3",
          "parameter_size": "4B",
          "quantization_level": "Q4_1"
        },
        "label": ["vision"],
        "hip_oga": {
          "provider": "AMDGPU",
          "profile": "hip",
          "config_path": "gemma3-4b-oga",
          "offload_policy": "hybrid_gpu",
          "hybrid_cpu_fallback_allowlist": ["Shape", "Reshape", "Gather"],
          "required_files": [
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json"
          ],
          "provider_options": {}
        }
      }
    },
    "qwen3vl-it": {
      "4b": {
        "name": "Qwen3-VL-4B-Instruct-NPU2",
        "files": ["config.json", "model.q4nx"],
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {
          "family": "qwen3vl",
          "parameter_size": "4B",
          "quantization_level": "Q4_1"
        },
        "label": ["vision"],
        "hip_oga": {
          "provider": "AMDGPU",
          "profile": "hip",
          "config_path": "qwen3vl-4b-oga",
          "required_files": [
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json"
          ],
          "provider_options": {}
        }
      }
    }
  }
})json");

        set_env("FLM_HIP_OGA_MODEL_ROOT", hip_root.string());

        std::string list_path = (config_dir / "model_list.json").string();
        std::string exe_dir = root.string();
        model_list models(list_path, exe_dir);

        if (!models.is_model_supported("llama3.2:1b")) {
            throw std::runtime_error("expected llama3.2:1b to be supported");
        }
        if (!models.has_hip_oga_model_path("llama3.2:1b")) {
            throw std::runtime_error("expected llama3.2:1b to have HIP OGA metadata");
        }
        if (!models.is_hip_oga_only_model("hip-test:1b")) {
            throw std::runtime_error("expected hip-test:1b to be marked HIP-only");
        }
        if (models.get_model_path("llama3.2:1b") != npu_model_dir.string()) {
            throw std::runtime_error("default NPU model path changed unexpectedly");
        }
        if (models.get_hip_oga_model_path("llama3.2:1b") !=
            hip_model_dir.string()) {
            throw std::runtime_error("HIP OGA tag path did not resolve through FLM_HIP_OGA_MODEL_ROOT");
        }
        model_list::validate_hip_oga_model_dir(hip_model_dir.string());

        nlohmann::json default_models = models.get_all_models();
        for (const auto& model : default_models["models"]) {
            if (model["name"] == "hip-test:1b") {
                throw std::runtime_error("default model list must hide HIP-only entries");
            }
        }

        if (models.get_hip_oga_offload_policy("llama3.2:1b") !=
            flm::hip_oga::kStrictGpuPolicy) {
            throw std::runtime_error("strict HIP OGA tag must report strict_gpu policy");
        }
        if (models.get_hip_oga_offload_policy("gemma3:4b") !=
            flm::hip_oga::kHybridGpuPolicy) {
            throw std::runtime_error("hybrid HIP OGA VLM tag must report hybrid_gpu policy");
        }
        if (models.get_hip_oga_offload_policy("qwen3vl-it:4b") !=
            flm::hip_oga::kUnqualifiedPolicy) {
            throw std::runtime_error("HIP OGA entries without policy must remain unqualified");
        }
        bool rejected_bad_policy = false;
        try {
            (void)flm::hip_oga::metadata_offload_policy(
                nlohmann::json{{"offload_policy", "gpu_maybe"}});
        } catch (const std::exception&) {
            rejected_bad_policy = true;
        }
        if (!rejected_bad_policy) {
            throw std::runtime_error("invalid HIP OGA offload policies must be rejected");
        }

        const nlohmann::json genai_config =
            nlohmann::json::parse((std::ifstream(hip_model_dir / "genai_config.json")));
        const nlohmann::json strict_overlay =
            flm::hip_oga::build_session_overlay(
                genai_config,
                nlohmann::json{{"provider", "AMDGPU"},
                               {"profile", "hip"},
                               {"offload_policy", "strict_gpu"}},
                false,
                "");
        for (const std::string section : {"decoder", "embedding", "vision", "speech"}) {
            const auto& session_options =
                strict_overlay["model"][section]["session_options"];
            if (session_options["session.disable_cpu_ep_fallback"] != "1") {
                throw std::runtime_error("strict_gpu overlay must disable CPU EP fallback for " + section);
            }
            if (session_options["provider_options"][0]["AMDGPU"]["profile"] != "hip") {
                throw std::runtime_error("strict_gpu overlay must preserve AMDGPU profile=hip for " + section);
            }
        }

        const nlohmann::json hybrid_overlay =
            flm::hip_oga::build_session_overlay(
                genai_config,
                nlohmann::json{{"provider", "AMDGPU"},
                               {"profile", "hip"},
                               {"offload_policy", "hybrid_gpu"}},
                true,
                "flm-hip-oga-profile");
        if (hybrid_overlay["model"]["vision"]["session_options"][
                "session.disable_cpu_ep_fallback"] != "0") {
            throw std::runtime_error("hybrid_gpu overlay must explicitly enable CPU EP fallback");
        }
        if (hybrid_overlay["model"]["vision"]["session_options"]["enable_profiling"] !=
            "flm-hip-oga-profile-vision") {
            throw std::runtime_error("diagnostics overlay must set section-specific profile prefix");
        }

        nlohmann::json hip_models = models.get_all_hip_oga_models();
        if (hip_models["models"].size() != 4) {
            throw std::runtime_error("expected four HIP OGA-capable model-list entries");
        }
        bool saw_strict = false;
        bool saw_hybrid = false;
        bool saw_unqualified = false;
        bool saw_uninstalled_hip_only_model = false;
        for (const auto& hip_model : hip_models["models"]) {
            if (hip_model["model"] == "hip-alias:1b") {
                throw std::runtime_error("legacy HIP alias must be hidden from list output");
            }
            if (hip_model["backend"] != "hip") {
                throw std::runtime_error("HIP OGA list entry did not expose backend status");
            }
            if (hip_model["name"] == "hip-test:1b") {
                if (hip_model["installed"].get<bool>()) {
                    throw std::runtime_error("missing HIP-only OGA package should remain uninstalled");
                }
                saw_uninstalled_hip_only_model = true;
                continue;
            }
            if (!hip_model["installed"].get<bool>()) {
                throw std::runtime_error("installed HIP OGA entries must report installed=true");
            }
            const std::string policy =
                hip_model["hip_oga"]["offload_policy"].get<std::string>();
            saw_strict = saw_strict || policy == flm::hip_oga::kStrictGpuPolicy;
            saw_hybrid = saw_hybrid || policy == flm::hip_oga::kHybridGpuPolicy;
            saw_unqualified = saw_unqualified ||
                (policy == flm::hip_oga::kUnqualifiedPolicy &&
                 hip_model["hip_oga"]["qualification"] ==
                     flm::hip_oga::kUnqualifiedPolicy);
            if (hip_model["model"].get<std::string>() == "llama3.2:1b" &&
                hip_model["hip_oga"]["resolved_config_path"] != hip_model_dir.string()) {
                throw std::runtime_error("HIP OGA list entry did not expose the resolved OGA package path");
            }
            if (policy == flm::hip_oga::kHybridGpuPolicy &&
                hip_model["hip_oga"]["policy_warning"].get<std::string>().find("CPU EP fallback") == std::string::npos) {
                throw std::runtime_error("hybrid_gpu list entry must include an explicit CPU fallback warning");
            }
        }
        if (!saw_strict || !saw_hybrid || !saw_unqualified ||
            !saw_uninstalled_hip_only_model) {
            throw std::runtime_error("HIP OGA list must distinguish policy/install states");
        }

        set_env("FLM_HIP_OGA_OFFLOAD_POLICY", flm::hip_oga::kHybridGpuPolicy);
        nlohmann::json openai_models =
            models.get_all_models_openai_hip_oga(hip_model_dir.string());
        bool saw_direct_model = false;
        bool saw_direct_hybrid_warning = false;
        bool saw_tag_model = false;
        for (const auto& model : openai_models["data"]) {
            saw_direct_model = saw_direct_model ||
                model["id"].get<std::string>() == hip_model_dir.string();
            if (model["id"].get<std::string>() == hip_model_dir.string()) {
                saw_direct_hybrid_warning =
                    model["hip_oga"]["offload_policy"] ==
                        flm::hip_oga::kHybridGpuPolicy &&
                    model["hip_oga"]["policy_warning"].get<std::string>().find(
                        "CPU EP fallback") != std::string::npos &&
                    model["hip_oga"]["qualification_warning"].get<std::string>().find(
                        "unqualified") != std::string::npos;
            }
            saw_tag_model = saw_tag_model ||
                model["id"].get<std::string>() == "llama3.2:1b";
            if (model["id"].get<std::string>() == "gemma3:4b" &&
                model["hip_oga"]["offload_policy"] != flm::hip_oga::kHybridGpuPolicy) {
                throw std::runtime_error("OpenAI HIP model list must report hybrid_gpu policy");
            }
        }
        if (!saw_direct_model || !saw_tag_model) {
            throw std::runtime_error("OpenAI HIP model list must include the direct OGA directory and installed HIP tag");
        }
        if (!saw_direct_hybrid_warning) {
            throw std::runtime_error("OpenAI HIP direct model list must warn when direct offload policy is hybrid_gpu");
        }

        std::cout << "PASS: HIP OGA model-list tag path resolves outside NPU model path\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
