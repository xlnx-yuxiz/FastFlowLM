/// \file test_hip_oga_public_pull.cpp
/// \brief Runtime check for public AMD Hugging Face HIP OGA catalog installation
#include "hip_oga_public_downloader.hpp"
#include "hip_oga_ensure_model.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

bool regular_file_exists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) && !ec;
}

bool has_temp_install_dirs(const std::filesystem::path& install_root) {
    std::error_code ec;
    if (!std::filesystem::exists(install_root, ec)) return false;
    for (const auto& entry : std::filesystem::directory_iterator(install_root)) {
        if (entry.path().filename().string().find(".tmp-") == 0 ||
            entry.path().filename().string().find(".backup-") == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 9) {
            throw std::runtime_error(
                "usage: test_hip_oga_public_pull <model_list> <exe_dir> <endpoint> <install_root> <success|failure> [force]");
        }

        const std::filesystem::path model_list_path = argv[1];
        const std::string exe_dir = argv[2];
        const std::string endpoint = argv[3];
        const std::filesystem::path install_root = argv[4];
        const std::string expected_result = argv[5];
        bool force = false;
        bool default_root = false;
        bool ensure = false;
        for (int i = 6; i < argc; ++i) {
            force = force || std::string(argv[i]) == "force";
            default_root = default_root || std::string(argv[i]) == "default-root";
            ensure = ensure || std::string(argv[i]) == "ensure";
        }
        const std::filesystem::path installed_model =
            install_root / "test-oga";

        set_env("FLM_HIP_OGA_HF_ENDPOINT", endpoint);
        set_env("FLM_HIP_OGA_SECRET_ENDPOINT", endpoint);
        set_env("AWS_SECRET_ACCESS_KEY", endpoint);
        set_env("FLM_HIP_OGA_MODEL_ROOT", default_root ? "" : install_root.string());

        std::string list_path = model_list_path.string();
        std::string exe_dir_string = exe_dir;
        model_list models(list_path, exe_dir_string);
        flm::hip_oga::PublicHfDownloader downloader(models);

        bool success = false;
        if (ensure) {
            try {
                success = flm::hip_oga::ensure_hip_oga_model(models, "hip-test:1b") ==
                    installed_model.string();
            } catch (const std::exception&) {
                success = false;
            }
        } else {
            success = downloader.pull("hip-test:1b", force);
        }
        if (expected_result == "failure") {
            if (success) {
                throw std::runtime_error("expected HIP OGA public pull to fail");
            }
            if (has_temp_install_dirs(install_root)) {
                throw std::runtime_error("failed pull left temporary install directories");
            }
            std::cout << "PASS: HIP OGA public pull failed safely\n";
            return 0;
        }
        if (expected_result != "success") {
            throw std::runtime_error("expected result must be success or failure");
        }
        if (!success) {
            throw std::runtime_error("expected HIP OGA public pull to succeed");
        }

        const std::string resolved_path =
            models.get_hip_oga_model_path("hip-test:1b");
        if (resolved_path != installed_model.string()) {
            throw std::runtime_error(
                "installed HIP OGA path did not resolve through FLM_HIP_OGA_MODEL_ROOT");
        }

        for (const char* required_file : {
                 "genai_config.json",
                 "tokenizer.json",
                 "tokenizer_config.json",
                 "text.onnx",
             }) {
            if (!regular_file_exists(installed_model / required_file)) {
                throw std::runtime_error(
                    std::string("missing installed file: ") + required_file);
            }
        }
        if (regular_file_exists(installed_model /
                                "Upload Qwen2.5 ONNX files" /
                                "genai_config.json")) {
            throw std::runtime_error("source_subdir was not flattened");
        }

        if (has_temp_install_dirs(install_root)) {
            throw std::runtime_error("temporary install directory was not cleaned up");
        }

        std::cout << "PASS: HIP OGA public pull installs a flattened validated package\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
