/// \file test_hip_oga_config_directory.cpp
/// \brief No-runtime tests for HIP OGA model config directory mirroring.
#include "hip_oga_config_directory.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TemporaryTree {
public:
    TemporaryTree() {
        static std::atomic<unsigned long long> next_id{0};
        const auto base = fs::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto stamp = std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count();
            root_ = base / ("fastflowlm_config_directory_test_" +
                            std::to_string(stamp) + "_" +
                            std::to_string(next_id.fetch_add(1)));
            std::error_code error;
            if (fs::create_directory(root_, error)) {
                return;
            }
            if (error) {
                throw fs::filesystem_error("create test directory", root_, error);
            }
        }
        throw std::runtime_error("could not allocate a unique test directory");
    }

    TemporaryTree(const TemporaryTree&) = delete;
    TemporaryTree& operator=(const TemporaryTree&) = delete;

    ~TemporaryTree() {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }

    const fs::path& path() const { return root_; }

private:
    fs::path root_;
};

struct ModelFiles {
    std::string config;
    std::string model;
    std::string nested_sidecar;
    std::string tokenizer;
};

void write_bytes(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not create fixture file " + path.string());
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("could not write fixture file " + path.string());
    }
}

ModelFiles create_model_fixture(const fs::path& source) {
    ModelFiles files{
        "{\"model\":{\"decoder\":{}},\"fixture\":\"original\"}",
        "onnx-model-fixture-bytes",
        "external-weight-sidecar-fixture-bytes",
        "{\"version\":1,\"fixture\":\"tokenizer\"}"};

    fs::create_directories(source / "vision" / "external");
    fs::create_directories(source / "tokenizer");
    write_bytes(source / "genai_config.json", files.config);
    write_bytes(source / "model.onnx", files.model);
    write_bytes(source / "vision" / "external" / "weights.bin",
                files.nested_sidecar);
    write_bytes(source / "tokenizer" / "tokenizer.json", files.tokenizer);
    return files;
}

std::string read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not read file " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::set<fs::path> direct_entries(const fs::path& directory) {
    std::set<fs::path> result;
    for (const auto& entry : fs::directory_iterator(directory)) {
        result.insert(entry.path());
    }
    return result;
}

nlohmann::json effective_config() {
    nlohmann::json config = nlohmann::json::object();
    config["model"]["decoder"]["session_options"]["provider"] = "AMDGPU";
    config["fixture"] = "effective";
    config["vision"]["session_options"]["provider"] = "AMDGPU";
    return config;
}

void test_effective_config_is_independent_and_model_resources_are_hardlinked() {
    TemporaryTree temporary;
    const fs::path source = temporary.path() / "source-model";
    const ModelFiles files = create_model_fixture(source);
    const auto replacement = effective_config();

    flm::hip_oga::ScopedConfigDirectory config_directory(source, replacement);
    const fs::path mirror = config_directory.path();

    require(mirror != source,
            "effective config must live in a directory distinct from the source");
    require(fs::exists(mirror / "genai_config.json"),
            "mirror must contain the effective genai_config.json");
    require(nlohmann::json::parse(read_bytes(mirror / "genai_config.json")) == replacement,
            "mirror genai_config.json must contain the effective config");
    require(!fs::equivalent(source / "genai_config.json",
                            mirror / "genai_config.json"),
            "effective genai_config.json must be an independent file");
    require(read_bytes(mirror / "genai_config.json") != files.config,
            "effective config bytes must replace the original config bytes");
    require(read_bytes(source / "genai_config.json") == files.config,
            "creating the mirror must leave the source config unchanged");

    require(fs::equivalent(source / "model.onnx", mirror / "model.onnx"),
            "model weights must be hardlinked rather than copied");
    const fs::path nested_sidecar = fs::path("vision") / "external" / "weights.bin";
    require(fs::exists(mirror / nested_sidecar),
            "nested ONNX external data path must retain its relative structure");
    require(fs::equivalent(source / nested_sidecar, mirror / nested_sidecar),
            "nested external data must be hardlinked rather than copied");
    require(fs::equivalent(source / "tokenizer" / "tokenizer.json",
                           mirror / "tokenizer" / "tokenizer.json"),
            "nested tokenizer resources must be retained as hardlinks");
    require(read_bytes(source / "model.onnx") == files.model &&
                read_bytes(source / nested_sidecar) == files.nested_sidecar &&
                read_bytes(source / "tokenizer" / "tokenizer.json") == files.tokenizer,
            "creating the mirror must leave all source resource bytes unchanged");
}

void test_simultaneous_instances_are_isolated_and_destruction_removes_mirrors() {
    TemporaryTree temporary;
    const fs::path source = temporary.path() / "source-model";
    const ModelFiles files = create_model_fixture(source);
    const auto first_config = nlohmann::json{{"fixture", "first"}};
    const auto second_config = nlohmann::json{{"fixture", "second"}};
    fs::path first_mirror;
    fs::path second_mirror;

    {
        flm::hip_oga::ScopedConfigDirectory first(source, first_config);
        flm::hip_oga::ScopedConfigDirectory second(source, second_config);
        first_mirror = first.path();
        second_mirror = second.path();

        require(first_mirror != second_mirror,
                "simultaneous instances must own separate mirror directories");
        require(nlohmann::json::parse(read_bytes(first_mirror / "genai_config.json")) ==
                    first_config,
                "first instance must retain its own effective config");
        require(nlohmann::json::parse(read_bytes(second_mirror / "genai_config.json")) ==
                    second_config,
                "second instance must retain its own effective config");
        require(fs::equivalent(source / "model.onnx", first_mirror / "model.onnx") &&
                    fs::equivalent(source / "model.onnx", second_mirror / "model.onnx"),
                "isolated mirrors must both reuse the original model file");
        require(read_bytes(source / "genai_config.json") == files.config,
                "live mirrors must not mutate the source config");
    }

    require(!fs::exists(first_mirror) && !fs::exists(second_mirror),
            "destroying each instance must remove its owned mirror directory");
    require(read_bytes(source / "genai_config.json") == files.config &&
                read_bytes(source / "model.onnx") == files.model &&
                read_bytes(source / "vision" / "external" / "weights.bin") ==
                    files.nested_sidecar &&
                read_bytes(source / "tokenizer" / "tokenizer.json") == files.tokenizer,
            "destroying mirrors must preserve source config and resource bytes");
}

void test_missing_source_fails() {
    TemporaryTree temporary;
    const fs::path missing_source = temporary.path() / "does-not-exist";
    bool threw = false;
    try {
        flm::hip_oga::ScopedConfigDirectory config_directory(
            missing_source, effective_config());
    } catch (const std::exception&) {
        threw = true;
    }

    require(threw, "constructing a mirror from a missing source must fail");
    require(!fs::exists(missing_source),
            "failed construction must not create the missing source directory");
}

void test_filesystem_root_is_rejected_before_mirroring() {
    const auto root = fs::canonical(fs::temp_directory_path()).root_path();
    bool rejected = false;
    try {
        flm::hip_oga::ScopedConfigDirectory config_directory(root, effective_config());
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("must not be a filesystem root") != std::string::npos;
    }
    require(rejected, "filesystem root must be rejected explicitly before allocating a nested mirror");
}

void test_model_root_symlink_is_accepted() {
    TemporaryTree temporary;
    const fs::path source = temporary.path() / "source-model";
    const ModelFiles files = create_model_fixture(source);
    const fs::path source_link = temporary.path() / "source-model-link";
    std::error_code symlink_error;
    fs::create_directory_symlink(source, source_link, symlink_error);
    if (symlink_error) {
        std::cout << "SKIP: model-root directory symlinks are unavailable: "
                  << symlink_error.message() << "\n";
        return;
    }

    flm::hip_oga::ScopedConfigDirectory config_directory(
        source_link, effective_config());
    const fs::path mirror = config_directory.path();
    require(mirror != source_link && mirror != source,
            "model-root symlink must be mirrored to a distinct directory");
    require(nlohmann::json::parse(read_bytes(mirror / "genai_config.json")) ==
                effective_config(),
            "model-root symlink must retain the effective config");
    require(fs::equivalent(source / "model.onnx", mirror / "model.onnx"),
            "model-root symlink must still expose hardlinked model resources");
    require(read_bytes(source / "genai_config.json") == files.config,
            "model-root symlink must not mutate the resolved source config");
}

void test_windows_case_insensitive_config_name_does_not_mutate_source() {
#ifdef _WIN32
    TemporaryTree temporary;
    const fs::path source = temporary.path() / "source-model";
    const ModelFiles files = create_model_fixture(source);
    const fs::path source_config = source / "GenAI_Config.json";
    fs::rename(source / "genai_config.json", source_config);

    const fs::path identity_anchor = temporary.path() / "original-config-identity";
    fs::create_hard_link(source_config, identity_anchor);
    const auto replacement = effective_config();

    flm::hip_oga::ScopedConfigDirectory config_directory(source, replacement);
    const fs::path mirror_config = config_directory.path() / "genai_config.json";

    require(read_bytes(source_config) == files.config,
            "writing the effective config must preserve original mixed-case config bytes");
    require(fs::equivalent(source_config, identity_anchor),
            "writing the effective config must preserve original config file identity");
    require(nlohmann::json::parse(read_bytes(mirror_config)) == replacement,
            "mixed-case source config must produce the effective config in the mirror");
    require(!fs::equivalent(source_config, mirror_config),
            "effective config must not share identity with the mixed-case source config");
#else
    std::cout << "SKIP: mixed-case config alias regression requires Windows semantics\n";
#endif
}

void test_directory_symlink_is_rejected_and_partial_mirror_is_cleaned() {
    TemporaryTree temporary;
    const fs::path source = temporary.path() / "source-model";
    const ModelFiles files = create_model_fixture(source);
    const fs::path symlink_target = temporary.path() / "linked-assets";
    fs::create_directories(symlink_target);
    write_bytes(symlink_target / "linked.bin", "linked-resource-bytes");

    const fs::path directory_link = source / "linked-assets";
    std::error_code symlink_error;
    fs::create_directory_symlink(symlink_target, directory_link, symlink_error);
    if (symlink_error) {
        std::cout << "SKIP: directory symlinks are unavailable: "
                  << symlink_error.message() << "\n";
        return;
    }
    require(fs::is_symlink(fs::symlink_status(directory_link)),
            "symlink fixture must be a directory symlink");

    const fs::path system_temp = fs::temp_directory_path();
    const auto fixture_entries_before = direct_entries(temporary.path());
    const auto system_temp_entries_before = direct_entries(system_temp);
    bool threw = false;
    try {
        flm::hip_oga::ScopedConfigDirectory config_directory(
            source, effective_config());
    } catch (const std::exception&) {
        threw = true;
    }

    require(threw, "directory symlink recursion must be rejected");
    require(direct_entries(temporary.path()) == fixture_entries_before,
            "failed symlink traversal must remove any partial sibling mirror");
    require(direct_entries(system_temp) == system_temp_entries_before,
            "failed symlink traversal must remove any same-volume temp mirror");
    require(read_bytes(source / "genai_config.json") == files.config &&
                read_bytes(source / "model.onnx") == files.model,
            "failed symlink traversal must leave source files unchanged");
}

}  // namespace

int main() {
    try {
        test_effective_config_is_independent_and_model_resources_are_hardlinked();
        test_simultaneous_instances_are_isolated_and_destruction_removes_mirrors();
        test_missing_source_fails();
        test_filesystem_root_is_rejected_before_mirroring();
        test_model_root_symlink_is_accepted();
        test_windows_case_insensitive_config_name_does_not_mutate_source();
        test_directory_symlink_is_rejected_and_partial_mirror_is_cleaned();
        std::cout << "PASS: HIP OGA config directory mirroring\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
