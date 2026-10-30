/// \file hip_oga_config_directory.hpp
/// \brief Scoped, linked model view for initializing OGA from effective JSON.
#pragma once

#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace flm::hip_oga {

// OGA discovers both configuration and model assets through one directory.
// Supply the complete configuration at creation time: post-create Overlay does
// not initialize provider lists for optional vision/embedding sessions in OGA.
// Assets are hardlinks, so even multi-GB external weights are never copied.
class ScopedConfigDirectory {
public:
    ScopedConfigDirectory(const std::filesystem::path& source_directory,
                          const nlohmann::json& effective_config) {
        namespace fs = std::filesystem;
        const auto source = fs::canonical(source_directory);
        if (!fs::is_directory(source)) {
            throw std::runtime_error("HIP OGA model path is not a directory: " + source.string());
        }
        if (source == source.root_path()) {
            throw std::runtime_error("HIP OGA model directory must not be a filesystem root");
        }
        const auto source_config = source / "genai_config.json";

        // A sibling normally shares the model's volume. If it is unwritable,
        // try OS temp; create_hard_link still enforces same-volume reuse.
        try {
            path_ = create_unique_directory(source.parent_path());
        } catch (const fs::filesystem_error&) {
            const auto temporary = fs::canonical(fs::temp_directory_path());
            const auto relative = temporary.lexically_relative(source);
            if (relative.empty() || *relative.begin() != "..") {
                throw std::runtime_error("HIP OGA temporary directory must be outside the model directory");
            }
            path_ = create_unique_directory(temporary);
        }

        try {
            // Write before linking any assets, so this stream can never
            // truncate a source file through a case-insensitive alias.
            std::ofstream output(path_ / "genai_config.json", std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << effective_config.dump(2) << '\n';
            output.close();
            for (const auto& entry : fs::recursive_directory_iterator(source)) {
                const auto relative = entry.path().lexically_relative(source);
                if (fs::equivalent(entry.path(), source_config)) {
                    continue;
                }
                const auto destination = path_ / relative;
                if (entry.is_directory()) {
                    // canonical also detects Windows directory junctions,
                    // which need not be reported as ordinary symlinks.
                    if (entry.is_symlink() || fs::canonical(entry.path()) != entry.path()) {
                        throw std::runtime_error("HIP OGA linked model view does not support nested directory links: " +
                                                 entry.path().string());
                    }
                    fs::create_directory(destination);
                } else if (entry.is_regular_file()) {
                    fs::create_hard_link(fs::canonical(entry.path()), destination);
                } else {
                    throw std::runtime_error("HIP OGA linked model view requires regular model assets: " +
                                             entry.path().string());
                }
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ScopedConfigDirectory(const ScopedConfigDirectory&) = delete;
    ScopedConfigDirectory& operator=(const ScopedConfigDirectory&) = delete;

    ~ScopedConfigDirectory() { cleanup(); }

    const std::filesystem::path& path() const { return path_; }

private:
    static std::filesystem::path create_unique_directory(const std::filesystem::path& parent) {
        static std::atomic<unsigned long long> next_id{0};
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            const auto candidate = parent / (".flm-hip-oga-config-" + std::to_string(stamp) + "-" +
                                             std::to_string(next_id.fetch_add(1)));
            if (std::filesystem::create_directory(candidate)) {
                return candidate;
            }
        }
        throw std::runtime_error("Could not create a unique HIP OGA config directory");
    }

    void cleanup() noexcept {
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }
    }

    std::filesystem::path path_;
};

}  // namespace flm::hip_oga
