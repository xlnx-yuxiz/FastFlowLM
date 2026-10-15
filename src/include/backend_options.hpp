/// \file backend_options.hpp
/// \brief Shared backend selection helpers
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace flm::backend {

inline constexpr const char* kNpuBackendId = "npu";
inline constexpr const char* kHipOgaBackendId = "hip";

inline std::string configured_backend(const std::string& requested) {
    if (!requested.empty()) return requested;
    if (const char* env_backend = std::getenv("FLM_BACKEND");
        env_backend && *env_backend) {
        return env_backend;
    }
    return {};
}

inline bool is_default_npu_backend(const std::string& backend) {
    return backend.empty() || backend == kNpuBackendId;
}

inline bool is_hip_oga_backend(const std::string& backend) {
    return backend == kHipOgaBackendId;
}

inline std::string normalize_backend(const std::string& requested) {
    const std::string backend = configured_backend(requested);
    if (is_default_npu_backend(backend)) return {};

    if (is_hip_oga_backend(backend)) {
#if defined(FLM_ENABLE_HIP_OGA)
        return backend;
#else
        throw std::invalid_argument(
            "Backend 'hip' requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
#endif
    }

    throw std::invalid_argument(
        "Unknown backend '" + backend + "'. Supported backends: npu, hip");
}

}  // namespace flm::backend
