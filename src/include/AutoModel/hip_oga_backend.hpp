#pragma once

#include "AutoModel/model_backend.hpp"
#include "backend_options.hpp"
#include <stdexcept>

namespace flm::backend {

inline constexpr const char* kHipOgaGenericFamily = "hip-oga";

// OGA identifies the architecture from its package. Custom catalogs and direct
// directories need no fabricated FLM architecture to select that engine.
inline std::string hip_oga_backend_family(const BackendRegistry& registry,
                                          const std::string& family) {
    if (registry.has(family, kHipOgaBackendId)) return family;
    if (registry.has(kHipOgaGenericFamily, kHipOgaBackendId)) {
        return kHipOgaGenericFamily;
    }
    throw std::runtime_error("This build has no HIP OGA backend for '" + family + "'");
}

void register_hip_oga_backends(BackendRegistry& registry);

}  // namespace flm::backend
