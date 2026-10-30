/// \file hip_oga_ensure_model.hpp
/// \brief Resolve an admitted HIP OGA catalog tag to a validated local package.
#pragma once

#include "hip_oga_public_downloader.hpp"

#include <stdexcept>
#include <string>

namespace flm::hip_oga {

inline std::string ensure_hip_oga_model(model_list& models,
                                        const std::string& tag) {
    models.require_hip_oga_runtime_supported(tag);
    if (models.has_hip_oga_public_hf(tag)) {
        PublicHfDownloader downloader(models);
        if (!downloader.pull(tag)) {
            throw std::runtime_error(
                "Failed to install or verify HIP OGA package for tag '" + tag + "'");
        }
    }
    return models.get_hip_oga_model_path(tag);
}

}  // namespace flm::hip_oga
