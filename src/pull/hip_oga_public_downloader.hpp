/// \file hip_oga_public_downloader.hpp
/// \brief Public Hugging Face HIP OGA package installer
#pragma once

#include "model_list.hpp"

#include <string>

namespace flm::hip_oga {

class PublicHfDownloader {
public:
    explicit PublicHfDownloader(model_list& models);

    /// \brief Install a HIP OGA package from hip_oga.public_hf metadata.
    /// \param model_tag model-list tag with hip_oga.public_hf metadata
    /// \param force_redownload replace a present package after verifying a new one
    /// \return true if the package is installed and validates locally
    bool pull(const std::string& model_tag, bool force_redownload = false);

    /// Verify all pinned payload sizes and hashes without modifying the package.
    bool check(const std::string& model_tag);

    /// Remove only a managed package below the explicit HIP model root.
    bool remove(const std::string& model_tag);

private:
    model_list& supported_models_;
};

}  // namespace flm::hip_oga
