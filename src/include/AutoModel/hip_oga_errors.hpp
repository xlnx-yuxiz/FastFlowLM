#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flm::hip_oga {
class ImageInputError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// Translate errors only at image loading/preprocessing, before inference.
template <typename Function>
decltype(auto) with_image_input_errors(Function&& operation) {
    try {
        return std::forward<Function>(operation)();
    } catch (const std::exception& error) {
        // This SDK exposes WIC failures as strings. Translate only the observed
        // WINCODEC_ERR_UNKNOWNIMAGEFORMAT; allocation/device/unknown failures
        // keep their original type. A future typed SDK error can replace this.
        if (std::string_view(error.what()) ==
            "[ImageDecoder]: Failed to create pIDecoder. HRESULT: -2003292409") {
            throw ImageInputError(error.what());
        }
        throw;
    }
}

inline nlohmann::ordered_json prefill_error_response(const std::exception& error) {
    const bool invalid_input = dynamic_cast<const ImageInputError*>(&error) != nullptr;
    return {{"error", {{"message", error.what()},
        {"type", invalid_input ? "invalid_request_error" : "model_error"},
        {"code", invalid_input ? 400 : 500}}}};
}
}
