#include "AutoModel/hip_oga_errors.hpp"
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

using namespace flm::hip_oga;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        const std::string corrupt = "[ImageDecoder]: Failed to create pIDecoder. HRESULT: -2003292409";
        bool translated = false;
        try {
            with_image_input_errors([&]() -> int { throw std::runtime_error(corrupt); });
        } catch (const ImageInputError& error) {
            translated = true;
            const auto response = prefill_error_response(error);
            require(response["error"]["code"] == 400 && response["error"]["type"] == "invalid_request_error",
                    "corrupt image must be an HTTP 400 input error");
            require(response["error"]["message"] == corrupt, "decoder diagnostic must be retained");
        } catch (const std::exception&) {}
        require(translated, "known malformed-image decoder failure was not classified");
        for (const std::string error : {"device lost", "Invalid rank for input: image_features",
                "[ImageDecoder]: Failed to create pIDecoder. HRESULT: -2147024882",
                "unrelated HRESULT: -2003292409"}) {
            try {
                with_image_input_errors([&]() -> int { throw std::runtime_error(error); });
            } catch (const ImageInputError&) {
                throw std::runtime_error("device/resource/unknown errors must not become client errors");
            } catch (const std::runtime_error& actual) {
                require(actual.what() == error, "unexpected runtime error changed");
                const auto response = prefill_error_response(actual);
                require(response["error"]["code"] == 500 && response["error"]["type"] == "model_error",
                        "runtime prefill failure must be HTTP 500");
            }
        }
        try {
            with_image_input_errors([]() -> int { throw std::bad_alloc(); });
        } catch (const std::bad_alloc&) {}
        require(with_image_input_errors([] { return 7; }) == 7, "successful processor return changed");
        std::cout << "PASS image input classification and runtime error separation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
