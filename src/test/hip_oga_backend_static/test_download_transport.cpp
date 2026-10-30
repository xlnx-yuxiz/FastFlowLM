/// \file test_download_transport.cpp
/// \brief Command driver for strict HTTP download transport tests.
#include "download_model.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {
bool flag(const char* value) {
    const std::string text(value);
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
    throw std::invalid_argument("flags must be 0 or 1");
}
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: test_download_transport URL DEST SIZE SHA256 STRICT EXPECT_SUCCESS\n";
        return 2;
    }
    try {
        download_utils::DownloadRequest request{};
        request.url = argv[1];
        request.destination = argv[2];
        request.expected_size = std::stoull(argv[3]);
        request.hash_algorithm = download_utils::HashAlgorithm::Sha256;
        request.expected_hash = argv[4];
        if (flag(argv[5])) request.strict_http = true;
        const bool expected_success = flag(argv[6]);
        if (fs::exists(request.destination)) {
            throw std::runtime_error("destination must not exist before the attempt");
        }

        download_utils::CurlInitializer curl;
        const bool success = download_utils::download_file_atomic(request);
        if (success != expected_success) {
            throw std::runtime_error("download result did not match expectation");
        }
        if (!success && fs::exists(request.destination)) {
            throw std::runtime_error("failed download published a destination");
        }
        if (success && (!fs::exists(request.destination) ||
            fs::file_size(request.destination) != request.expected_size ||
            download_utils::calculate_file_sha256(request.destination.string()) != request.expected_hash)) {
            throw std::runtime_error("published destination failed size or SHA256 verification");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
