/// \file hip_oga_public_downloader.cpp
/// \brief Public Hugging Face HIP OGA package installer
#include "hip_oga_public_downloader.hpp"

#include "picosha2.h"
#include "utils/debug_utils.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace flm::hip_oga {
namespace {

constexpr const char* kDefaultEndpointEnv = "FLM_HIP_OGA_HF_ENDPOINT";
constexpr const char* kDefaultEndpoint = "https://huggingface.co";
constexpr int kMaxDownloadAttempts = 3;
constexpr const char* kInstallReceipt = "fastflowlm_hip_oga_install.json";

struct FileEntry {
    std::string path;
    std::string sha256;
    std::uint64_t size = 0;
};

struct CurlGlobal {
    CurlGlobal() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    ~CurlGlobal() {
        curl_global_cleanup();
    }
};

void ensure_curl_global() {
    static CurlGlobal curl_global;
    (void)curl_global;
}

size_t write_file_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* output = static_cast<std::ofstream*>(userdata);
    output->write(static_cast<char*>(ptr),
                  static_cast<std::streamsize>(size * nmemb));
    return output->good() ? size * nmemb : 0;
}

std::string trim_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string getenv_string(const std::string& name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name.c_str()) == 0 && value != nullptr) {
        std::string result(value);
        free(value);
        return result;
    }
    return {};
#else
    const char* value = std::getenv(name.c_str());
    return value && *value ? std::string(value) : std::string();
#endif
}

void configure_curl_transport(CURL* curl) {
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
#if LIBCURL_VERSION_NUM >= 0x075500
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "HTTP,HTTPS");
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "HTTP,HTTPS");
#else
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "FastFlowLM/1.0");
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    picosha2::hash256(file, hash.begin(), hash.end());
    return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
}

bool is_hex_digest(const std::string& value, std::size_t length) {
    return value.size() == length &&
        std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isxdigit(c) != 0;
        });
}

bool is_drive_relative_or_rooted(const std::string& value) {
    if (value.size() >= 2 &&
        std::isalpha(static_cast<unsigned char>(value[0])) &&
        value[1] == ':') {
        return true;
    }
    return value.rfind("//", 0) == 0 || value.rfind("\\\\", 0) == 0;
}

std::vector<std::string> split_validated_path_segments(
    const std::string& value,
    const std::string& field_name) {
    if (value.empty()) {
        throw std::runtime_error(field_name + " must not be empty");
    }
    if (value.find('?') != std::string::npos ||
        value.find('#') != std::string::npos) {
        throw std::runtime_error(field_name + " must not contain query or fragment characters");
    }
    if (value.find('\\') != std::string::npos) {
        throw std::runtime_error(field_name + " must use forward slashes only");
    }
    if (value.front() == '/' || is_drive_relative_or_rooted(value)) {
        throw std::runtime_error(field_name + " must be relative");
    }

    std::vector<std::string> segments;
    size_t cursor = 0;
    while (cursor <= value.size()) {
        const size_t slash = value.find('/', cursor);
        const std::string segment =
            value.substr(cursor, slash == std::string::npos ? slash : slash - cursor);
        if (segment.empty() || segment == "." || segment == "..") {
            throw std::runtime_error(field_name + " contains an unsafe path component");
        }
        if (segment.back() == '.' || segment.back() == ' ' ||
            segment.find_first_of(":<>\"|*%") != std::string::npos ||
            std::any_of(segment.begin(), segment.end(), [](unsigned char c) { return c < 32; })) {
            throw std::runtime_error(field_name + " contains an unsafe filesystem name");
        }
        std::string device = segment.substr(0, segment.find('.'));
        std::transform(device.begin(), device.end(), device.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (device == "CON" || device == "PRN" || device == "AUX" || device == "NUL" ||
            (device.size() == 4 && (device.rfind("COM", 0) == 0 || device.rfind("LPT", 0) == 0) &&
             device[3] >= '1' && device[3] <= '9')) {
            throw std::runtime_error(field_name + " contains a reserved filesystem name");
        }
        segments.push_back(segment);
        if (slash == std::string::npos) break;
        cursor = slash + 1;
    }
    return segments;
}

std::string join_path_segments(const std::vector<std::string>& segments) {
    std::string result;
    for (const std::string& segment : segments) {
        if (!result.empty()) result += "/";
        result += segment;
    }
    return result;
}

std::string validate_relative_path(const std::string& value,
                                   const std::string& field_name) {
    return join_path_segments(split_validated_path_segments(value, field_name));
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool contains_credential_fragment(const std::string& name) {
    static const std::vector<std::string> blocked_fragments = {
        "token",
        "password",
        "passwd",
        "secret",
        "cookie",
        "authorization",
        "credential",
        "access_key",
        "api_key",
        "private_key",
        "bearer",
    };
    const std::string lower_name = lowercase_ascii(name);
    for (const std::string& fragment : blocked_fragments) {
        if (lower_name.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void reject_credential_fields_recursive(const nlohmann::json& value,
                                        const std::string& path) {
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            if (contains_credential_fragment(key)) {
                throw std::runtime_error(
                    "hip_oga.public_hf must not contain credential field: " +
                    path + key);
            }
            reject_credential_fields_recursive(child, path + key + ".");
        }
    } else if (value.is_array()) {
        for (const auto& child : value) {
            reject_credential_fields_recursive(child, path);
        }
    }
}

bool is_env_name_shape_safe(const std::string& name) {
    if (name.empty() || std::isdigit(static_cast<unsigned char>(name.front()))) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isupper(c) || std::isdigit(c) || c == '_';
    });
}

void validate_endpoint_env_name(const std::string& name) {
    if (!is_env_name_shape_safe(name)) {
        throw std::runtime_error(
            "hip_oga.public_hf.endpoint_env must be an uppercase environment variable name");
    }
    if (name != kDefaultEndpointEnv &&
        name.rfind("FLM_HIP_OGA_", 0) != 0) {
        throw std::runtime_error(
            "hip_oga.public_hf.endpoint_env must be " +
            std::string(kDefaultEndpointEnv) + " or FLM_HIP_OGA_*");
    }
    if (contains_credential_fragment(name)) {
        throw std::runtime_error(
            "hip_oga.public_hf.endpoint_env must not name a credential-bearing environment variable");
    }
}

bool endpoint_contains_userinfo(const std::string& endpoint) {
    size_t authority_start = std::string::npos;
    const size_t scheme = endpoint.find("://");
    if (scheme != std::string::npos) {
        authority_start = scheme + 3;
    } else if (endpoint.rfind("//", 0) == 0) {
        authority_start = 2;
    } else {
        return false;
    }
    const size_t authority_end = endpoint.find_first_of("/?#", authority_start);
    const std::string authority = endpoint.substr(
        authority_start,
        authority_end == std::string::npos
            ? std::string::npos
            : authority_end - authority_start);
    return authority.find('@') != std::string::npos;
}

void validate_endpoint_value(const std::string& endpoint) {
    if (endpoint.empty()) {
        throw std::runtime_error("HIP OGA public Hugging Face endpoint must not be empty");
    }
    if (endpoint.find('?') != std::string::npos ||
        endpoint.find('#') != std::string::npos) {
        throw std::runtime_error(
            "HIP OGA public Hugging Face endpoint must not contain query or fragment characters");
    }
    if (endpoint_contains_userinfo(endpoint)) {
        throw std::runtime_error(
            "HIP OGA public Hugging Face endpoint must not contain URL userinfo");
    }
    const std::string trimmed = trim_trailing_slashes(endpoint);
    if (trimmed != kDefaultEndpoint &&
        trimmed.rfind("http://127.0.0.1:", 0) != 0 &&
        trimmed.rfind("http://localhost:", 0) != 0) {
        throw std::runtime_error(
            "HIP OGA public Hugging Face endpoint must be https://huggingface.co or a localhost test endpoint");
    }
}

std::string resolve_endpoint(const nlohmann::json& public_hf) {
    const std::string endpoint_env =
        public_hf.value("endpoint_env", std::string(kDefaultEndpointEnv));
    validate_endpoint_env_name(endpoint_env);

    std::string endpoint = getenv_string(endpoint_env);
    if (endpoint.empty()) {
        endpoint = kDefaultEndpoint;
    }
    validate_endpoint_value(endpoint);
    return trim_trailing_slashes(endpoint);
}

std::string validate_repo_id(const std::string& value) {
    const std::string repo_id =
        join_path_segments(split_validated_path_segments(value, "hip_oga.public_hf.repo_id"));
    if (repo_id.rfind("amd/", 0) != 0 || repo_id.size() <= 4 ||
        repo_id.find('/', 4) != std::string::npos) {
        throw std::runtime_error(
            "hip_oga.public_hf.repo_id must name a public AMD Hugging Face repo under amd/*");
    }
    return repo_id;
}

std::string validate_revision(const std::string& value) {
    if (!is_hex_digest(value, 40)) {
        throw std::runtime_error(
            "hip_oga.public_hf.revision must be a pinned commit SHA");
    }
    return value;
}

void validate_license(const nlohmann::json& public_hf) {
    const auto license = public_hf.find("license");
    if (license == public_hf.end() || !license->is_string() ||
        license->get<std::string>().empty()) {
        throw std::runtime_error(
            "hip_oga.public_hf.license must be a non-empty string");
    }
    const std::string lower_license = lowercase_ascii(license->get<std::string>());
    if (lower_license == "unknown" ||
        lower_license.find("no-redistribution") != std::string::npos) {
        throw std::runtime_error(
            "hip_oga.public_hf.license must be known before auto-pull");
    }
}

void reject_case_colliding_path(
    const std::string& relative_path,
    std::unordered_set<std::string>& casefolded_paths) {
    const std::string casefolded = lowercase_ascii(relative_path);
    if (!casefolded_paths.insert(casefolded).second) {
        throw std::runtime_error(
            "hip_oga.public_hf.files contains case-colliding path: " +
            relative_path);
    }
}

std::vector<FileEntry> parse_file_entries(const nlohmann::json& public_hf) {
    const auto files = public_hf.find("files");
    if (files == public_hf.end() || !files->is_array() || files->empty()) {
        throw std::runtime_error(
            "hip_oga.public_hf.files must be a non-empty array");
    }

    std::vector<FileEntry> entries;
    std::unordered_set<std::string> exact_paths;
    std::unordered_set<std::string> casefolded_paths;
    for (const auto& file : *files) {
        if (!file.is_object()) {
            throw std::runtime_error("hip_oga.public_hf.files entries must be objects");
        }
        const auto path = file.find("path");
        const auto sha256 = file.find("sha256");
        const auto size = file.find("size");
        if (path == file.end() || !path->is_string() ||
            sha256 == file.end() || !sha256->is_string() ||
            size == file.end() || !(size->is_number_unsigned() || size->is_number_integer())) {
            throw std::runtime_error(
                "hip_oga.public_hf.files entries need string path, string sha256, and numeric size");
        }
        if (size->is_number_integer() && size->get<long long>() < 0) {
            throw std::runtime_error(
                "hip_oga.public_hf.files size must not be negative");
        }
        FileEntry entry{
            validate_relative_path(path->get<std::string>(),
                                   "hip_oga.public_hf.files.path"),
            sha256->get<std::string>(),
            size->get<std::uint64_t>(),
        };
        if (!is_hex_digest(entry.sha256, 64)) {
            throw std::runtime_error(
                "hip_oga.public_hf.files sha256 must be a SHA256 hex digest");
        }
        if (!exact_paths.insert(entry.path).second) {
            throw std::runtime_error(
                "hip_oga.public_hf.files contains duplicate path: " +
                entry.path);
        }
        reject_case_colliding_path(entry.path, casefolded_paths);
        if (lowercase_ascii(entry.path) == kInstallReceipt ||
            lowercase_ascii(entry.path).rfind(std::string(kInstallReceipt) + "/", 0) == 0) {
            throw std::runtime_error("package files must not replace the install receipt");
        }
        entries.push_back(std::move(entry));
    }
    for (const auto& entry : entries) {
        auto parent = std::filesystem::path(entry.path).parent_path();
        while (!parent.empty()) {
            if (casefolded_paths.count(lowercase_ascii(parent.generic_string()))) {
                throw std::runtime_error("package manifest has a file/directory collision");
            }
            parent = parent.parent_path();
        }
    }
    return entries;
}

std::vector<std::string> required_hip_oga_files(const nlohmann::json& options) {
    std::vector<std::string> required = {
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    };
    const auto required_files = options.find("required_files");
    if (required_files != options.end()) {
        if (!required_files->is_array()) {
            throw std::runtime_error("hip_oga.required_files must be an array");
        }
        for (const auto& file : *required_files) {
            if (!file.is_string()) {
                throw std::runtime_error("hip_oga.required_files must contain strings");
            }
            required.push_back(file.get<std::string>());
        }
    }
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    return required;
}

void ensure_file_entries_cover_required_files(
    const std::vector<FileEntry>& entries,
    const nlohmann::json& options) {
    std::unordered_set<std::string> paths;
    for (const FileEntry& entry : entries) {
        paths.insert(entry.path);
    }
    for (const std::string& required_file : required_hip_oga_files(options)) {
        const std::string normalized_required_file =
            validate_relative_path(required_file, "hip_oga.required_files");
        if (normalized_required_file != required_file) {
            throw std::runtime_error(
                "hip_oga.required_files contains unsafe path: " + required_file);
        }
        if (paths.find(normalized_required_file) == paths.end()) {
            throw std::runtime_error(
                "hip_oga.public_hf.files is missing required file: " +
                normalized_required_file);
        }
    }
}

std::string escape_url_segment(CURL* curl, const std::string& segment) {
    char* escaped = curl_easy_escape(curl, segment.c_str(),
                                     static_cast<int>(segment.size()));
    if (!escaped) {
        throw std::runtime_error("failed to URL-encode Hugging Face path segment");
    }
    std::string result(escaped);
    curl_free(escaped);
    return result;
}

std::string encode_slash_separated_path(const std::string& value,
                                        const std::string& field_name) {
    ensure_curl_global();
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("failed to initialize CURL for URL encoding");
    }
    const std::vector<std::string> segments =
        split_validated_path_segments(value, field_name);
    std::string encoded;
    try {
        for (const std::string& segment : segments) {
            if (!encoded.empty()) encoded += "/";
            encoded += escape_url_segment(curl, segment);
        }
    } catch (...) {
        curl_easy_cleanup(curl);
        throw;
    }
    curl_easy_cleanup(curl);
    return encoded;
}

std::string prefixed_source_path(const std::string& source_subdir,
                                 const std::string& relative_path) {
    if (source_subdir.empty()) {
        return relative_path;
    }
    return source_subdir + "/" + relative_path;
}

std::string build_resolve_url(const std::string& endpoint,
                              const std::string& repo_id,
                              const std::string& revision,
                              const std::string& source_relative_path) {
    return trim_trailing_slashes(endpoint) + "/" +
        encode_slash_separated_path(repo_id, "hip_oga.public_hf.repo_id") +
        "/resolve/" +
        encode_slash_separated_path(revision, "hip_oga.public_hf.revision") +
        "/" +
        encode_slash_separated_path(source_relative_path, "hip_oga.public_hf source path") +
        "?download=true";
}

void http_download_file(const std::string& url, const std::filesystem::path& path) {
    ensure_curl_global();
    std::filesystem::create_directories(path.parent_path());

    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("failed to initialize CURL");
    }

    std::ofstream output(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("failed to open " + path.string() + " for writing");
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output);
    configure_curl_transport(curl);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L);

    CURLcode result = curl_easy_perform(curl);
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    output.close();
    curl_easy_cleanup(curl);

    if (result != CURLE_OK) {
        std::filesystem::remove(path);
        throw std::runtime_error(
            "CURL error while downloading HIP OGA public Hugging Face URL: " +
            std::string(curl_easy_strerror(result)));
    }
    if (response_code < 200 || response_code >= 300) {
        std::filesystem::remove(path);
        throw std::runtime_error(
            "HTTP " + std::to_string(response_code) +
            " while downloading HIP OGA public Hugging Face URL");
    }
}

void download_with_sha256_retry(const std::string& url,
                                const std::filesystem::path& path,
                                const FileEntry& entry) {
    std::exception_ptr last_error;
    for (int attempt = 1; attempt <= kMaxDownloadAttempts; ++attempt) {
        try {
            std::filesystem::remove(path);
            http_download_file(url, path);
            std::error_code ec;
            const std::uint64_t actual_size =
                static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
            if (ec) {
                throw std::runtime_error(
                    "failed to stat downloaded file " + path.string() + ": " +
                    ec.message());
            }
            if (actual_size != entry.size) {
                std::filesystem::remove(path);
                throw std::runtime_error(
                    "size mismatch for " + path.string() + ": expected " +
                    std::to_string(entry.size) + ", got " +
                    std::to_string(actual_size));
            }
            const std::string actual_sha256 = sha256_file(path);
            if (actual_sha256 == lowercase_ascii(entry.sha256)) {
                return;
            }
            std::filesystem::remove(path);
            throw std::runtime_error(
                "sha256 mismatch for " + path.string() + ": expected " +
                entry.sha256 + ", got " + actual_sha256);
        } catch (...) {
            last_error = std::current_exception();
            if (attempt < kMaxDownloadAttempts) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }
    std::rethrow_exception(last_error);
}

std::filesystem::path make_temp_install_path(const std::filesystem::path& install_path) {
    const std::filesystem::path parent = install_path.parent_path();
    const std::string basename = install_path.filename().string();
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return parent / (".tmp-" + basename + "-" + std::to_string(ticks));
}

std::filesystem::path make_backup_install_path(const std::filesystem::path& install_path) {
    const std::filesystem::path parent = install_path.parent_path();
    const std::string basename = install_path.filename().string();
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return parent / (".backup-" + basename + "-" + std::to_string(ticks));
}

bool path_is_within(const std::filesystem::path& child,
                    const std::filesystem::path& root) {
    auto child_it = child.begin();
    auto root_it = root.begin();
    for (; root_it != root.end(); ++root_it, ++child_it) {
        if (child_it == child.end() || *child_it != *root_it) {
            return false;
        }
    }
    return true;
}

std::filesystem::path weakly_canonical_or_throw(
    const std::filesystem::path& path,
    const std::string& description) {
    std::error_code ec;
    const std::filesystem::path result =
        std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        throw std::runtime_error(
            "failed to resolve " + description + " for HIP OGA public pull: " +
            ec.message());
    }
    return result;
}

void reject_symlinked_existing_components(
    const std::filesystem::path& root_path,
    const std::filesystem::path& relative_path,
    const std::string& tag) {
    std::filesystem::path current = root_path;
    for (const auto& component : relative_path) {
        current /= component;
        std::error_code ec;
        const std::filesystem::file_status status =
            std::filesystem::symlink_status(current, ec);
        if (ec) {
            if (ec == std::errc::no_such_file_or_directory) {
                continue;
            }
            throw std::runtime_error(
                "failed to inspect HIP OGA install path for tag '" + tag +
                "': " + ec.message());
        }
        if (status.type() == std::filesystem::file_type::not_found) {
            continue;
        }
        bool linked = std::filesystem::is_symlink(status);
#ifdef _WIN32
        const DWORD attributes = GetFileAttributesW(current.c_str());
        linked = linked || (attributes != INVALID_FILE_ATTRIBUTES &&
                            (attributes & FILE_ATTRIBUTE_REPARSE_POINT));
#endif
        if (linked) {
            throw std::runtime_error(
                "Backend 'hip' pull refuses symlinked path component for tag '" +
                tag + "' under FLM_HIP_OGA_MODEL_ROOT");
        }
    }
}

std::filesystem::path ensure_install_path_under_root(
    const model_list& models, const nlohmann::json& options,
    const std::string& tag, bool create_root = true) {
    const auto configured_path = options.find("config_path");
    if (configured_path == options.end() || !configured_path->is_string()) {
        throw std::runtime_error(
            "Backend 'hip' pull needs string hip_oga.config_path for tag '" +
            tag + "'");
    }
    const std::string relative_config_path =
        validate_relative_path(configured_path->get<std::string>(),
                               "hip_oga.config_path");
    const std::string override_root = getenv_string("FLM_HIP_OGA_MODEL_ROOT");
    const std::string root = override_root.empty()
        ? models.get_model_root_path() : override_root;
    const std::filesystem::path root_input =
        std::filesystem::absolute(std::filesystem::path(root)).lexically_normal();
    if (create_root) std::filesystem::create_directories(root_input);
    const std::filesystem::path root_path =
        weakly_canonical_or_throw(root_input, "FLM_HIP_OGA_MODEL_ROOT");
    if (std::filesystem::exists(root_path) && !std::filesystem::is_directory(root_path)) {
        throw std::runtime_error(
            "FLM_HIP_OGA_MODEL_ROOT must resolve to a directory");
    }
    std::filesystem::path install_path =
        (root_path / std::filesystem::path(relative_config_path)).lexically_normal();
    reject_symlinked_existing_components(
        root_path, std::filesystem::path(relative_config_path), tag);
    const std::filesystem::path install_parent =
        weakly_canonical_or_throw(install_path.parent_path(), "HIP OGA install parent");
    const std::filesystem::path physical_install_path =
        weakly_canonical_or_throw(install_path, "HIP OGA install path");
    if (!path_is_within(install_path, root_path) || install_path == root_path) {
        throw std::runtime_error(
            "Backend 'hip' pull resolved tag '" + tag +
            "' outside FLM_HIP_OGA_MODEL_ROOT");
    }
    if (!path_is_within(install_parent, root_path) ||
        !path_is_within(physical_install_path, root_path) ||
        physical_install_path == root_path) {
        throw std::runtime_error(
            "Backend 'hip' pull resolved tag '" + tag +
            "' outside FLM_HIP_OGA_MODEL_ROOT");
    }
    return install_path;
}

void restore_existing_install_backup(const std::filesystem::path& backup_path,
                                     const std::filesystem::path& install_path) {
    if (backup_path.empty()) return;
    std::error_code ec;
    if (std::filesystem::exists(backup_path, ec) &&
        !std::filesystem::exists(install_path, ec)) {
        std::filesystem::rename(backup_path, install_path, ec);
    }
}

void replace_install_transactionally(const std::filesystem::path& temp_path,
                                     const std::filesystem::path& install_path) {
    std::filesystem::create_directories(install_path.parent_path());
    std::filesystem::path backup_path;
    try {
        if (std::filesystem::exists(install_path)) {
            backup_path = make_backup_install_path(install_path);
            std::filesystem::rename(install_path, backup_path);
        }
        std::filesystem::rename(temp_path, install_path);
        if (!backup_path.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(backup_path, ec);
        }
    } catch (...) {
        restore_existing_install_backup(backup_path, install_path);
        throw;
    }
}

std::string read_required_string(const nlohmann::json& object,
                                 const std::string& key,
                                 const std::string& owner) {
    const auto value = object.find(key);
    if (value == object.end() || !value->is_string() ||
        value->get<std::string>().empty()) {
        throw std::runtime_error(owner + "." + key + " must be a non-empty string");
    }
    return value->get<std::string>();
}

std::string read_source_subdir(const nlohmann::json& public_hf) {
    const auto value = public_hf.find("source_subdir");
    if (value == public_hf.end() || value->is_null()) {
        return {};
    }
    if (!value->is_string()) {
        throw std::runtime_error("hip_oga.public_hf.source_subdir must be a string");
    }
    if (value->get<std::string>().empty()) {
        return {};
    }
    return validate_relative_path(value->get<std::string>(),
                                  "hip_oga.public_hf.source_subdir");
}

void write_install_metadata(const std::filesystem::path& temp_path,
                            const nlohmann::json& public_hf,
                            const std::vector<FileEntry>& files,
                            const std::string& endpoint) {
    nlohmann::json metadata = {
        {"schema_version", "fastflowlm-hip-oga-public-hf/v1"},
        {"endpoint", endpoint},
        {"repo_id", public_hf["repo_id"]},
        {"revision", public_hf["revision"]},
        {"source_subdir", public_hf.value("source_subdir", std::string())},
        {"license", public_hf["license"]},
        {"files", nlohmann::json::array()},
    };
    for (const FileEntry& file : files) {
        metadata["files"].push_back({
            {"path", file.path},
            {"sha256", file.sha256},
            {"size", file.size},
        });
    }
    const std::filesystem::path metadata_path =
        temp_path / kInstallReceipt;
    std::ofstream output(metadata_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write " + metadata_path.string());
    }
    output << metadata.dump(2) << "\n";
    output.close();
    if (!output) throw std::runtime_error("failed to flush HIP OGA install receipt");
}

std::vector<FileEntry> validated_manifest(const nlohmann::json& public_hf,
                                          const nlohmann::json& options) {
    reject_credential_fields_recursive(public_hf, "");
    validate_repo_id(read_required_string(public_hf, "repo_id", "hip_oga.public_hf"));
    validate_revision(read_required_string(public_hf, "revision", "hip_oga.public_hf"));
    read_source_subdir(public_hf);
    auto files = parse_file_entries(public_hf);
    ensure_file_entries_cover_required_files(files, options);
    return files;
}

void verify_payload(const std::filesystem::path& root, const std::vector<FileEntry>& files) {
    for (const auto& file : files) {
        reject_symlinked_existing_components(root, file.path, "payload");
        const auto path = root / file.path;
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != file.size ||
            sha256_file(path) != lowercase_ascii(file.sha256)) {
            throw std::runtime_error("HIP OGA size/sha256 mismatch or missing file: " + file.path);
        }
    }
}

// Audit the entire tree before mutation. Unlisted files/directories belong to
// the caller; even --force must preserve them. Never traverse links/junctions.
std::vector<std::filesystem::path> audit_owned_tree(
    const std::filesystem::path& root, const std::vector<FileEntry>& files) {
    std::set<std::string> allowed_files{kInstallReceipt}, allowed_dirs;
    for (const auto& file : files) {
        allowed_files.insert(file.path);
        auto parent = std::filesystem::path(file.path).parent_path();
        while (!parent.empty()) {
            allowed_dirs.insert(parent.generic_string());
            parent = parent.parent_path();
        }
    }
    std::vector<std::filesystem::path> directories;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        const auto relative = entry.path().lexically_relative(root);
        reject_symlinked_existing_components(root, relative, "managed package");
        if (entry.is_directory()) {
            if (!allowed_dirs.count(relative.generic_string())) {
                throw std::runtime_error("HIP OGA package contains an untracked directory: " + relative.generic_string());
            }
            directories.push_back(relative);
        } else if (!entry.is_regular_file() || !allowed_files.count(relative.generic_string())) {
            throw std::runtime_error("HIP OGA package contains an untracked file: " + relative.generic_string());
        }
    }
    std::sort(directories.begin(), directories.end(), [](const auto& a, const auto& b) {
        return std::distance(a.begin(), a.end()) > std::distance(b.begin(), b.end());
    });
    return directories;
}

void require_matching_receipt(const std::filesystem::path& root, const nlohmann::json& public_hf) {
    reject_symlinked_existing_components(root, kInstallReceipt, "install receipt");
    const auto path = root / kInstallReceipt;
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > 4 * 1024 * 1024) {
        throw std::runtime_error("HIP OGA remove requires a managed install receipt");
    }
    std::ifstream input(path);
    const auto receipt = nlohmann::json::parse(input);
    if (!receipt.is_object() || receipt.value("schema_version", "") != "fastflowlm-hip-oga-public-hf/v1" ||
        receipt.value("repo_id", "") != public_hf.at("repo_id") ||
        receipt.value("revision", "") != public_hf.at("revision") ||
        receipt.value("source_subdir", "") != public_hf.value("source_subdir", "") ||
        !receipt.contains("files") || receipt["files"] != public_hf.at("files")) {
        throw std::runtime_error("HIP OGA install receipt does not match the pinned catalog manifest");
    }
}

void require_download_space(const std::filesystem::path& path, const std::vector<FileEntry>& files) {
    std::uint64_t required = 64 * 1024 * 1024; // receipt + filesystem margin
    for (const auto& file : files) {
        if (file.size > std::numeric_limits<std::uint64_t>::max() - required) {
            throw std::runtime_error("HIP OGA package size overflow");
        }
        required += file.size;
    }
    auto parent = path.parent_path();
    while (!std::filesystem::exists(parent)) parent = parent.parent_path();
    if (std::filesystem::space(parent).available < required) {
        throw std::runtime_error("Insufficient free disk space for HIP OGA staged download: need " +
                                 std::to_string(required) + " bytes");
    }
}

}  // namespace

PublicHfDownloader::PublicHfDownloader(model_list& models)
    : supported_models_(models) {}

bool PublicHfDownloader::check(const std::string& model_tag) {
    try {
        const auto path = supported_models_.get_hip_oga_model_path(model_tag);
        const auto options = supported_models_.get_hip_oga_model_options(model_tag);
        const auto public_hf = supported_models_.get_hip_oga_public_hf_config(model_tag);
        verify_payload(path, validated_manifest(public_hf, options));
        header_print("FLM", "HIP OGA pinned package integrity verified: " << path);
        return true;
    } catch (const std::exception& e) {
        header_print("ERROR", "HIP OGA check failed: " + std::string(e.what()));
        return false;
    }
}

bool PublicHfDownloader::remove(const std::string& model_tag) {
    try {
        const auto options = supported_models_.get_hip_oga_model_options(model_tag);
        const auto public_hf = supported_models_.get_hip_oga_public_hf_config(model_tag);
        const auto files = validated_manifest(public_hf, options);
        const auto path = ensure_install_path_under_root(supported_models_, options, model_tag, false);
        if (!std::filesystem::exists(path)) {
            header_print("FLM", "HIP OGA package already absent: " << path.string());
            return true;
        }
        require_matching_receipt(path, public_hf);
        const auto directories = audit_owned_tree(path, files);
        // Delete named files only. A concurrently added untracked file prevents
        // the final directory removal instead of being recursively erased.
        for (const auto& file : files) {
            reject_symlinked_existing_components(path, file.path, model_tag);
            std::filesystem::remove(path / file.path);
        }
        for (const auto& directory : directories) {
            reject_symlinked_existing_components(path, directory, model_tag);
            std::filesystem::remove(path / directory);
        }
        std::filesystem::remove(path / kInstallReceipt);
        std::filesystem::remove(path);
        header_print("FLM", "Removed managed HIP OGA package: " << path.string());
        return true;
    } catch (const std::exception& e) {
        header_print("ERROR", "HIP OGA remove failed: " + std::string(e.what()));
        return false;
    }
}

bool PublicHfDownloader::pull(const std::string& model_tag, bool force_redownload) {
    try {
        auto [resolved_tag, model_info] = supported_models_.get_model_info(model_tag);
        const nlohmann::json options =
            supported_models_.get_hip_oga_model_options(resolved_tag);
        supported_models_.require_hip_oga_runtime_supported(resolved_tag);
        const nlohmann::json public_hf =
            supported_models_.get_hip_oga_public_hf_config(resolved_tag);
        reject_credential_fields_recursive(public_hf, "");
        validate_license(public_hf);

        const std::string repo_id =
            validate_repo_id(read_required_string(
                public_hf, "repo_id", "hip_oga.public_hf"));
        const std::string revision =
            validate_revision(read_required_string(
                public_hf, "revision", "hip_oga.public_hf"));
        const std::string model_kind =
            read_required_string(public_hf, "model_kind", "hip_oga.public_hf");
        if (model_kind != "text" && model_kind != "vlm") {
            throw std::runtime_error(
                "HIP OGA public pull expects model_kind text or vlm");
        }
        const std::string source_subdir = read_source_subdir(public_hf);
        const std::string endpoint = resolve_endpoint(public_hf);
        const std::vector<FileEntry> files = parse_file_entries(public_hf);
        ensure_file_entries_cover_required_files(files, options);

        const std::filesystem::path install_path =
            ensure_install_path_under_root(supported_models_, options, resolved_tag);
        if (std::filesystem::exists(install_path) && !force_redownload) {
            try {
                supported_models_.validate_hip_oga_model_path_for_tag(
                    resolved_tag, install_path.string());
                verify_payload(install_path, files);
                header_print("FLM", "HIP OGA package already installed: " <<
                    install_path.string());
                return true;
            } catch (const std::exception&) {
                header_print("FLM", "Existing HIP OGA package is incomplete; reinstalling from public Hugging Face.");
            }
        }

        if (std::filesystem::exists(install_path)) {
            require_matching_receipt(install_path, public_hf);
            audit_owned_tree(install_path, files);
        }
        require_download_space(install_path, files);

        const std::filesystem::path temp_path = make_temp_install_path(install_path);
        std::filesystem::remove_all(temp_path);
        std::filesystem::create_directories(temp_path);

        try {
            for (const FileEntry& file : files) {
                const std::string source_path =
                    prefixed_source_path(source_subdir, file.path);
                const std::string file_url =
                    build_resolve_url(endpoint, repo_id, revision, source_path);
                download_with_sha256_retry(file_url, temp_path / file.path, file);
            }
            write_install_metadata(temp_path, public_hf, files, endpoint);

            supported_models_.validate_hip_oga_model_path_for_tag(
                resolved_tag, temp_path.string());

            replace_install_transactionally(temp_path, install_path);
            header_print("FLM", "Installed HIP OGA package: " << install_path.string());
            return true;
        } catch (...) {
            std::filesystem::remove_all(temp_path);
            throw;
        }
    } catch (const std::exception& e) {
        header_print("ERROR", "HIP OGA public pull failed: " + std::string(e.what()));
        return false;
    }
}

}  // namespace flm::hip_oga
