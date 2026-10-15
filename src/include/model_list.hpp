/// \file model_list.hpp
/// \brief model_list class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to manage the model list.
#pragma once
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <any>
#include <unordered_set>
#include <filesystem>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <cctype>
#include "hip_oga_policy.hpp"
// Only the header_print* macros are needed here. Including utils/utils.hpp
// would drag NPU runtime headers into catalog-only checks.
#include "utils/debug_utils.hpp"

/// \note This class is used to manage the model list.
class model_list {
    public:
        std::unordered_set<std::string> all_tags;
        /// \brief constructor
        model_list(){}


        /// \brief constructor
        /// \param list_path the path to the model list
        /// \param exe_dir the executable directory for resolving relative paths
        model_list(std::string& list_path, std::string& exe_dir){
            this->list_path = list_path;
            std::ifstream config_file(list_path);
            if (!config_file.is_open()) {
                std::cerr << "Failed to open config file: " << list_path << std::endl;
                exit(1);
            }
            this->config = nlohmann::json::parse(config_file);
            // Resolve model_root_path relative to executable directory
            std::string relative_model_path = this->config["model_path"];
            std::filesystem::path root_path = std::filesystem::path(exe_dir) / relative_model_path;
            this->model_root_path = root_path.string();
            config_file.close();

            // Populate all_tags set
            for (const auto& [model_type, sizes] : this->config["models"].items()) {
                // insert default tag without size
                all_tags.insert(model_type);
                for (const auto& [size, model_info] : sizes.items()) {
                    all_tags.insert(model_type + ":" + size);
                }
            }
        }

        /// \brief get the model info
        /// \param tag the tag of the model
        /// \return the model info
        std::pair<std::string, nlohmann::json> get_model_info(const std::string tag) const {
            static std::string last_error_tag = "";
            std::string new_tag = rectify_model_tag(tag);
            bool model_found = false;
            // get model type, the string before ':' in the tag
            std::string model_type;
            std::string model_size;

            if (new_tag.find(':') != std::string::npos) {
                model_type = new_tag.substr(0, new_tag.find(':'));
                model_size = new_tag.substr(new_tag.find(':') + 1);
            }
            else {
                model_type = new_tag;
                model_size = "";
            }
            
            // find the model subset first, compare with the key of the model
            bool model_subset_found = false;
            for (const auto& [key, model] : this->config["models"].items()) {
                if (key == model_type) {
                    model_subset_found = true;
                    break;
                }
            } 
            if (model_subset_found) {
                bool model_found = false;
                for (const auto& [key, model] : this->config["models"][model_type].items()) {
                    if (key == model_size) { // if the size is found, return the model
                        model_found = true;
                        return std::make_pair(new_tag, model);
                    }
                } 
                if (!model_found) {
                    if (last_error_tag != new_tag) {
                        last_error_tag = new_tag;
                        header_print_r("ERROR", "Model not found: " + model_size + " in subset " + model_type);
                        header_print_r("ERROR", "Using default model: llama3.2-1B");
                    }
                    return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
                }
            }
            else{
                if (last_error_tag != new_tag) {
                    last_error_tag = new_tag;
                    header_print_r("ERROR", "Model subset not found: " << model_type << "; using default model: llama3.2-1B");
                }
                return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
            }
            return std::make_pair("llama3.2:1b", this->config["models"]["llama3.2"]["1b"]);
        }

        /// \brief cut the tag, some program adds a prefix to the tag, like "Ollama/llama3.2-1B", we need to cut the prefix
        /// \param tag the tag of the model
        /// \return the model type, string
        std::string cut_tag(const std::string tag) const {
            std::string new_tag = tag;
            if (tag.find('/') != std::string::npos) {
                new_tag = tag.substr(tag.find('/') + 1);
            }
            return new_tag;
        }

        /// \brief rectify the model tag, remove / and replace with actuall tag if size is not specified
        /// \param original_tag the original tag of the model
        /// \return the rectified model tag, string
        std::string rectify_model_tag(const std::string original_tag) const {
            std::string new_tag = this->cut_tag(original_tag);
            // check if size is specified
            if (new_tag.find(':') == std::string::npos) {
                // get the first size in the subset
                std::string model_type = new_tag;
                std::string model_size = this->config["models"][model_type].begin().key();
                new_tag = model_type + ":" + model_size;
            }
            return new_tag;
        }

        /// \brief get the model root path
        /// \return the model root path, string
        std::string get_model_root_path(){
            return this->model_root_path;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models(){
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    if (is_hip_oga_only_entry(model_info)) continue;
                    nlohmann::json model_entry = model_info;
                    model_entry["name"] = model_type + ":" + size;
                    model_entry["model"] = model_type + ":" + size;
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get HIP OGA-capable model-list entries with OGA install status
        /// \return model-list entries whose metadata contains hip_oga.config_path
        nlohmann::json get_all_hip_oga_models(){
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    const std::string tag = model_type + ":" + size;
                    if (!has_hip_oga_model_path(tag)) continue;

                    nlohmann::json model_entry = model_info;
                    model_entry["name"] = tag;
                    model_entry["model"] = tag;
                    model_entry["backend"] = "hip";
                    model_entry["installed"] = false;
                    const std::string offload_policy =
                        get_hip_oga_offload_policy(tag);
                    model_entry["offload_policy"] = offload_policy;
                    model_entry["hip_oga"]["offload_policy"] = offload_policy;
                    model_entry["hip_oga"]["qualification"] = offload_policy;
                    const std::string warning =
                        flm::hip_oga::policy_warning(offload_policy);
                    if (!warning.empty()) {
                        model_entry["hip_oga"]["policy_warning"] = warning;
                    }

                    try {
                        const std::string resolved_path = get_hip_oga_model_path(tag);
                        model_entry["installed"] = true;
                        model_entry["hip_oga"]["resolved_config_path"] = resolved_path;
                        const nlohmann::json genai_config =
                            read_hip_oga_json_object(tag,
                                                     std::filesystem::path(resolved_path) /
                                                         "genai_config.json");
                        model_entry["hip_oga"]["pipeline_sections"] =
                            flm::hip_oga::pipeline_sections(genai_config);
                    } catch (const std::exception&) {
                        // Missing or incomplete OGA packages are still listable,
                        // but they should not be treated as installed.
                    }

                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_ollama() {
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    if (is_hip_oga_only_entry(model_info)) continue;
                    nlohmann::json model_entry = {
                        {"name", model_type + ":" + size},
                        {"model", model_type + ":" + size},
                        {"details", {
                            {"family", model_info["details"]["family"]},
                            {"parameter_size", model_info["details"]["parameter_size"]},
                            {"quantization_level", model_info["details"]["quantization_level"]}
                        }}
                    };
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_openai() {
            nlohmann::json response = {
                {"object", "list"},
                {"data", nlohmann::json::array()},
                {"object", "list" }
            };

            std::time_t now = std::time(nullptr);

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    if (is_hip_oga_only_entry(model_info)) continue;
                    // id uses the same "type:size" convention; created uses current epoch seconds
                    nlohmann::json model_entry = {
                        {"id", model_type + ":" + size},
                        {"object", "model"},
                        {"created", static_cast<long long>(now)},
                        {"owned_by", "FastFlowLM"}
                    };
                    response["data"].push_back(model_entry);
                }
            }

            return response;
        }

        /// \brief get OpenAI model-list entries that are loadable by backend=hip
        /// \param direct_model_path optional direct OGA directory used to start the server
        /// \return OpenAI /v1/models response scoped to HIP OGA-loadable models
        nlohmann::json get_all_models_openai_hip_oga(
            const std::string& direct_model_path = "") {
            nlohmann::json response = {
                {"object", "list"},
                {"data", nlohmann::json::array()}
            };
            std::unordered_set<std::string> seen_ids;
            std::time_t now = std::time(nullptr);

            auto add_model = [&](const std::string& id,
                                 const nlohmann::json& hip_oga = nlohmann::json::object()) {
                if (id.empty() || seen_ids.find(id) != seen_ids.end()) return;
                seen_ids.insert(id);
                nlohmann::json model = {
                    {"id", id},
                    {"object", "model"},
                    {"created", static_cast<long long>(now)},
                    {"owned_by", "FastFlowLM"},
                    {"backend", "hip"}
                };
                if (!hip_oga.empty()) {
                    model["hip_oga"] = hip_oga;
                }
                response["data"].push_back(std::move(model));
            };

            if (!direct_model_path.empty()) {
                const std::string direct_policy =
                    flm::hip_oga::direct_model_offload_policy();
                nlohmann::json direct_hip_oga = {
                    {"offload_policy", direct_policy},
                    {"qualification", flm::hip_oga::kUnqualifiedPolicy},
                    {"qualification_warning",
                     flm::hip_oga::policy_warning(flm::hip_oga::kUnqualifiedPolicy)}
                };
                const std::string direct_policy_warning =
                    flm::hip_oga::policy_warning(direct_policy);
                if (!direct_policy_warning.empty()) {
                    direct_hip_oga["policy_warning"] = direct_policy_warning;
                }
                try {
                    const nlohmann::json genai_config =
                        read_hip_oga_json_object("direct OGA directory",
                                                 std::filesystem::path(direct_model_path) /
                                                     "genai_config.json");
                    direct_hip_oga["pipeline_sections"] =
                        flm::hip_oga::pipeline_sections(genai_config);
                } catch (const std::exception&) {
                    // Direct paths are optional list entries; omit section
                    // metadata if the package disappeared between checks.
                }
                add_model(direct_model_path, direct_hip_oga);
            }

            const nlohmann::json hip_models = get_all_hip_oga_models();
            for (const auto& model : hip_models["models"]) {
                if (!model.value("installed", false)) continue;
                add_model(model.value("model", std::string()),
                          model.value("hip_oga", nlohmann::json::object()));
            }

            return response;
        }

        /// \brief get the declared HIP OGA offload policy for a model-list tag
        /// \param tag the model tag
        /// \return strict_gpu, hybrid_gpu, or unqualified
        std::string get_hip_oga_offload_policy(const std::string& tag) const {
            if (all_tags.find(tag) == all_tags.end()) {
                return flm::hip_oga::kUnqualifiedPolicy;
            }
            auto [new_tag_unused, model_info] = this->get_model_info(tag);
            const auto options = model_info.find("hip_oga");
            if (options == model_info.end()) {
                return flm::hip_oga::kUnqualifiedPolicy;
            }
            return flm::hip_oga::metadata_offload_policy(*options);
        }

        /// \brief get the model path
        /// \param tag the tag of the model
        /// \return the model path, string
        std::string get_model_path(const std::string& tag){
            std::string new_tag = this->rectify_model_tag(tag);
            auto [new_tag_unused, model_info] = this->get_model_info(new_tag);
            std::string model_name = model_info["name"];
            std::filesystem::path full_path = std::filesystem::path(this->model_root_path) / model_name;
            return full_path.string();
        }

        /// \brief whether a model-list tag has HIP OGA routing metadata
        /// \param tag the model tag
        /// \return true if the entry includes hip_oga.config_path
        bool has_hip_oga_model_path(const std::string& tag) const {
            if (all_tags.find(tag) == all_tags.end()) return false;
            auto [new_tag_unused, model_info] = this->get_model_info(tag);
            const auto options = model_info.find("hip_oga");
            if (options == model_info.end() || !options->is_object()) return false;
            const auto configured_path = options->find("config_path");
            return configured_path != options->end() &&
                configured_path->is_string() &&
                !configured_path->get<std::string>().empty();
        }

        /// \brief whether a model-list entry should only be used with backend=hip
        bool is_hip_oga_only_model(const std::string& tag) const {
            if (all_tags.find(tag) == all_tags.end()) return false;
            auto [new_tag_unused, model_info] = this->get_model_info(tag);
            const auto options = model_info.find("hip_oga");
            return options != model_info.end() &&
                options->is_object() &&
                options->value("hip_only", false);
        }

        /// \brief get the HIP OGA model-list metadata for a tag
        /// \param tag the model tag
        /// \return the hip_oga metadata object
        nlohmann::json get_hip_oga_model_options(const std::string& tag) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            const auto options = model_info.find("hip_oga");
            if (options == model_info.end() || !options->is_object()) {
                throw std::runtime_error(
                    "Backend 'hip' needs model-list metadata hip_oga for tag '" +
                    new_tag + "'");
            }
            return *options;
        }

        /// \brief whether a HIP OGA tag is supported by the current runtime
        bool is_hip_oga_runtime_supported(const std::string& tag) const {
            if (!has_hip_oga_model_path(tag)) return false;
            const nlohmann::json options = get_hip_oga_model_options(tag);
            const auto unsupported_reason = options.find("unsupported_reason");
            return unsupported_reason == options.end() ||
                !unsupported_reason->is_string() ||
                unsupported_reason->get<std::string>().empty();
        }

        /// \brief fail when a HIP OGA tag is cataloged but unsupported here
        void require_hip_oga_runtime_supported(const std::string& tag) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            const nlohmann::json options = get_hip_oga_model_options(new_tag);
            const auto unsupported_reason = options.find("unsupported_reason");
            if (unsupported_reason != options.end() &&
                unsupported_reason->is_string() &&
                !unsupported_reason->get<std::string>().empty()) {
                throw std::runtime_error(
                    "Backend 'hip' does not support tag '" + new_tag + "': " +
                    unsupported_reason->get<std::string>());
            }
        }

        /// \brief get the resolved HIP OGA install path without validating files
        /// \param tag the model tag
        /// \return local OGA package directory for the tag
        std::string get_hip_oga_install_path(const std::string& tag) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            const nlohmann::json options = get_hip_oga_model_options(new_tag);
            const auto configured_path = options.find("config_path");
            if (configured_path == options.end() || !configured_path->is_string() ||
                configured_path->get<std::string>().empty()) {
                throw std::runtime_error(
                    "Backend 'hip' needs non-empty model-list metadata hip_oga.config_path for tag '" +
                    new_tag + "'");
            }
            return resolve_hip_oga_config_path(configured_path->get<std::string>()).string();
        }

        /// \brief whether a HIP OGA tag has public Hugging Face install metadata
        bool has_hip_oga_public_hf(const std::string& tag) const {
            if (!has_hip_oga_model_path(tag)) return false;
            const nlohmann::json options = get_hip_oga_model_options(tag);
            const auto public_hf = options.find("public_hf");
            return public_hf != options.end() && public_hf->is_object();
        }

        /// \brief get validated public Hugging Face metadata for a HIP OGA tag
        nlohmann::json get_hip_oga_public_hf_config(const std::string& tag) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            const nlohmann::json options = get_hip_oga_model_options(new_tag);
            const auto public_hf = options.find("public_hf");
            if (public_hf == options.end() || !public_hf->is_object()) {
                throw std::runtime_error(
                    "Backend 'hip' pull needs model-list metadata hip_oga.public_hf for tag '" +
                    new_tag + "'");
            }

            for (const char* key : {"repo_id", "revision", "license", "model_kind"}) {
                const auto value = public_hf->find(key);
                if (value == public_hf->end() || !value->is_string() ||
                    value->get<std::string>().empty()) {
                    throw std::runtime_error(
                        "Backend 'hip' pull needs string hip_oga.public_hf." +
                        std::string(key) + " for tag '" + new_tag + "'");
                }
            }

            const auto endpoint_env = public_hf->find("endpoint_env");
            if (endpoint_env != public_hf->end() &&
                (!endpoint_env->is_string() ||
                 endpoint_env->get<std::string>().empty())) {
                throw std::runtime_error(
                    "Backend 'hip' pull needs string hip_oga.public_hf.endpoint_env for tag '" +
                    new_tag + "'");
            }
            if (public_hf->contains("endpoint")) {
                throw std::runtime_error(
                    "Backend 'hip' pull does not allow embedded hip_oga.public_hf.endpoint URLs for tag '" +
                    new_tag + "'");
            }

            const auto source_subdir = public_hf->find("source_subdir");
            if (source_subdir != public_hf->end() &&
                !source_subdir->is_string()) {
                throw std::runtime_error(
                    "Backend 'hip' pull needs string hip_oga.public_hf.source_subdir for tag '" +
                    new_tag + "'");
            }

            const auto files = public_hf->find("files");
            if (files == public_hf->end() || !files->is_array() || files->empty()) {
                throw std::runtime_error(
                    "Backend 'hip' pull needs non-empty array hip_oga.public_hf.files for tag '" +
                    new_tag + "'");
            }

            return *public_hf;
        }

        /// \brief get the resolved OGA model directory for a HIP model-list tag
        /// \param tag the model tag
        /// \return directory containing genai_config.json
        /// \note Relative hip_oga.config_path values resolve below
        ///       FLM_HIP_OGA_MODEL_ROOT when set, otherwise below model_path.
        std::string get_hip_oga_model_path(const std::string& tag) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            require_hip_oga_runtime_supported(new_tag);
            const std::filesystem::path model_path(get_hip_oga_install_path(new_tag));
            validate_hip_oga_package(new_tag, model_path,
                                     get_hip_oga_model_options(new_tag));
            return model_path.string();
        }

        /// \brief validate an explicit local OGA package path against a tag's HIP metadata
        /// \param tag the model tag
        /// \param model_path directory containing genai_config.json
        void validate_hip_oga_model_path_for_tag(const std::string& tag,
                                                 const std::string& model_path) const {
            auto [new_tag, model_info] = this->get_model_info(tag);
            require_hip_oga_runtime_supported(new_tag);
            validate_hip_oga_package(new_tag,
                                     std::filesystem::path(model_path),
                                     get_hip_oga_model_options(new_tag));
        }

        /// \brief validate a direct OGA directory used outside the model list
        /// \param model_path directory containing genai_config.json
        static void validate_hip_oga_model_dir(const std::string& model_path) {
            validate_hip_oga_package("direct OGA directory",
                                     std::filesystem::path(model_path),
                                     nlohmann::json::object());
        }

        bool is_model_supported(const std::string& tag) const {
            return all_tags.find(tag) != all_tags.end();
        }
        
    private:
        std::string list_path;
        nlohmann::json config;
        std::string model_root_path;

        static std::string getenv_string(const char* name) {
#ifdef _WIN32
            char* value = nullptr;
            size_t len = 0;
            if (_dupenv_s(&value, &len, name) == 0 && value != nullptr) {
                std::string result(value);
                free(value);
                return result;
            }
            return {};
#else
            const char* value = std::getenv(name);
            return value && *value ? std::string(value) : std::string();
#endif
        }

        std::filesystem::path resolve_hip_oga_config_path(
            const std::string& configured_path) const {
            std::filesystem::path path(configured_path);
            if (path.is_absolute()) return path;

            const std::string hip_model_root =
                getenv_string("FLM_HIP_OGA_MODEL_ROOT");
            if (!hip_model_root.empty()) {
                return std::filesystem::path(hip_model_root) / path;
            }
            return std::filesystem::path(this->model_root_path) / path;
        }

        static bool hip_oga_file_exists(const std::filesystem::path& path) {
            std::error_code ec;
            return std::filesystem::exists(path, ec) && !ec;
        }

        static bool is_hip_oga_only_entry(const nlohmann::json& model_info) {
            const auto options = model_info.find("hip_oga");
            return options != model_info.end() &&
                options->is_object() &&
                options->value("hip_only", false);
        }

        static bool hip_oga_drive_relative_or_rooted(const std::string& value) {
            if (value.size() >= 2 &&
                std::isalpha(static_cast<unsigned char>(value[0])) &&
                value[1] == ':') {
                return true;
            }
            return value.rfind("//", 0) == 0 || value.rfind("\\\\", 0) == 0;
        }

        static bool is_safe_hip_oga_package_relative_path(const std::string& value) {
            if (value.empty() || value.find('\\') != std::string::npos ||
                value.find('?') != std::string::npos ||
                value.find('#') != std::string::npos ||
                value.front() == '/' || hip_oga_drive_relative_or_rooted(value)) {
                return false;
            }
            size_t cursor = 0;
            while (cursor <= value.size()) {
                const size_t slash = value.find('/', cursor);
                const std::string segment =
                    value.substr(cursor, slash == std::string::npos ? slash : slash - cursor);
                if (segment.empty() || segment == "." || segment == "..") {
                    return false;
                }
                if (slash == std::string::npos) break;
                cursor = slash + 1;
            }
            return true;
        }

        static nlohmann::json read_hip_oga_json_object(
            const std::string& tag,
            const std::filesystem::path& path) {
            std::ifstream file(path, std::ios::in | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    path.parent_path().string() + "', but " +
                    path.filename().string() + " is missing.");
            }
            auto parsed = nlohmann::json::parse(file, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object()) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    path.parent_path().string() + "', but " +
                    path.filename().string() + " is not a JSON object.");
            }
            return parsed;
        }

        static bool hip_oga_token_id_present(const nlohmann::json& object,
                                             const char* key) {
            if (!object.contains(key)) return false;
            const auto& value = object[key];
            if (value.is_number_integer() || value.is_number_unsigned()) return true;
            if (!value.is_array()) return false;
            for (const auto& token : value) {
                if (token.is_number_integer() || token.is_number_unsigned()) {
                    return true;
                }
            }
            return false;
        }

        static void require_hip_oga_file(const std::string& tag,
                                         const std::filesystem::path& model_path,
                                         const std::string& relative_path) {
            const std::filesystem::path package_relative_path(relative_path);
            if (!is_safe_hip_oga_package_relative_path(relative_path) ||
                package_relative_path.is_absolute()) {
                throw std::runtime_error(
                    "Backend 'hip' model-list metadata hip_oga.required_files for tag '" +
                    tag + "' must contain package-relative paths.");
            }
            const std::filesystem::path required_path =
                model_path / package_relative_path;
            if (!hip_oga_file_exists(required_path)) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    model_path.string() + "', but required file '" +
                    relative_path + "' is missing.");
            }
        }

        static void validate_hip_oga_package(const std::string& tag,
                                             const std::filesystem::path& model_path,
                                             const nlohmann::json& options) {
            require_hip_oga_file(tag, model_path, "genai_config.json");
            require_hip_oga_file(tag, model_path, "tokenizer.json");
            require_hip_oga_file(tag, model_path, "tokenizer_config.json");

            const auto required_files = options.find("required_files");
            if (required_files != options.end()) {
                if (!required_files->is_array()) {
                    throw std::runtime_error(
                        "Backend 'hip' model-list metadata hip_oga.required_files for tag '" +
                        tag + "' must be an array.");
                }
                for (const auto& required_file : *required_files) {
                    if (!required_file.is_string()) {
                        throw std::runtime_error(
                            "Backend 'hip' model-list metadata hip_oga.required_files for tag '" +
                            tag + "' must contain strings.");
                    }
                    require_hip_oga_file(tag, model_path, required_file.get<std::string>());
                }
            }

            const nlohmann::json genai_config =
                read_hip_oga_json_object(tag, model_path / "genai_config.json");
            const nlohmann::json tokenizer_config =
                read_hip_oga_json_object(tag, model_path / "tokenizer_config.json");
            if (!tokenizer_config.contains("eos_token") ||
                !tokenizer_config["eos_token"].is_string()) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    model_path.string() +
                    "', but tokenizer_config.json must contain string eos_token.");
            }
            const bool has_eos_token_id =
                hip_oga_token_id_present(tokenizer_config, "eos_token_id") ||
                (genai_config.contains("model") && genai_config["model"].is_object() &&
                 hip_oga_token_id_present(genai_config["model"], "eos_token_id"));
            if (!has_eos_token_id) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    model_path.string() +
                    "', but eos_token_id is missing from tokenizer_config.json and genai_config.json.");
            }

            const bool has_template_file =
                hip_oga_file_exists(model_path / "chat_template.jinja");
            const bool has_inline_template =
                tokenizer_config.contains("chat_template") &&
                tokenizer_config["chat_template"].is_string() &&
                !tokenizer_config["chat_template"].get<std::string>().empty();
            if (!has_template_file && !has_inline_template) {
                throw std::runtime_error(
                    "Backend 'hip' resolved tag '" + tag + "' to package '" +
                    model_path.string() +
                    "', but it needs chat_template.jinja or tokenizer_config.json chat_template.");
            }
        }

};
