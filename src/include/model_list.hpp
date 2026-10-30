/// \file model_list.hpp
/// \brief model_list class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to manage the model list.
#pragma once
#include "nlohmann/json.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <any>
#include <unordered_set>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include "hip_oga_policy.hpp"
// Only the header_print* macros are needed here. Including utils/utils.hpp
// instead would drag in the NPU runtime headers and make this class impossible
// to unit test without an XRT/HRX toolchain.
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
        /// \param platform the NPU generation this machine has ("aie2p" or
        ///        "dev04"); an entry whose supported_platforms names a
        ///        different generation is pruned, because these artifacts do
        ///        not run on this silicon
        /// \param backends the kernel flows this build links, as backend ids;
        ///        an entry whose flow is not among them is pruned too
        /// \note Two independent axes. The generation says what the silicon can
        ///       run; the backend list says which kernels were linked in. Which
        ///       flow runs an entry is not a third thing to look up: a tag is
        ///       served by exactly one flow and its name says which, so a
        ///       family ending in "-core" is ryzenai-corelib's and anything else
        ///       is the FastFlowLM kernels'. That is why the two never contend
        ///       for a tag -- phi4-mini-it and phi4-mini-it-core are separate
        ///       entries, with their own artifacts and their own silicon.
        model_list(std::string& list_path, std::string& exe_dir,
                   std::string platform = "aie2p",
                   std::vector<std::string> backends = {"flm"}){
            this->list_path = list_path;
            this->platform_ = std::move(platform);
            this->backends_ = std::move(backends);
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

            // Prune before indexing: all_tags must describe what this machine can
            // actually run, so an unsupported tag fails at validation instead of
            // failing much later inside the model backend.
            this->apply_support_filter();

            // Populate all_tags set
            for (const auto& [model_type, sizes] : this->config["models"].items()) {
                // insert default tag without size
                all_tags.insert(model_type);
                for (const auto& [size, model_info] : sizes.items()) {
                    all_tags.insert(model_type + ":" + size);
                }
            }

            // An empty catalog is a real state, not a crash: it is what an
            // install looks like when every entry names other silicon or other
            // kernels. Say so once, here, where the reason is still known --
            // "Models:" followed by nothing tells the user only that something
            // is wrong -- and then let the commands answer for themselves.
            if (all_tags.empty()) {
                header_print_r("ERROR",
                               "No models in " + this->list_path +
                                   " run on this NPU (" + this->platform_ +
                                   ") with the kernels this build has (" +
                                   this->backend_summary() + ")");
            }
        }

        /// \brief the NPU generation this catalog was filtered for
        /// \return "aie2p" or "dev04"
        const std::string& platform() const { return this->platform_; }

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
                        return std::make_pair(new_tag, this->resolve_source(new_tag, model));
                    }
                } 
                if (!model_found) {
                    auto fallback = this->fallback_model();
                    if (last_error_tag != new_tag) {
                        last_error_tag = new_tag;
                        header_print_r("ERROR", "Model not found: " + model_size + " in subset " + model_type);
                        header_print_r("ERROR", "Using default model: " + fallback.first);
                    }
                    return fallback;
                }
            }
            else{
                auto fallback = this->fallback_model();
                if (last_error_tag != new_tag) {
                    last_error_tag = new_tag;
                    header_print_r("ERROR", "Model subset not found: " << model_type << "; using default model: " << fallback.first);
                }
                return fallback;
            }
            return this->fallback_model();
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
                const std::string model_type = new_tag;
                // A family this build has no kernels for (or a misspelled one) has no
                // sizes to pick from. Return the tag untouched so get_model_info
                // reports it rather than dereferencing a null subset.
                const auto& models = this->config["models"];
                if (!models.contains(model_type) || models.at(model_type).empty()) {
                    return new_tag;
                }
                // get the first size in the subset
                std::string model_size = models.at(model_type).begin().key();
                new_tag = model_type + ":" + model_size;
            }
            return new_tag;
        }

        /// \brief get the model root path
        /// \return the model root path, string
        std::string get_model_root_path() const {
            return this->model_root_path;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models(bool hip_oga = false){
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    const std::string tag = model_type + ":" + size;
                    if (hip_oga) {
                        if (!has_hip_oga_model_path(tag) ||
                            model_info["hip_oga"].contains("legacy_alias_for")) continue;
                    } else if (is_hip_oga_only_entry(model_info)) continue;
                    nlohmann::json model_entry = model_info;
                    model_entry["name"] = tag;
                    model_entry["model"] = tag;
                    if (hip_oga) annotate_hip_oga_model(tag, model_entry);
                    response["models"].push_back(std::move(model_entry));
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
                else if (model_type == "d3") continue;   // decision models answer /v1/systemone only
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
        nlohmann::json get_all_models_openai(bool hip_oga = false,
                                              const std::string& direct_model_path = "") {
            nlohmann::json response = {{"object", "list"}, {"data", nlohmann::json::array()}};
            const auto now = static_cast<long long>(std::time(nullptr));
            std::unordered_set<std::string> seen_ids;
            auto add_model = [&](const nlohmann::json& entry) {
                const std::string id = entry.at("model");
                if (id.empty() || !seen_ids.insert(id).second) return;
                nlohmann::json model = {{"id", id}, {"object", "model"},
                    {"created", now}, {"owned_by", "FastFlowLM"}};
                if (hip_oga) {
                    model["backend"] = "hip";
                    model["hip_oga"] = entry.at("hip_oga");
                }
                response["data"].push_back(std::move(model));
            };
            if (hip_oga && !direct_model_path.empty()) {
                add_model(direct_hip_oga_model(direct_model_path));
            }
            const auto models = get_all_models(hip_oga);
            for (const auto& entry : models["models"]) {
                if (hip_oga) {
                    if (!entry.value("installed", false)) continue;
                } else {
                    const std::string tag = entry.at("model");
                    const std::string family = tag.substr(0, tag.find(':'));
                    if (family == "whisper-v3" || family == "embed-gemma" || family == "d3") continue;
                }
                add_model(entry);
            }
            return response;
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

        bool is_model_supported(const std::string& tag) {
            return all_tags.find(tag) != all_tags.end();
        }

        /// \brief pin which source get_model_info resolves a tag to
        /// \param tag the tag of the model
        /// \param preferred true for the preferred source, false for the entry's
        ///        own, nullopt to go back to deciding by what is on disk
        /// \note A pull has to settle the source before it looks at anything,
        ///       since the source decides which files the model has; outside a
        ///       pull the files on disk are what say which source was pulled.
        void choose_source(const std::string& tag, std::optional<bool> preferred) {
            const std::string new_tag = this->rectify_model_tag(tag);
            if (preferred) this->source_choice_[new_tag] = *preferred;
            else this->source_choice_.erase(new_tag);
        }

        /// \brief the files the preferred source has and the entry's own does not
        /// \param tag the tag of the model
        /// \return empty when the entry names no preferred source
        std::vector<std::string> preferred_source_files(const std::string& tag) const {
            const auto entry = this->find_entry(this->rectify_model_tag(tag));
            if (!entry || !entry->contains("preferred_source")) return {};
            return files_only_in(entry->at("preferred_source"), *entry);
        }
        
        /// \brief list HIP packages through the shared catalog enumeration
        nlohmann::json get_all_hip_oga_models() { return get_all_models(true); }

        /// \brief list installed HIP packages and an optional direct OGA directory
        nlohmann::json get_all_models_openai_hip_oga(const std::string& direct_model_path = "") {
            return get_all_models_openai(true, direct_model_path);
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

    private:
        std::string list_path;
        nlohmann::json config;
        std::string model_root_path;
        std::string platform_;
        /// \brief the source a pull has settled on, by tag
        std::map<std::string, bool> source_choice_;
        /// \brief the backend ids this build links
        /// \note Spelled as plain strings rather than reached through
        ///       flm::backend: AutoModel/model_backend.hpp pulls in causal_lm
        ///       and the NPU runtime headers, and keeping those out is what
        ///       lets this class be unit tested without an XRT toolchain. The
        ///       key name mirrors flm::backend::kBackendKey.
        std::vector<std::string> backends_;

        /// \brief the backend ids, rendered for an error message
        std::string backend_summary() const {
            std::string joined;
            for (const auto& id : this->backends_) {
                if (!joined.empty()) joined += ", ";
                joined += id;
            }
            return joined.empty() ? std::string("none") : joined;
        }

        /// \brief the raw catalog entry for a rectified "family:size" tag
        const nlohmann::json* find_entry(const std::string& tag) const {
            const auto colon = tag.find(':');
            if (colon == std::string::npos) return nullptr;
            const auto& models = this->config["models"];
            const auto family = models.find(tag.substr(0, colon));
            if (family == models.end()) return nullptr;
            const auto entry = family->find(tag.substr(colon + 1));
            return entry == family->end() ? nullptr : &*entry;
        }

        static std::vector<std::string> files_only_in(const nlohmann::json& overlay,
                                                      const nlohmann::json& entry) {
            std::vector<std::string> own;
            if (!overlay.contains("files")) return own;
            const auto& base = entry.value("files", nlohmann::json::array());
            for (const auto& file : overlay.at("files")) {
                if (std::find(base.begin(), base.end(), file) == base.end()) {
                    own.push_back(file.get<std::string>());
                }
            }
            return own;
        }

        /// \brief an entry as the source it resolves to
        /// \param tag the rectified tag
        /// \param entry the raw catalog entry
        /// \note "preferred_source" is a JSON merge patch over the entry. It
        ///       applies when a pull chose it, or else when every file only it
        ///       has is on disk: downloads land under their final name only once
        ///       verified, so a present file is a pulled one.
        nlohmann::json resolve_source(const std::string& tag, nlohmann::json entry) const {
            const auto preferred = entry.find("preferred_source");
            if (preferred == entry.end()) return entry;
            const nlohmann::json overlay = *preferred;
            entry.erase(preferred);

            bool use_preferred;
            if (const auto chosen = this->source_choice_.find(tag);
                chosen != this->source_choice_.end()) {
                use_preferred = chosen->second;
            } else {
                const auto own = files_only_in(overlay, entry);
                const auto root = std::filesystem::path(this->model_root_path) /
                                  entry.at("name").get<std::string>();
                use_preferred = !own.empty() &&
                    std::all_of(own.begin(), own.end(), [&](const std::string& file) {
                        std::error_code error;
                        return std::filesystem::is_regular_file(root / file, error);
                    });
            }
            if (use_preferred) entry.merge_patch(overlay);
            return entry;
        }

        /// \brief the kernel flow that runs a tag, read off the tag itself
        /// \param model_type the family part of the tag, e.g. "phi4-mini-it-core"
        /// \return "core" for a family ending in "-core", "flm" otherwise
        /// \note The name is the mechanism, not a label on top of one. A
        ///       corelib model is packaged differently from its FastFlowLM
        ///       namesake -- different weights, different quantization -- so it
        ///       is a different model to pull and a different tag to ask for,
        ///       and there is nothing left for a catalog key to disambiguate.
        static std::string backend_for_family(const std::string& model_type) {
            if (model_type.rfind("hip-", 0) == 0) return "hip";
            static const std::string suffix = "-core";
            if (model_type.size() > suffix.size() &&
                model_type.compare(model_type.size() - suffix.size(),
                                   suffix.size(), suffix) == 0) {
                return "core";
            }
            return "flm";
        }

        /// \brief whether an entry claims support for this machine's silicon
        /// \param entry the size entry
        /// \param tag the "family:size" tag, used only in error messages
        /// \return true if the entry runs on this->platform_
        /// \note An entry that says nothing runs on every generation. The
        ///       shipped catalog names one on every entry, because artifacts
        ///       are built for a generation and saying so is how a model that
        ///       cannot run here fails at validation rather than deep inside a
        ///       kernel; the default is what keeps an older catalog, or a
        ///       hand-made development tree, readable.
        bool entry_supports(const nlohmann::json& entry, const std::string& tag) const {
            const auto supported = entry.find("supported_platforms");
            if (supported == entry.end()) return true;
            if (!supported->is_array() || supported->empty()) {
                throw std::runtime_error(
                    "supported_platforms must be a non-empty array: " + tag);
            }
            for (const auto& value : *supported) {
                if (!value.is_string()) {
                    throw std::runtime_error(
                        "supported_platforms must contain strings: " + tag);
                }
                if (tag.rfind("hip-", 0) == 0 &&
                    value.get<std::string>() == "amdgpu") return true;
                if (value.get<std::string>() == this->platform_) return true;
            }
            return false;
        }

        /// \brief whether this build links a given kernel flow
        bool build_has(const std::string& id) const {
            for (const auto& have : this->backends_) {
                if (have == id) return true;
            }
            return false;
        }

        /// \brief drop entries this build cannot run and settle their backend
        /// \note After this runs the config has exactly the shape it had before
        ///       any of this existed, plus one key: "backend", the flow that
        ///       will run the entry. It is written here rather than read from
        ///       the file because the tag is what decides it, and the tag is
        ///       not part of the entry -- by the time AutoModel sees the entry,
        ///       the name it came from is gone. The generation, by contrast, is
        ///       settled once the entry is kept, so its key is erased.
        void apply_support_filter() {
            std::vector<std::string> empty_families;

            for (auto& [model_type, model_subset] : this->config["models"].items()) {
                std::vector<std::string> unsupported_sizes;
                const std::string backend = backend_for_family(model_type);

                for (auto& [size, model_info] : model_subset.items()) {
                    const std::string tag = model_type + ":" + size;
                    if (!entry_supports(model_info, tag) ||
                        !this->build_has(backend)) {
                        unsupported_sizes.push_back(size);
                        continue;
                    }
                    model_info.erase("supported_platforms");
                    model_info["backend"] = backend;
                }

                for (const auto& size : unsupported_sizes) model_subset.erase(size);
                if (model_subset.empty()) empty_families.push_back(model_type);
            }

            for (const auto& model_type : empty_families) {
                this->config["models"].erase(model_type);
            }
        }

        /// \brief the entry to fall back on when a tag cannot be resolved
        /// \return the fallback tag and its info
        /// \note llama3.2:1b is the historical default, but it is pruned on
        ///       silicon that cannot run it, so fall back to whatever survived.
        std::pair<std::string, nlohmann::json> fallback_model() const {
            const auto& models = this->config["models"];
            if (models.contains("llama3.2") &&
                models["llama3.2"].contains("1b")) {
                return std::make_pair("llama3.2:1b",
                                      this->resolve_source("llama3.2:1b", models["llama3.2"]["1b"]));
            }
            for (const auto& [model_type, model_subset] : models.items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    const std::string tag = model_type + ":" + size;
                    return std::make_pair(tag, this->resolve_source(tag, model_info));
                }
            }
            throw std::runtime_error("No models available for NPU platform " +
                                     this->platform_ + " with backends " +
                                     this->backend_summary());
        }

        /// \brief attach HIP package status without changing the native catalog entry
        void annotate_hip_oga_model(const std::string& tag, nlohmann::json& entry) const {
            const std::string policy = get_hip_oga_offload_policy(tag);
            entry["backend"] = "hip";
            entry["installed"] = false;
            entry["offload_policy"] = policy;
            auto& hip = entry["hip_oga"];
            hip["offload_policy"] = policy;
            hip["qualification"] = policy;
            const std::string warning = flm::hip_oga::policy_warning(policy);
            if (!warning.empty()) hip["policy_warning"] = warning;
            try {
                const std::string path = get_hip_oga_model_path(tag);
                hip["pipeline_sections"] = flm::hip_oga::pipeline_sections(
                    read_hip_oga_json_object(tag, std::filesystem::path(path) / "genai_config.json"));
                hip["resolved_config_path"] = path;
                entry["installed"] = true;
            } catch (const std::exception&) {
                // Missing/incomplete packages stay listable, but not installed.
            }
        }

        static nlohmann::json direct_hip_oga_model(const std::string& path) {
            const std::string policy = flm::hip_oga::direct_model_offload_policy();
            nlohmann::json hip = {{"offload_policy", policy},
                {"qualification", flm::hip_oga::kUnqualifiedPolicy},
                {"qualification_warning", flm::hip_oga::policy_warning(flm::hip_oga::kUnqualifiedPolicy)}};
            const std::string warning = flm::hip_oga::policy_warning(policy);
            if (!warning.empty()) hip["policy_warning"] = warning;
            try {
                hip["pipeline_sections"] = flm::hip_oga::pipeline_sections(
                    read_hip_oga_json_object("direct OGA directory",
                        std::filesystem::path(path) / "genai_config.json"));
            } catch (const std::exception&) {
                // A direct path may have disappeared since the server started.
            }
            return {{"model", path}, {"hip_oga", std::move(hip)}};
        }

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
