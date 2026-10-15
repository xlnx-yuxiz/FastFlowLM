/// \file hip_oga_model.cpp
/// \brief Prototype AutoModel backed by ONNX Runtime GenAI over HIP EP
#include "AutoModel/hip_oga_model.hpp"
#include "AutoModel/hip_oga_prefix_cache.hpp"
#include "AutoModel/hip_oga_tool_parser.hpp"
#include "base64.hpp"
#include "hip_oga_policy.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace {

std::filesystem::path genai_config_path_for(const std::string& model_path) {
    return std::filesystem::path(model_path) / "genai_config.json";
}

int read_hip_oga_context_length(const std::string& model_path) {
    const auto config_path = genai_config_path_for(model_path);
    std::ifstream config_file(config_path, std::ios::in | std::ios::binary);
    if (!config_file.is_open()) return 4096;

    auto config = nlohmann::ordered_json::parse(config_file, nullptr, false);
    if (config.is_discarded()) return 4096;
    if (config.contains("model") && config["model"].contains("context_length") &&
        config["model"]["context_length"].is_number_integer()) {
        return config["model"]["context_length"].get<int>();
    }
    if (config.contains("search") && config["search"].contains("max_length") &&
        config["search"]["max_length"].is_number_integer()) {
        return config["search"]["max_length"].get<int>();
    }
    return 4096;
}

nlohmann::json read_json_file_if_present(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) return nlohmann::json::object();

    auto parsed = nlohmann::json::parse(file, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return nlohmann::json::object();
    }
    return parsed;
}

template <typename JsonObject>
bool has_integer_field(const JsonObject& object, const char* key) {
    return object.contains(key) &&
        (object[key].is_number_integer() || object[key].is_number_unsigned());
}

void load_hip_oga_lm_config(LM_Config& lm_config,
                            const std::string& model_path,
                            const nlohmann::ordered_json& model_info) {
    const std::filesystem::path model_root(model_path);
    lm_config.model_path = model_path;
    lm_config.model_name = model_root.filename().string();
    lm_config.exec_path.clear();
    lm_config.flm_version = "0.0.0";

    lm_config._json_config = read_json_file_if_present(model_root / "config.json");
    const auto genai_config = read_json_file_if_present(genai_config_path_for(model_path));

    if (!has_integer_field(lm_config._json_config, "vocab_size")) {
        if (genai_config.contains("model") && genai_config["model"].is_object() &&
            has_integer_field(genai_config["model"], "vocab_size")) {
            lm_config._json_config["vocab_size"] =
                genai_config["model"]["vocab_size"].get<std::uint32_t>();
        } else if (has_integer_field(model_info, "vocab_size")) {
            lm_config._json_config["vocab_size"] =
                model_info["vocab_size"].get<std::uint32_t>();
        } else {
            throw std::runtime_error(
                "HIP OGA backend could not resolve vocab_size from config.json or genai_config.json: " +
                model_path);
        }
    }

    if (lm_config._json_config.contains("flm_version") &&
        lm_config._json_config["flm_version"].is_string()) {
        lm_config.flm_version = lm_config._json_config["flm_version"].get<std::string>();
    } else {
        lm_config._json_config["flm_version"] = lm_config.flm_version;
    }
    if (!lm_config._json_config.contains("is_vlm")) {
        lm_config._json_config["is_vlm"] = false;
    }
    if (!lm_config._json_config.contains("is_audio")) {
        lm_config._json_config["is_audio"] = false;
    }
    const std::string policy = flm::hip_oga::runtime_offload_policy(
        nlohmann::json::parse(model_info.dump()));
    lm_config._json_config["hip_oga_offload_policy"] = policy;
    const std::string warning = flm::hip_oga::policy_warning(policy);
    if (!warning.empty()) {
        lm_config._json_config["hip_oga_policy_warning"] = warning;
    }
}

void append_token_ids(std::vector<int>& out, const nlohmann::json& value) {
    if (value.is_array()) {
        for (const auto& token : value) {
            if (token.is_number_integer() || token.is_number_unsigned()) {
                out.push_back(token.get<int>());
            }
        }
    } else if (value.is_number_integer() || value.is_number_unsigned()) {
        out.push_back(value.get<int>());
    }
}

StreamEventType to_auto_stream_type(flm::hip_oga::StreamEventType type) {
    switch (type) {
        case flm::hip_oga::StreamEventType::WAITING:
            return StreamEventType::WAITING;
        case flm::hip_oga::StreamEventType::CONTENT:
            return StreamEventType::CONTENT;
        case flm::hip_oga::StreamEventType::REASONING:
            return StreamEventType::REASONING;
        case flm::hip_oga::StreamEventType::TOOL_DONE:
            return StreamEventType::TOOL_DONE;
        case flm::hip_oga::StreamEventType::ERROR_EVENT:
            return StreamEventType::ERROR_EVENT;
    }
    return StreamEventType::ERROR_EVENT;
}

flm::hip_oga::StreamEventType to_hip_stream_type(StreamEventType type) {
    switch (type) {
        case StreamEventType::WAITING:
            return flm::hip_oga::StreamEventType::WAITING;
        case StreamEventType::CONTENT:
            return flm::hip_oga::StreamEventType::CONTENT;
        case StreamEventType::REASONING:
            return flm::hip_oga::StreamEventType::REASONING;
        case StreamEventType::TOOL_DONE:
            return flm::hip_oga::StreamEventType::TOOL_DONE;
        case StreamEventType::ERROR_EVENT:
            return flm::hip_oga::StreamEventType::ERROR_EVENT;
    }
    return flm::hip_oga::StreamEventType::ERROR_EVENT;
}

StreamResult to_auto_stream_result(const flm::hip_oga::ParsedStreamEvent& parsed) {
    StreamResult result;
    result.type = to_auto_stream_type(parsed.type);
    result.content = parsed.content;
    result.tool_id = parsed.tool_id;
    result.tool_name = parsed.tool_name;
    result.tool_args_str = parsed.tool_args_str;
    result.error_message = parsed.error_message;
    return result;
}

}  // namespace

bool HipOgaModel::is_model_dir(const std::string& model_path) {
    if (model_path.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(genai_config_path_for(model_path), ec) && !ec;
}

json HipOgaModel::model_info_for_path(const std::string& model_path) {
    const int context_length = read_hip_oga_context_length(model_path);
    const auto genai_config = read_json_file_if_present(genai_config_path_for(model_path));
    json model_info = json::object();
    model_info["default_context_length"] = context_length;
    model_info["max_prefill_len"] = context_length;
    if (genai_config.contains("model") && genai_config["model"].is_object() &&
        has_integer_field(genai_config["model"], "vocab_size")) {
        model_info["vocab_size"] =
            genai_config["model"]["vocab_size"].get<std::uint32_t>();
    }
    model_info["details"] = json{{"family", "llama3"}};
    model_info["hip_oga"] = json{
        {"config_path", model_path},
        {"offload_policy", flm::hip_oga::direct_model_offload_policy()},
        {"qualification", flm::hip_oga::kUnqualifiedPolicy},
        {"pipeline_sections", flm::hip_oga::pipeline_sections(genai_config)}
    };
    return model_info;
}

#if defined(FLM_ENABLE_HIP_OGA)

#include "ort_genai.h"

namespace {

constexpr std::uint32_t kDefaultRequestedMaxTokens = 4096;
constexpr std::uint32_t kHipOgaSharedGenerateLookaheadTokens = 1;

bool hip_oga_prefix_cache_enabled() {
    return flm::hip_oga::prefix_cache_env_enabled(
        std::getenv("FLM_HIP_OGA_PREFIX_CACHE"));
}

std::string get_hip_oga_string(const nlohmann::ordered_json& model_info,
                               const char* key,
                               std::string fallback) {
    const auto options = model_info.find("hip_oga");
    if (options == model_info.end() || !options->is_object()) return fallback;
    const auto value = options->find(key);
    if (value == options->end() || !value->is_string()) return fallback;
    return value->get<std::string>();
}

nlohmann::json get_hip_oga_provider_options(
    const nlohmann::ordered_json& model_info) {
    const auto options = model_info.find("hip_oga");
    if (options == model_info.end() || !options->is_object()) {
        return nlohmann::json::object();
    }
    const auto provider_options = options->find("provider_options");
    if (provider_options == options->end() || !provider_options->is_object()) {
        return nlohmann::json::object();
    }
    return *provider_options;
}

std::filesystem::path resolve_oga_config_path(
    const std::string& model_path,
    const nlohmann::ordered_json& model_info) {
    const auto override_path =
        get_hip_oga_string(model_info, "config_path", "");
    if (!override_path.empty()) return override_path;

    const std::filesystem::path model_root(model_path);
    const std::filesystem::path genai_config = model_root / "genai_config.json";
    if (!std::filesystem::exists(genai_config)) {
        throw std::runtime_error(
            "HIP OGA backend needs an OGA model directory containing genai_config.json: " +
            model_root.string());
    }
    return model_root;
}

std::vector<int32_t> to_int32_tokens(const std::vector<int>& tokens) {
    std::vector<int32_t> converted;
    converted.reserve(tokens.size());
    for (int token : tokens) {
        if (token < 0) {
            throw std::runtime_error("HIP OGA backend received a negative token id");
        }
        converted.push_back(static_cast<int32_t>(token));
    }
    return converted;
}

void ensure_oga_process_handle() {
    static OgaHandle handle;
}

struct HipOgaImageInput {
    std::string payload;
    input_payload_type_t payload_type = BASE64_ENCODED;
};

struct HipOgaLoadedImages {
    std::vector<std::vector<std::uint8_t>> buffers;
    std::unique_ptr<OgaImages> images;
};

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
        value.compare(0, prefix.size(), prefix) == 0;
}

bool is_remote_url_like(const std::string& value) {
    return starts_with(value, "http://") || starts_with(value, "https://");
}

bool path_exists_without_throw(const std::string& value) {
    std::error_code ec;
    const bool exists = std::filesystem::exists(value, ec);
    return exists && !ec;
}

std::string strip_image_data_url_prefix(std::string payload) {
    if (!starts_with(payload, "data:")) return payload;
    const std::size_t comma = payload.find(',');
    if (comma == std::string::npos ||
        payload.find(";base64") == std::string::npos ||
        !starts_with(payload, "data:image/")) {
        throw std::runtime_error(
            "HIP OGA backend supports only base64 data:image inputs");
    }
    return payload.substr(comma + 1);
}

void reject_hip_oga_audio_inputs(const lm_uniform_input_t& input) {
    if (!input.audios.empty() || !input.audio_payload_types.empty()) {
        throw std::runtime_error(
            "HIP OGA backend supports image inputs only; audio inputs are not supported");
    }
    for (const auto& message : input.messages) {
        if (!message.is_object()) continue;
        if (message.contains("audios") && !message["audios"].empty()) {
            throw std::runtime_error(
                "HIP OGA backend supports image inputs only; message audio inputs are not supported");
        }
        if (!message.contains("content") || !message["content"].is_array()) continue;
        for (const auto& item : message["content"]) {
            const std::string type = item.value("type", std::string());
            if (type == "input_audio" || type == "audio") {
                throw std::runtime_error(
                    "HIP OGA backend supports image inputs only; structured audio content is not supported");
            }
            if (type == "image" || type == "image_url") {
                throw std::runtime_error(
                    "HIP OGA backend expects image content to be normalized before insertion");
            }
        }
    }
}

void append_hip_oga_image_input(std::vector<HipOgaImageInput>& images,
                                std::string payload,
                                input_payload_type_t payload_type) {
    if (payload.empty()) {
        throw std::runtime_error(
            "HIP OGA backend image entries must be non-empty strings");
    }
    if (payload_type == URL_LINK || is_remote_url_like(payload)) {
        throw std::runtime_error(
            "HIP OGA backend does not fetch remote image URLs; send base64 image data or a local file path");
    }
    images.push_back({std::move(payload), payload_type});
}

std::vector<HipOgaImageInput> collect_hip_oga_image_inputs(
    const lm_uniform_input_t& input) {
    if (!input.messages.empty() && !input.images.empty()) {
        throw std::runtime_error(
            "HIP OGA backend does not support mixing top-level images with message images");
    }
    if (!input.image_payload_types.empty() &&
        input.image_payload_types.size() != input.images.size()) {
        throw std::runtime_error(
            "HIP OGA backend image payload type count must match image count");
    }

    std::vector<HipOgaImageInput> images;
    for (std::size_t i = 0; i < input.images.size(); ++i) {
        append_hip_oga_image_input(images,
                                   input.images[i],
                                   input.image_payload_types.empty()
                                       ? FILE_NAME
                                       : input.image_payload_types[i]);
    }

    for (const auto& message : input.messages) {
        if (!message.is_object()) {
            throw std::runtime_error("HIP OGA backend messages must be JSON objects");
        }
        if (!message.contains("images") || message["images"].is_null()) continue;
        if (!message["images"].is_array()) {
            throw std::runtime_error("HIP OGA backend expects message images to be an array");
        }
        if (message["images"].empty()) continue;
        for (const auto& image : message["images"]) {
            if (!image.is_string()) {
                throw std::runtime_error(
                    "HIP OGA backend image entries must be non-empty strings");
            }
            const std::string payload = image.get<std::string>();
            append_hip_oga_image_input(images,
                                       payload,
                                       path_exists_without_throw(payload)
                                           ? FILE_NAME
                                           : BASE64_ENCODED);
        }
    }

    return images;
}

nlohmann::ordered_json build_hip_oga_prompt_messages(
    const lm_uniform_input_t& input,
    const std::vector<HipOgaImageInput>& images) {
    if (input.messages.empty()) {
        nlohmann::ordered_json message;
        message["role"] = "user";
        if (images.empty()) {
            message["content"] = input.prompt;
        } else {
            message["content"] = nlohmann::ordered_json::array();
            for (std::size_t i = 0; i < images.size(); ++i) {
                message["content"].push_back({{"type", "image"}});
            }
            message["content"].push_back({{"type", "text"}, {"text", input.prompt}});
        }
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        messages.push_back(std::move(message));
        return messages;
    }

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    for (const auto& message : input.messages) {
        if (!message.is_object()) {
            throw std::runtime_error("HIP OGA backend messages must be JSON objects");
        }
        if (!message.contains("images") || message["images"].is_null()) {
            messages.push_back(message);
            continue;
        }
        if (!message["images"].is_array()) {
            throw std::runtime_error("HIP OGA backend expects message images to be an array");
        }
        if (message["images"].empty()) {
            messages.push_back(message);
            continue;
        }

        nlohmann::ordered_json content = nlohmann::ordered_json::array();
        for (std::size_t i = 0; i < message["images"].size(); ++i) {
            content.push_back({{"type", "image"}});
        }
        if (message.contains("content") && message["content"].is_string() &&
            !message["content"].get_ref<const std::string&>().empty()) {
            content.push_back({{"type", "text"},
                               {"text", message["content"].get<std::string>()}});
        } else if (message.contains("content") && message["content"].is_array()) {
            for (const auto& item : message["content"]) {
                if (!item.is_object() || item.value("type", std::string()) != "text") {
                    throw std::runtime_error(
                        "HIP OGA backend supports only text content beside normalized images");
                }
                if (!item.contains("text") || !item["text"].is_string()) {
                    throw std::runtime_error(
                        "HIP OGA backend text content needs a string text field");
                }
                content.push_back({{"type", "text"},
                                   {"text", item["text"].get<std::string>()}});
            }
        }

        nlohmann::ordered_json rebuilt = message;
        rebuilt.erase("images");
        rebuilt["content"] = std::move(content);
        messages.push_back(std::move(rebuilt));
    }
    return messages;
}

std::vector<std::uint8_t> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("HIP OGA backend could not open image file: " + path);
    }
    std::string bytes_string((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    if (bytes_string.empty()) {
        throw std::runtime_error("HIP OGA backend image file is empty: " + path);
    }
    return std::vector<std::uint8_t>(bytes_string.begin(), bytes_string.end());
}

std::vector<std::uint8_t> decode_base64_image_bytes(const std::string& image) {
    if (starts_with(image, "http://") || starts_with(image, "https://")) {
        throw std::runtime_error(
            "HIP OGA backend does not fetch remote image URLs; send base64 image data or a local file path");
    }
    const std::string payload = strip_image_data_url_prefix(image);
    try {
        const std::string decoded = base64::from_base64(payload);
        if (decoded.empty()) {
            throw std::runtime_error("decoded payload is empty");
        }
        return std::vector<std::uint8_t>(decoded.begin(), decoded.end());
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("HIP OGA backend could not decode base64 image input: ") +
            e.what());
    }
}

HipOgaLoadedImages load_hip_oga_images(
    const std::vector<HipOgaImageInput>& images) {
    if (images.empty()) {
        throw std::runtime_error("HIP OGA backend needs at least one image input");
    }

    HipOgaLoadedImages loaded;
    std::vector<const void*> image_data;
    std::vector<std::size_t> image_sizes;
    loaded.buffers.reserve(images.size());
    image_data.reserve(images.size());
    image_sizes.reserve(images.size());

    for (const auto& image : images) {
        if (image.payload_type == URL_LINK) {
            throw std::runtime_error(
                "HIP OGA backend does not fetch remote image URLs; send base64 image data or a local file path");
        }
        loaded.buffers.push_back(image.payload_type == FILE_NAME
            ? read_binary_file(image.payload)
            : decode_base64_image_bytes(image.payload));
        image_data.push_back(loaded.buffers.back().data());
        image_sizes.push_back(loaded.buffers.back().size());
    }

    loaded.images = OgaImages::Load(image_data.data(), image_sizes.data(), image_sizes.size());
    return loaded;
}

std::size_t read_oga_prompt_token_count(OgaNamedTensors& inputs) {
    auto input_ids = inputs.Get("input_ids");
    const auto shape = input_ids->Shape();
    if (shape.empty()) {
        throw std::runtime_error(
            "HIP OGA multimodal processor produced scalar input_ids");
    }
    if (shape.size() > 1 && shape.front() != 1) {
        throw std::runtime_error(
            "HIP OGA backend multimodal path supports batch size 1 only");
    }
    if (shape.back() <= 0) {
        throw std::runtime_error(
            "HIP OGA multimodal processor produced empty input_ids");
    }
    return static_cast<std::size_t>(shape.back());
}

class HipOgaCausalLm final : public causal_lm {
public:
    HipOgaCausalLm(std::string model_path,
                   nlohmann::ordered_json model_info,
                   std::uint32_t context_length)
        : model_path_(std::move(model_path)),
          model_info_(std::move(model_info)),
          context_length_(std::max<std::uint32_t>(context_length, 1)),
          provider_(get_hip_oga_string(model_info_, "provider", "AMDGPU")),
          profile_(get_hip_oga_string(model_info_, "profile", "hip")),
          offload_policy_(flm::hip_oga::runtime_offload_policy(
              nlohmann::json::parse(model_info_.dump()))) {
        ensure_oga_process_handle();
        const auto config_path = resolve_oga_config_path(model_path_, model_info_);
        config_ = OgaConfig::Create(config_path.string().c_str());
        const auto genai_config =
            read_json_file_if_present(config_path / "genai_config.json");
        nlohmann::json hip_oga_options = nlohmann::json::object();
        const auto options = model_info_.find("hip_oga");
        if (options != model_info_.end() && options->is_object()) {
            hip_oga_options = nlohmann::json::parse(options->dump());
        }
        hip_oga_options["provider"] = provider_;
        hip_oga_options["profile"] = profile_;
        hip_oga_options["offload_policy"] = offload_policy_;

        const bool diagnostics_enabled =
            flm::hip_oga::getenv_flag_enabled("FLM_HIP_OGA_DIAGNOSTICS");
        const std::string diagnostics_prefix =
            flm::hip_oga::getenv_string("FLM_HIP_OGA_PROFILE_PREFIX");
        const auto overlay =
            flm::hip_oga::build_session_overlay(genai_config,
                                                hip_oga_options,
                                                diagnostics_enabled,
                                                diagnostics_prefix);
        // strict_gpu maps to OGA session.disable_cpu_ep_fallback="1" via
        // OGA's per-section disable_cpu_ep_fallback session option.
        config_->Overlay(overlay.dump().c_str());

        model_ = OgaModel::Create(*config_);
        tokenizer_ = OgaTokenizer::Create(*model_);
    }

    void set_requested_max_tokens(int max_tokens) {
        requested_max_tokens_ = max_tokens > 0
            ? static_cast<std::uint32_t>(max_tokens)
            : kDefaultRequestedMaxTokens;
    }

    buffer<bf16> forward(int id) override {
        std::vector<int> one{id};
        return append_and_compute(one);
    }

    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override {
        if (payload != nullptr) {
            throw std::runtime_error(
                "HIP OGA backend prototype does not support multimodal payloads");
        }
        clear_context();
        return append_and_compute(ids);
    }

    void set_context_length(int length) override {
        if (length < 1) {
            throw std::runtime_error("HIP OGA context length must be positive");
        }
        context_length_ = static_cast<std::uint32_t>(length);
        release_generator();
    }

    void load_weights(Q4NX&) override {
        // OGA owns ONNX weight loading through genai_config.json.
    }

    void update_max_length(std::uint32_t length) override {
        if (length == 0) {
            throw std::runtime_error("HIP OGA context length must be positive");
        }
        context_length_ = length;
        release_generator();
    }

    void clear_context() override { release_generator(); }

    buffer<bf16> get_k_cache(int, int) override {
        throw std::runtime_error("HIP OGA backend does not expose KV cache buffers");
    }

    buffer<bf16> get_v_cache(int, int) override {
        throw std::runtime_error("HIP OGA backend does not expose KV cache buffers");
    }

    int get_current_context_length() override {
        return static_cast<int>(accepted_token_count_);
    }

    int checkpoint() override {
        checkpoint_token_count_ = accepted_token_count_;
        return static_cast<int>(checkpoint_token_count_);
    }

    int restore() override {
        if (!generator_) return 0;
        generator_->RewindTo(checkpoint_token_count_);
        accepted_token_count_ = checkpoint_token_count_;
        return static_cast<int>(accepted_token_count_);
    }

    void restore_text_checkpoint() {
        if (!generator_) {
            throw std::runtime_error("HIP OGA prefix cache has no generator to restore");
        }
        generator_->RewindTo(checkpoint_token_count_);
        accepted_token_count_ = checkpoint_token_count_;
        token_stream_.reset();
    }

    void append_text_suffix(const std::vector<int>& suffix_tokens) {
        if (suffix_tokens.empty()) return;
        if (!generator_) {
            throw std::runtime_error("HIP OGA prefix cache has no generator for suffix append");
        }
        const auto converted = to_int32_tokens(suffix_tokens);
        generator_->AppendTokens(converted.data(), converted.size());
        accepted_token_count_ += converted.size();
    }

    int checkpoint_text_prompt() {
        checkpoint_token_count_ = accepted_token_count_;
        return static_cast<int>(checkpoint_token_count_);
    }

    void reset_decoder_stream() {
        token_stream_ = OgaTokenizerStream::Create(*tokenizer_);
    }

    std::uint32_t native_text_required_max_length(std::size_t prompt_token_count) const {
        return static_cast<std::uint32_t>(
            std::min<std::size_t>(prompt_token_count,
                                  std::numeric_limits<std::uint32_t>::max()));
    }

    std::uint32_t generator_capacity() const {
        return generator_max_length_;
    }

    std::string prefix_cache_signature(const std::string& request_signature) const {
        std::ostringstream signature;
        signature << "provider=" << provider_
                  << ";profile=" << profile_
                  << ";ctx=" << context_length_
                  << ";request=" << request_signature;
        return signature.str();
    }

    std::vector<int> encode(const std::string& text) const {
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(text.c_str(), *sequences);
        if (sequences->Count() != 1) {
            throw std::runtime_error("HIP OGA tokenizer produced an invalid prompt batch");
        }

        const std::size_t token_count = sequences->SequenceCount(0);
        const int32_t* token_data = sequences->SequenceData(0);
        std::vector<int> tokens;
        tokens.reserve(token_count);
        for (std::size_t i = 0; i < token_count; ++i) {
            tokens.push_back(static_cast<int>(token_data[i]));
        }
        return tokens;
    }

    void begin_turn(const std::vector<int>& tokens) {
        const auto converted = to_int32_tokens(tokens);
        reset_generator(context_length_);
        if (converted.empty()) {
            throw std::runtime_error("HIP OGA tokenizer produced no prompt tokens");
        }
        generator_->AppendTokens(converted.data(), converted.size());
        accepted_token_count_ = converted.size();
        checkpoint_token_count_ = accepted_token_count_;
    }

    std::size_t begin_multimodal_turn(
        const std::string& prompt,
        const std::vector<HipOgaImageInput>& images) {
        reset_generator(effective_oga_max_length(1));
        if (!processor_) {
            try {
                processor_ = OgaMultiModalProcessor::Create(*model_);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    std::string("HIP OGA backend could not create OGA multimodal processor for image input: ") +
                    e.what());
            }
        }

        auto loaded_images = load_hip_oga_images(images);
        auto inputs = processor_->ProcessImages(prompt.c_str(), loaded_images.images.get());
        const std::size_t prompt_token_count = read_oga_prompt_token_count(*inputs);
        if (prompt_token_count >= context_length_) {
            throw std::runtime_error(
                "HIP OGA multimodal prompt reached the model context length");
        }

        reset_generator(effective_oga_max_length(prompt_token_count));
        generator_->SetInputs(*inputs);
        token_stream_ = OgaTokenizerStream::Create(*processor_);
        accepted_token_count_ = prompt_token_count;
        checkpoint_token_count_ = accepted_token_count_;
        return prompt_token_count;
    }

    bool is_done() {
        return generator_ == nullptr || generator_->IsDone();
    }

    bool generate_next_token(int& token) {
        if (!generator_) {
            throw std::runtime_error("HIP OGA generator is not initialized for this turn");
        }
        if (generator_->IsDone()) return false;

        const std::size_t before_count = generator_->GetSequenceCount(0);
        generator_->GenerateNextToken();
        const std::size_t after_count = generator_->GetSequenceCount(0);
        if (after_count <= before_count) return false;

        const int32_t* sequence = generator_->GetSequenceData(0);
        token = static_cast<int>(sequence[after_count - 1]);
        accepted_token_count_ = after_count;
        return true;
    }

    std::string decode_token(int token) {
        if (!token_stream_) {
            token_stream_ = OgaTokenizerStream::Create(*tokenizer_);
        }
        const char* decoded = token_stream_->Decode(static_cast<int32_t>(token));
        return decoded == nullptr ? std::string() : std::string(decoded);
    }

private:
    std::uint32_t effective_oga_max_length(std::size_t prompt_token_count) const {
        const std::uint64_t requested_max_tokens = requested_max_tokens_;
        const std::uint64_t minimum_total =
            static_cast<std::uint64_t>(prompt_token_count) + 1;
        const std::uint64_t requested_total =
            prompt_token_count + requested_max_tokens + kHipOgaSharedGenerateLookaheadTokens;
        const std::uint64_t bounded_total =
            std::max<std::uint64_t>(minimum_total, requested_total);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(context_length_, bounded_total));
    }

    void reset_generator(std::uint32_t max_length) {
        params_ = OgaGeneratorParams::Create(*model_);
        params_->SetSearchOption("batch_size", 1);
        params_->SetSearchOption("max_length", static_cast<double>(max_length));
        generator_ = OgaGenerator::Create(*model_, *params_);
        generator_max_length_ = max_length;
        token_stream_ = OgaTokenizerStream::Create(*tokenizer_);
        accepted_token_count_ = 0;
        checkpoint_token_count_ = 0;
    }

    void release_generator() {
        generator_.reset();
        params_.reset();
        token_stream_.reset();
        generator_max_length_ = 0;
        accepted_token_count_ = 0;
        checkpoint_token_count_ = 0;
    }

    buffer<bf16> append_and_compute(const std::vector<int>& tokens) {
        const auto converted = to_int32_tokens(tokens);
        if (!generator_) {
            reset_generator(effective_oga_max_length(converted.size()));
        }
        if (!converted.empty()) {
            generator_->AppendTokens(converted.data(), converted.size());
            accepted_token_count_ += converted.size();
        }

        if (accepted_token_count_ >= context_length_) {
            throw std::runtime_error("HIP OGA backend reached its context length");
        }

        generator_->GenerateNextToken();
        auto logits = copy_logits();
        generator_->RewindTo(accepted_token_count_);
        return logits;
    }

    buffer<bf16> copy_logits() {
        auto logits_tensor = generator_->GetLogits();
        const auto shape = logits_tensor->Shape();
        if (shape.empty()) {
            throw std::runtime_error("HIP OGA backend received scalar logits");
        }
        if (shape.size() > 1 && shape.front() != 1) {
            throw std::runtime_error(
                "HIP OGA backend prototype supports batch size 1 only");
        }

        std::size_t element_count = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("HIP OGA logits have an invalid shape");
            }
            element_count *= static_cast<std::size_t>(dim);
        }
        const std::size_t vocab_size = static_cast<std::size_t>(shape.back());
        const std::size_t offset = element_count - vocab_size;

        buffer<bf16> out(vocab_size);
        switch (logits_tensor->Type()) {
            case OgaElementType_float32: {
                const auto* data = static_cast<const float*>(logits_tensor->Data());
                for (std::size_t i = 0; i < vocab_size; ++i) {
                    out[i] = data[offset + i];
                }
                break;
            }
            case OgaElementType_bfloat16: {
                static_assert(sizeof(bf16) == sizeof(std::uint16_t),
                              "FastFlowLM bf16 must be storage-compatible with OGA bfloat16");
                const auto* data =
                    static_cast<const std::uint8_t*>(logits_tensor->Data());
                std::memcpy(out.data(), data + offset * sizeof(bf16),
                            vocab_size * sizeof(bf16));
                break;
            }
            default:
                throw std::runtime_error(
                    "HIP OGA backend only supports float32 or bfloat16 logits");
        }
        return out;
    }

    std::string model_path_;
    nlohmann::ordered_json model_info_;
    std::uint32_t context_length_;
    std::string provider_;
    std::string profile_;
    std::string offload_policy_;
    std::uint32_t requested_max_tokens_ = kDefaultRequestedMaxTokens;
    std::unique_ptr<OgaConfig> config_;
    std::unique_ptr<OgaModel> model_;
    std::unique_ptr<OgaTokenizer> tokenizer_;
    std::unique_ptr<OgaMultiModalProcessor> processor_;
    std::unique_ptr<OgaTokenizerStream> token_stream_;
    std::unique_ptr<OgaGeneratorParams> params_;
    std::unique_ptr<OgaGenerator> generator_;
    std::uint32_t generator_max_length_ = 0;
    std::size_t accepted_token_count_ = 0;
    std::size_t checkpoint_token_count_ = 0;
};

}  // namespace

HipOgaModel::HipOgaModel() : AutoModel(nullptr, "HIP OGA") {
    forward_on_eos = false;
}

void HipOgaModel::clear_context() {
    prefix_cache_.reset_checkpoint("clear_context");
    AutoModel::clear_context();
}

void HipOgaModel::load_model(std::string model_path,
                             json model_info,
                             int default_context_length,
                             bool enable_preemption) {
    if (enable_preemption) {
        throw std::runtime_error("HIP OGA backend does not support NPU preemption");
    }
    if (this->is_model_loaded && this->model_path == model_path) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }

    this->model_path = std::move(model_path);
    header_print("FLM", "Loading model: " << this->model_path);
    this->lm_config = std::make_unique<LM_Config>();
    load_hip_oga_lm_config(*this->lm_config, this->model_path, model_info);

    const int context_length = default_context_length != -1
        ? default_context_length
        : model_info["default_context_length"].get<int>();
    if (context_length < 1) {
        throw std::runtime_error("HIP OGA context length must be positive");
    }
    this->MAX_L = static_cast<std::uint32_t>(context_length);
    this->token_history.clear();
    this->token_history.reserve(this->MAX_L);
    this->last_token = -1;
    this->total_tokens = 0;
    this->npu.reset();
    this->enable_preemption = false;
    this->tool_start_token_id_ = -1;
    this->prefix_cache_.reset_session();
    this->prefix_cache_request_signature_ = "{}";

    try {
        this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
        const auto model_root = std::filesystem::path(this->model_path);
        auto tokenizer_config =
            read_json_file_if_present(model_root / "tokenizer_config.json");
        if (tokenizer_config.empty()) {
            throw std::runtime_error("Cannot open " +
                                     (model_root / "tokenizer_config.json").string());
        }

        const auto chat_template_path = model_root / "chat_template.jinja";
        if (std::filesystem::exists(chat_template_path)) {
            std::ifstream template_file(chat_template_path, std::ios::in | std::ios::binary);
            if (!template_file.is_open()) {
                throw std::runtime_error("Cannot open " + chat_template_path.string());
            }
            std::stringstream template_buffer;
            template_buffer << template_file.rdbuf();
            tokenizer_config["chat_template"] = template_buffer.str();
        } else if (!tokenizer_config.contains("chat_template") ||
                   !tokenizer_config["chat_template"].is_string()) {
            throw std::runtime_error(
                "No template file found and no chat_template in tokenizer_config.json");
        }

        this->has_bos_token = tokenizer_config.contains("bos_token") &&
            !tokenizer_config["bos_token"].is_null();
        this->bos_token = this->has_bos_token
            ? tokenizer_config["bos_token"].get<std::string>()
            : "";
        this->bos_token_id = -1;
        if (this->has_bos_token) {
            if (has_integer_field(tokenizer_config, "bos_token_id")) {
                this->bos_token_id = tokenizer_config["bos_token_id"].get<int>();
            } else {
                const auto genai_config =
                    read_json_file_if_present(genai_config_path_for(this->model_path));
                if (genai_config.contains("model") && genai_config["model"].is_object() &&
                    has_integer_field(genai_config["model"], "bos_token_id")) {
                    this->bos_token_id = genai_config["model"]["bos_token_id"].get<int>();
                } else {
                    throw std::runtime_error(
                        "bos_token is set but bos_token_id is missing from tokenizer_config.json and genai_config.json");
                }
            }
        }
        if (!tokenizer_config.contains("eos_token") ||
            tokenizer_config["eos_token"].is_null()) {
            throw std::runtime_error("eos_token is null in tokenizer_config.json");
        }
        this->eos_token = tokenizer_config["eos_token"].get<std::string>();
        this->eos_token_ids.clear();
        if (tokenizer_config.contains("eos_token_id")) {
            append_token_ids(this->eos_token_ids, tokenizer_config["eos_token_id"]);
        }
        if (this->eos_token_ids.empty()) {
            const auto genai_config =
                read_json_file_if_present(genai_config_path_for(this->model_path));
            if (genai_config.contains("model") && genai_config["model"].is_object() &&
                genai_config["model"].contains("eos_token_id")) {
                append_token_ids(this->eos_token_ids, genai_config["model"]["eos_token_id"]);
            }
        }
        if (this->eos_token_ids.empty()) {
            throw std::runtime_error(
                "eos_token_id is missing from tokenizer_config.json and genai_config.json");
        }
        this->chat_tmpl = std::make_unique<minja::chat_template>(
            tokenizer_config["chat_template"].get<std::string>(),
            this->bos_token,
            this->eos_token);
        const std::vector<int> tool_start_tokens =
            this->tokenizer->encode("<tool_call>");
        if (tool_start_tokens.size() == 1) {
            this->tool_start_token_id_ = tool_start_tokens.front();
        }
        this->user_system_prompt = "";
        this->extra_context["user_system_prompt"] = this->user_system_prompt;
        this->lm_engine = std::make_unique<HipOgaCausalLm>(
            this->model_path, model_info, this->MAX_L);
    } catch (...) {
        this->lm_engine.reset();
        this->tokenizer.reset();
        this->lm_config.reset();
        this->is_model_loaded = false;
        throw;
    }
    this->is_model_loaded = true;

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;
    this->set_sampler(config);

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

std::string HipOgaModel::apply_chat_template(nlohmann::ordered_json& messages,
                                             nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty()) {
        inputs.tools = tools;
    }
    return this->chat_tmpl->apply(inputs);
}

bool HipOgaModel::insert(chat_meta_info_t& meta_info,
                         lm_uniform_input_t& input,
                         std::function<bool()> is_cancelled) {
    reject_hip_oga_audio_inputs(input);
    const std::vector<HipOgaImageInput> images =
        collect_hip_oga_image_inputs(input);

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    nlohmann::ordered_json messages =
        build_hip_oga_prompt_messages(input, images);
    templated_text = this->apply_chat_template(messages, input.tools);

    auto* hip_engine = dynamic_cast<HipOgaCausalLm*>(this->lm_engine.get());
    if (hip_engine == nullptr) {
        throw std::runtime_error("HIP OGA model is not backed by an OGA engine");
    }

    std::size_t prompt_tokens = 0;
    if (images.empty()) {
        std::vector<int> tokens = hip_engine->encode(templated_text);
        this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            return false;
        }
        if (tokens.size() >= this->MAX_L) {
            header_print("WARNING", "Max length reached, stopping prefilling...");
            return false;
        }

        const auto required_capacity =
            hip_engine->native_text_required_max_length(tokens.size());
        const std::string signature =
            hip_engine->prefix_cache_signature(prefix_cache_request_signature_);
        auto cache_decision = prefix_cache_.evaluate(
            hip_oga_prefix_cache_enabled(),
            meta_info.restore_allowed,
            true,
            signature,
            required_capacity,
            hip_engine->generator_capacity(),
            tokens);

        auto prefill_start_time = this->profiler_list[PREFILL_TIME].start();
        try {
            if (cache_decision.reuse) {
                hip_engine->restore_text_checkpoint();
                std::vector<int> suffix_tokens(
                    tokens.begin() + static_cast<std::ptrdiff_t>(cache_decision.suffix_start),
                    tokens.end());
                hip_engine->append_text_suffix(suffix_tokens);
            } else {
                hip_engine->begin_turn(tokens);
            }
        } catch (const std::exception& e) {
            const std::string reason =
                std::string("provider_exception: ") + e.what();
            header_print("FLM", "HIP OGA prefix cache fallback: " + reason);
            prefix_cache_.disable_for_session(reason);
            cache_decision = flm::hip_oga::PrefixCacheDecision{};
            cache_decision.attempted = true;
            cache_decision.reason = "provider_exception";
            cache_decision.newly_prefilled_tokens = tokens.size();
            hip_engine->begin_turn(tokens);
        }
        const int checkpoint_idx = hip_engine->checkpoint_text_prompt();
        auto prefill_end_time =
            this->profiler_list[PREFILL_TIME].stop(cache_decision.newly_prefilled_tokens);

        this->token_history = tokens;
        prompt_tokens = tokens.size();
        meta_info.cached_prompt_tokens = static_cast<int>(cache_decision.cached_tokens);
        meta_info.prefill_duration =
            (uint64_t)time_utils::duration_ns(prefill_start_time, prefill_end_time).first;
        prefix_cache_.commit_checkpoint(this->token_history, signature,
                                        hip_engine->generator_capacity());
        header_print("FLM", "HIP OGA prefix cache attempted=" +
            std::string(cache_decision.attempted ? "1" : "0") +
            " hit=" + std::string(cache_decision.reuse ? "1" : "0") +
            " fallback=" + (cache_decision.reuse ? std::string("none") : cache_decision.reason) +
            " cached_tokens=" + std::to_string(cache_decision.cached_tokens) +
            " newly_prefilled_tokens=" + std::to_string(cache_decision.newly_prefilled_tokens) +
            " checkpoint_tokens=" + std::to_string(checkpoint_idx));
    } else {
        prefix_cache_.reset_checkpoint("multimodal_forced_miss");
        this->profiler_list[TKOEN_ENCODE_TIME].stop(0);

        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            return false;
        }

        auto prefill_start_time = this->profiler_list[PREFILL_TIME].start();
        prompt_tokens = hip_engine->begin_multimodal_turn(templated_text, images);
        auto prefill_end_time =
            this->profiler_list[PREFILL_TIME].stop(prompt_tokens);

        this->token_history.clear();
        this->checkpoint_his.clear();
        meta_info.cached_prompt_tokens = 0;
        meta_info.prefill_duration =
            (uint64_t)time_utils::duration_ns(prefill_start_time, prefill_end_time).first;
    }

    if (prompt_tokens > std::numeric_limits<std::uint32_t>::max() ||
        prompt_tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("HIP OGA prompt token count is too large");
    }
    this->total_tokens = static_cast<std::uint32_t>(prompt_tokens);
    this->last_token = -1;
    meta_info.prompt_tokens = static_cast<int>(prompt_tokens);
    return true;
}

std::string HipOgaModel::generate(chat_meta_info_t& meta_info,
                                  int length_limit,
                                  std::ostream& os,
                                  std::function<bool()> is_cancelled) {
    auto* hip_engine = dynamic_cast<HipOgaCausalLm*>(this->lm_engine.get());
    if (hip_engine == nullptr) {
        throw std::runtime_error("HIP OGA model is not backed by an OGA engine");
    }

    std::string result;
    stop_reason_t reason = EOT_DETECTED;
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    hip_engine->reset_decoder_stream();

    while (this->total_tokens < this->MAX_L) {
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            this->buffer_.clear();
            this->current_mode_ = StreamEventType::CONTENT;
            this->tool_name_.clear();
            this->is_in_tool_block_ = false;
            break;
        }
        if (hip_engine->is_done()) {
            break;
        }

        int sampled_token = -1;
        this->profiler_list[DECODING_TIME].start();
        const bool generated = hip_engine->generate_next_token(sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);
        if (!generated) {
            break;
        }

        this->last_token = sampled_token;
        this->token_history.push_back(sampled_token);
        this->total_tokens++;

        if (this->is_eos(sampled_token)) {
            meta_info.generated_tokens++;
            break;
        }

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)) {
            std::string token_str = hip_engine->decode_token(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        meta_info.generated_tokens++;

        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (this->total_tokens >= this->MAX_L) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }

    meta_info.decoding_duration =
        (uint64_t)(time_utils::cast_to_us(
            this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    if (this->log_raw_output) {
        std::cout << std::endl;
        header_print("FLM", "Model RAW Output: \n" + result);
    }
    return result;
}

std::string HipOgaModel::generate_with_prompt(chat_meta_info_t& meta_info,
                                              lm_uniform_input_t& input,
                                              int length_limit,
                                              std::ostream& os) {
    (void)this->configure_parameter("hip_oga_requested_max_tokens", length_limit);
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

bool HipOgaModel::configure_parameter(std::string parameter_name,
                                      const std::any& value) {
    if (parameter_name == "hip_oga_requested_max_tokens") {
        try {
            auto* hip_engine = dynamic_cast<HipOgaCausalLm*>(this->lm_engine.get());
            if (hip_engine == nullptr) return false;
            hip_engine->set_requested_max_tokens(std::any_cast<int>(value));
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "hip_oga_request_signature") {
        try {
            this->prefix_cache_request_signature_ =
                std::any_cast<std::string>(value);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "hip_oga_prefix_cache_disable_session") {
        try {
            this->prefix_cache_.disable_for_session(std::any_cast<std::string>(value));
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    return AutoModel::configure_parameter(std::move(parameter_name), value);
}

#else

HipOgaModel::HipOgaModel() : AutoModel(nullptr, "HIP OGA") {}

void HipOgaModel::clear_context() {
    AutoModel::clear_context();
}

void HipOgaModel::load_model(std::string, json, int, bool) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

bool HipOgaModel::insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

std::string HipOgaModel::generate(chat_meta_info_t&, int, std::ostream&,
                                  std::function<bool()>) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

std::string HipOgaModel::generate_with_prompt(chat_meta_info_t&,
                                              lm_uniform_input_t&,
                                              int,
                                              std::ostream&) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

bool HipOgaModel::configure_parameter(std::string, const std::any&) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

std::string HipOgaModel::apply_chat_template(nlohmann::ordered_json&,
                                             nlohmann::ordered_json) {
    throw std::runtime_error(
        "HIP OGA backend requires a build configured with -DFLM_ENABLE_HIP_OGA=ON");
}

#endif  // FLM_ENABLE_HIP_OGA

int HipOgaModel::get_tool_start_token_id() const {
    return this->tool_start_token_id_;
}

NonStreamResult HipOgaModel::parse_nstream_content(const std::string response_text) {
    const auto parsed = flm::hip_oga::parse_tool_call_markup(response_text);
    NonStreamResult result;
    result.content = parsed.content;
    result.reasoning_content = parsed.reasoning_content;
    result.tool_calls_list = parsed.tool_calls_list;
    result.error_message = parsed.error_message;
    if (!result.tool_calls_list.empty()) {
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
    }
    return result;
}

StreamResult HipOgaModel::parse_stream_content(const std::string content) {
    auto mode = to_hip_stream_type(this->current_mode_);
    const auto parsed = flm::hip_oga::parse_stream_markup(
        this->buffer_, this->tool_name_, mode, this->is_in_tool_block_, content, false);
    this->current_mode_ = to_auto_stream_type(mode);
    return to_auto_stream_result(parsed);
}

StreamResult HipOgaModel::parse_stream_content_final(const std::string content) {
    auto mode = to_hip_stream_type(this->current_mode_);
    const auto parsed = flm::hip_oga::parse_stream_markup(
        this->buffer_, this->tool_name_, mode, this->is_in_tool_block_, content, true);
    this->current_mode_ = to_auto_stream_type(mode);
    return to_auto_stream_result(parsed);
}
