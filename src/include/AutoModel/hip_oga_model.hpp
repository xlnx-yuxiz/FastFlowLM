/// \file hip_oga_model.hpp
/// \brief Prototype AutoModel backed by ONNX Runtime GenAI over HIP EP
#pragma once

#include "AutoModel/automodel.hpp"
#include "AutoModel/hip_oga_prefix_cache.hpp"

class HipOgaModel : public AutoModel {
public:
    HipOgaModel();

    static bool is_model_dir(const std::string& model_path);
    static json model_info_for_path(const std::string& model_path);

    void load_model(std::string model_path, json model_info,
                    int default_context_length = -1,
                    bool enable_preemption = false) override;
    void clear_context() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit,
                         std::ostream& os,
                         std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info,
                                     lm_uniform_input_t& input,
                                     int length_limit,
                                     std::ostream& os = std::cout) override;
    bool configure_parameter(std::string parameter_name,
                             const std::any& value) override;
    int get_tool_start_token_id() const override;
    std::string apply_chat_template(
        nlohmann::ordered_json& messages,
        nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text) override;
    StreamResult parse_stream_content(const std::string content) override;
    StreamResult parse_stream_content_final(const std::string content) override;

private:
    int tool_start_token_id_ = -1;
    flm::hip_oga::PrefixCacheState prefix_cache_;
    std::string prefix_cache_request_signature_ = "{}";
};
