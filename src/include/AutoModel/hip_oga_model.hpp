/// \file hip_oga_model.hpp
/// \brief Prototype AutoModel backed by ONNX Runtime GenAI over HIP EP
#pragma once

#include "AutoModel/automodel.hpp"
#include "AutoModel/hip_oga_prefix_cache.hpp"
#include "AutoModel/hip_oga_text_codec.hpp"
#include "AutoModel/hip_oga_codec_options.hpp"
#include "chat/chat_grammar.hpp"

class HipOgaModel : public AutoModel {
public:
    HipOgaModel();
    std::shared_ptr<decode_constraint> make_constraint(const chat_constraint_request& request) override;

    static bool is_model_dir(const std::string& model_path);
    static json model_info_for_path(const std::string& model_path);

    void load_model(std::string model_path, json model_info,
                    int default_context_length = -1,
                    bool enable_preemption = false, const std::string& backend = "") override;
    void clear_context() override;
    void set_sampling_provider(const std::string& name) override;
    void configure_sampling_request(const nlohmann::json& options, bool reset = true) override;
    void set_topk(int value) override;
    void set_topp(float value) override;
    void set_minp(float value) override;
    void set_temperature(float value) override;
    void set_presence_penalty(float value) override;
    void set_repetition_penalty(float value) override;
    void set_frequency_penalty(float value) override;
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
    void update_sampling_option(const char* name, double value);
    int tool_start_token_id_ = -1;
    flm::hip_oga::PrefixCacheState prefix_cache_;
    std::string prefix_cache_request_signature_ = "{}";
    flm::hip_oga::TextCodecSelector tokenizer_provider_selector_;
    flm::hip_oga::TextCodecSelector decoder_provider_selector_;
    flm::hip_oga::TextCodecSelector active_decoder_provider_selector_;
    bool active_media_input_ = false;
    std::uint64_t diagnostic_request_sequence_ = 0;
    flm::hip_oga::DecoderCanonicalizationPolicy decoder_canonicalization_policy_;
    std::string tokenizer_fingerprint_ = "{}";
    std::unique_ptr<chat_grammar_engine> grammar_engine_;
    std::optional<std::vector<int>> constrained_decoder_bans_;
};
