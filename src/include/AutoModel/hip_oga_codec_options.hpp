#pragma once

#include <cstdlib>
#include "AutoModel/hip_oga_text_codec.hpp"

namespace flm::hip_oga {

struct TextCodecConfig {
    TextCodecSelector tokenizer;
    TextCodecSelector decoder;
};

inline TextCodecConfig resolve_text_codec_config(
    const std::string& codec_cli, const std::string& tokenizer_cli,
    const std::string& decoder_cli, const std::string& codec_env,
    const std::string& tokenizer_env, const std::string& decoder_env,
    bool legacy_compare) {
    // Validate all CLI values, including overridden ones, before loading a model.
    for (const auto& value : {codec_cli, tokenizer_cli, decoder_cli}) {
        if (!value.empty()) (void)parse_text_codec_selector(value);
    }
    auto resolve = [&](const std::string& layer_cli, const std::string& layer_env) {
        if (!layer_cli.empty()) return parse_text_codec_selector(layer_cli);
        if (!codec_cli.empty()) return parse_text_codec_selector(codec_cli);
        // Preserve the old global compare flag's authority over env selectors.
        if (legacy_compare) return parse_text_codec_selector("compare");
        return parse_text_codec_selector(layer_env.empty() ? codec_env : layer_env);
    };
    return {resolve(tokenizer_cli, tokenizer_env), resolve(decoder_cli, decoder_env)};
}

inline std::string text_codec_environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

inline TextCodecConfig text_codec_config_from_environment(
    const std::string& codec_cli = {}, const std::string& tokenizer_cli = {},
    const std::string& decoder_cli = {}) {
    const auto compare = normalize_selector_name(
        text_codec_environment("FLM_HIP_OGA_TEXT_CODEC_COMPARE"));
    return resolve_text_codec_config(
        codec_cli, tokenizer_cli, decoder_cli,
        text_codec_environment("FLM_HIP_OGA_TEXT_CODEC"),
        text_codec_environment("FLM_HIP_OGA_TOKENIZER_PROVIDER"),
        text_codec_environment("FLM_HIP_OGA_DECODER_PROVIDER"),
        compare == "1" || compare == "true" || compare == "yes" || compare == "on");
}

inline void set_text_codec_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    const int result = _putenv_s(name, value.c_str());
#else
    const int result = setenv(name, value.c_str(), 1);
#endif
    if (result != 0) throw std::runtime_error(std::string("Cannot set ") + name);
}

// Called once by the CLI before any runner/server threads or models are created.
// The environment also propagates these process-wide modes to child processes.
inline void apply_text_codec_cli_options(
    const std::string& codec, const std::string& tokenizer, const std::string& decoder) {
    if (codec.empty() && tokenizer.empty() && decoder.empty()) return;
    const auto config = text_codec_config_from_environment(codec, tokenizer, decoder);
    set_text_codec_environment("FLM_HIP_OGA_TOKENIZER_PROVIDER", config.tokenizer.normalized_name);
    set_text_codec_environment("FLM_HIP_OGA_DECODER_PROVIDER", config.decoder.normalized_name);
    set_text_codec_environment("FLM_HIP_OGA_TEXT_CODEC_COMPARE", "0");
}

} // namespace flm::hip_oga
