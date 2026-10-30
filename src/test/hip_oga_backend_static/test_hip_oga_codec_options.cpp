#include "AutoModel/hip_oga_codec_options.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace flm::hip_oga;

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

int main() {
    const auto native = text_codec_execution(parse_text_codec_selector("oga"));
    require(native.use_oga && !native.use_flm && !native.compare && !native.output_flm,
            "native mode must not invoke FLM codecs");
    const auto flm = text_codec_execution(parse_text_codec_selector(" FLM "));
    require(!flm.use_oga && flm.use_flm && !flm.compare && flm.output_flm,
            "FLM mode must not invoke OGA codecs");
    const auto compare = text_codec_execution(parse_text_codec_selector("compare"));
    require(compare.use_oga && compare.use_flm && compare.compare && !compare.output_flm,
            "compare must invoke both and use OGA output");
    const auto legacy = text_codec_execution(parse_text_codec_selector("external"));
    require(legacy.use_oga && legacy.use_flm && legacy.compare && legacy.output_flm,
            "legacy external must retain checked FLM output");

    const auto defaults = resolve_text_codec_config({}, {}, {}, {}, {}, {}, false);
    require(defaults.tokenizer.normalized_name == "oga" && defaults.decoder.normalized_name == "oga",
            "unset configuration must remain native");
    const auto env = resolve_text_codec_config({}, {}, {}, "flm", {}, "oga", false);
    require(env.tokenizer.normalized_name == "flm" && env.decoder.normalized_name == "oga",
            "per-layer environment must override combined environment");
    const auto old_compare = resolve_text_codec_config({}, {}, {}, {}, "external", "external", true);
    require(old_compare.tokenizer.normalized_name == "compare" && old_compare.decoder.normalized_name == "compare",
            "legacy compare must retain OGA authority");
    const auto cli = resolve_text_codec_config("flm", {}, "oga", "oga", "external", "external", true);
    require(cli.tokenizer.normalized_name == "flm" && cli.decoder.normalized_name == "oga",
            "layer CLI > combined CLI > legacy environment compare");
    const auto partial = resolve_text_codec_config({}, "flm", {}, {}, "oga", "external", true);
    require(partial.tokenizer.normalized_name == "flm" && partial.decoder.normalized_name == "compare",
            "partial CLI must leave the other layer's environment policy intact");
    bool rejected = false;
    try { (void)resolve_text_codec_config("typo", "flm", "oga", {}, {}, {}, false); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid CLI values must fail even if overridden");

    // Exercise process-local CLI propagation used by both run and serve.
    set_text_codec_environment("FLM_HIP_OGA_TOKENIZER_PROVIDER", "external");
    set_text_codec_environment("FLM_HIP_OGA_DECODER_PROVIDER", "external");
    set_text_codec_environment("FLM_HIP_OGA_TEXT_CODEC_COMPARE", "1");
    apply_text_codec_cli_options({}, "flm", {});
    const auto applied = text_codec_config_from_environment();
    require(applied.tokenizer.normalized_name == "flm" && applied.decoder.normalized_name == "compare",
            "model loading must see resolved CLI modes without legacy compare overriding FLM");
    std::cout << "HIP OGA codec options tests passed\n";
}
