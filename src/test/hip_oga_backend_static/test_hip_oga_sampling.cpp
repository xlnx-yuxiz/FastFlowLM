// Behavioral checks for request normalization and per-model OGA sampling state.
#include "AutoModel/hip_oga_sampling.hpp"
#include "AutoModel/hip_oga_logits.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace flm::hip_oga;
using nlohmann::json;

void require(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

template<class F> void rejects(F action, const std::string& field) {
    bool rejected = false;
    try { action(); }
    catch (const SamplingRequestError& e) { rejected = std::string(e.what()).find(field) != std::string::npos; }
    require(rejected, ("invalid/unsupported field was silently accepted: " + field).c_str());
}

struct CapturedParams {
    bool do_sample = false;
    double min_length = 0, no_repeat_ngram_size = 0;
    std::map<std::string, double> numbers;

    void SetSearchOptionBool(const char*, bool value) { do_sample = value; }
    void SetSearchOption(const char* name, double value) { numbers[name] = value; }
    bool GetSearchBool(const char*) const { return do_sample; }
    double GetSearchNumber(const char* name) const { if (std::string(name) == "min_length") return min_length;
        if (std::string(name) == "no_repeat_ngram_size") return no_repeat_ngram_size;
        return numbers.at(name); }
};

int main() {
    require(parse_sampling_seed("-1") == -1 && parse_sampling_seed("2147483647") == 2147483647,
            "CLI seed boundaries must be preserved");
    for (const auto* text : {"1.5", "1junk", "2147483648", "999999999999999999999", "-2", ""})
        rejects([&]{ parse_sampling_seed(text); }, "seed");
    rejects([&]{ parse_sampling_integer("1.5", "top_k", 0); }, "top_k");
    for (const auto* text : {"1junk", "nan", "inf", "1e100", "1e-100"})
        rejects([&]{ parse_sampling_float(text, "temperature"); }, "temperature");
    require(parse_sampling_float("0.25", "min_p") == 0.25f, "CLI fractions must be preserved");

    // OGA has no extension fields; SamplingState supplies their canonical zeros.
    const json oga_defaults = {{"do_sample",true},{"temperature",0.75},{"top_k",20},
        {"top_p",0.875},{"repetition_penalty",1.0},{"random_seed",-1}};
    const json defaults = oga_defaults;

    auto normalized = normalize_sampling_request(
        {{"temperature",1.5},{"top_k",3},{"min_p",0.25},{"presence_penalty",3.5},{"num_predict",4}},
        {{"temperature",0},{"top_p",0.5},{"min_p",0},{"presence_penalty",-4.25},{"model","fixture"}});
    require(normalized == json({{"temperature",0},{"top_k",3},{"min_p",0},
                                {"top_p",0.5},{"presence_penalty",-4.25}}),
            "normalization must preserve fractions, signed penalties, and top-level precedence");
    rejects([&]{ normalize_sampling_request(json::array(),json::object()); }, "options");

    SamplingState state(oga_defaults);
    SamplingState small_vocab(oga_defaults, 64);
    require(state.effective() == defaults, "missing processor options must default to canonical zeros");
    rejects([&]{ small_vocab.reset_request({{"top_k",65}}); }, "top_k");
    require(small_vocab.effective() == defaults, "out-of-vocabulary top-k must fail atomically");

    state.reset_request({{"temperature",0},{"seed",42}});
    require(state.effective().at("do_sample") == false, "temperature zero must select greedy");
    require(state.effective().at("temperature").get<double>() > 0,
            "OGA temperature must stay positive for greedy");
    require(state.effective().at("random_seed") == 42, "seed must map to OGA random_seed");
    require(!state.requires_fresh_generator(), "greedy requests do not need RNG reset");

    state.update({{"top_k",0},{"top_p",0},{"seed",2147483647}});
    state.update({{"temperature",0.5}});
    require(state.effective().at("random_seed") == 2147483647 &&
            state.effective().at("top_k") == 0 && state.effective().at("top_p") == 0,
            "CLI updates must preserve earlier overrides");

    state.reset_request({{"temperature",1.25},{"top_k",3},{"top_p",0.5},
                         {"seed",0},{"repetition_penalty",1.25},
                         {"min_p",0.25},{"presence_penalty",3.25},{"frequency_penalty",-4.5}});
    const json extended = {{"do_sample",true},{"temperature",1.25},{"top_k",3},
        {"top_p",0.5},{"random_seed",0},{"repetition_penalty",1.25},
        {"min_p",0.25},{"presence_penalty",3.25},{"frequency_penalty",-4.5}};
    require(state.effective() == extended,
            "sampling settings must preserve finite signed processor values beyond plus or minus two");
    require(state.requires_fresh_generator(), "seeded stochastic extension requires a fresh generator");
    require(requires_logits_processing(state.effective()), "nonzero processor options select the extension path");
    const auto mapped = sampling_logits_options(state.effective());
    require(mapped.do_sample && mapped.temperature == 1.25 && mapped.top_k == 3 &&
            mapped.top_p == 0.5 && mapped.min_p == 0.25 &&
            mapped.repetition_penalty == 1.25 && mapped.presence_penalty == 3.25 &&
            mapped.frequency_penalty == -4.5,
            "effective request fields must map to the logits processor without loss");

    CapturedParams extension_params;
    apply_oga_sampling_options(extension_params, state.effective());
    require(extension_params.do_sample, "extension path must retain stochastic sampling");
    require(extension_params.numbers == std::map<std::string, double>{
                {"temperature",1.0},{"top_k",0},{"top_p",1.0},
                {"repetition_penalty",1.0},{"random_seed",0}},
            "extension path must neutralize OGA filters but retain its seed");
    require(read_oga_sampling_options(extension_params) == json({
                {"do_sample",true},{"temperature",1.0},{"top_k",0},{"top_p",1.0},
                {"repetition_penalty",1.0},{"random_seed",0}}),
            "OGA readback must report neutral engine settings separately from effective settings");

    CapturedParams incompatible;
    incompatible.no_repeat_ngram_size = 2;
    rejects([&]{ apply_oga_sampling_options(incompatible, state.effective()); }, "no_repeat_ngram_size");
    require(incompatible.numbers.empty(), "incompatible model processors must fail before changing OGA options");

    state.reset_request({{"temperature",1.25},{"top_k",3},{"top_p",0.5},
                         {"seed",42},{"repetition_penalty",1.25}});
    require(!requires_logits_processing(state.effective()), "ordinary sampling keeps the native OGA path");
    CapturedParams ordinary_params;
    apply_oga_sampling_options(ordinary_params, state.effective());
    require(ordinary_params.do_sample && ordinary_params.numbers == std::map<std::string, double>{
                {"temperature",1.25},{"top_k",3},{"top_p",0.5},
                {"repetition_penalty",1.25},{"random_seed",42}},
            "ordinary sampling settings must pass through to OGA unchanged");

    for (double value : {0.0, 1.0}) {
        state.reset_request({{"min_p",value}});
        if (value == 0.0) {
            require(!state.effective().contains("min_p") && state.effective() == defaults,
                    "zero min_p must be omitted as a canonical no-op");
        } else {
            require(state.effective().at("min_p") == value,
                    "min_p upper endpoint must be accepted and preserved");
        }
    }
    state.reset_request({{"presence_penalty",3.5},{"frequency_penalty",-4.25}});
    require(state.effective().at("presence_penalty") == static_cast<float>(3.5) &&
            state.effective().at("frequency_penalty") == static_cast<float>(-4.25),
            "signed penalties outside plus or minus two must remain valid float32 values");
    rejects([&]{ state.reset_request({{"presence_penalty",-1e-50}}); }, "presence_penalty");
    rejects([&]{ state.reset_request({{"frequency_penalty",1e-50}}); }, "frequency_penalty");
    rejects([&]{ state.reset_request({{"min_p",1e-50}}); }, "min_p");

    state.reset_request({{"min_p",0},{"presence_penalty",0},{"frequency_penalty",0}});
    require(state.effective() == defaults && !state.effective().contains("min_p") &&
            !state.effective().contains("presence_penalty") && !state.effective().contains("frequency_penalty") &&
            !requires_logits_processing(state.effective()),
            "explicit zeros are canonical no-ops");
    state.reset_request({{"top_k",0},{"top_p",0.5}});
    require(state.effective().at("do_sample") == true && state.effective().at("top_k") == 0 &&
            state.effective().at("top_p") == 0.5,
            "sampled pure top-p must be accepted and retained");
    require(requires_logits_processing(state.effective()), "sampled pure top-p selects the extension path");
    state.reset_request({{"do_sample",false},{"top_k",0},{"top_p",0.5}});
    require(!requires_logits_processing(state.effective()),
            "greedy pure-top-p skips stochastic filters and the extension path");

    state.reset_request({{"min_p",0.2},{"presence_penalty",2.5},{"seed",123}});
    const auto prior = state.effective();
    rejects([&]{ state.update({{"temperature",2},{"frequency_penalty",1e100}}); }, "frequency_penalty");
    require(state.effective() == prior, "failed processor update must preserve all previous effective options");
    state.reset_request(json::object());
    require(state.effective() == defaults, "request B must restore model defaults after extension overrides");
    require(!state.requires_fresh_generator(), "request B must not retain the previous seed");

    state.reset_request({{"min_p",0.2},{"presence_penalty",-2.5},{"seed",7}});
    require(state.requires_fresh_generator(), "fixed-seed stochastic extension must reset the generator");
    state.reset_request({{"do_sample",false},{"presence_penalty",2.5},{"seed",7}});
    require(!state.requires_fresh_generator(), "greedy processor requests do not need seeded RNG reset");
    require(requires_logits_processing(state.effective()),
            "greedy presence penalties still require logits processing");

    for (const auto& value : {json(true),json("0.8"),json(nullptr),json(-0.1),
                              json(std::numeric_limits<double>::infinity()),json(1e100)}) {
        rejects([&]{ state.reset_request({{"temperature",value}}); }, "temperature");
    }
    for (const auto& pair : std::vector<std::pair<std::string,json>>{
        {"top_k",-1},{"top_k",1.5},{"top_p",-0.1},{"top_p",1.1},
        {"min_p",-0.1},{"min_p",1.1},{"repetition_penalty",0},
        {"repetition_penalty",-1},{"seed",2147483648LL},{"seed",1.5},
        {"seed",-2},{"do_sample",1}}) {
        rejects([&]{state.reset_request({{pair.first,pair.second}});},pair.first);
    }
    for (const auto& field : {"min_p", "presence_penalty", "frequency_penalty"}) {
        rejects([&]{ state.reset_request({{field,std::numeric_limits<double>::infinity()}}); }, field);
        rejects([&]{ state.reset_request({{field,std::numeric_limits<double>::quiet_NaN()}}); }, field);
    }

    state.reset_request({{"do_sample",false},{"temperature",1.5}});
    require(!state.effective().at("do_sample").get<bool>(), "explicit do_sample=false must be honored");
    state.reset_request({{"do_sample",true},{"temperature",0}});
    require(!state.effective().at("do_sample").get<bool>(), "temperature zero remains greedy");
    auto greedy_defaults = defaults; greedy_defaults["do_sample"] = false;
    SamplingState from_greedy(greedy_defaults);
    from_greedy.reset_request({{"top_p",0.5}});
    require(from_greedy.effective().at("do_sample").get<bool>(), "explicit filter must enable sampling");
    std::cout << "PASS: HIP OGA sampling request contract\n";
}
