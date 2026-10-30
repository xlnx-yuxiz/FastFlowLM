// Controlled scores at the real OGA boundary, using the production mapping and processor.
#include "AutoModel/hip_oga_sampling.hpp"
#include "AutoModel/hip_oga_constraint.hpp"
#include <ort_genai.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

using nlohmann::json;
using namespace flm::hip_oga;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    OgaHandle handle;
    json report = {{"status", "FAIL"}, {"cases", json::array()}};
    try {
        auto model = OgaModel::Create(argv[1]);
        auto model_params = OgaGeneratorParams::Create(*model);
        const auto defaults = read_oga_sampling_options(*model_params);
        std::ifstream input(std::string(argv[1]) + "/genai_config.json");
        const auto config = json::parse(input);
        const int vocab = config.at("model").at("vocab_size").get<int>();
        require(vocab > 20, "fixture vocabulary too small");
        const auto& eos_value = config.at("model").at("eos_token_id");
        const std::vector<int> eos_ids = eos_value.is_array() ? eos_value.get<std::vector<int>>() :
            std::vector<int>{eos_value.get<int>()};
        require(!eos_ids.empty(), "fixture needs model EOS IDs");
        report["model_search"] = config.at("search");
        auto record = [&](const char* name, const std::function<void(json&)>& check) {
            json result = {{"name", name}, {"status", "FAIL"}, {"runs", json::array()}};
            try { check(result); result["status"] = "PASS"; }
            catch (const std::exception& e) { result["error"] = e.what(); }
            report["cases"].push_back(result);
        };
        using Scores = std::vector<std::pair<int, float>>;
        const Scores two = {{10, 4}, {11, 3}};
        const json stochastic = {{"temperature", 1}, {"top_k", 0}, {"top_p", 1}, {"seed", 42}, {"repetition_penalty", 1}};
        auto settings = [&](const json& updates) { auto value = stochastic; value.update(updates); return value; };

        auto run = [&](json& result, const json& overrides, const Scores& supplied, int count,
                       const std::vector<int32_t>& prompt = std::vector<int32_t>{10},
                       const std::vector<int>& excluded = std::vector<int>{}, bool rewind_append = false, int min_length = 0) {
            SamplingState state(defaults, vocab);
            state.reset_request(overrides);
            auto params = OgaGeneratorParams::Create(*model);
            params->SetSearchOption("max_length", 128);
            params->SetSearchOption("batch_size", 1);
            params->SetSearchOption("min_length", min_length);
            apply_oga_sampling_options(*params, state.effective());
            const auto runtime = read_oga_sampling_options(*params);
            const bool process = requires_logits_processing(state.effective());
            json run_report = {{"effective", state.effective()}, {"runtime", runtime},
                               {"processor", process}, {"prompt", prompt}, {"rewind_append", rewind_append},
                               {"min_length", min_length}, {"eos_ids", eos_ids}};
            auto generator = OgaGenerator::Create(*model, *params);
            generator->AppendTokens(prompt.data(), prompt.size());
            std::vector<int> selected;
            auto step = [&](const Scores& injected) {
                const auto before_count = generator->GetSequenceCount(0);
                const auto* before_data = generator->GetSequenceData(0);
                std::vector<int32_t> before(before_data, before_data + before_count);
                auto current_logits = generator->GetLogits();
                require(generator->GetSequenceCount(0) == before.size() &&
                        std::equal(before.begin(), before.end(), generator->GetSequenceData(0)),
                        "GetLogits changed the current penalty history");
                std::vector<float> logits(vocab, -std::numeric_limits<float>::infinity());
                for (auto score : injected) logits.at(score.first) = score.second;
                // Apply the same grammar/decoder mask before sampling processors.
                if (!excluded.empty()) {
                    std::vector<std::uint32_t> mask((vocab + 31) / 32, 0);
                    for (auto score : injected) mask[score.first / 32] |= 1u << (score.first % 32);
                    exclude_constraint_tokens(mask, vocab, excluded);
                    mask_constraint_logits(logits, mask);
                }
                if (process) process_oga_sampling_logits(logits, generator->GetSequenceData(0),
                    generator->GetSequenceCount(0), state.effective(), *params, eos_ids);
                std::vector<bool> allowed(vocab);
                for (int id = 0; id < vocab; ++id) allowed[id] = std::isfinite(logits[id]);
                const int64_t shape[] = {1, vocab};
                auto tensor = OgaTensor::Create(logits.data(), shape, 2, OgaElementType_float32);
                generator->SetLogits(*tensor);
                generator->GenerateNextToken();
                const auto next_tokens = generator->GetNextTokens();
                require(next_tokens.size() == 1, "generation did not return one token");
                const int token = next_tokens[0];
                const bool eos = std::find(eos_ids.begin(), eos_ids.end(), token) != eos_ids.end();
                if (eos) {
                    require(generator->IsDone() && before_count >= static_cast<std::size_t>(min_length),
                            "EOS selected before minimum length or did not terminate");
                } else {
                    require(generator->GetSequenceCount(0) == before_count + 1, "generation did not append exactly one token");
                    require(!generator->IsDone(), "non-EOS token unexpectedly terminated");
                }
                require(std::find(excluded.begin(), excluded.end(), token) == excluded.end(), "sampled a masked token");
                require(token >= 0 && token < vocab && allowed[token], "sampled outside processed support");
                return token;
            };
            if (rewind_append) {
                require(prompt == std::vector<int32_t>({10, 10}), "rewind fixture prompt drifted");
                require(step({{10, 10}, {11, -100}}) == 10, "warmup did not append the discarded token");
                generator->RewindTo(prompt.size());
                const int32_t suffix[] = {11};
                generator->AppendTokens(suffix, 1);
                const std::vector<int32_t> wanted{10, 10, 11};
                require(generator->GetSequenceCount(0) == wanted.size() &&
                        std::equal(wanted.begin(), wanted.end(), generator->GetSequenceData(0)),
                        "rewind/append retained discarded history or lost suffix");
            }
            for (int i = 0; i < count; ++i) selected.push_back(step(supplied));
            run_report["tokens"] = selected;
            result["runs"].push_back(run_report);
            if (process) {
                require(runtime.at("temperature") == 1 && runtime.at("repetition_penalty") == 1 &&
                        runtime.at("top_k") == 0 && runtime.at("top_p") == 1,
                        "extension runtime processors were not neutralized");
                require(runtime.at("do_sample") == state.effective().at("do_sample") &&
                        runtime.at("random_seed") == state.effective().at("random_seed"), "OGA RNG settings changed");
            }
            return selected;
        };
        const auto only = [](const std::vector<int>& tokens, int id) {
            return !tokens.empty() && std::all_of(tokens.begin(), tokens.end(), [=](int value) { return value == id; });
        };
        const auto both = [](const std::vector<int>& tokens) {
            return std::all_of(tokens.begin(), tokens.end(), [](int id) { return id == 10 || id == 11; }) &&
                   std::find(tokens.begin(), tokens.end(), 10) != tokens.end() &&
                   std::find(tokens.begin(), tokens.end(), 11) != tokens.end();
        };
        record("ordinary_greedy", [&](json& c) { require(only(run(c, settings({{"temperature", 0}}), two, 1), 10), "greedy argmax mismatch"); });
        record("ordinary_repetition_prompt", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"repetition_penalty", 2}}), two, 1, {10, 10}), 11), "prompt repetition mismatch"); });
        record("ordinary_top_k", [&](json& c) { require(both(run(c, settings({{"top_k", 2}}), {{10, 4}, {11, 4}, {12, 3}}, 24)), "top-k support mismatch"); });
        record("ordinary_top_p_with_k", [&](json& c) { require(only(run(c, settings({{"top_k", 2}, {"top_p", 0.65}}), two, 16), 10), "combined nucleus mismatch"); });
        record("pure_top_p", [&](json& c) { require(only(run(c, settings({{"top_p", 0.65}}), two, 24), 10), "pure nucleus admitted lower candidate"); });
        record("min_p", [&](json& c) { require(only(run(c, settings({{"min_p", 0.5}}), two, 24), 10), "min-p admitted lower candidate"); });
        record("presence_once", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"presence_penalty", 0.75}}), two, 1, {10, 10}), 10), "presence counted duplicate history"); });
        record("presence_negative", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"presence_penalty", -2}}), two, 1, {11}), 11), "negative presence did not promote seen token"); });
        record("frequency_counts", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"frequency_penalty", 0.75}}), two, 1, {10, 10}), 11), "frequency did not count prompt duplicates"); });
        record("frequency_negative", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"frequency_penalty", -0.75}}), two, 1, {11, 11}), 11), "negative frequency mismatch"); });
        record("repetition_before_presence", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"repetition_penalty", 2}, {"presence_penalty", 1.6}}), {{10, 4}, {11, 1}}, 1), 11), "penalty ordering mismatch"); });
        record("repetition_once", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"repetition_penalty", 2}, {"presence_penalty", 0.1}}), {{10, 4}, {11, 1}}, 1), 10), "repetition ran twice"); });
        record("temperature_once_before_nucleus", [&](json& c) {
            const auto transformed = run(c, settings({{"temperature", 2}, {"top_k", 2}, {"top_p", 0.65}, {"min_p", 0.1}}), two, 32);
            const auto reference = run(c, stochastic, {{10, 0}, {11, -0.5}}, 32);
            require(both(transformed) && transformed == reference, "temperature/nucleus order or double processing mismatch");
        });
        record("top_k_then_nucleus_ties", [&](json& c) { require(only(run(c, settings({{"top_k", 2}, {"top_p", 0.4}, {"min_p", 0.1}}), {{10, 4}, {11, 4}, {12, 4}, {13, 4}}, 16), 10), "top-k/nucleus tie support mismatch"); });
        record("mask_with_negative_penalties", [&](json& c) { require(only(run(c, settings({{"presence_penalty", -100}, {"frequency_penalty", -100}, {"top_p", 0.65}}), two, 8, {10}, {10}), 11), "negative penalties revived a masked token"); });
        record("greedy_skips_filters", [&](json& c) { require(only(run(c, settings({{"temperature", 0}, {"top_k", 1}, {"top_p", 0.01}, {"min_p", 1}, {"presence_penalty", 2}}), two, 1), 11), "greedy applied filters before penalty"); });
        record("extension_seed_repeat", [&](json& c) {
            const auto options = settings({{"top_p", 0.9}, {"presence_penalty", 0.2}, {"frequency_penalty", -0.1}});
            const auto a = run(c, options, two, 24);
            require(a == run(c, options, two, 24), "extension fixed-seed fresh generators did not repeat");
        });
        record("rewind_append_history", [&](json& c) {
            const auto options = settings({{"temperature", 0}, {"frequency_penalty", 0.6}});
            const auto reused = run(c, options, two, 1, {10, 10}, {}, true);
            const auto fresh = run(c, options, two, 1, {10, 10, 11});
            require(reused == fresh && only(fresh, 10), "rewind history differs from fresh sequence");
        });
        for (bool extension : {false, true}) {
            record(extension ? "extension_min_length_eos" : "ordinary_min_length_eos", [&](json& c) {
                Scores candidates{{11, 3}};
                for (int eos : eos_ids) candidates.push_back({eos, 4});
                const auto options = settings(extension ? json{{"min_p", 1}} : json{{"top_k", 1}});
                const auto tokens = run(c, options, candidates, 3, {10}, {}, false, 3);
                require(tokens[0] == 11 && tokens[1] == 11 &&
                        std::find(eos_ids.begin(), eos_ids.end(), tokens[2]) != eos_ids.end(),
                        "model EOS eligibility must precede truncation and allow termination at minimum length");
            });
        }
        report["case_count"] = report["cases"].size();
        report["status"] = std::all_of(report["cases"].begin(), report["cases"].end(),
            [](const json& c) { return c.at("status") == "PASS"; }) ? "PASS" : "FAIL";
    } catch (const std::exception& e) { report["error"] = e.what(); }
    std::ofstream output(argv[2]);
    if (!output) return 2;
    output << report.dump(2) << '\n';
    std::cout << report.dump(2) << std::endl;
    return report["status"] == "PASS" ? 0 : 1;
}
