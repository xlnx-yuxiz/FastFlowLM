// Raw OGA observations for sampling order and sequence visibility. This is a
// diagnostic fixture; it does not qualify any filter behavior.
#include <ort_genai.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

template <class Generator>
json sequence_snapshot(Generator& generator) {
    const std::size_t count = generator.GetSequenceCount(0);
    const int32_t* data = generator.GetSequenceData(0);
    if (count > 0 && data == nullptr) throw std::runtime_error("OGA sequence data is null for a nonempty sequence");
    json head = json::array();
    json tail = json::array();
    const std::size_t edge = std::min<std::size_t>(4, count);
    for (std::size_t i = 0; i < edge; ++i) head.push_back(data[i]);
    const std::size_t tail_start = count > edge ? count - edge : 0;
    for (std::size_t i = tail_start; i < count; ++i) tail.push_back(data[i]);
    return {{"count", count}, {"head", head}, {"tail", tail}};
}

json run_case(OgaModel& model, int vocab, const std::string& name,
              const json& settings, const std::vector<int32_t>& prompt,
              int steps, int expected_token = -1) {
    json result = {
        {"name", name}, {"status", "ERROR"}, {"parameters", settings},
        {"prompt_tokens", prompt}, {"steps_requested", steps},
        {"selected_ids", json::array()}, {"sequence_snapshots", json::object()}
    };
    if (expected_token >= 0) result["control_expected_token"] = expected_token;
    try {
        if (vocab <= 11 || steps <= 0 || prompt.empty()) throw std::runtime_error("invalid fixture dimensions or prompt");
        for (int32_t token : prompt) {
            if (token < 0 || token >= vocab)
                throw std::runtime_error("prompt token is outside the model vocabulary");
        }

        auto params = OgaGeneratorParams::Create(model);
        params->SetSearchOptionBool("do_sample", settings.at("do_sample").get<bool>());
        for (const char* key : {"temperature", "top_k", "top_p", "repetition_penalty",
                                "random_seed", "max_length", "batch_size"}) {
            params->SetSearchOption(key, settings.at(key).get<double>());
        }
        auto generator = OgaGenerator::Create(model, *params);
        generator->AppendTokens(prompt.data(), prompt.size());

        std::vector<int> selected;
        bool candidates_valid = true;
        for (int step = 0; step < steps; ++step) {
            const bool capture_first = step == 0;
            const bool capture_last = step == steps - 1;
            if (capture_first)
                result["sequence_snapshots"]["first"]["before_get_logits"] =
                    sequence_snapshot(*generator);
            if (capture_last)
                result["sequence_snapshots"]["last"]["before_get_logits"] =
                    sequence_snapshot(*generator);

            // Keep the current logits alive through replacement/sampling, as in
            // test_sampling_runtime.cpp; SetLogits only replaces a prepared row.
            auto current_logits = generator->GetLogits();
            if (capture_first)
                result["sequence_snapshots"]["first"]["after_get_logits"] =
                    sequence_snapshot(*generator);
            if (capture_last)
                result["sequence_snapshots"]["last"]["after_get_logits"] =
                    sequence_snapshot(*generator);

            std::vector<float> logits(static_cast<std::size_t>(vocab), -100.0f);
            logits[10] = 4.0f;
            logits[11] = 3.0f;
            const int64_t shape[] = {1, vocab};
            auto tensor = OgaTensor::Create(logits.data(), shape, 2, OgaElementType_float32);
            generator->SetLogits(*tensor);
            const std::size_t before_count = generator->GetSequenceCount(0);
            generator->GenerateNextToken();
            const std::size_t after_count = generator->GetSequenceCount(0);
            if (after_count <= before_count)
                throw std::runtime_error("GenerateNextToken did not append a sequence token");
            const int token = generator->GetSequenceData(0)[after_count - 1];
            selected.push_back(token);
            result["selected_ids"].push_back(token);
            result["selected_count"] = selected.size();
            if (token != 10 && token != 11) candidates_valid = false;
        }

        result["sequence_count_final"] = generator->GetSequenceCount(0);
        result["selected_ids_within_injected_candidates"] = candidates_valid;
        if (!candidates_valid) {
            result["status"] = "FAIL";
            result["error"] = "selected an ID outside the injected candidates {10,11}";
        } else if (expected_token >= 0 &&
                   !std::all_of(selected.begin(), selected.end(),
                                [expected_token](int token) { return token == expected_token; })) {
            result["status"] = "FAIL";
            result["error"] = "known control selected an unexpected token";
        } else {
            result["status"] = expected_token >= 0 ? "PASS" : "EXECUTED";
        }
    } catch (const std::exception& error) {
        result["status"] = "ERROR";
        result["error"] = error.what();
    }
    return result;
}

json settings(bool do_sample, int top_k, double top_p, double temperature,
              double repetition_penalty) {
    return {{"do_sample", do_sample}, {"top_k", top_k}, {"top_p", top_p},
            {"temperature", temperature}, {"repetition_penalty", repetition_penalty},
            {"random_seed", 42}, {"batch_size", 1}, {"max_length", 128}};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: probe_sampling_order <model_dir> <report.json>\n";
        return 2;
    }

    OgaHandle handle;
    json report = {{"status", "FAIL"}, {"model_dir", argv[1]}, {"cases", json::array()}};
    try {
        auto model = OgaModel::Create(argv[1]);
        std::ifstream input(std::string(argv[1]) + "/genai_config.json");
        if (!input) throw std::runtime_error("could not open genai_config.json");
        const int vocab = json::parse(input).at("model").at("vocab_size").get<int>();
        const std::vector<int32_t> diagnostic_prompt{10, 10, 11};

        report["cases"].push_back(run_case(*model, vocab, "topk2_p065_temp1", settings(true, 2, 0.65, 1.0, 1.0), diagnostic_prompt, 48));
        report["cases"].push_back(run_case(*model, vocab, "topk2_p065_temp2", settings(true, 2, 0.65, 2.0, 1.0), diagnostic_prompt, 48));
        report["cases"].push_back(run_case(*model, vocab, "pure_topp_p065_temp1", settings(true, 0, 0.65, 1.0, 1.0), diagnostic_prompt, 48));
        report["cases"].push_back(run_case(*model, vocab, "unfiltered_topk0_p1_temp1", settings(true, 0, 1.0, 1.0, 1.0), diagnostic_prompt, 48));
        report["cases"].push_back(run_case(*model, vocab, "control_topk1_temp1_p1", settings(true, 1, 1.0, 1.0, 1.0), diagnostic_prompt, 48, 10));
        report["cases"].push_back(run_case(*model, vocab, "control_greedy_repetition2", settings(false, 0, 1.0, 1.0, 2.0), {10, 10}, 1, 11));

        bool complete = true;
        for (const auto& item : report["cases"])
            complete = complete && item.at("status") != "FAIL" && item.at("status") != "ERROR";
        report["status"] = complete ? "DIAGNOSTIC_COMPLETE" : "FAIL";
        report["qualification"] = false;
        report["note"] = "Diagnostic completion and controls are not feature qualification; no filter order is asserted.";
    } catch (const std::exception& error) {
        report["error"] = error.what();
    }

    std::ofstream output(argv[2]);
    if (!output) {
        std::cerr << "could not open report path\n";
        return 2;
    }
    output << report.dump(2) << '\n';
    json summary = {{"status", report["status"]}, {"case_count", report["cases"].size()},
                    {"cases", json::array()}};
    for (const auto& item : report["cases"])
        summary["cases"].push_back({{"name", item["name"]}, {"status", item["status"]},
                                    {"selected_count", item.value("selected_count", 0)},
                                    {"error", item.value("error", "")}});
    std::cout << summary.dump() << std::endl;
    return report["status"] == "DIAGNOSTIC_COMPLETE" ? 0 : 1;
}
