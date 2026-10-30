// Isolated upstream-runtime reproducer. Exit 1 reproduces the known nucleus failure.
#include <ort_genai.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    OgaHandle handle;
    nlohmann::json report;
    try {
        auto model = OgaModel::Create(argv[1]);
        std::ifstream input(std::string(argv[1]) + "/genai_config.json");
        const int vocab = nlohmann::json::parse(input).at("model").at("vocab_size");
        bool matches = true;
        for (double p : {0.5, 0.1}) {
            auto params = OgaGeneratorParams::Create(*model);
            params->SetSearchOptionBool("do_sample", true);
            params->SetSearchOption("temperature", 1);
            params->SetSearchOption("top_k", 0);
            params->SetSearchOption("top_p", p);
            params->SetSearchOption("repetition_penalty", 1);
            params->SetSearchOption("random_seed", 42);
            params->SetSearchOption("max_length", 32);
            auto generator = OgaGenerator::Create(*model, *params);
            const int32_t prompt[] = {10}; generator->AppendTokens(prompt, 1);
            std::vector<int> selected;
            for (int i=0; i<12; ++i) {
                auto current = generator->GetLogits();
                std::vector<float> logits(vocab, -100); logits[10]=4; logits[11]=3;
                const int64_t shape[] = {1, vocab};
                auto tensor = OgaTensor::Create(logits.data(), shape, 2, OgaElementType_float32);
                generator->SetLogits(*tensor); generator->GenerateNextToken();
                selected.push_back(generator->GetSequenceData(0)[generator->GetSequenceCount(0)-1]);
            }
            matches = matches && std::all_of(selected.begin(),selected.end(),[](int id){return id==10;});
            report["cases"].push_back({{"top_k",params->GetSearchNumber("top_k")},{"top_p",params->GetSearchNumber("top_p")},{"tokens",selected}});
        }
        report["expected_token"] = 10;
        report["matches_nucleus_contract"] = matches;
    } catch (const std::exception& e) { report["error"]=e.what(); }
    std::ofstream(argv[2]) << report.dump(2) << '\n';
    std::cout << report.dump(2) << std::endl;
    return report.value("matches_nucleus_contract",false)?0:1;
}
