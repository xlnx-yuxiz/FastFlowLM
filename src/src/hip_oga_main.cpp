/// \file hip_oga_main.cpp
/// \brief Minimal FastFlowLM HIP OGA smoke runner.
///
/// This executable is intentionally separate from the NPU CLI/server target.
/// It validates that FastFlowLM can load an OGA model through the AMDGPU
/// umbrella provider without pulling in XRT, FFmpeg, CURL, Boost, or the
/// FastFlowLM NPU engine DLLs.

#include "ort_genai.h"
#include "hip_oga_policy.hpp"
#include "hip_oga_config_directory.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string command = "run";
    std::string model_path;
    std::string backend;
    std::string provider = "AMDGPU";
    std::string profile = "hip";
    std::string offload_policy;
    std::string prompt = "Hello";
    int max_tokens = 1;
    bool completion_only = false;
    bool dump_tokens = false;
    bool help = false;
};

void PrintHelp() {
    std::cout
        << "Usage: flm run <oga_model_dir> --backend hip [options]\n"
        << "\n"
        << "HIP OGA-only options:\n"
        << "  --backend hip          Required unless FLM_BACKEND=hip is set\n"
        << "  --provider <name>      OGA provider name (default: AMDGPU)\n"
        << "  --profile <name>       AMDGPU provider profile (default: hip)\n"
        << "  --offload-policy <p>   strict_gpu or hybrid_gpu (default: strict_gpu)\n"
        << "  --prompt <text>        Prompt text (default: Hello)\n"
        << "  --max-tokens <n>       Decode token count, at least 1 (default: 1)\n"
        << "  --completion-only      Print only generated text, not prompt+generated text\n"
        << "  --dump-tokens          Print prompt/generated token IDs and decoded text\n"
        << "  --help                 Show this help\n";
}

bool StartsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.substr(0, prefix.size()) == prefix;
}

std::string ConsumeValue(int& index, int argc, char** argv, std::string_view flag) {
    const std::string current = argv[index];
    const std::string prefix = std::string(flag) + "=";
    if (StartsWith(current, prefix)) {
        return current.substr(prefix.size());
    }
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++index;
    return argv[index];
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    bool saw_command = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            options.help = true;
        } else if (arg == "run" && !saw_command) {
            options.command = "run";
            saw_command = true;
        } else if (arg == "--backend" || StartsWith(arg, "--backend=")) {
            options.backend = ConsumeValue(i, argc, argv, "--backend");
        } else if (arg == "--provider" || StartsWith(arg, "--provider=")) {
            options.provider = ConsumeValue(i, argc, argv, "--provider");
        } else if (arg == "--profile" || StartsWith(arg, "--profile=")) {
            options.profile = ConsumeValue(i, argc, argv, "--profile");
        } else if (arg == "--offload-policy" || StartsWith(arg, "--offload-policy=")) {
            options.offload_policy = ConsumeValue(i, argc, argv, "--offload-policy");
        } else if (arg == "--prompt" || StartsWith(arg, "--prompt=")) {
            options.prompt = ConsumeValue(i, argc, argv, "--prompt");
        } else if (arg == "--max-tokens" || StartsWith(arg, "--max-tokens=") ||
                   arg == "-n") {
            options.max_tokens =
                std::stoi(ConsumeValue(i, argc, argv, arg == "-n" ? "-n" : "--max-tokens"));
        } else if (arg == "--completion-only") {
            options.completion_only = true;
        } else if (arg == "--dump-tokens") {
            options.dump_tokens = true;
        } else if (arg == "--ctx-len" || StartsWith(arg, "--ctx-len=") ||
                   arg == "-c") {
            // Accepted for command-line compatibility. OGA owns max length from
            // genai_config.json; --max-tokens controls this smoke run.
            (void)ConsumeValue(i, argc, argv, arg == "-c" ? "-c" : "--ctx-len");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("Unknown option: " + arg);
        } else if (options.model_path.empty()) {
            options.model_path = arg;
        } else {
            throw std::invalid_argument("Unexpected positional argument: " + arg);
        }
    }

    if (const char* env_backend = std::getenv("FLM_BACKEND");
        options.backend.empty() && env_backend && *env_backend) {
        options.backend = env_backend;
    }
    if (options.offload_policy.empty()) {
        options.offload_policy = flm::hip_oga::direct_model_offload_policy();
    }
    if (!flm::hip_oga::is_runtime_offload_policy(options.offload_policy)) {
        throw std::invalid_argument(
            "--offload-policy must be strict_gpu or hybrid_gpu");
    }

    if (options.max_tokens < 1) {
        throw std::invalid_argument("--max-tokens must be at least 1");
    }
    if (options.command != "run") {
        throw std::invalid_argument("HIP OGA-only flm currently supports only the run command");
    }
    if (options.backend != "hip") {
        throw std::invalid_argument(
            "HIP OGA-only flm requires --backend hip or FLM_BACKEND=hip");
    }
    if (!options.help && options.model_path.empty()) {
        throw std::invalid_argument("Missing OGA model directory");
    }
    return options;
}

std::filesystem::path ResolveModelDir(const std::string& model_path) {
    std::filesystem::path path(model_path);
    const std::filesystem::path config = path / "genai_config.json";
    if (!std::filesystem::exists(config)) {
        throw std::runtime_error(
            "OGA model directory must contain genai_config.json: " + path.string());
    }
    return path;
}

nlohmann::json ReadJsonFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    auto parsed = nlohmann::json::parse(file, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        throw std::runtime_error(path.string() + " is not a JSON object");
    }
    return parsed;
}

std::string JoinTokenIds(const std::vector<int32_t>& tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) out << ",";
        out << tokens[i];
    }
    return out.str();
}

int RunHipOga(const Options& options) {
    OgaHandle oga_handle;
    const auto model_dir = ResolveModelDir(options.model_path);

    const auto genai_config = ReadJsonFile(model_dir / "genai_config.json");
    const auto effective_config = flm::hip_oga::build_effective_config(
        genai_config,
        nlohmann::json{{"provider", options.provider},
                       {"profile", options.profile},
                       {"offload_policy", options.offload_policy}},
        flm::hip_oga::getenv_flag_enabled("FLM_HIP_OGA_DIAGNOSTICS"),
        flm::hip_oga::getenv_string("FLM_HIP_OGA_PROFILE_PREFIX"));
    flm::hip_oga::ScopedConfigDirectory config_directory(model_dir, effective_config);
    auto config = OgaConfig::Create(config_directory.path().string().c_str());

    std::cerr << "FLM HIP OGA provider=" << options.provider
              << " profile=" << options.profile
              << " offload_policy=" << options.offload_policy << "\n";
    if (options.offload_policy == flm::hip_oga::kStrictGpuPolicy) {
        std::cerr << "FLM HIP OGA strict_gpu enforces session.disable_cpu_ep_fallback=1"
                  << " for each OGA session section\n";
    }
    const std::string warning = flm::hip_oga::policy_warning(options.offload_policy);
    if (!warning.empty()) {
        std::cerr << "FLM HIP OGA WARNING: " << warning << "\n";
    }

    auto model = OgaModel::Create(*config);
    auto tokenizer = OgaTokenizer::Create(*model);
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(options.prompt.c_str(), *sequences);
    if (sequences->Count() != 1 || sequences->SequenceCount(0) == 0) {
        throw std::runtime_error("Tokenizer produced no prompt tokens");
    }
    const size_t prompt_token_count = sequences->SequenceCount(0);
    const int32_t* prompt_token_data = sequences->SequenceData(0);
    std::vector<int32_t> prompt_tokens(
        prompt_token_data, prompt_token_data + prompt_token_count);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("batch_size", 1);
    params->SetSearchOption(
        "max_length",
        static_cast<double>(sequences->SequenceCount(0) + options.max_tokens));

    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*sequences);
    auto stream = OgaTokenizerStream::Create(*tokenizer);
    std::vector<int32_t> generated_tokens;
    std::string completion_text;
    for (int i = 0; i < options.max_tokens && !generator->IsDone(); ++i) {
        const size_t before_count = generator->GetSequenceCount(0);
        generator->GenerateNextToken();
        const size_t after_count = generator->GetSequenceCount(0);
        const int32_t* sequence_data = generator->GetSequenceData(0);
        for (size_t j = before_count; j < after_count; ++j) {
            const int32_t token = sequence_data[j];
            generated_tokens.push_back(token);
            const char* piece = stream->Decode(token);
            if (piece != nullptr) {
                completion_text += piece;
            }
        }
    }

    const int32_t* data = generator->GetSequenceData(0);
    const size_t count = generator->GetSequenceCount(0);
    const auto decoded = tokenizer->Decode(data, count);
    if (options.dump_tokens) {
        std::cout << "prompt_token_count=" << prompt_tokens.size() << "\n";
        std::cout << "prompt_token_ids=" << JoinTokenIds(prompt_tokens) << "\n";
        std::cout << "generated_token_count=" << generated_tokens.size() << "\n";
        std::cout << "generated_token_ids=" << JoinTokenIds(generated_tokens) << "\n";
        std::cout << "completion_text_begin\n"
                  << completion_text
                  << "\ncompletion_text_end\n";
        std::cout << "full_text_begin\n"
                  << static_cast<const char*>(decoded)
                  << "\nfull_text_end\n";
    } else if (options.completion_only) {
        std::cout << completion_text << "\n";
    } else {
        std::cout << static_cast<const char*>(decoded) << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        if (options.help) {
            PrintHelp();
            return 0;
        }
        return RunHipOga(options);
    } catch (const std::exception& ex) {
        std::cerr << "flm HIP OGA error: " << ex.what() << "\n";
        std::cerr << "Run `flm --help` for usage.\n";
        return 1;
    }
}
