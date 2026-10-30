/*
*  Copyright (c) 2026 Advanced Micro Devices, Inc.
*  \file runner.cpp
*  \brief Runner implementation for interactive model execution
*  \author FastFlowLM Team
*  \date 2025-08-05
*  \version 0.9.21
*/
#include "runner.hpp"
#include "backend_options.hpp"
#include "chat/chat_constraint.hpp"
#include "chat/auto_tools.hpp"
#include "chat/chat_output.hpp"
#include "harmony_filter.hpp"
#include "hip_oga_policy.hpp"
#include "hip_oga_ensure_model.hpp"
#include "AutoModel/hip_oga_sampling.hpp"
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "AutoEmbeddingModel/all_embedding_model.hpp"
#endif
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <map>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <stdexcept>

/// \brief Command map for command line input
std::map<std::string, runner_cmd_t> cmd_map = {
    {"/set", CMD_SET},
    {"/show", CMD_SHOW},
    {"/load", CMD_LOAD},
    {"/save", CMD_SAVE},
    {"/clear", CMD_CLEAR},
    {"/bye", CMD_BYE},
    {"/pull", CMD_PULL},
    {"/?", CMD_HELP},
    {"/? shortcuts", CMD_HELP_SHOTCUT},
};

namespace {

void report_hip_oga_offload_policy(const json& model_info) {
    const std::string policy = flm::hip_oga::runtime_offload_policy(
        nlohmann::json::parse(model_info.dump()));
    header_print("FLM", "HIP OGA offload_policy=" << policy);
    if (policy == flm::hip_oga::kStrictGpuPolicy) {
        header_print("FLM", "strict_gpu enforces session.disable_cpu_ep_fallback=1 for each OGA session section.");
    }
    const std::string warning = flm::hip_oga::policy_warning(policy);
    if (!warning.empty()) {
        header_print("WARNING", warning);
    }
}

}  // namespace

/// \brief Constructor
/// \param supported_models - the list of supported models
/// \param downloader - the downloader for the models
/// \param tag - the tag of the model to load
Runner::Runner(model_list& supported_models, ModelDownloader& downloader, program_args_t& args)
    : supported_models(supported_models), downloader(downloader), tag(args.model_tag),
      initial_prompt(args.input_file_name), modelscope(args.modelscope), asr(args.asr), embed(args.embed),
      img_pre_resize(args.img_pre_resize), preemption(args.preemption),
      backend(flm::backend::normalize_backend(args.backend)), sampler_provider(args.sampler) {

    if (flm::backend::is_hip_oga_backend(this->backend) && (this->asr || this->embed)) {
        throw std::runtime_error(
            "ASR and embedding models remain NPU-only; do not combine them with --backend hip");
    }

    if (!args.tools_file.empty()) {
        if (!flm::backend::is_hip_oga_backend(this->backend) || this->asr || this->embed)
            throw std::invalid_argument("--tools currently requires a text model with --backend hip");
        this->tool_session = std::make_unique<flm::chat::cli_tools_session>(
            std::filesystem::u8path(args.tools_file), args.tool_executor,
            std::filesystem::u8path(args.tool_data_dir), args.max_tool_turns, args.tools_choice);
    }

    if (args.ctx_length != -1) {
        this->ctx_length = args.ctx_length >= 512 ? args.ctx_length : 512;
    } 
    else {
        this->ctx_length = -1;
    }

    if (this->auto_chat_engine != nullptr) {
        this->auto_chat_engine.reset();
    }
    const bool using_hip_oga = flm::backend::is_hip_oga_backend(this->backend);
    const bool hip_direct_path =
        using_hip_oga && HipOgaModel::is_model_dir(this->tag);
    json model_info;
    std::string model_path;
    if (hip_direct_path) {
        this->auto_chat_engine = std::make_unique<HipOgaModel>();
        model_info = HipOgaModel::model_info_for_path(this->tag);
        model_path = this->tag;
    } else if (using_hip_oga) {
        std::pair<std::string, std::unique_ptr<AutoModel>> auto_model =
            get_auto_model(this->tag, this->supported_models, nullptr, this->backend);
        this->auto_chat_engine = std::move(auto_model.second);
        this->tag = auto_model.first;

        auto [new_tag, selected_model_info] = this->supported_models.get_model_info(this->tag);
        model_info = std::move(selected_model_info);
        model_path = flm::hip_oga::ensure_hip_oga_model(this->supported_models, new_tag);
        model_info["hip_oga"]["config_path"] = model_path;
    } else {
        std::pair<std::string, std::unique_ptr<AutoModel>> auto_model =
            get_auto_model(this->tag, this->supported_models, ensure_npu_device(), this->backend);
        this->auto_chat_engine = std::move(auto_model.second);
        this->tag = auto_model.first;

        switch (this->downloader.is_model_downloaded(this->tag)) {
            case ModelDownloader::ModelStatus::Ready:
                break;
            case ModelDownloader::ModelStatus::Outdated:
            case ModelDownloader::ModelStatus::Missing:
                this->downloader.pull_model(this->tag, this->modelscope);
                break;
            case ModelDownloader::ModelStatus::Incompatible:
                exit(EXIT_FAILURE);
        }
        auto [new_tag, selected_model_info] = this->supported_models.get_model_info(this->tag);
        model_info = std::move(selected_model_info);
        model_path = this->supported_models.get_model_path(new_tag);
    }
    this->asr_supported = model_info.contains("asr") && model_info["asr"];
    // header_print("ASR", asr_supported);
    this->auto_chat_engine->configure_parameter("img_pre_resize", this->img_pre_resize);
    if (using_hip_oga) {
        report_hip_oga_offload_policy(model_info);
    }
    try {
        this->auto_chat_engine->load_model(model_path, model_info, this->ctx_length, this->preemption, this->backend);
        this->auto_chat_engine->set_sampling_provider(this->sampler_provider);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }

    try {
        if (args.prefill_chunk_len != -1) {
            this->prefill_chunk_len = args.prefill_chunk_len >= 512 ? args.prefill_chunk_len : 512;
        }
        else {
            this->prefill_chunk_len = model_info["max_prefill_len"].get<int>();;
        }
    }
    catch (const std::exception& e) {
        this->prefill_chunk_len = 4096; // default value
    }

    this->generate_limit = (this->tool_session && args.max_tokens == -1) ? 256 : args.max_tokens;
    if (this->tool_session) {
        this->auto_chat_engine->set_log_raw_output(false);
        this->auto_chat_engine->set_temperature(0.0f);
    }


    if (this->embed) {
        header_print("Warning", "Embed model not supported in CLI; Use 'flm serve -e 1'");
    }

    header_print("FLM", "Loading model: " << this->tag);

#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    if (this->asr) {
        // check if multi-modal model with ASR support is loaded, if not, load the default whisper model for ASR
        if (!asr_supported) 
        {
            header_print("FLM", "The loaded model does not support ASR. Loading default Whisper model for ASR...");
            std::string whisper_tag = "whisper-v3:turbo";
            switch (this->downloader.is_model_downloaded(whisper_tag)) {
                case ModelDownloader::ModelStatus::Ready:
                    break;
                case ModelDownloader::ModelStatus::Outdated:
                case ModelDownloader::ModelStatus::Missing:
                    this->downloader.pull_model(whisper_tag, this->modelscope);
                    break;
                case ModelDownloader::ModelStatus::Incompatible:
                    header_print("ERROR", "Whisper is incompatible with this version of FastFlowLM, skipping... ");
                    return;
            }
            this->whisper_engine = std::make_unique<Whisper>(ensure_npu_device());
            auto [new_whisper_tag, whisper_model_info] = this->supported_models.get_model_info(whisper_tag);
            std::string whisper_model_path = this->supported_models.get_model_path(new_whisper_tag);
            try {
                this->whisper_engine->load_model(whisper_model_path, whisper_model_info, this->preemption);
            }
            catch (const std::exception& e) {
                header_print("WARNING", "Failed to load ASR model: " + std::string(e.what()));
                exit(EXIT_FAILURE);
            }
        }
        else {
            header_print("FLM", "The loaded model (" + this->tag + ") already supports ASR. No additional ASR model is needed.");
        }
    }
#else
    if (this->asr) {
        header_print("Error", "ASR models are not supported in this build");
    }
#endif
}





flm_rt::device* Runner::ensure_npu_device() {
    if (!this->npu_device_inst) {
        this->npu_device_inst = std::make_unique<flm_rt::device>(0);
    }
    return this->npu_device_inst.get();
}

bool Runner::generate_tools(const std::string& input, std::ostream& output) {
    try {
        this->tool_session->run(input, [&](const json& messages, const json& tools) {
            this->auto_chat_engine->clear_context();
            const bool automatic = this->tool_session->automatic();
            const json request = {{"tool_choice", automatic ? "auto" : "required"}, {"parallel_tool_calls", false},
                                  {"tool_routing", automatic ? "structured" : "template"}};
            chat_constraint_request constraint_request;
            std::string error;
            if (!chat_constraint_request_from_openai(request, tools, constraint_request, error))
                throw std::invalid_argument(error);
            chat_meta_info_t meta;
            meta.max_prefill_len = this->prefill_chunk_len;
            meta.constraint = this->auto_chat_engine->make_constraint(constraint_request);
            if (!meta.constraint) throw std::invalid_argument("Model cannot enforce tool calls");
            lm_uniform_input_t request_input;
            request_input.messages = automatic ? chat_prepare_auto_tools(messages, tools) : messages;
            request_input.tools = automatic ? json::array() : tools;
            (void)this->auto_chat_engine->configure_parameter("hip_oga_requested_max_tokens", this->generate_limit);
            this->auto_chat_engine->start_total_timer();
            this->auto_chat_engine->start_ttft_timer();
            if (!this->auto_chat_engine->insert(meta, request_input))
                throw std::runtime_error("Tool prompt exceeds model context");
            this->auto_chat_engine->stop_ttft_timer();
            std::ostringstream hidden;
            const auto raw = this->auto_chat_engine->generate(meta, this->generate_limit, hidden);
            this->auto_chat_engine->stop_total_timer();
            const auto& format = this->auto_chat_engine->get_chat_format();
            chat_output parser(format, chat_format_prefill(format, meta.prompt_tail), tools, true, true, false, automatic);
            const auto parsed = parser.update(raw, true);
            if (!parsed.error.empty()) throw std::runtime_error(parsed.error);
            return parsed.message;
        }, output);
        this->auto_chat_engine->clear_context();
        return true;
    } catch (const std::exception& error) {
        this->auto_chat_engine->clear_context();
        header_print("ERROR", "Tool request failed: " << error.what());
        return false;
    }
}

bool Runner::generate_once(const std::string& input, std::ostream& base_ostream) {
    if (this->tool_session) return this->generate_tools(input, base_ostream);
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    meta_info.max_prefill_len = this->prefill_chunk_len;
    uniformed_input.prompt = input;

    this->auto_chat_engine->start_total_timer();
    this->auto_chat_engine->start_ttft_timer();
    try {
        bool success = this->auto_chat_engine->insert(meta_info, uniformed_input);
        if (!success) {
            header_print("WARNING", "Max length reached, stopping generation...");
            return false;
        }
    } catch (const std::exception& e) {
        header_print("ERROR", "Insertion error: " + std::string(e.what()));
        this->auto_chat_engine->clear_context();
        return false;
    }

    this->auto_chat_engine->stop_ttft_timer();

    if (this->auto_chat_engine->get_current_model().find("gpt-oss") != std::string::npos) {
        cli_harmony_filter harmony_filter_ostream(std::cout);
        try {
            this->auto_chat_engine->generate(meta_info, this->generate_limit, harmony_filter_ostream);
        } catch (const std::exception& e) {
            header_print("ERROR", "Generation error: " + std::string(e.what()));
            this->auto_chat_engine->clear_context();
            return false;
        }
    } else {
        try {
            this->auto_chat_engine->generate(meta_info, this->generate_limit, base_ostream);
        } catch (const std::exception& e) {
            header_print("ERROR", "Generation error: " + std::string(e.what()));
            this->auto_chat_engine->clear_context();
            return false;
        }
    }

    this->auto_chat_engine->stop_total_timer();
    std::cout << std::endl;
    return true;
}

/// \brief Run the runner
void Runner::run() {
    bool verbose = false;
    this->system_prompt = "";
    this->auto_chat_engine->configure_parameter("system_prompt", this->system_prompt);
    if (!this->initial_prompt.empty()) {
        if (!this->generate_once(this->initial_prompt, std::cout)) {
            // Let main report a failure after the model and its temporary
            // linked configuration view have been destroyed.
            throw std::runtime_error("One-shot generation failed");
        }
        return;
    }
    std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8conv;
    wstream_buf obuf(std::cout);
    std::ostream base_ostream(&obuf);
    header_print("FLM", "Type /? for help");
    int empty_line_count = 0;
    bool is_image = false;
    bytes image;
    while (true) {
        std::string input = cli.get_interactive_input();
        is_image = false;
        if (input.empty()) {
            empty_line_count++;
            if (empty_line_count > 2) {
                header_print("FLM", "Type /? for help");
                empty_line_count = 0;
            }
            continue;
        }

        // Convert line (UTF-8 bytes) → wide string (Unicode codepoints)
        std::wstring winput = utf8conv.from_bytes(input);

        // Split on *any* Unicode whitespace
        std::wistringstream wiss(winput);
        std::wstring wtoken;
        std::vector<std::string> input_list;
        std::cout << std::endl;

        while (wiss >> wtoken) {
            // Convert each token back → UTF-8 bytes
            input_list.push_back(utf8conv.to_bytes(wtoken));
        }

        // now input_list contains your UTF-8 tokens
        if (input_list.empty()) {
            continue;
        }
        
        // For commands, we only need to check the first token
        std::string first_token = input_list[0];
        

        // Check if this is a command (starts with /)
        bool is_command = (first_token[0] == '/');
        
        if (is_command && first_token != "/input") {
            if (first_token == "/bye") {
                break;
            }
            else if (first_token == "/clear") {
                this->cmd_clear(input_list);
            }
            else if (first_token == "/status") {
                this->cmd_status(input_list);
            }
            else if (first_token == "/load") {
                this->cmd_load(input_list);
            }
            else if (first_token == "/save") {
                this->cmd_save(input_list);
            }
            else if (first_token == "/show") {
                this->cmd_show(input_list);
            }
            else if (first_token == "/set") {
                try {
                    this->cmd_set(input_list);
                } catch (const std::exception& e) {
                    header_print("ERROR", "Setting rejected: " + std::string(e.what()));
                }
            }
            else if (first_token == "/list") {
                std::cout << "Models:" << std::endl;
                nlohmann::json models = supported_models.get_all_models();
                for (const auto& model : models["models"]) {
                    std::string name = model["name"].get<std::string>();
                    std::cout << "  - " << name;
                    switch (downloader.is_model_downloaded(name)) {
                        case ModelDownloader::ModelStatus::Ready:        std::cout << " ✅"; break;
                        case ModelDownloader::ModelStatus::Missing:      std::cout << " ⏬"; break;
                        case ModelDownloader::ModelStatus::Outdated:     std::cout << " ⚠️"; break;
                        case ModelDownloader::ModelStatus::Incompatible: std::cout << " ⚠️"; break;
                    }
                    std::cout << std::endl;
                }
            }
            else if (first_token == "/think") {
                if (!this->auto_chat_engine->configure_parameter("toggle_think", std::any{})) {
                    header_print("WARNING", "Think is not toggleable for this model!");
                }
            }
            else if (first_token == "/help") {
                this->cmd_help(input_list);
            }
            else if (first_token == "/?") {
                this->cmd_help(input_list);
            }
            else if (first_token == "/verbose") {
                verbose = !verbose;
            }
            else if (first_token == "/history") {
                std::pair<std::string, std::vector<int>> history = this->auto_chat_engine->get_history();
                std::cout << "History: " << std::endl;
                std::cout << history.first << std::endl;
                std::cout << "Tokens: " << history.second.size() << std::endl;
                for (int i = 0; i < history.second.size(); i++) {
                    std::cout << std::dec << history.second[i] << " ";
                }
                std::cout << std::endl;
            }
            else if (first_token == "/pull") {
                std::string model_name = input_list[1];
                this->downloader.pull_model(model_name, this->modelscope);
            }
        } else {
            // This is a regular message, not a command
            // std::cout << std::endl;  // Add newline before AI response
            lm_uniform_input_t uniformed_input;
            int last_file_name_idx = 0;
            std::string audio_context = "";
            if (first_token == "/input") {
                input = "\n";
                std::string filename;
                if (input_list[1][0] == '\"'){
                    for (int i = 1; i < input_list.size(); i++) {
                        filename += input_list[i];
                        if (input_list[i][input_list[i].size() - 1] == '\"') {
                            last_file_name_idx = i;
                            break;
                        }
                        filename += " ";
                    }
                    filename = filename.substr(1, filename.size() - 2);
                }
                else{
                    filename = input_list[1];
                    last_file_name_idx = 1;
                }

                header_print("FLM", "Loading file: " << filename);
                std::cout << std::endl;
                if (filename.find(".jpg") != std::string::npos || filename.find(".png") != std::string::npos || filename.find(".jpeg") != std::string::npos) {
                    uniformed_input.images.push_back(filename);
                    uniformed_input.image_payload_types.push_back(FILE_NAME);
                }
                else if (filename.find(".wav") != std::string::npos || filename.find(".mp3") != std::string::npos || filename.find(".ogg") != std::string::npos || filename.find(".m4a") != std::string::npos) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
                    if (this->asr_supported) {
                        // gemma4 process
                        uniformed_input.audios.push_back(filename);
                        uniformed_input.audio_payload_types.push_back(FILE_NAME);
                    }
                    else if (this->asr) {
                        // check if the file exists
                        if (!std::filesystem::exists(filename)) {
                            header_print("FLM", "Error: Could not open file: " << filename);
                            header_print("FLM", "Please check if the file exists and is readable.");
                            continue;
                        }
                        this->whisper_engine->load_audio(filename);
                        std::cout << "Audio content: " << std::flush;
                        std::pair<std::string, std::string> audio_result = this->whisper_engine->generate(Whisper::whisper_task_type_t::e_transcribe, true, false, std::cout);
                        audio_context = audio_result.first;
                        std::cout << std::endl;
                        input = "Audio content (" + audio_result.second + "): " + audio_context + "\n";
                    }
                    else {
                        header_print("Warning", "No asr model loaded, cannot load audio file");
                        continue;
                    }
#else
                    header_print("Warning", "ASR models are not supported in this build");
                    continue;
#endif

                }
                else{
#ifdef _WIN32
                    std::wifstream file(utf8_to_wstring(filename));
                    if (!file.is_open()) {
                        header_print("FLM", "Error: Could not open file: " << filename);
                        header_print("FLM", "Please check if the file exists and is readable.");
                        continue;
                    }
                    file.imbue(std::locale(file.getloc(), new std::codecvt_utf8<wchar_t>));  // treat file content as UTF-8
                    std::wstring file_content_original((std::istreambuf_iterator<wchar_t>(file)), std::istreambuf_iterator<wchar_t>());
                    std::string file_content = utf8conv.to_bytes(file_content_original);
                    file.close();
#else
                    std::ifstream file(filename);
                    if (!file.is_open()) {
                        header_print("FLM", "Error: Could not open file: " << filename);
                        header_print("FLM", "Please check if the file exists and is readable.");
                        continue;
                    }
                    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                    file.close();
#endif
                    input = file_content + "\n";
                    std::cout << std::endl;
                }
                if (last_file_name_idx < input_list.size() - 1) {
                    for (int i = last_file_name_idx + 1; i < input_list.size() - 1; i++) {
                        input += input_list[i] + " ";
                    }
                    input += input_list[input_list.size() - 1];
                }
            }

            if (this->tool_session) {
                if (!uniformed_input.images.empty() || !uniformed_input.audios.empty()) {
                    header_print("ERROR", "Native CLI tools currently accept text input only");
                    continue;
                }
                (void)this->generate_tools(input, std::cout);
                continue;
            }

            chat_meta_info_t meta_info;
            meta_info.max_prefill_len = this->prefill_chunk_len;
            uniformed_input.prompt = input;
            
            this->auto_chat_engine->start_total_timer();
            
            this->auto_chat_engine->start_ttft_timer();
            try {
                (void)this->auto_chat_engine->configure_parameter(
                    "hip_oga_requested_max_tokens", this->generate_limit);
                bool success = this->auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success){
                    header_print("WARNING", "Max length reached, stopping generation...");
                    break;
                }
            }
            catch (const std::exception& e) {
                header_print("ERROR", "Insertion error: " + std::string(e.what()));
                this->auto_chat_engine->clear_context();
                continue;
            }

            this->auto_chat_engine->stop_ttft_timer();
            
            // Use harmony filter for gpt-oss models
            if (this->auto_chat_engine->get_current_model().find("gpt-oss") != std::string::npos) { // contains gpt-oss:20b and gpt-oss
                cli_harmony_filter harmony_filter_ostream(std::cout);
                try {
                    this->auto_chat_engine->generate(meta_info, this->generate_limit, harmony_filter_ostream);
                }
                catch (const std::exception& e) {
                    header_print("ERROR", "Generation error: " + std::string(e.what()));
                    this->auto_chat_engine->clear_context();
                    continue;
                }
            } else {
                try {
                    this->auto_chat_engine->generate(meta_info, this->generate_limit, base_ostream);
                }
                catch (const std::exception& e) {
                    header_print("ERROR", "Generation error: " + std::string(e.what()));
                    this->auto_chat_engine->clear_context();
                    continue;
                }
            }
            
            this->auto_chat_engine->stop_total_timer();
            std::cout << std::endl;
            if (verbose) {
                this->auto_chat_engine->verbose();
            }
        }
    }
}

/// \brief Clear the context
/// \param input_list, std::vector<std::string>
void Runner::cmd_clear(std::vector<std::string>& input_list) {
    this->auto_chat_engine->clear_context();
}

/// \brief Show the status
/// \param input_list, std::vector<std::string>
void Runner::cmd_status(std::vector<std::string>& input_list) {
    std::cout << this->auto_chat_engine->show_profile() << std::endl;
}

/// \brief Load a model
/// \param input_list, std::vector<std::string>
void Runner::cmd_load(std::vector<std::string>& input_list) {
    std::string model_name = input_list[1];

    const bool using_hip_oga = flm::backend::is_hip_oga_backend(this->backend);
    const bool hip_direct_path =
        using_hip_oga && HipOgaModel::is_model_dir(model_name);
    std::pair<std::string, std::unique_ptr<AutoModel>> auto_model;
    if (hip_direct_path) {
        auto_model = std::make_pair(model_name, std::make_unique<HipOgaModel>());
    } else if (using_hip_oga) {
        auto_model = get_auto_model(model_name, this->supported_models, nullptr, this->backend);
        model_name = auto_model.first;
    } else {
        auto_model = get_auto_model(model_name, this->supported_models, ensure_npu_device(), this->backend);
        model_name = auto_model.first;
    }

    if (model_name != this->tag) {
        this->tag = model_name;

        if (!using_hip_oga) {
            switch (this->downloader.is_model_downloaded(this->tag)) {
                case ModelDownloader::ModelStatus::Ready:
                    break;
                case ModelDownloader::ModelStatus::Outdated:
                case ModelDownloader::ModelStatus::Missing:
                    this->downloader.pull_model(this->tag, this->modelscope);
                    break;
                case ModelDownloader::ModelStatus::Incompatible:
                    header_print("ERROR", "Model is incompatible with this version of FastFlowLM: " + this->tag);
                    exit(EXIT_FAILURE);
            }
        }
        auto_chat_engine.reset();
        if(model_name=="gpt-oss:20b")
            std::this_thread::sleep_for(std::chrono::milliseconds(2800));
        this->auto_chat_engine = std::move(auto_model.second);

        json model_info;
        std::string model_path;
        if (hip_direct_path) {
            model_info = HipOgaModel::model_info_for_path(this->tag);
            model_path = this->tag;
        } else if (using_hip_oga) {
            auto [new_tag, selected_model_info] = this->supported_models.get_model_info(this->tag);
            model_info = std::move(selected_model_info);
            model_path = flm::hip_oga::ensure_hip_oga_model(this->supported_models, new_tag);
            model_info["hip_oga"]["config_path"] = model_path;
        } else {
            auto [new_tag, selected_model_info] = this->supported_models.get_model_info(this->tag);
            model_info = std::move(selected_model_info);
            model_path = this->supported_models.get_model_path(new_tag);
        }
        this->auto_chat_engine->configure_parameter("img_pre_resize", this->img_pre_resize);
        if (using_hip_oga) {
            report_hip_oga_offload_policy(model_info);
        }
        try {
            this->auto_chat_engine->load_model(model_path, model_info, this->ctx_length, this->preemption, this->backend);
            this->auto_chat_engine->set_sampling_provider(this->sampler_provider);
        }
        catch (const std::exception& e) {
            header_print("ERROR", "Failed to load model: " + std::string(e.what()));
            exit(EXIT_FAILURE);
        }

        this->auto_chat_engine->configure_parameter("system_prompt", this->system_prompt);
        if (this->tool_session) {
            this->auto_chat_engine->set_log_raw_output(false);
            this->auto_chat_engine->set_temperature(0.0f);
        }

    }
    else
        header_print("FLM", "Model already loaded: " << model_name);

}

/// \brief Save the history
/// \param input_list, std::vector<std::string>
void Runner::cmd_save(std::vector<std::string>& input_list) {
    std::pair<std::string, std::vector<int>> history = this->auto_chat_engine->get_history();
    // Get the FLM_MODEL_PATH environment variable for the history directory
    std::string history_dir;
    const char* path_sep = "/";
#ifdef _WIN32
    path_sep = "\\";
    char* model_path_env = nullptr;
    size_t len = 0;

    history_dir = utils::get_models_directory() + path_sep + "history";

#else
    const char* model_path_env = std::getenv("FLM_MODEL_PATH");
    if (model_path_env && *model_path_env) {
        history_dir = std::string(model_path_env) + path_sep + "history";
    } else {
        std::string documents_dir = utils::get_user_directory();
        history_dir = documents_dir + path_sep + "flm" + path_sep + "history";
    }
#endif
    
    // Create the history directory if it doesn't exist
    if (!std::filesystem::exists(history_dir)) {
        std::filesystem::create_directories(history_dir);
    }
    
    // save file to history_hh_mm_mm_dd_yyyy.txt
    // 1) get current date
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream date_ss;
    date_ss << std::setw(2) << std::setfill('0') << tm.tm_hour
            << '_'
            << std::setw(2) << std::setfill('0') << tm.tm_min
            << '_'
            << std::setw(2) << std::setfill('0') << tm.tm_mon + 1
            << '_'
            << std::setw(2) << std::setfill('0') << tm.tm_mday
            << '_'
            << (tm.tm_year + 1900);
    std::string date_str = date_ss.str();

    // 2) build filename in the history directory
    std::string file_name = history_dir + path_sep + "history_" + date_str + ".txt";
    std::ofstream file(file_name);
    if (file.is_open()) {
        file << history.first << std::endl;
        file.close();
        std::cout << "History saved to " << file_name << std::endl;
    }
    else {
        std::cout << "Failed to open file: " << file_name << std::endl;
    }
}

/// \brief Show the model information
/// \param input_list, std::vector<std::string>
void Runner::cmd_show(std::vector<std::string>& input_list) {
    std::cout << this->auto_chat_engine->show_model_info() << std::endl;
    std::cout << "    max context length    : " << this->auto_chat_engine->get_max_length() << std::endl;
    std::cout << std::endl;

}

/// \brief Set the model parameters
/// \param input_list, std::vector<std::string>
void Runner::cmd_set(std::vector<std::string>& input_list) {
    // remove the /set
    if (input_list.size() < 3){
        std::cout << "Usage: /set [context] [value]" << std::endl;
        std::cout << "Available parameters: " << std::endl;
        std::cout << "  /set topk [value] - set the top-k" << std::endl;
        std::cout << "  /set topp [value] - set the top-p" << std::endl;
        std::cout << "  /set minp [value] - set the min-p" << std::endl;
        std::cout << "  /set temp [value] - set the temperature" << std::endl;
        std::cout << "  /set rep-pen [value] - set the repetition penalty" << std::endl;
        std::cout << "  /set freq-pen [value] - set the frequency penalty" << std::endl;
        std::cout << "  /set pres-pen [value] - set the presence penalty" << std::endl;
        std::cout << "  /set sampler [backend|common] - select the sampling provider" << std::endl;
        std::cout << "  /set seed [value] - set request random seed (-1 for random)" << std::endl;
        std::cout << "  /set sys-msg [value] - set the system message" << std::endl;
        std::cout << "  /set ctx-len [value] - set the max context length" << std::endl;
        std::cout << "  /set gen-lim [value] - Limit tokens generated per round" << std::endl;
        std::cout << "  /set r-eff [low|medium|high] - set the reasoning effort level (GPT-OSS only, default = medium)" << std::endl;
        return;
    }
    
    std::string set_context = input_list[1];
    
    // Handle system_prompt specially since it can be multi-line
    if (set_context == "sys-msg") {
        // Reconstruct the original input to get the full system prompt
        std::string full_input;
        for (size_t i = 2; i < input_list.size(); i++) {
            if (i > 2) full_input += " ";
            full_input += input_list[i];
        }
        this->system_prompt = full_input;
        this->auto_chat_engine->configure_parameter("system_prompt", this->system_prompt);
        return;
    }
    
    // For other parameters, use the third token as the value
    if (input_list.size() < 3) {
        std::cout << "Usage: /set [context] [value]" << std::endl;
        return;
    }
    
    std::string set_value = input_list[2];

    if (set_context == "topk"){
        this->auto_chat_engine->set_topk(flm::hip_oga::parse_sampling_integer(set_value, "top_k", 0));
    }
    else if (set_context == "topp"){
        this->auto_chat_engine->set_topp(flm::hip_oga::parse_sampling_float(set_value, "top_p"));
    }
    else if (set_context == "minp") {
        this->auto_chat_engine->set_minp(flm::hip_oga::parse_sampling_float(set_value, "min_p"));
    }
    else if (set_context == "temp"){
        this->auto_chat_engine->set_temperature(flm::hip_oga::parse_sampling_float(set_value, "temperature"));
    }
    else if (set_context == "rep-pen"){
        this->auto_chat_engine->set_repetition_penalty(flm::hip_oga::parse_sampling_float(set_value, "repetition_penalty"));
    }
    else if (set_context == "freq-pen"){
        this->auto_chat_engine->set_frequency_penalty(flm::hip_oga::parse_sampling_float(set_value, "frequency_penalty"));
    }
    else if (set_context == "pres-pen") {
        this->auto_chat_engine->set_presence_penalty(flm::hip_oga::parse_sampling_float(set_value, "presence_penalty"));
    }
    else if (set_context == "seed") {
        const nlohmann::json update = {{"seed", flm::hip_oga::parse_sampling_seed(set_value)}};
        this->auto_chat_engine->configure_sampling_request(update, false);
    }
    else if (set_context == "sampler") {
        this->auto_chat_engine->configure_sampling_request({{"sampler", set_value}}, false);
    }
    else if (set_context == "ctx-len"){
        try {
            this->auto_chat_engine->set_max_length(std::stoi(set_value));
        }
        catch (const std::exception& e) {
            header_print("ERROR", "Failed to set context length: " + std::string(e.what()));
            exit(EXIT_FAILURE);
        }
    }
    else if (set_context == "gen-lim"){
        this->generate_limit = std::stoi(set_value);
    }
    else if (set_context == "prefill-chunk-len"){ 
        
    }
    else if (set_context == "r-eff"){
        if (this->auto_chat_engine->get_current_model() == "gpt-oss")
            this->auto_chat_engine->configure_parameter("reasoning_effort", set_value);
        else
            header_print("WARNING", "Reasoning effort only support for gpt-oss");
    }
    else{
        std::cout << "Invalid context: " << set_context << std::endl;
        std::cout << "Available parameters: " << std::endl;
        std::cout << "  /set ctx-len [value] - set the max context length" << std::endl;
        std::cout << "  /set topk [value] - set the top-k" << std::endl;
        std::cout << "  /set topp [value] - set the top-p" << std::endl;
        std::cout << "  /set minp [value] - set the min-p" << std::endl;
        std::cout << "  /set temp [value] - set the temperature" << std::endl;
        std::cout << "  /set rep-pen [value] - set the repetition penalty" << std::endl;
        std::cout << "  /set freq-pen [value] - set the frequency penalty" << std::endl;
        std::cout << "  /set pres-pen [value] - set the presence penalty" << std::endl;
        std::cout << "  /set sampler [backend|common] - select the sampling provider" << std::endl;
        std::cout << "  /set seed [value] - set request random seed (-1 for random)" << std::endl;
        std::cout << "  /set sys-msg [value] - set the system message" << std::endl;
        std::cout << "  /set gen-lim [value] - Limit tokens generated per round" << std::endl;
        std::cout << "  /set r-eff [low|medium|high] - set the reasoning effort level (GPT-OSS only, default = medium)" << std::endl;
    }
}

/// \brief Show the help
/// \param input_list, std::vector<std::string>
void Runner::cmd_help(std::vector<std::string>& input_list) {
    std::cout << "Available commands:" << std::endl;
    std::cout << "  /show - show the model information" << std::endl;
    std::cout << "  /load [model_name] - load a model" << std::endl;
    std::cout << "  /input [filename] [follow_up_prompt] - load a file and follow up with a prompt" << std::endl;
    std::cout << "                                       - If space is in the filename, use quotes to wrap it" << std::endl;
    std::cout << "  /save - save the history" << std::endl;
    std::cout << "  /clear - clear the context" << std::endl;
    std::cout << "  /status - show perf. metrics" << std::endl;
    std::cout << "  /history - show the history" << std::endl;
    std::cout << "  /verbose - toggle the verbose" << std::endl;
    std::cout << "  /think - toggle the think" << std::endl;
    std::cout << "  /set [variable] [value] - set the variable" << std::endl;
    std::cout << "  /list - list all the models" << std::endl;
    std::cout << "  /bye - exit the program" << std::endl;
    std::cout << "  /? - show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive input:" << std::endl;
    std::cout << "  - Press Enter to submit single-line input" << std::endl;
    std::cout << "  - Paste multi-line text and it will be detected automatically" << std::endl;
    std::cout << "  - Use 'Shift + Enter' to explicitly continue on next line" << std::endl;
    std::cout << "  - Commands (starting with /) are processed immediately" << std::endl;
}

/// \brief Show the help shotcut
/// \param input_list, std::vector<std::string>
void Runner::cmd_help_shotcut(std::vector<std::string>& input_list) {
    std::cout << "Help shotcut" << std::endl;
}
