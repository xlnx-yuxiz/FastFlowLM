// Test the actual SSE streambuf. Only the device/generation boundary is absent.
#include "streaming_ostream_openai.hpp"
#include "chat/chat_format.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

class FixtureModel : public AutoModel {
public:
    explicit FixtureModel(chat_format_kind format = chat_format_kind::HERMES_JSON) : AutoModel(nullptr) {
        MAX_L = 1024;
        chat_format = chat_format_make(format, true);
    }
    int get_current_context_length() override { return 15; }
    std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>) override {
        throw std::logic_error("SSE fixture does not generate");
    }
    bool insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) override {
        throw std::logic_error("SSE fixture does not insert");
    }
    std::string generate_with_prompt(chat_meta_info_t&, lm_uniform_input_t&, int, std::ostream&) override {
        throw std::logic_error("SSE fixture does not generate");
    }
    std::string apply_chat_template(json&, json) override {
        throw std::logic_error("SSE fixture does not render templates");
    }
};

struct Events {
    std::vector<json> chunks;
    int done = 0;
    void receive(const std::string& raw, bool final) {
        require(done == 0, "event after DONE");
        if (raw == "data: [DONE]\n\n") {
            require(final, "DONE is not terminal"); ++done; return;
        }
        require(!final && raw.substr(0, 6) == "data: ", "invalid SSE envelope");
        chunks.push_back(json::parse(raw.substr(6))); // also rejects malformed UTF-8
    }
    std::string text(const char* field) const {
        std::string result;
        for (const auto& c : chunks) {
            if (!c.contains("choices")) continue;
            for (const auto& choice : c["choices"]) {
                if (choice.contains("delta") && choice["delta"].contains(field) && choice["delta"][field].is_string())
                    result += choice["delta"][field].get<std::string>();
            }
        }
        return result;
    }
    std::string finish() const {
        require(done == 1 && !chunks.empty(), "missing terminal event");
        return chunks.back()["choices"][0]["finish_reason"].get<std::string>();
    }
};

static chat_meta_info_t metadata(stop_reason_t reason = EOT_DETECTED) {
    chat_meta_info_t meta;
    meta.prompt_tokens = 11; meta.cached_prompt_tokens = 5; meta.generated_tokens = 4;
    meta.prefill_duration = 1000000000; meta.decoding_duration = 1000000000;
    meta.stop_reason = reason;
    return meta;
}

static Events stream(FixtureModel& model, const std::string& raw, bool bytes,
                     stop_reason_t reason = EOT_DETECTED, bool allow = true,
                     const std::string& prefill = "") {
    Events events;
    auto meta = metadata(reason); meta.prompt_tail = prefill;
    json tools = json::array({{{"type", "function"}, {"function", {{"name", "lookup"}}}}});
    streaming_ostream_openai_chat out("fixture", &model,
        [&](const std::string& data, bool final) { events.receive(data, final); }, allow, tools, true);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.start(meta);
    if (bytes) for (char c : raw) out.put(c).flush();
    else out << raw << std::flush;
    out.finalize(meta);
    return events;
}

int main() {
    try {
        FixtureModel model;
        {
            const auto events = stream(model, "你好 🌍 <thi", true);
            require(events.text("content") == "你好 🌍 <thi", "UTF-8 or final literal lost");
            require(events.finish() == "stop", "plain finish");
            const auto& usage = events.chunks.back()["usage"];
            require(usage["prompt_tokens"] == 11 && usage["prompt_tokens_details"]["cached_tokens"] == 5 &&
                    usage["completion_tokens"] == 4 && usage["total_tokens"] == 15, "usage accounting");
            std::cout << "PASS actual SSE byte-wise UTF-8 final flush and usage\n";
        }
        require(stream(model, std::string("tail ") + "\xE4\xBD", true).text("content") ==
                std::string("tail ") + "\xEF\xBF\xBD", "invalid final UTF-8 escaped SSE handling");
        std::cout << "PASS actual SSE truncated final UTF-8\n";
        require(stream(model, "    print('hi')\n", true).text("content") == "    print('hi')\n",
                "actual SSE code indentation or final newline lost");
        require(stream(model, " \t\n", true).text("content") == " \t\n", "actual SSE whitespace-only reply lost");
        std::cout << "PASS actual SSE code whitespace preserved\n";

        const std::string call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}}</tool_call>";
        {
            auto events = stream(model, "reason</think>answer" + call + call, false, EOT_DETECTED, true, "<think>\n");
            require(events.text("reasoning_content") == "reason" && events.text("content") == "answer", "multi-delta fields lost");
            std::vector<std::string> ids;
            for (const auto& chunk : events.chunks) {
                if (!chunk.contains("choices")) continue;
                const auto& delta = chunk["choices"][0]["delta"];
                if (!delta.contains("tool_calls")) continue;
                for (const auto& c : delta["tool_calls"]) {
                    require(c["index"] == ids.size(), "tool index");
                    require(c["function"]["name"] == "lookup" && c["function"]["arguments"] == "{\"n\":2}", "tool fields");
                    ids.push_back(c["id"].get<std::string>());
                }
            }
            require(ids.size() == 2 && ids[0] != ids[1] && events.finish() == "tool_calls", "tool identities or finish");
            std::cout << "PASS actual SSE multiple deltas and calls in one flush\n";
        }
        for (const auto reason : {MAX_LENGTH_REACHED, CANCEL_DETECTED, ERROR_DETECTED}) {
            auto events = stream(model, call, true, reason);
            require(events.finish() == stop_reason_to_string(reason), "tool completion overrode stop reason");
        }
        std::cout << "PASS actual SSE length cancel error precedence\n";
        // Cancellation may interrupt the model after a tool delta but before its
        // arguments/closing tag. Those bytes are not a completed model response.
        for (const auto format : {chat_format_kind::HERMES_JSON, chat_format_kind::QWEN_XML}) {
            FixtureModel cancelled_model(format);
            const std::string complete = format == chat_format_kind::HERMES_JSON ? call :
                "<tool_call><function=lookup><parameter=n>2</parameter></function></tool_call>";
            for (const auto raw : {std::string("<tool_call>"),
                     std::string("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":"),
                     complete + "<tool_call>"}) {
                const auto events = stream(cancelled_model, raw, true, CANCEL_DETECTED);
                for (const auto& chunk : events.chunks)
                    require(!chunk.contains("error"), "cancelled partial tool emitted model output error");
                require(events.finish() == "cancel", "cancelled partial tool lost cancel finish");
            }
            const auto next = stream(cancelled_model, complete, true);
            require(next.finish() == "tool_calls", "cancelled parser contaminated next tool request");
        }
        std::cout << "PASS actual SSE partial tool cancellation and next request\n";
        for (const auto stop : {EOT_DETECTED, CANCEL_DETECTED}) {
            Events events;
            auto meta = metadata();
            meta.stop_reason = stop;
            const json declared = json::array({{{"type", "function"}, {"function", {{"name", "lookup"}}}}});
            streaming_ostream_openai_chat out("fixture", &model,
                [&](const std::string& s, bool final) { events.receive(s, final); },
                true, declared, true, false, true);
            out.start(meta);
            out << R"({"type":"tool_call","name":"lookup","arguments":{"n":2}})" << std::flush;
            require(events.chunks.empty(), "structured decision emitted before final release");
            out.finalize(meta);
            require(events.finish() == (stop == CANCEL_DETECTED ? "cancel" : "tool_calls"),
                    "structured SSE final reason");
            int calls = 0;
            for (const auto& chunk : events.chunks) {
                require(!chunk.contains("error"), "structured SSE produced error");
                if (chunk.contains("choices")) for (const auto& choice : chunk["choices"])
                    if (choice.contains("delta") && choice["delta"].contains("tool_calls")) ++calls;
            }
            require(calls == (stop == CANCEL_DETECTED ? 0 : 1), "cancelled structured decision became executable");
        }
        std::cout << "PASS actual SSE structured decision release and cancellation\n";
        for (const auto reason : {EOT_DETECTED, CANCEL_DETECTED}) {
            const auto malformed = stream(model, "<tool_call>{bad}</tool_call>", true, reason);
            int errors = 0;
            for (const auto& chunk : malformed.chunks) if (chunk.contains("error")) ++errors;
            require(errors == 1 && malformed.finish() == "error", "cancellation hid an existing output error");
        }
        for (const auto raw : {call, std::string("<tool_call>{bad}</tool_call>")}) {
            auto events = stream(model, raw, false, EOT_DETECTED, false);
            int errors = 0;
            for (const auto& chunk : events.chunks) if (chunk.contains("error")) {
                ++errors;
                require(chunk["error"]["code"] == 422 && chunk["error"]["type"] == "model_output_error", "output error contract");
            }
            require(errors == 1 && events.finish() == "error", "output error terminal event");
        }
        std::cout << "PASS actual SSE malformed and undeclared tool errors\n";
        {
            FixtureModel xml(chat_format_kind::QWEN_XML);
            for (const auto raw : {std::string("<tool_call>"), std::string("<tool_call>{bad}</tool_call>")}) {
                for (bool allow : {false, true}) {
                    const auto events = stream(xml, raw, true, EOT_DETECTED, allow);
                    int errors = 0;
                    for (const auto& chunk : events.chunks) if (chunk.contains("error")) {
                        ++errors;
                        require(chunk["error"]["code"] == 422, "XML tool output error status");
                    }
                    require(errors == 1 && events.finish() == "error", "XML tool attempt disappeared from SSE");
                }
            }
            std::cout << "PASS actual SSE strict XML tool errors\n";
        }

        {
            Events discarded;
            auto meta = metadata();
            streaming_ostream_openai_chat out("fixture", &model,
                [&](const std::string& s, bool final) { discarded.receive(s, final); }, true, json::array(), true);
            out.start(meta); out << "<think>unfinished" << std::flush;
        }
        require(stream(model, "fresh", true).text("content") == "fresh", "discarded stream contaminated next request");
        std::cout << "PASS actual SSE request parser disposal\n";
        {
            Events events; auto meta = metadata();
            streaming_ostream_openai raw("fixture", [&](const std::string& s, bool final) { events.receive(s, final); });
            raw << "<think>literal</think>" << std::flush; raw.finalize(meta);
            std::string text;
            for (const auto& c : events.chunks) for (const auto& choice : c["choices"])
                if (choice.contains("text") && choice["text"].is_string()) text += choice["text"].get<std::string>();
            require(text == "<think>literal</think>", "raw completion was chat parsed");
            std::cout << "PASS raw completion protocol remains literal\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
    return 0;
}
