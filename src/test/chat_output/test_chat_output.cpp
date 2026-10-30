// Exercise the request-owned adapter, including the existing HIP error policy.
#include "chat/chat_output.hpp"
#include "chat/auto_tools.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using json = nlohmann::ordered_json;
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const auto hermes = chat_format_make(chat_format_kind::HERMES_JSON, true);
        const json tools = json::array({{{"type", "function"}, {"function", {
            {"name", "lookup"}, {"parameters", {{"type", "object"}}}}}}});

        {
            const json original = json::array({{{"role", "system"}, {"content", "Keep this instruction."}},
                {{"role", "user"}, {"content", "Hello"}}});
            const auto prepared = chat_prepare_auto_tools(original, tools);
            require(prepared[0]["content"].get<std::string>().find("Keep this instruction.") == 0,
                    "routing preparation replaced caller system content");
            require(prepared[1] == original[1] && original[0]["content"] == "Keep this instruction.",
                    "routing preparation mutated caller messages");
            const std::string answer = R"({"type":"message","content":"Literal <tool_call> example; 中文"})";
            for (const auto& raw : {answer, std::string(R"({"type":"tool_call","name":"lookup","arguments":{"path":"a.json"}})")}) {
                chat_output output(hermes, "", tools, true, true, false, true);
                for (char c : raw) {
                    const auto part = output.update(std::string(1, c), false);
                    require(part.error.empty() && part.deltas.empty() && part.message.tool_calls.empty(),
                            "partial auto envelope escaped before completion");
                }
                const auto full = output.update("", true);
                require(full.error.empty(), "valid auto envelope rejected");
                if (raw == answer) require(full.message.tool_calls.empty() &&
                    full.message.content == "Literal <tool_call> example; 中文", "answer parsed as an invocation");
                else require(full.message.content.empty() && full.message.tool_calls.size() == 1 &&
                    full.message.tool_calls[0].name == "lookup" && !full.message.tool_calls[0].id.empty() &&
                    json::parse(full.message.tool_calls[0].arguments)["path"] == "a.json", "tool envelope lost fields");
                require(output.update("", true).deltas.empty(), "auto finalization duplicated output");
            }
            for (const std::string raw : {"{}", R"({"type":"message","content":""})",
                R"({"type":"message","content":"Hi","name":"lookup"})",
                R"({"type":"tool_call","name":"unknown","arguments":{}})",
                R"({"type":"tool_call","name":"lookup","arguments":"{}"})",
                R"({"type":"tool_call","name":"lookup","arguments":{})"}) {
                chat_output output(hermes, "", tools, true, true, false, true);
                const auto result = output.update(raw, true);
                require(!result.error.empty() && result.deltas.empty() && result.message.tool_calls.empty(),
                        "invalid auto envelope became a call");
            }
            for (const std::string name : {"message", "answer", "finish_task"}) {
                const json named = json::array({{{"type", "function"}, {"function", {{"name", name}}}}});
                const auto result = chat_parse_auto_tools(json{{"type", "tool_call"}, {"name", name},
                    {"arguments", json::object()}}.dump(), named);
                require(result.content.empty() && result.tool_calls.size() == 1 && result.tool_calls[0].name == name,
                        "application function name treated as completion signal");
            }
        }
        std::cout << "PASS structured auto envelopes, buffered chunks and name isolation\n";

        for (const std::string raw : {
                "<tool_call>{bad}</tool_call>",
                "<tool_call>{\"name\":\"lookup\"}</tool_call>",
                "<tool_call>{\"arguments\":{}}</tool_call>",
                "<tool_call>{\"name\":\"\",\"arguments\":{}}</tool_call>",
                "<tool_call>{\"name\":\"lookup\",\"arguments\":null}</tool_call>",
                "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}"}) {
            chat_output output(hermes, "", tools, true, true);
            auto result = output.update(raw, true);
            require(!result.error.empty(), "malformed HIP tool output accepted");
            require(result.deltas.empty(), "malformed final emitted tool delta");
        }
        std::cout << "PASS malformed HIP tool output\n";

        {
            chat_output output(hermes, "", tools, true, true);
            auto first = output.update("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":", false);
            require(first.error.empty(), "incomplete streaming tool rejected early");
            auto last = output.update("\"Paris\"}}</tool_call>", true);
            require(last.error.empty(), "valid completed tool rejected");
            require(last.message.tool_calls.size() == 1, "call lost");
            require(last.message.tool_calls[0].arguments == "{\"city\":\"Paris\"}", "arguments changed");
            require(output.update("", true).deltas.empty(), "duplicate final delta");
        }
        std::cout << "PASS partial tool and idempotent final\n";

        for (const auto spec : {hermes, chat_format_make(chat_format_kind::CONTENT_ONLY, false)}) {
            chat_output output(spec, "", json::array(), true, false);
            auto result = output.update("<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>", true);
            require(!result.error.empty() && result.deltas.empty(), "undeclared tool output accepted");
        }
        std::cout << "PASS undeclared tools including content-only format\n";

        const std::string literal_json = R"({"s":"<think>x</think><tool_call>","not":{"if":1}})";
        for (const auto spec : {hermes, chat_format_make(chat_format_kind::CONTENT_ONLY, false)}) {
            for (bool reasoning : {false, true}) {
                if (reasoning && !spec.has_reasoning()) continue;
                const std::string prefix = reasoning ? "<think>actual reasoning</think>" : "";
                chat_output whole(spec, "", json::array(), true, false, true);
                auto full = whole.update(prefix + literal_json, true);
                require(full.error.empty() && full.message.content == literal_json &&
                        full.message.tool_calls.empty(), "JSON string markers interpreted as controls");
                require(full.message.reasoning == (reasoning ? "actual reasoning" : ""),
                        "actual leading reasoning was not separated from JSON");
                chat_output stream(spec, "", json::array(), true, false, true);
                std::string emitted;
                for (char byte : prefix + literal_json) {
                    auto part = stream.update(std::string(1, byte), false);
                    require(part.error.empty(), "partial JSON marker rejected");
                    for (const auto& delta : part.deltas) emitted += delta.content;
                }
                auto last = stream.update("", true);
                for (const auto& delta : last.deltas) emitted += delta.content;
                require(last.error.empty() && last.message.content == literal_json && emitted == literal_json,
                        "streaming JSON bytes differ from whole response");
            }
        }
        std::cout << "PASS literal JSON markers and leading reasoning across byte splits\n";


        {
            chat_output output(hermes, "<think>", json::array(), true, false);
            auto result = output.update("Example <tool_call>{not a call}</tool_call></think>answer", true);
            require(result.error.empty(), "tool example inside reasoning rejected");
            require(result.message.content == "answer" && result.message.tool_calls.empty(), "reasoning leaked tool");
        }
        std::cout << "PASS tool markup in prefilled reasoning is inert\n";

        {
            json typed_tools = json::array({{{"type", "function"}, {"function", {
                {"name", "lookup"}, {"parameters", {{"type", "object"}, {"properties", {
                    {"value", {{"type", "string"}}}}}}}}}}});
            chat_output output(chat_format_make(chat_format_kind::QWEN_XML, false), "", typed_tools, false, true);
            typed_tools.clear();
            auto result = output.update("<tool_call><function=lookup><parameter=value>7</parameter></function></tool_call>", true);
            require(result.error.empty() && result.message.tool_calls.size() == 1, "typed XML call missing");
            require(result.message.tool_calls[0].arguments == "{\"value\":\"7\"}", "request tool schema lifetime lost");
        }
        std::cout << "PASS request owns schema used for argument typing\n";

        {
            chat_output cancelled(hermes, "", tools, true, true);
            cancelled.update("<think>cancelled partial", false);
            chat_output next(hermes, "", json::array(), true, false);
            auto result = next.update("fresh answer", true);
            require(result.error.empty() && result.message.content == "fresh answer" &&
                    result.message.reasoning.empty() && result.message.tool_calls.empty(), "cross-request parser state");
        }
        std::cout << "PASS independent request state after unfinished turn\n";
        {
            chat_output output(hermes, "", json::array(), true, false);
            const auto result = output.update(std::string("tail ") + "\xE4\xBD", true);
            require(result.error.empty() && result.message.content == std::string("tail ") + "\xEF\xBF\xBD",
                    "truncated final UTF-8 must become a replacement character");
        }
        std::cout << "PASS truncated final UTF-8 remains serializable\n";
        for (const auto spec : {hermes, chat_format_make(chat_format_kind::CONTENT_ONLY, false)}) {
            for (const std::string raw : {std::string("    print('hi')\n"), std::string(" \t\n")}) {
                chat_output full(spec, "", json::array(), true, false);
                require(full.update(raw, true).message.content == raw, "nonstream code whitespace lost");
                chat_output streamed(spec, "", json::array(), true, false);
                std::string content;
                for (char c : raw) for (const auto& delta : streamed.update(std::string(1, c), false).deltas)
                    content += delta.content;
                for (const auto& delta : streamed.update("", true).deltas) content += delta.content;
                require(content == raw, "streamed code whitespace lost");
            }
        }
        {
            chat_output output(hermes, "", tools, true, true);
            const auto result = output.update("  before\n<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>\n  after\n", true);
            require(result.message.content == "  before\n\n  after\n", "content around tool markers rewritten");
        }
        std::cout << "PASS content whitespace and indentation preserved\n";
        for (const std::string raw : {std::string("<tool_call>"), std::string("<tool_call>{bad}</tool_call>")}) {
            for (bool allow : {false, true}) {
                const auto xml = chat_format_make(chat_format_kind::QWEN_XML, true);
                chat_output full(xml, "", tools, true, allow);
                require(!full.update(raw, true).error.empty(), "malformed XML tool attempt silently accepted");
                chat_output streamed(xml, "", tools, true, allow);
                for (char c : raw) streamed.update(std::string(1, c), false);
                require(!streamed.update("", true).error.empty(), "streamed malformed XML tool attempt accepted");
            }
        }
        {
            const auto xml = chat_format_make(chat_format_kind::QWEN_XML, true);
            const std::string call = "<tool_call><function=lookup><parameter=n>2</parameter></function></tool_call>";
            chat_output allowed(xml, "", tools, true, true);
            const auto result = allowed.update(call, true);
            require(result.error.empty() && result.message.tool_calls.size() == 1 &&
                    result.message.tool_calls[0].arguments == "{\"n\":2}", "valid native XML call rejected");
            chat_output reasoning(xml, "<think>", tools, true, false);
            const auto inert = reasoning.update("Example <tool_call>{bad}</tool_call></think>answer", true);
            require(inert.error.empty() && inert.message.content == "answer", "XML tool example inside reasoning rejected");
        }
        std::cout << "PASS strict XML tool validation and reasoning exclusion\n";


    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
