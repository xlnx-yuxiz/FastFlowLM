// Real shared-parser fixtures. Catch marker leaks, lost fields, unstable tool
// indices/IDs and non-monotone arguments independently of model generation.
#include "chat/chat_stream.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

static chat_msg assemble(const std::vector<chat_msg_delta>& deltas) {
    chat_msg result;
    for (const auto& d : deltas) {
        result.content += d.content;
        result.reasoning += d.reasoning;
        if (!d.is_tool_call()) continue;
        require(d.tool_index <= static_cast<int>(result.tool_calls.size()), "tool index gap");
        if (d.tool_index == static_cast<int>(result.tool_calls.size())) {
            require(d.tool_id.size() == 29 && d.tool_id.substr(0, 5) == "call_", "missing initial ID");
            require(!d.tool_name.empty(), "missing initial name");
            result.tool_calls.push_back({d.tool_id, d.tool_name, d.tool_arguments});
        } else {
            require(d.tool_id.empty() && d.tool_name.empty(), "duplicate tool identity");
            result.tool_calls[d.tool_index].arguments += d.tool_arguments;
        }
    }
    return result;
}

int main() {
    int passed = 0;
    auto test = [&](const char* name, const std::function<void()>& body) {
        try { body(); ++passed; std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& e) { std::cerr << "FAIL " << name << ": " << e.what() << '\n'; throw; }
    };
    try {
        test("format detection uses template and vocabulary", [] {
            auto no_token = [](const std::string&) { return false; };
            require(chat_format_detect("ordinary template", no_token).kind == chat_format_kind::CONTENT_ONLY,
                    "plain format");
            require(chat_format_detect("<tool_call>", no_token).kind == chat_format_kind::HERMES_JSON,
                    "Hermes template");
            auto think_token = [](const std::string& s) { return s == "<think>" || s == "</think>"; };
            require(chat_format_detect("ordinary template", think_token).has_reasoning(), "vocabulary reasoning");
        });
        test("template-opened reasoning and Harmony prefill", [] {
            auto spec = chat_format_make(chat_format_kind::HERMES_JSON, true);
            require(chat_format_prefill(spec, "user text\nassistant\n<think>\n") == "<think>", "open reasoning");
            require(chat_format_prefill(spec, "<think>\n</think>\n").empty(), "closed reasoning");
            auto harmony = chat_format_make(chat_format_kind::HARMONY, false);
            require(chat_format_prefill(harmony, "old turn<|start|>assistant<|channel|>analysis") ==
                    "assistant<|channel|>analysis", "Harmony header");
        });
        test("reasoning content and tools survive every split", [] {
            const std::string raw = "<think>check</think>answer<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
                                    "<tool_call>{\"name\":\"count\",\"arguments\":{\"n\":2}}</tool_call>";
            const auto spec = chat_format_make(chat_format_kind::HERMES_JSON, true);
            for (size_t split = 0; split <= raw.size(); ++split) {
                chat_stream stream(spec, "", {});
                auto deltas = stream.update(raw.substr(0, split), false);
                auto tail = stream.update(raw.substr(split), true);
                deltas.insert(deltas.end(), tail.begin(), tail.end());
                auto msg = assemble(deltas);
                require(msg.reasoning == "check" && msg.content == "answer", "reasoning/content split");
                require(msg.tool_calls.size() == 2, "two calls");
                require(msg.tool_calls[0].name == "lookup" && msg.tool_calls[0].arguments == "{\"city\":\"Paris\"}", "first call");
                require(msg.tool_calls[1].name == "count" && msg.tool_calls[1].arguments == "{\"n\":2}", "second call");
                require(msg.tool_calls[0].id != msg.tool_calls[1].id, "unique per-call IDs");
                require(stream.update("", true).empty(), "repeated final emits nothing");
            }
        });
        test("partial markers are held and final literal text is flushed", [] {
            chat_stream stream(chat_format_make(chat_format_kind::HERMES_JSON, true), "", {});
            auto first = assemble(stream.update("hello <thi", false));
            require(first.content == "hello", "partial marker leaked");
            auto last = assemble(stream.update("", true));
            require(first.content + last.content == "hello <thi", "final text lost");
        });
        test("prefilled reasoning yields content in the same update", [] {
            chat_stream stream(chat_format_make(chat_format_kind::HERMES_JSON, true), "<think>", {});
            auto ds = stream.update("reason</think>result", true);
            require(ds.size() == 2, "both deltas required");
            auto msg = assemble(ds);
            require(msg.reasoning == "reason" && msg.content == "result", "prefill split");
        });
        test("complete UTF-8 chunks retain Chinese and emoji", [] {
            chat_stream stream(chat_format_make(chat_format_kind::CONTENT_ONLY, false), "", {});
            auto ds = stream.update(u8"你好", false);
            auto tail = stream.update(u8" 🌍", true);
            ds.insert(ds.end(), tail.begin(), tail.end());
            require(assemble(ds).content == u8"你好 🌍", "UTF-8 changed");
        });
        test("byte-wise JSON string arguments append without duplication", [] {
            const std::string raw = R"(<tool_call>{"arguments":"{\"n\":2}","name":"count"}</tool_call>)";
            chat_stream stream(chat_format_make(chat_format_kind::HERMES_JSON, false), "", {});
            std::vector<chat_msg_delta> all;
            for (char c : raw) {
                auto ds = stream.update(std::string(1, c), false);
                all.insert(all.end(), ds.begin(), ds.end());
            }
            auto tail = stream.update("", true);
            all.insert(all.end(), tail.begin(), tail.end());
            auto msg = assemble(all);
            require(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "count", "late name lost");
            require(msg.tool_calls[0].arguments == "{\"n\":2}", "string arguments duplicated or lost");
        });
        test("Harmony analysis final and function recipient", [] {
            auto spec = chat_format_make(chat_format_kind::HARMONY, false);
            auto msg = chat_parse(spec,
                "assistant<|channel|>analysis<|message|>check<|end|>"
                "<|start|>assistant<|channel|>final<|message|>answer<|end|>"
                "<|start|>assistant to=functions.count<|channel|>commentary<|message|>{\"n\":2}<|call|>",
                false, {});
            require(msg.reasoning == "check" && msg.content == "answer", "Harmony channels mixed");
            require(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "count" &&
                    msg.tool_calls[0].arguments == "{\"n\":2}", "Harmony recipient lost");
        });
    } catch (...) { return 1; }
    std::cout << passed << " fixture groups passed\n";
    return 0;
}
