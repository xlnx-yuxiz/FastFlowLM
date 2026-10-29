#pragma once
#include "chat/chat_stream.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

// Request-owned protocol. A final answer is never an application function.
inline nlohmann::ordered_json chat_prepare_auto_tools(nlohmann::ordered_json messages,
                                                       const nlohmann::ordered_json& tools) {
    using json = nlohmann::ordered_json;
    const std::string instruction =
        "\n\nAvailable tool definitions (JSON): " + tools.dump() +
        "\nChoose the next action based on the user's request and actual tool results. "
        "If external data or an action is needed, call a declared tool. Never invent tool results. "
        "After receiving a result, either choose the next necessary tool or answer the user. "
        "For questions that need no tools, answer directly. "
        "Return exactly one JSON object using one of these mutually exclusive forms:\n"
        "{\"type\":\"message\",\"content\":\"your answer to the user\"}\n"
        "{\"type\":\"tool_call\",\"name\":\"declared tool name\",\"arguments\":{}}\n"
        "Use the chosen tool's argument schema. The message branch contains the final answer "
        "in content, including any output format the user requested. The tool_call branch "
        "contains no answer or content. Do not repeat a completed action unless fresh data is needed.";
    if (!messages.is_array() || messages.empty())
        throw std::invalid_argument("Structured tool routing requires chat messages");
    if (messages.front().value("role", "") == "system") {
        auto& content = messages.front()["content"];
        if (!content.is_string()) throw std::invalid_argument("Structured routing requires a text system message");
        content = content.get<std::string>() + instruction;
    } else {
        messages.insert(messages.begin(), json{{"role", "system"}, {"content", instruction}});
    }
    return messages;
}

inline chat_msg chat_parse_auto_tools(const std::string& raw, const nlohmann::ordered_json& tools) {
    using json = nlohmann::ordered_json;
    const auto value = json::parse(raw, nullptr, false);
    const auto fail = [] { throw std::invalid_argument("Invalid structured tool routing output"); };
    if (!value.is_object() || !value.contains("type") || !value["type"].is_string()) fail();
    chat_msg result;
    if (value["type"] == "message") {
        if (value.size() != 2 || !value.contains("content") || !value["content"].is_string() ||
            value["content"].get_ref<const std::string&>().empty()) fail();
        result.content = value["content"].get<std::string>();
    } else if (value["type"] == "tool_call") {
        if (value.size() != 3 || !value.contains("name") || !value["name"].is_string() ||
            !value.contains("arguments") || !value["arguments"].is_object()) fail();
        const auto name = value["name"].get<std::string>();
        bool declared = false;
        for (const auto& tool : tools) {
            if (tool.contains("function") && tool["function"].is_object() &&
                tool["function"].value("name", "") == name) declared = true;
        }
        if (!declared) fail();
        result.tool_calls.push_back({chat_new_tool_call_id(), name, value["arguments"].dump()});
    } else fail();
    return result;
}
