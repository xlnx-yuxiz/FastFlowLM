#pragma once

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace flm::chat {

// Validate before rendering or changing model state. Some model templates omit
// tool_call_id, so emit each completed result group in its declared call order.
// Commit the normalized copy only after the entire history has been validated.
inline bool normalize_openai_tool_history(nlohmann::ordered_json& messages,
                                         std::string& error) {
    using json = nlohmann::ordered_json;
    error.clear();
    const auto reject = [&](const std::string& detail) {
        error = "Invalid tool history: " + detail;
        return false;
    };
    if (!messages.is_array()) return reject("messages must be an array");
    json normalized = json::array();
    std::map<std::string, std::size_t> pending;
    std::vector<json> results;
    std::size_t received = 0;
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        const std::string at = "messages[" + std::to_string(index) + "]: ";
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string())
            return reject(at + "each message must have a string role");
        const std::string role = message["role"].get<std::string>();
        if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
            if (role != "assistant" || !message["tool_calls"].is_array())
                return reject(at + "tool_calls must be an assistant array");
        }
        if (role == "tool") {
            if (!message.contains("tool_call_id") || !message["tool_call_id"].is_string() ||
                message["tool_call_id"].get_ref<const std::string&>().empty())
                return reject(at + "tool_call_id must be a nonempty string");
            const auto found = pending.find(message["tool_call_id"].get<std::string>());
            if (found == pending.end()) return reject(at + "tool result has no matching pending call");
            if (!results[found->second].is_null()) return reject(at + "duplicate tool result");
            results[found->second] = message;
            if (++received == pending.size()) {
                for (auto& result : results) normalized.push_back(std::move(result));
                pending.clear();
                results.clear();
                received = 0;
            }
            continue;
        }
        if (!pending.empty()) return reject(at + "all tool calls must be resolved before the next message");
        if (role == "assistant" && message.contains("tool_calls") && message["tool_calls"].is_array()) {
            for (const auto& call : message["tool_calls"]) {
                if (!call.is_object() || !call.contains("id") || !call["id"].is_string() ||
                    call["id"].get_ref<const std::string&>().empty() ||
                    !call.contains("type") || call["type"] != "function" ||
                    !call.contains("function") || !call["function"].is_object())
                    return reject(at + "each tool call needs an id, type function and function object");
                const auto& function = call["function"];
                if (!function.contains("name") || !function["name"].is_string() ||
                    function["name"].get_ref<const std::string&>().empty() ||
                    !function.contains("arguments"))
                    return reject(at + "each function needs a name and arguments");
                const auto& arguments = function["arguments"];
                if (arguments.is_string()) {
                    const auto parsed = json::parse(arguments.get_ref<const std::string&>(), nullptr, false);
                    if (!parsed.is_object()) return reject(at + "function arguments must encode a JSON object");
                } else if (!arguments.is_object()) {
                    return reject(at + "function arguments must be a JSON object or its encoded string");
                }
                const auto id = call["id"].get<std::string>();
                if (!pending.emplace(id, results.size()).second) return reject(at + "duplicate tool call id");
                results.emplace_back(nullptr);
            }
        }
        normalized.push_back(message);
    }
    if (!pending.empty()) return reject("unresolved tool calls at end of messages");
    messages = std::move(normalized);
    return true;
}

} // namespace flm::chat
