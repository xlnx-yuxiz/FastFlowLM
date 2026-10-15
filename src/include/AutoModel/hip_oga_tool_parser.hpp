/// \file hip_oga_tool_parser.hpp
/// \brief No-hardware parser for HIP OGA OpenAI tool-call markup.
#pragma once

#include "nlohmann/json.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace flm::hip_oga {

enum class StreamEventType {
    WAITING,
    CONTENT,
    REASONING,
    TOOL_DONE,
    ERROR_EVENT,
};

struct ParsedToolMarkup {
    std::string content;
    std::string reasoning_content;
    std::vector<std::pair<std::string, std::string>> tool_calls_list;
    std::string error_message;
};

struct ParsedStreamEvent {
    StreamEventType type = StreamEventType::WAITING;
    std::string content;
    std::string tool_id;
    std::string tool_name;
    std::string tool_args_str;
    std::string error_message;
};

inline ParsedStreamEvent make_error_event(std::string message) {
    ParsedStreamEvent result;
    result.type = StreamEventType::ERROR_EVENT;
    result.error_message = std::move(message);
    return result;
}

inline bool decode_tool_call_json(const std::string& raw_json,
                                  std::string& name,
                                  std::string& arguments,
                                  std::string& error_message) {
    auto parsed = nlohmann::json::parse(raw_json, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        error_message =
            "HIP OGA model emitted an invalid <tool_call> block: expected a JSON object";
        return false;
    }

    if (!parsed.contains("name") || !parsed["name"].is_string() ||
        parsed["name"].get<std::string>().empty()) {
        error_message =
            "HIP OGA model emitted an invalid <tool_call> block: missing string name";
        return false;
    }
    if (!parsed.contains("arguments") || parsed["arguments"].is_null()) {
        error_message =
            "HIP OGA model emitted an invalid <tool_call> block: missing arguments";
        return false;
    }

    name = parsed["name"].get<std::string>();
    if (parsed["arguments"].is_string()) {
        arguments = parsed["arguments"].get<std::string>();
    } else {
        arguments = parsed["arguments"].dump();
    }
    return true;
}

inline ParsedToolMarkup parse_tool_call_markup(const std::string& response_text) {
    static const std::string kThinkStart = "<think>";
    static const std::string kThinkEnd = "</think>";
    static const std::string kToolStart = "<tool_call>";
    static const std::string kToolEnd = "</tool_call>";

    ParsedToolMarkup result;
    std::size_t cursor = 0;

    while (cursor < response_text.size()) {
        const std::size_t think_pos = response_text.find(kThinkStart, cursor);
        const std::size_t tool_pos = response_text.find(kToolStart, cursor);
        const bool next_is_think =
            think_pos != std::string::npos &&
            (tool_pos == std::string::npos || think_pos < tool_pos);
        const std::size_t marker_pos = next_is_think ? think_pos : tool_pos;

        if (marker_pos == std::string::npos) {
            result.content += response_text.substr(cursor);
            break;
        }

        result.content += response_text.substr(cursor, marker_pos - cursor);

        if (next_is_think) {
            const std::size_t body_start = marker_pos + kThinkStart.size();
            const std::size_t body_end = response_text.find(kThinkEnd, body_start);
            if (body_end == std::string::npos) {
                result.content += response_text.substr(marker_pos);
                break;
            }
            result.reasoning_content +=
                response_text.substr(body_start, body_end - body_start);
            cursor = body_end + kThinkEnd.size();
            continue;
        }

        const std::size_t body_start = marker_pos + kToolStart.size();
        const std::size_t body_end = response_text.find(kToolEnd, body_start);
        if (body_end == std::string::npos) {
            result = ParsedToolMarkup{};
            result.error_message =
                "HIP OGA model emitted an unterminated <tool_call> block";
            return result;
        }

        std::string tool_name;
        std::string tool_args;
        const std::string raw_tool_json =
            response_text.substr(body_start, body_end - body_start);
        if (!decode_tool_call_json(raw_tool_json,
                                   tool_name,
                                   tool_args,
                                   result.error_message)) {
            result.content.clear();
            result.reasoning_content.clear();
            result.tool_calls_list.clear();
            return result;
        }
        result.tool_calls_list.emplace_back(std::move(tool_name),
                                            std::move(tool_args));
        cursor = body_end + kToolEnd.size();
    }

    return result;
}

inline ParsedStreamEvent parse_stream_markup(std::string& buffer,
                                             std::string& tool_buffer,
                                             StreamEventType& current_mode,
                                             bool& is_in_tool_block,
                                             const std::string& content,
                                             bool is_final) {
    static const std::string kThinkStart = "<think>";
    static const std::string kThinkEnd = "</think>";
    static const std::string kToolStart = "<tool_call>";
    static const std::string kToolEnd = "</tool_call>";

    buffer += content;

    while (true) {
        if (is_in_tool_block) {
            const std::size_t tool_end_pos = buffer.find(kToolEnd);
            if (tool_end_pos != std::string::npos) {
                tool_buffer += buffer.substr(0, tool_end_pos);
                buffer = buffer.substr(tool_end_pos + kToolEnd.size());
                is_in_tool_block = false;

                std::string tool_name;
                std::string tool_args;
                std::string error_message;
                if (!decode_tool_call_json(tool_buffer,
                                           tool_name,
                                           tool_args,
                                           error_message)) {
                    tool_buffer.clear();
                    return make_error_event(std::move(error_message));
                }

                tool_buffer.clear();
                ParsedStreamEvent result;
                result.type = StreamEventType::TOOL_DONE;
                result.tool_name = std::move(tool_name);
                result.tool_args_str = std::move(tool_args);
                return result;
            }

            if (is_final) {
                tool_buffer += buffer;
                buffer.clear();
                tool_buffer.clear();
                is_in_tool_block = false;
                return make_error_event(
                    "HIP OGA model emitted an unterminated <tool_call> block");
            }

            tool_buffer += buffer;
            buffer.clear();
            ParsedStreamEvent result;
            result.type = StreamEventType::WAITING;
            return result;
        }

        std::size_t pos_tool_start = buffer.find(kToolStart);
        std::size_t pos_think = std::string::npos;
        if (current_mode == StreamEventType::CONTENT) {
            pos_think = buffer.find(kThinkStart);
        } else if (current_mode == StreamEventType::REASONING) {
            pos_think = buffer.find(kThinkEnd);
        }

        std::size_t min_pos = std::string::npos;
        if (pos_tool_start != std::string::npos) min_pos = pos_tool_start;
        if (pos_think != std::string::npos) {
            min_pos = min_pos == std::string::npos
                ? pos_think
                : std::min(min_pos, pos_think);
        }

        if (min_pos != std::string::npos && min_pos > 0) {
            ParsedStreamEvent result;
            result.content = buffer.substr(0, min_pos);
            result.type = current_mode;
            buffer = buffer.substr(min_pos);
            return result;
        }

        if (min_pos == 0) {
            if (pos_tool_start == 0) {
                is_in_tool_block = true;
                tool_buffer.clear();
                buffer = buffer.substr(kToolStart.size());
                continue;
            }
            if (pos_think == 0) {
                if (current_mode == StreamEventType::CONTENT) {
                    buffer = buffer.substr(kThinkStart.size());
                    current_mode = StreamEventType::REASONING;
                } else {
                    buffer = buffer.substr(kThinkEnd.size());
                    current_mode = StreamEventType::CONTENT;
                }
                continue;
            }
        }

        std::vector<std::string> active_markers = {kToolStart};
        active_markers.push_back(current_mode == StreamEventType::CONTENT
                                     ? kThinkStart
                                     : kThinkEnd);

        std::size_t safe_flush_len = buffer.length();
        for (const auto& marker : active_markers) {
            for (std::size_t i = 1; i <= marker.length() && i <= buffer.length(); ++i) {
                if (buffer.compare(buffer.length() - i, i, marker, 0, i) == 0) {
                    safe_flush_len = std::min(safe_flush_len, buffer.length() - i);
                }
            }
        }

        if (safe_flush_len > 0) {
            ParsedStreamEvent result;
            result.content = buffer.substr(0, safe_flush_len);
            result.type = current_mode;
            buffer = buffer.substr(safe_flush_len);
            return result;
        }

        if (is_final && !buffer.empty()) {
            ParsedStreamEvent result;
            result.content = buffer;
            result.type = current_mode;
            buffer.clear();
            return result;
        }

        ParsedStreamEvent result;
        result.type = StreamEventType::WAITING;
        return result;
    }
}

}  // namespace flm::hip_oga
