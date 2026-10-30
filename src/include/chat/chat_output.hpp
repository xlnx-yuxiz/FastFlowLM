/// Request-owned shared parsing with the HIP tool-output error contract.
#pragma once

#include "chat/chat_stream.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct chat_output_result {
    chat_msg message;
    std::vector<chat_msg_delta> deltas;
    std::string error;
};

class chat_output {
public:
    chat_output(chat_format_spec spec, std::string prefill,
                nlohmann::ordered_json tools, bool strict_tools, bool allow_tool_calls,
                bool json_content = false, bool structured_auto = false);

    // chat_stream borrows our tools. Keep the owner at a stable address.
    chat_output(const chat_output&) = delete;
    chat_output& operator=(const chat_output&) = delete;
    chat_output(chat_output&&) = delete;
    chat_output& operator=(chat_output&&) = delete;

    chat_output_result update(const std::string& text, bool final);

private:
    chat_format_spec spec_;
    nlohmann::ordered_json tools_;
    bool strict_tools_;
    bool allow_tool_calls_;
    std::string raw_;
    chat_stream stream_;
    bool finalized_ = false;
    std::string error_;
    bool structured_auto_ = false;
    chat_msg auto_message_;
};
