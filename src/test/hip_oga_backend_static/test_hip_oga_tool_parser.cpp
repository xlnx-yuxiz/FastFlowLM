#include "AutoModel/hip_oga_tool_parser.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void require_eq(const std::string& actual,
                const std::string& expected,
                const std::string& message) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message << "\n"
                  << "expected: " << expected << "\n"
                  << "actual: " << actual << "\n";
        std::exit(1);
    }
}

void test_nonstream_multiple_tool_calls() {
    const std::string text =
        "Checking. "
        "<think>Need two calls.</think>"
        "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
        "<tool_call>{\"name\":\"get_time\",\"arguments\":\"{\\\"zone\\\":\\\"UTC\\\"}\"}</tool_call>";

    const flm::hip_oga::ParsedToolMarkup parsed =
        flm::hip_oga::parse_tool_call_markup(text);

    require(parsed.error_message.empty(), "valid tool markup should not report an error");
    require_eq(parsed.content, "Checking. ", "nonstream content before tool call should be preserved");
    require_eq(parsed.reasoning_content, "Need two calls.", "reasoning content should be extracted");
    require(parsed.tool_calls_list.size() == 2, "multiple tool calls should be preserved");
    require_eq(parsed.tool_calls_list[0].first, "get_weather", "first tool name mismatch");
    require_eq(parsed.tool_calls_list[0].second, "{\"city\":\"Paris\"}", "object arguments should serialize to JSON");
    require_eq(parsed.tool_calls_list[1].first, "get_time", "second tool name mismatch");
    require_eq(parsed.tool_calls_list[1].second, "{\"zone\":\"UTC\"}", "string arguments should pass through unchanged");
}

void test_nonstream_invalid_tool_json_is_explicit_error() {
    const flm::hip_oga::ParsedToolMarkup parsed =
        flm::hip_oga::parse_tool_call_markup("<tool_call>not-json</tool_call>");

    require(!parsed.error_message.empty(), "invalid tool JSON should be an explicit parser error");
    require(parsed.tool_calls_list.empty(), "invalid tool JSON must not be surfaced as a tool call");
    require(parsed.content.empty(), "invalid tool JSON must not be downgraded to assistant text");
}

void test_streaming_tool_call_across_chunks() {
    std::string buffer;
    std::string tool_buffer;
    flm::hip_oga::StreamEventType mode = flm::hip_oga::StreamEventType::CONTENT;
    bool in_tool_block = false;

    flm::hip_oga::ParsedStreamEvent event = flm::hip_oga::parse_stream_markup(
        buffer, tool_buffer, mode, in_tool_block, "prefix <tool_", false);
    require(event.type == flm::hip_oga::StreamEventType::CONTENT, "prefix should stream before partial marker");
    require_eq(event.content, "prefix ", "prefix content mismatch");

    event = flm::hip_oga::parse_stream_markup(
        buffer, tool_buffer, mode, in_tool_block,
        "call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"rocm\"}}", false);
    require(event.type == flm::hip_oga::StreamEventType::WAITING, "open tool block should wait for end marker");

    event = flm::hip_oga::parse_stream_markup(
        buffer, tool_buffer, mode, in_tool_block, "</tool_call>", false);
    require(event.type == flm::hip_oga::StreamEventType::TOOL_DONE, "complete tool block should emit a tool event");
    require_eq(event.tool_name, "lookup", "streaming tool name mismatch");
    require_eq(event.tool_args_str, "{\"q\":\"rocm\"}", "streaming tool args mismatch");
}

void test_streaming_final_incomplete_tool_call_is_explicit_error() {
    std::string buffer;
    std::string tool_buffer;
    flm::hip_oga::StreamEventType mode = flm::hip_oga::StreamEventType::CONTENT;
    bool in_tool_block = false;

    flm::hip_oga::ParsedStreamEvent event = flm::hip_oga::parse_stream_markup(
        buffer, tool_buffer, mode, in_tool_block,
        "<tool_call>{\"name\":\"lookup\",\"arguments\":", false);
    require(event.type == flm::hip_oga::StreamEventType::WAITING, "incomplete streaming tool block should wait");

    event = flm::hip_oga::parse_stream_markup(
        buffer, tool_buffer, mode, in_tool_block, "", true);
    require(event.type == flm::hip_oga::StreamEventType::ERROR_EVENT, "final incomplete tool block should report an error");
    require(!event.error_message.empty(), "stream parser error should include a message");
}

}  // namespace

int main() {
    test_nonstream_multiple_tool_calls();
    test_nonstream_invalid_tool_json_is_explicit_error();
    test_streaming_tool_call_across_chunks();
    test_streaming_final_incomplete_tool_call_is_explicit_error();

    std::cout << "PASS: HIP OGA production tool parser contract\n";
    return 0;
}
