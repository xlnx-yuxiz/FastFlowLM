#include "chat/chat_constraint.hpp"
#include "chat/chat_grammar.hpp"
#include <iostream>
#include <stdexcept>

using json = nlohmann::ordered_json;
static void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void check_string_bounds(chat_grammar_engine& grammar);

static int run_checks(int argc, char** argv) {
    chat_constraint_request parsed;
    std::string error;
    const json no_tools = json::array();
    for (const auto& format : {json("json_object"), json::array(), json{{"type", "unknown"}},
            json{{"type", "json_schema"}}, json{{"type", "json_schema"}, {"json_schema", {{"schema", nullptr}}}}}) {
        require(!chat_constraint_request_from_openai({{"response_format", format}}, no_tools, parsed, error),
                "malformed response_format was silently ignored");
    }
    require(!chat_constraint_request_from_openai({{"tool_choice", "required"}}, no_tools, parsed, error),
            "required without tools accepted");
    const json tools = json::parse(R"([
        {"type":"function","function":{"name":"lookup","parameters":{
            "type":"object","properties":{"city":{"type":"string","enum":["Paris"]}},
            "required":["city"],"additionalProperties":false
        }}}
    ])");
    require(!chat_constraint_request_from_openai({{"tool_choice", {{"type", "function"}, {"function", {{"name", "missing"}}}}}},
            tools, parsed, error), "unknown forced tool accepted");
    require(chat_constraint_request_from_openai({{"response_format", {{"type", "json_object"}}}}, no_tools, parsed, error),
            "valid JSON object request rejected");
    require(parsed.json_object, "JSON object request lost its constraint");
    for (const auto& schema : {
            json::parse(R"({"type":"array","items":{"const":1},"minItems":2,"maxItems":2,"uniqueItems":true})"),
            json::parse(R"({"type":"string","pattern":"^x$","minLength":2})"),
            json::parse(R"({"type":"integer","enum":[1],"minimum":2})"),
            json::parse(R"({"oneOf":[{"type":"integer"},{"type":"number"}]})"),
            json::parse(R"({"type":"integer","multipleOf":2})")}) {
        require(!chat_constraint_request_from_openai({{"response_format", {{"type", "json_schema"},
            {"json_schema", {{"schema", schema}, {"strict", true}}}}}}, no_tools, parsed, error),
            "schema restriction that XGrammar may ignore was accepted");
    }
    require(chat_constraint_request_from_openai({{"response_format", {{"type", "json_object"}}}}, no_tools, parsed, error),
            "JSON object rejected after invalid schema");
    for (const json malformed_parameters : {json("not-a-schema"), json::array(), json(nullptr)}) {
        json bad_tools = tools;
        bad_tools[0]["function"]["parameters"] = malformed_parameters;
        require(!chat_constraint_request_from_openai({{"tool_choice", "required"}}, bad_tools, parsed, error),
                "malformed constrained tool schema accepted");
    }
    json bad_strict = tools;
    bad_strict[0]["function"]["strict"] = "true";
    require(!chat_constraint_request_from_openai({{"tool_choice", "required"}}, bad_strict, parsed, error),
            "malformed strict field accepted");
    require(chat_constraint_request_from_openai({{"response_format", {{"type", "json_object"}}}}, no_tools, parsed, error),
            "JSON object rejected after invalid tool definition");
    require(argc == 2, "provide existing tokenizer.json path");
    chat_vocab vocab;
    vocab.tokenizer_json = argv[1];
    // Use the tokenizer's full vocabulary. Grammar construction indexes its bytes directly.
    auto plain = chat_format_make(chat_format_kind::CONTENT_ONLY, false);
    chat_grammar_engine grammar(vocab, plain);
    require(grammar.accepts(plain, parsed, "", "{\"ok\":true}", error), "valid JSON rejected");
    require(!grammar.accepts(plain, parsed, "", "hello", error), "plain text escaped JSON constraint");
    auto constraint = grammar.make(plain, parsed);
    bool rejected=false;
    try { constraint->start("not-json"); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected, "grammar desynchronization fell back to unconstrained generation");
    bool new_case_failed = false;
    const auto run_new_case = [&](const char* name, const auto& check) {
        try { check(); }
        catch (const std::exception& failure) {
            new_case_failed = true;
            std::cerr << "new constraint case failed (" << name << "): " << failure.what() << std::endl;
        }
    };
    run_new_case("JSON response with auto tool strict metadata", [&] {
        json bad_strict_auto = tools;
        bad_strict_auto[0]["function"]["strict"] = "bad";
        const json json_only_schema = {{"type", "object"}, {"properties", {{"s", {{"type", "string"}, {"const", "JSON_ONLY"}}}}},
                                       {"required", {"s"}}, {"additionalProperties", false}};
        const json json_only_body = {{"tool_choice", "auto"}, {"response_format", {{"type", "json_schema"},
                                       {"json_schema", {{"name", "json_only"}, {"strict", true},
                                                        {"schema", json_only_schema}}}}}};
        require(chat_constraint_request_from_openai(json_only_body, bad_strict_auto, parsed, error),
                "optional tool strict metadata rejected an unrelated JSON response constraint");
        require(grammar.accepts(plain, parsed, "", R"({"s":"JSON_ONLY"})", error),
                "JSON response format was not enforced alongside auto tool metadata");
    });
    const json schema = {{"type", "object"}, {"properties", {{"value", {{"type", "integer"}, {"enum", {7}}}}}},
                         {"required", {"value"}}, {"additionalProperties", false}};
    require(chat_constraint_request_from_openai({{"response_format", {{"type", "json_schema"},
        {"json_schema", {{"name", "answer"}, {"strict", true}, {"schema", schema}}}}}}, no_tools, parsed, error),
        "valid schema request rejected");
    require(grammar.accepts(plain, parsed, "", "{\"value\":7}", error), "valid schema output rejected");
    require(!grammar.accepts(plain, parsed, "", "{\"value\":8}", error), "wrong enum value accepted");
    require(!grammar.accepts(plain, parsed, "", "{\"value\":7,\"extra\":0}", error), "extra key accepted");
    auto hermes=chat_format_make(chat_format_kind::HERMES_JSON, false);
    require(chat_constraint_request_from_openai({{"tool_choice", {{"type", "function"}, {"function", {{"name", "lookup"}}}}}},
        tools, parsed, error), "valid forced tool rejected");
    require(grammar.accepts(hermes, parsed, "", "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>", error),
        "valid forced tool output rejected");
    require(!grammar.accepts(hermes, parsed, "", "<tool_call>\n{\"name\": \"missing\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>", error),
        "wrong forced tool output accepted");
    run_new_case("forced Hermes literal const data", [&] {
        const json literal_tool = json::parse(R"([{"type":"function","function":{"name":"lookup","parameters":{
            "type":"object","properties":{"shape":{"const":{"type":"array"}},"values":{"type":"array"}},
            "required":["shape","values"],"additionalProperties":false}}}])");
        require(chat_constraint_request_from_openai({{"tool_choice", {{"type", "function"}, {"function", {{"name", "lookup"}}}}}},
            literal_tool, parsed, error), "forced tool with const object and unconstrained array rejected");
        require(grammar.accepts(hermes, parsed, "",
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"shape\":{\"type\":\"array\"},\"values\":[1,\"x\"]}}\n</tool_call>", error),
            "forced tool changed const object data or rejected array without items");
        require(!grammar.accepts(hermes, parsed, "",
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"shape\":{\"type\":\"array\",\"items\":{}},\"values\":[1,\"x\"]}}\n</tool_call>", error),
            "forced tool accepted extra schema-looking key in const object data");
    });
    for (const auto kind : {chat_format_kind::GEMMA4, chat_format_kind::LFM2_PYTHONIC,
                            chat_format_kind::QWEN_XML, chat_format_kind::HARMONY}) {
        bool unsupported = false;
        try { grammar.make(chat_format_make(kind, false), parsed); }
        catch (const std::invalid_argument&) { unsupported = true; }
        require(unsupported, "unqualified forced-tool dialect accepted");
    }
    const json literal = json::parse(R"({"type":"object","properties":{"not":{"const":{"if":1,"uniqueItems":true}}},"required":["not"],"additionalProperties":false})");
    require(chat_constraint_request_from_openai({{"response_format", {{"type", "json_schema"},
        {"json_schema", {{"schema", literal}}}}}}, no_tools, parsed, error),
        "property names or const object data were interpreted as schema keywords");
    require(!new_case_failed, "new constraint regression cases failed");
    // Automatic routing must preserve a real answer branch and reject invalid
    // combinations instead of silently behaving like ordinary template auto.
    for (const json choice : {json("none"), json("required"),
            json{{"type", "function"}, {"function", {{"name", "lookup"}}}}}) {
        require(!chat_constraint_request_from_openai({{"tool_choice", choice}, {"tool_routing", "structured"}},
            tools, parsed, error), "structured routing silently accepted a non-auto choice");
    }
    require(!chat_constraint_request_from_openai({{"tool_routing", "unknown"}}, tools, parsed, error),
            "unknown routing mode was ignored");
    require(!chat_constraint_request_from_openai({{"tool_routing", "structured"}}, no_tools, parsed, error),
            "structured routing without tools accepted");
    require(!chat_constraint_request_from_openai({{"tool_routing", "structured"},
        {"response_format", {{"type", "json_object"}}}}, tools, parsed, error),
        "structured routing and response schema were conflated");
    require(chat_constraint_request_from_openai({{"tool_routing", "structured"}}, tools, parsed, error),
            "structured auto was rejected");
    require(grammar.accepts(hermes, parsed, "", R"({"type": "message", "content": "Hello!"})", error),
            "ordinary no-tool answer is not allowed");
    require(grammar.accepts(hermes, parsed, "", R"({"type": "message", "content": "{\"label\":\"oak\",\"code\":7}"})", error),
            "JSON answer cannot be escaped inside the message envelope");
    require(grammar.accepts(hermes, parsed, "<think>", R"({"type": "message", "content": "line\nC:\\path\t\u4e2d"})", error),
            "escaped answer or literal prefill is not supported");
    require(!grammar.accepts(hermes, parsed, "", R"({"type": "message", "content": ""})", error),
            "empty automatic answer accepted by grammar");
    require(grammar.accepts(hermes, parsed, "", R"({"type": "tool_call", "name": "lookup", "arguments": {"city":"Paris"}})", error),
            "declared tool route is not allowed");
    for (const std::string raw : {
            R"({"type": "tool_call", "name": "missing", "arguments": {"city":"Paris"}})",
            R"({"type": "tool_call", "name": "lookup", "arguments": {"city":"London"}})",
            R"({"type": "tool_call", "name": "lookup", "arguments": {"city":"Paris"},"content":"answer"})",
            R"({"type": "message", "content": "Hi", "arguments": {}})",
            "<tools>{\"name\":\"lookup\",\"arguments\":{}}</tools>"}) {
        require(!grammar.accepts(hermes, parsed, "", raw, error), "invalid or ambiguous auto envelope accepted");
    }
    auto routed = grammar.make(hermes, parsed);
    // Template-prefill heuristics may see a literal marker in a user message.
    // The request-owned envelope starts at the generated JSON, independently.
    routed->start("<think>");
    check_string_bounds(grammar);
    std::cout << "request, grammar and fail-closed checks PASS\n";
    return 0;
}

int main(int argc, char** argv) {
    try { return run_checks(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "constraint check failed: " << error.what() << std::endl;
        return 1;
    }
}
