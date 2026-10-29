#include "chat/chat_constraint.hpp"
#include "chat/chat_grammar.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Test complete JSON spellings, not C++ string byte counts. Each payload is
// nested in an object/array so the same schema exercises all public routes.
void check_string_bounds(chat_grammar_engine& grammar) {
    struct example { std::string name, spelling; int length; };
    std::vector<example> examples = {
        {"empty", R"("")", 0}, {"ascii", R"("A")", 1}, {"two-ascii", R"("AB")", 2},
        {"quote", R"("\"")", 1}, {"backslash", R"("\\")", 1}, {"slash", R"("\/")", 1},
        {"backspace", R"("\b")", 1}, {"formfeed", R"("\f")", 1},
        {"newline", R"("\n")", 1}, {"return", R"("\r")", 1}, {"tab", R"("\t")", 1},
        {"literal-backslash-n", R"("\\n")", 2},
        {"escaped-ascii", R"("\u0061")", 1}, {"escaped-nul", R"("\u0000")", 1},
        {"escaped-control", R"("\u001f")", 1},
        {"cjk", u8R"("中")", 1}, {"escaped-cjk", R"("\u4e2d")", 1},
        {"emoji", u8R"("😀")", 1}, {"escaped-emoji", R"("\uD83D\uDE00")", 1},
        {"mixed-case-pair", R"("\ud83D\uDe00")", 1},
        {"first-pair", R"("\ud800\udc00")", 1}, {"last-pair", R"("\udbff\udfff")", 1},
        {"before-surrogates", R"("\uD7FF")", 1}, {"after-surrogates", R"("\uE000")", 1},
        {"last-bmp", R"("\uFFFF")", 1},
        {"two-cjk-ascii", u8R"("中A")", 2}, {"two-pair-ascii", R"("\uD83D\uDE00A")", 2},
        {"two-pairs", R"("\uD83D\uDE00\uD83D\uDE00")", 2},
        {"combining", R"("e\u0301")", 2}, {"precomposed", R"("\u00e9")", 1},
        {"path", R"("C:\\tmp")", 6}, {"three", R"("A\nB")", 3},
        {"invalid-escape", R"("\x")", -1}, {"invalid-hex", R"("\u00xz")", -1},
        {"short-unicode", R"("\u123")", -1}, {"trailing-slash", R"("\")", -1},
        {"high-alone", R"("\uD83D")", -1}, {"low-alone", R"("\uDE00")", -1},
        {"high-bmp", R"("\uD83D\u0041")", -1}, {"reversed-pair", R"("\uDE00\uD83D")", -1},
        {"high-high", R"("\uD83D\uD83D")", -1},
    };
    for (int c = 0; c < 32; ++c) {
        examples.push_back({"raw-control-" + std::to_string(c), std::string("\"") + char(c) + "\"", -1});
    }
    struct bounds { int min, max; };
    const std::vector<bounds> limits = {{1, 1}, {2, 2}, {0, 0}, {0, 1}, {1, 3}, {1, -1}, {-1, 2}};
    size_t total = 0, failed = 0;
    for (const auto& limit : limits) {
        json value = {{"type", "string"}};
        if (limit.min >= 0) value["minLength"] = limit.min;
        if (limit.max >= 0) value["maxLength"] = limit.max;
        const json schema = {{"type", "object"}, {"properties", {{"values", {
            {"type", "array"}, {"items", value}, {"minItems", 1}, {"maxItems", 1}}}}},
            {"required", {"values"}}, {"additionalProperties", false}};
        const json tools = json::array({{{"type", "function"}, {"function", {
            {"name", "echo"}, {"parameters", schema}}}}});
        for (const std::string route : {"response", "named", "required", "structured"}) {
            json body;
            if (route == "response") body = {{"response_format", {{"type", "json_schema"},
                {"json_schema", {{"name", "bounded"}, {"strict", true}, {"schema", schema}}}}}};
            else if (route == "named") body = {{"tool_choice", {{"type", "function"},
                {"function", {{"name", "echo"}}}}}};
            else if (route == "required") body = {{"tool_choice", "required"}, {"parallel_tool_calls", false}};
            else body = {{"tool_choice", "auto"}, {"tool_routing", "structured"}};
            chat_constraint_request request;
            std::string error;
            if (!chat_constraint_request_from_openai(body, tools, request, error))
                throw std::runtime_error("bounded schema admission: " + error);
            const auto format = chat_format_make(route == "response" ? chat_format_kind::CONTENT_ONLY
                : chat_format_kind::HERMES_JSON, false);
            for (const auto& ex : examples) {
                const std::string arguments = "{\"values\":[" + ex.spelling + "]}";
                std::string raw = arguments;
                if (route == "named" || route == "required") raw =
                    "<tool_call>\n{\"name\": \"echo\", \"arguments\": " + arguments + "}\n</tool_call>";
                if (route == "structured") raw =
                    "{\"type\": \"tool_call\", \"name\": \"echo\", \"arguments\": " + arguments + "}";
                const bool expected = ex.length >= 0 && (limit.min < 0 || ex.length >= limit.min) &&
                    (limit.max < 0 || ex.length <= limit.max);
                const bool actual = grammar.accepts(format, request, "", raw, error);
                ++total;
                if (actual != expected) {
                    ++failed;
                    std::cerr << "string bounds FAIL " << route << " [" << limit.min << "," << limit.max
                        << "] " << ex.name << " expected=" << expected << " actual=" << actual << '\n';
                }
            }
        }
    }
    std::cout << "string bounds: " << total << " cases, " << failed << " failures\n";
    if (failed) throw std::runtime_error("string length/escape regression");
}
