#include "chat/tool_history.hpp"
#include <iostream>
#include <stdexcept>

using json = nlohmann::ordered_json;
static void require(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}

int main() {
    try {
        const json valid = json::parse(R"([
            {"role":"user","content":"Read the records"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"a","type":"function","function":{"name":"old_lookup","arguments":"{\"key\":\"first\"}"}},
                {"id":"b","type":"function","function":{"name":"old_lookup","arguments":{"key":"second"}}}
            ]},
            {"role":"tool","tool_call_id":"b","content":"BETA"},
            {"role":"tool","tool_call_id":"a","content":"ALPHA"},
            {"role":"user","content":"Report the first and second results"}
        ])");
        std::string error;
        json messages = valid;
        require(flm::chat::normalize_openai_tool_history(messages, error), "valid history rejected");
        require(error.empty(), "successful validation left an error");
        require(messages[2]["tool_call_id"] == "a" && messages[2]["content"] == "ALPHA" &&
                messages[3]["tool_call_id"] == "b" && messages[3]["content"] == "BETA",
                "tool results lost their call association");
        require(messages[1] == valid[1], "assistant content or argument representation changed");
        messages[1].erase("content");
        require(flm::chat::normalize_openai_tool_history(messages, error), "omitted assistant content rejected");
        require(!messages[1].contains("content"), "omitted content was rewritten");
        const auto normalized = messages;
        require(flm::chat::normalize_openai_tool_history(messages, error) && messages == normalized,
                "history normalization is not idempotent");
        auto rejected = [&](json bad) {
            const json before = bad;
            require(!flm::chat::normalize_openai_tool_history(bad, error), "malformed tool history accepted");
            require(!error.empty(), "malformed history has no error detail");
            require(bad == before, "failed validation partially rewrote history");
        };
        rejected(json::object());
        rejected(json::array({42}));
        messages = valid; messages[0]["role"] = 7; rejected(messages);
        messages = valid; messages.erase(messages.begin() + 1); rejected(messages); // orphan
        messages = valid; messages[2]["tool_call_id"] = "unknown"; rejected(messages);
        messages = valid; messages[2].erase("tool_call_id"); rejected(messages);
        messages = valid; messages[2]["tool_call_id"] = ""; rejected(messages);
        messages = valid; messages[2]["tool_call_id"] = "a"; rejected(messages); // duplicate result
        messages = valid; messages.erase(messages.begin() + 2); rejected(messages); // incomplete group
        messages = valid; messages.erase(messages.begin() + 2, messages.end()); rejected(messages);
        messages = valid; messages[1]["tool_calls"][1]["id"] = "a"; rejected(messages);
        messages = valid; messages[1]["tool_calls"][0].erase("id"); rejected(messages);
        messages = valid; messages[1]["tool_calls"][0]["function"]["arguments"] = "{bad}"; rejected(messages);
        messages = valid; messages[1]["tool_calls"][0]["function"]["arguments"] = "[]"; rejected(messages);
        messages = valid; messages[1]["tool_calls"][0]["function"]["arguments"] = 3; rejected(messages);
        messages = valid; messages[1]["tool_calls"][0]["function"]["name"] = ""; rejected(messages);
        messages = valid; messages[1]["tool_calls"][0]["type"] = "other"; rejected(messages);
        messages = valid; messages[1]["role"] = "user"; rejected(messages);
        messages = valid; messages[1]["tool_calls"] = json::object(); rejected(messages);
        messages = valid; messages.push_back(valid[2]); rejected(messages); // closed group cannot reopen
        messages = json::parse(R"([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi","tool_calls":null},{"role":"user","content":"Continue"}])");
        const auto plain = messages;
        require(flm::chat::normalize_openai_tool_history(messages, error) && messages == plain,
                "ordinary chat was changed");
        messages = valid;
        messages.insert(messages.end(), valid.begin(), valid.end());
        require(flm::chat::normalize_openai_tool_history(messages, error),
                "independent completed groups cannot reuse a historical call ID");
        std::cout << "tool history validation, association and preservation PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
