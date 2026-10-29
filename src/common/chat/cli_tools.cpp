#include "chat/cli_tools.hpp"
#include "chat/chat_constraint.hpp"
#include "chat/tool_history.hpp"
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace flm::chat {
namespace {
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
json read_json(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("Cannot read JSON file: " + path.string());
    std::vector<char> bytes(65537);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const auto count = input.gcount();
    if (count > 65536) throw std::invalid_argument("JSON file exceeds 64 KiB");
    return json::parse(bytes.begin(), bytes.begin()+count);
}
const std::map<std::string, std::map<std::string, std::string>> signatures = {
    {"add_numbers", {{"a", "integer"}, {"b", "integer"}}},
    {"lookup_record", {{"name", "string"}}}, {"read_record", {{"path", "string"}}}
};
const json finish_tool = {{"type", "function"}, {"function", {
    {"name", "finish_task"}, {"description", "Deliver the final answer after using the tool results."},
    {"parameters", {{"type", "object"}, {"properties", {{"answer", {{"type", "string"}}}}},
                    {"required", json::array({"answer"})}, {"additionalProperties", false}}}}}};
void validate_demo(const json& tools) {
    for (const auto& tool : tools) {
        const auto name = tool.at("function").at("name").get<std::string>();
        const auto found = signatures.find(name);
        if (found == signatures.end()) throw std::invalid_argument("No demo executor for tool: " + name);
        const auto& schema = tool.at("function").at("parameters");
        const auto& properties = schema.at("properties");
        const auto& required = schema.at("required");
        if (schema.size() != 4 || !schema.contains("additionalProperties") || schema["additionalProperties"] != false ||
            properties.size() != found->second.size() || required.size() != found->second.size())
            throw std::invalid_argument("Demo tool parameters must match the registered signature: " + name);
        std::set<std::string> required_names;
        for (const auto& item : required) required_names.insert(item.get<std::string>());
        for (const auto& [key, kind] : found->second) {
            if (!properties.contains(key) || !required_names.count(key) || properties[key].value("type", "") != kind)
                throw std::invalid_argument("Incorrect demo parameter: " + name + "." + key);
            for (const auto& item : properties[key].items())
                if (item.key() != "type" && item.key() != "description")
                    throw std::invalid_argument("Demo parameters support only type and description");
        }
    }
}
}  // namespace

cli_tools_session::cli_tools_session(const std::filesystem::path& file, const std::string& executor,
                                     const std::filesystem::path& data_dir, int max_turns, const std::string& choice)
    : tools_(read_json(file)), demo_(executor == "demo"), automatic_(choice == "auto"), max_turns_(max_turns),
      data_dir_(std::filesystem::weakly_canonical(data_dir.empty() ? file.parent_path()/"data" : data_dir)) {
    if (!executor.empty() && !demo_) throw std::invalid_argument("--tool-executor supports only demo");
    if (choice != "required" && choice != "auto") throw std::invalid_argument("--tool-choice supports auto or required");
    if (max_turns < 1 || max_turns > 12) throw std::invalid_argument("--max-tool-turns must be 1..12");
    if (!tools_.is_array() || tools_.empty()) throw std::invalid_argument("--tools requires a nonempty OpenAI function array");
    const json request = {{"tool_choice", choice}, {"parallel_tool_calls", false},
                          {"tool_routing", automatic_ ? "structured" : "template"}};
    chat_constraint_request parsed;
    std::string error;
    if (!chat_constraint_request_from_openai(request, tools_, parsed, error)) throw std::invalid_argument(error);
    for (const auto& tool : tools_) {
        const auto& name = tool.at("function").at("name").get_ref<const std::string&>();
        if (name.size() > 64 || name.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::invalid_argument("Tool name must match [A-Za-z0-9_-]{1,64}");
    }
    if (demo_) {
        validate_demo(tools_);
        if (!automatic_) tools_.push_back(finish_tool);
    }
}

cli_tools_session::json cli_tools_session::record(const std::string& relative) const {
    namespace fs = std::filesystem;
    if (relative.empty() || relative.find('\\') != std::string::npos || relative.find(':') != std::string::npos)
        throw std::invalid_argument("Invalid tool data path");
    const auto path = fs::u8path(relative);
    if (path.is_absolute()) throw std::invalid_argument("Tool data path must be relative");
    for (const auto& part : path) if (part == "..") throw std::invalid_argument("Tool data path escapes data directory");
    const auto target = fs::weakly_canonical(data_dir_/path);
    const auto inside = target.lexically_relative(data_dir_);
    if (inside.empty() || inside.is_absolute()) throw std::invalid_argument("Invalid tool data path");
    for (const auto& part : inside) if (part == "..") throw std::invalid_argument("Tool data path escapes data directory");
    auto value = read_json(target);
    if (!value.is_object()) throw std::invalid_argument("Tool data must be a JSON object");
    if (value.dump().size() > 4096) throw std::invalid_argument("Tool result exceeds 4096 UTF-8 bytes");
    return value;
}

cli_tools_session::json cli_tools_session::execute(const std::string& name, const json& args) const {
    bool enabled = false;
    for (const auto& tool : tools_) if (tool["function"]["name"] == name) enabled = true;
    const auto found = signatures.find(name);
    if (!demo_ || !enabled || found == signatures.end()) throw std::invalid_argument("Tool has no enabled executor: " + name);
    if (!args.is_object() || args.size() != found->second.size()) throw std::invalid_argument("Incorrect tool arguments");
    for (const auto& [key, kind] : found->second) {
        if (!args.contains(key)) throw std::invalid_argument("Missing tool argument: " + key);
        if (kind == "integer") {
            constexpr int64_t limit = 9007199254740991LL;
            if (!args[key].is_number_integer() ||
                (args[key].is_number_unsigned() && args[key].get<uint64_t>() > static_cast<uint64_t>(limit)) ||
                args[key].get<int64_t>() < -limit || args[key].get<int64_t>() > limit)
                throw std::invalid_argument("Expected integer within +/- (2^53-1): " + key);
        } else if (!args[key].is_string() || args[key].get_ref<const std::string&>().empty() ||
                   args[key].get_ref<const std::string&>().size() > 256) {
            throw std::invalid_argument("Expected nonempty string of at most 256 bytes: " + key);
        }
    }
    if (name == "add_numbers") return {{"result", args["a"].get<int64_t>()+args["b"].get<int64_t>()}};
    if (name == "lookup_record") {
        auto index = record("index.json");
        const auto key = args["name"].get<std::string>();
        if (!index.contains(key) || !index[key].is_string()) throw std::invalid_argument("Record not found: " + key);
        const auto path = index[key].get<std::string>();
        (void)record(path);
        return {{"path", path}};
    }
    return record(args["path"].get<std::string>());
}

void cli_tools_session::run(const std::string& prompt, const generate_fn& generate, std::ostream& output) const {
    if (prompt.empty()) throw std::invalid_argument("Tool prompt must not be empty");
    const std::string system = automatic_
        ? "Answer the user's request. Use the provided tools when needed, read their actual results, and give a final answer once the work is complete. For requests that do not need tools, answer directly."
        : demo_
        ? "You are a tool-using assistant. Use the provided functions to do the requested work. Read tool results before choosing your next action. Once the task is complete, call finish_task with the final answer as a string. Do not call finish_task until you have obtained the requested data from the tools. Do not invent tool results."
        : "Use one of the provided functions to answer the user's request. Return the appropriate function call with its arguments.";
    json messages = json::array({{{"role", "system"}, {"content", system}}, {{"role", "user"}, {"content", prompt}}});
    std::set<std::string> ids;
    for (int turn = 0; turn < (demo_ ? max_turns_ : 1); ++turn) {
        std::string error;
        if (!normalize_openai_tool_history(messages, error)) throw std::runtime_error(error);
        auto message = generate(messages, tools_);
        if (automatic_ && message.tool_calls.empty()) {
            if (message.content.empty()) throw std::runtime_error("Expected a tool call or a nonempty answer");
            if (demo_) output << "[answer] " << message.content << std::endl;
            else output << json{{"role", "assistant"}, {"content", message.content}}.dump(2) << std::endl;
            return;
        }
        if (message.tool_calls.size() != 1) throw std::runtime_error("Expected exactly one required tool call");
        auto call = message.tool_calls.front();
        // Parsers without wire IDs leave them empty; the local client assigns a per-round ID.
        if (call.id.empty()) call.id = "cli-call-" + std::to_string(turn+1);
        if (!ids.insert(call.id).second) throw std::runtime_error("Duplicate tool call ID");
        bool enabled = false;
        for (const auto& tool : tools_) if (tool["function"]["name"] == call.name) enabled = true;
        if (!enabled) throw std::runtime_error("Model selected an unknown tool: " + call.name);
        const auto arguments = json::parse(call.arguments);
        if (!arguments.is_object()) throw std::runtime_error("Tool arguments must be a JSON object");
        const json wire_call = {{"id", call.id}, {"type", "function"},
                               {"function", {{"name", call.name}, {"arguments", arguments.dump()}}}};
        const json assistant = {{"role", "assistant"}, {"content", message.content},
                                {"tool_calls", json::array({wire_call})}};
        if (!demo_) { output << assistant.dump(2) << std::endl; return; }
        output << "[call] " << call.name << " " << arguments.dump() << std::endl;
        messages.push_back(assistant);
        if (!automatic_ && call.name == "finish_task") {
            if (arguments.size() != 1 || !arguments.contains("answer") || !arguments["answer"].is_string() ||
                arguments["answer"].get_ref<const std::string&>().empty())
                throw std::runtime_error("finish_task requires a nonempty answer string");
            output << "[answer] " << arguments["answer"].get<std::string>() << std::endl;
            return;
        }
        const auto result = execute(call.name, arguments);
        output << "[result] " << result.dump() << std::endl;
        messages.push_back({{"role", "tool"}, {"tool_call_id", call.id}, {"content", result.dump()}});
    }
    throw std::runtime_error(automatic_ ? "Tool turn limit reached before a final answer" :
                                         "Tool turn limit reached before finish_task");
}
}  // namespace flm::chat
