/// \file chat_constraint.cpp
/// \brief Build the decoding constraint a chat request asks for

#include "chat/chat_constraint.hpp"
#include "chat/chat_grammar.hpp"
#include <cmath>
#include <limits>
#include <set>

using json = nlohmann::ordered_json;

namespace {

/// \brief A field of a request object, null when it is missing
/// \note Some clients send null for every option they leave unset, so null
///       has to read as missing; json::value() throws on it instead.
const json& field(const json& object, const char* key) {
    static const json missing;
    auto it = object.find(key);
    return it == object.end() ? missing : *it;
}

// The pinned grammar compiler can warn and ignore schema keywords. Admit only
// the subset whose restrictions it actually enforces, including nested schemas.
// Property names and const/enum values are data, never schema keyword locations.
bool supported_schema(const json& schema, std::string& error, const std::string& path = "schema") {
    const auto reject = [&](const std::string& reason) {
        error = path + ": " + reason;
        return false;
    };
    if (schema.is_boolean()) return true;
    if (!schema.is_object()) return reject("must be an object or boolean schema");
    static const std::set<std::string> annotations = {
        "$schema", "$comment", "title", "description", "default", "examples",
        "deprecated", "readOnly", "writeOnly"};
    static const std::set<std::string> types = {
        "object", "array", "string", "integer", "number", "boolean", "null"};
    const auto& type_field = field(schema, "type");
    const bool has_type = schema.contains("type");
    if (has_type && (!type_field.is_string() || !types.count(type_field.get<std::string>())))
        return reject("type must be one supported primitive type name");
    const std::string type = has_type ? type_field.get<std::string>() : "";
    const bool literal = schema.contains("const") || schema.contains("enum");
    std::set<std::string> allowed = annotations;
    allowed.insert("type");
    if (literal) {
        if (schema.contains("const") && schema.contains("enum"))
            return reject("combining const and enum is not supported");
        allowed.insert(schema.contains("const") ? "const" : "enum");
    } else if (type == "object") {
        allowed.insert({"properties", "required", "additionalProperties"});
    } else if (type == "array") {
        allowed.insert({"items", "minItems", "maxItems"});
    } else if (type == "string") {
        allowed.insert({"minLength", "maxLength"});
    }
    for (const auto& [key, value] : schema.items()) {
        if (!allowed.count(key)) return reject("unsupported schema keyword or combination: " + key);
    }
    if (literal) {
        json values = schema.contains("const") ? json::array({schema["const"]}) : schema["enum"];
        if (!values.is_array() || values.empty()) return reject("enum must be a nonempty array");
        for (const auto& value : values) {
            bool matches = type.empty() || (type == "object" && value.is_object()) ||
                (type == "array" && value.is_array()) || (type == "string" && value.is_string()) ||
                (type == "boolean" && value.is_boolean()) || (type == "null" && value.is_null()) ||
                (type == "number" && value.is_number()) ||
                (type == "integer" && value.is_number() &&
                 std::floor(value.get<double>()) == value.get<double>());
            if (!matches) return reject("const/enum value does not match type");
        }
        return true;
    }
    if (type == "object") {
        const auto& properties = field(schema, "properties");
        if (schema.contains("properties") && !properties.is_object())
            return reject("properties must be an object");
        if (properties.is_object()) {
            for (const auto& [name, child] : properties.items())
                if (!supported_schema(child, error, path + ".properties[" + name + "]")) return false;
        }
        if (schema.contains("required")) {
            const auto& required = schema["required"];
            if (!required.is_array()) return reject("required must be an array");
            std::set<std::string> seen;
            for (const auto& name : required) {
                if (!name.is_string() || !seen.insert(name.get<std::string>()).second)
                    return reject("required must contain unique property names");
                if (!properties.is_object() || !properties.contains(name.get<std::string>()))
                    return reject("required properties must be declared in properties");
            }
        }
        if (schema.contains("additionalProperties") &&
            !supported_schema(schema["additionalProperties"], error, path + ".additionalProperties")) return false;
    }
    if (type == "array" && schema.contains("items") &&
        !supported_schema(schema["items"], error, path + ".items")) return false;
    const char* lower = type == "array" ? "minItems" : "minLength";
    const char* upper = type == "array" ? "maxItems" : "maxLength";
    if (type == "array" || type == "string") {
        for (const char* key : {lower, upper}) {
            if (schema.contains(key) && (!schema[key].is_number_integer() ||
                schema[key].get<double>() < 0 || schema[key].get<double>() > std::numeric_limits<int>::max()))
                return reject(std::string(key) + " must be a nonnegative supported integer");
        }
        if (schema.contains(lower) && schema.contains(upper) && schema[lower] > schema[upper])
            return reject("minimum length exceeds maximum length");
    }
    return true;
}

} // namespace

bool chat_constraint_request_from_openai(const json& request, const json& tools,
                                         chat_constraint_request& out, std::string& error) {
    out = chat_constraint_request();
    error.clear();
    if (!request.is_object() || !tools.is_array()) {
        error = "The chat request must be an object and tools must be an array";
        return false;
    }
    out.tools = &tools;
    const auto& routing = field(request, "tool_routing");
    if (!routing.is_null() && routing != "template" && routing != "structured") {
        error = "tool_routing must be template or structured";
        return false;
    }
    out.structured_auto = routing == "structured";
    const json& parallel = field(request, "parallel_tool_calls");
    if (!parallel.is_null() && !parallel.is_boolean()) {
        error = "parallel_tool_calls must be a boolean";
        return false;
    }
    out.parallel_tool_calls = parallel.is_null() || parallel.get<bool>();
    // thinking turned off: some models still get an empty think block fed in, but
    // the grammar must not offer room to reason
    const json& think = field(request, "think");
    if ((think.is_boolean() && !think.get<bool>()) || field(request, "reasoning_effort") == "none") {
        out.allow_reasoning = false;
    }

    const json& choice_field = field(request, "tool_choice");
    const json choice = choice_field.is_null() ? json("auto") : choice_field;
    if (choice.is_string()) {
        const std::string mode = choice.get<std::string>();
        if (mode == "auto") {
            out.tool_choice = chat_tool_choice::AUTO;
        } else if (mode == "none") {
            out.tool_choice = chat_tool_choice::NONE;
        } else if (mode == "required") {
            out.tool_choice = chat_tool_choice::REQUIRED;
        } else {
            error = "tool_choice must be \"auto\", \"none\", \"required\" or a function, got \"" + mode + "\"";
            return false;
        }
    } else if (choice.is_object() && field(choice, "type") == "function" &&
               field(field(choice, "function"), "name").is_string()) {
        out.tool_choice = chat_tool_choice::FUNCTION;
        out.function_name = choice["function"]["name"].get<std::string>();
    } else {
        error = "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}";
        return false;
    }

    if (out.tool_choice == chat_tool_choice::REQUIRED || out.tool_choice == chat_tool_choice::FUNCTION) {
        if (!tools.is_array() || tools.empty()) {
            error = "tool_choice requires a tool call but the request has no tools";
            return false;
        }
    }
    if (out.tool_choice == chat_tool_choice::FUNCTION) {
        bool found = false;
        for (const auto& tool : tools) {
            const json& name = field(field(tool, "function"), "name");
            if (name.is_string() && name.get<std::string>() == out.function_name) {
                found = true;
            }
        }
        if (!found) {
            error = "tool_choice names function \"" + out.function_name + "\" which is not in tools";
            return false;
        }
    }

    if (request.contains("response_format") && !request["response_format"].is_null()) {
        const json& format = request["response_format"];
        if (!format.is_object()) {
            error = "response_format must be an object";
            return false;
        }
        const json& type_field = field(format, "type");
        const std::string type = type_field.is_string() ? type_field.get<std::string>() : "";
        if (type == "json_object") {
            out.json_object = true;
        } else if (type == "json_schema") {
            if (!format.contains("json_schema") || !format["json_schema"].is_object() ||
                !format["json_schema"].contains("schema")) {
                error = "response_format json_schema needs json_schema.schema";
                return false;
            }
            out.json_schema = format["json_schema"]["schema"];
            if (!out.json_schema.is_object() && !out.json_schema.is_boolean()) {
                error = "response_format json_schema.schema must be an object or boolean schema";
                return false;
            }
            if (!supported_schema(out.json_schema, error)) return false;
            const auto& strict = field(format["json_schema"], "strict");
            if (!strict.is_null() && !strict.is_boolean()) {
                error = "response_format json_schema.strict must be a boolean";
                return false;
            }
        } else if (type != "text") {
            error = "response_format type must be text, json_object or json_schema";
            return false;
        }
    }
    if ((out.json_object || !out.json_schema.is_null()) &&
        (out.tool_choice == chat_tool_choice::REQUIRED || out.tool_choice == chat_tool_choice::FUNCTION)) {
        error = "Combining response_format with required or named tool_choice is not supported";
        return false;
    }
    if (out.structured_auto && (out.tool_choice != chat_tool_choice::AUTO || tools.empty() ||
                              out.json_object || !out.json_schema.is_null())) {
        error = "Structured tool routing requires auto, nonempty tools and text response_format";
        return false;
    }
    const bool constrain_tools = out.structured_auto || out.tool_choice == chat_tool_choice::REQUIRED ||
        out.tool_choice == chat_tool_choice::FUNCTION ||
        (out.tool_choice == chat_tool_choice::AUTO && !out.parallel_tool_calls);
    if (constrain_tools) {
        std::set<std::string> names;
        for (const auto& tool : tools) {
            const auto& fn = field(tool, "function");
            const auto& name = field(fn, "name");
            const auto& strict = field(fn, "strict");
            if (!tool.is_object() || field(tool, "type") != "function" || !fn.is_object() ||
                !name.is_string() || name.get<std::string>().empty() ||
                !names.insert(name.get<std::string>()).second ||
                (!strict.is_null() && !strict.is_boolean())) {
                error = "Constrained tools require unique function names and boolean strict";
                return false;
            }
            if (out.structured_auto && (name.get_ref<const std::string&>().size() > 64 ||
                name.get_ref<const std::string&>().find_first_not_of(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)) {
                error = "Tool name must match [A-Za-z0-9_-]{1,64}";
                return false;
            }
            if (fn.contains("parameters")) {
                const auto& parameters = fn["parameters"];
                if (!parameters.is_object() ||
                    (!parameters.empty() && field(parameters, "type") != "object")) {
                    error = "Constrained tool parameters must be an object schema";
                    return false;
                }
                if ((out.structured_auto || strict.is_null() || strict.get<bool>()) &&
                    !supported_schema(parameters, error, "tools[" + name.get<std::string>() + "].parameters")) return false;
            }
        }
    }
    return true;
}

std::shared_ptr<decode_constraint> chat_make_constraint(const chat_format_spec& spec,
                                                        const chat_constraint_request& request,
                                                        const chat_vocab& vocab,
                                                        const std::function<chat_grammar_engine&()>& grammar_engine) {
    if (!chat_structural_tag(spec, request).is_null()) {
        return grammar_engine().make(spec, request);
    }
    if (request.tool_choice == chat_tool_choice::NONE && !spec.tool_start.empty()) {
        // a single token opens every tool call block: taking it away is enough
        std::vector<int> ids = vocab.encode(spec.tool_start);
        if (ids.size() == 1) {
            return std::make_shared<token_ban_constraint>(vocab.size, ids);
        }
    }
    return nullptr;
}
