#include "chat/chat_output.hpp"
#include "chat/auto_tools.hpp"

#include <utility>

namespace {

chat_format_spec effective_spec(chat_format_spec spec, bool strict_tools) {
    if (strict_tools && spec.kind == chat_format_kind::CONTENT_ONLY) {
        // The old HIP path rejected undeclared Hermes calls even for a template
        // with no tool support. Do not turn those calls into ordinary content.
        spec.kind = chat_format_kind::HERMES_JSON;
        spec.tool_start = "<tool_call>";
        spec.tool_end = "</tool_call>";
    }
    return spec;
}

// Validate the existing HIP JSON-block contract without producing a second
// parsed message. The shared parser remains the only source of output fields.
std::string validate_tool_blocks(const chat_format_spec& spec, const std::string& raw, bool final,
                                 const nlohmann::ordered_json& tools, bool allow_tool_calls) {
    if (spec.tool_start.empty() || spec.tool_end.empty()) return "";
    const auto npos = std::string::npos;
    size_t cursor = 0;
    while (cursor < raw.size()) {
        const size_t think = spec.reasoning_start.empty() ? npos : raw.find(spec.reasoning_start, cursor);
        const size_t tool = raw.find(spec.tool_start, cursor);
        if (think != npos && (tool == npos || think < tool)) {
            const size_t end = raw.find(spec.reasoning_end, think + spec.reasoning_start.size());
            if (end == npos) return ""; // tool examples inside reasoning are inert
            cursor = end + spec.reasoning_end.size();
            continue;
        }
        if (tool == npos) return "";
        if (!allow_tool_calls) {
            return "HIP OGA model emitted a tool call even though no tools were enabled for this request.";
        }
        const size_t begin = tool + spec.tool_start.size();
        const size_t end = raw.find(spec.tool_end, begin);
        if (end == npos) {
            return final ? "HIP OGA model emitted an unterminated " + spec.tool_start + " block" : "";
        }
        if (spec.kind != chat_format_kind::HERMES_JSON) {
            // The shared parser is deliberately forgiving. A completed native
            // block that yielded no call must still be an output error on HIP.
            // Reuse that parser for recognition, never a separate result parser.
            const auto parsed = chat_parse(spec, raw.substr(tool, end + spec.tool_end.size() - tool),
                                           false, chat_parse_options{&tools, true, true});
            if (parsed.tool_calls.empty()) {
                return "HIP OGA model emitted an invalid " + spec.tool_start + " block";
            }
            for (const auto& call : parsed.tool_calls) {
                if (call.name.empty()) return "HIP OGA model emitted a tool call with an empty name";
            }
            cursor = end + spec.tool_end.size();
            continue;
        }
        const auto value = nlohmann::ordered_json::parse(raw.substr(begin, end - begin), nullptr, false);
        if (value.is_discarded() || !value.is_object()) {
            return "HIP OGA model emitted an invalid <tool_call> block: expected a JSON object";
        }
        if (!value.contains("name") || !value["name"].is_string() || value["name"].get<std::string>().empty()) {
            return "HIP OGA model emitted an invalid <tool_call> block: missing string name";
        }
        if (!value.contains("arguments") || value["arguments"].is_null()) {
            return "HIP OGA model emitted an invalid <tool_call> block: missing arguments";
        }
        cursor = end + spec.tool_end.size();
    }
    return "";
}

} // namespace

chat_output::chat_output(chat_format_spec spec, std::string prefill,
                         nlohmann::ordered_json tools, bool strict_tools, bool allow_tool_calls,
                         bool json_content, bool structured_auto)
    : spec_(effective_spec(std::move(spec), strict_tools && !json_content)), tools_(std::move(tools)),
      strict_tools_(strict_tools && !json_content), allow_tool_calls_(allow_tool_calls), raw_(prefill),
      stream_(spec_, std::move(prefill), chat_parse_options{&tools_, !json_content, true, json_content}),
      structured_auto_(structured_auto) {
    if (structured_auto_) raw_.clear(); // The internal envelope has no template prefill.
}

chat_output_result chat_output::update(const std::string& text, bool final) {
    if (!error_.empty()) return {{}, {}, error_};
    if (finalized_) return {structured_auto_ ? auto_message_ : stream_.message(), {}, {}};
    // Stream callers buffer complete UTF-8 before update. A length-limited turn
    // can still end halfway through a character; nonstream output has the same
    // boundary. Use JSON's UTF-8 replacement policy once at finalization so both
    // routes remain serializable without changing valid text or argument bytes.
    const std::string chunk = final
        ? nlohmann::ordered_json::parse(nlohmann::ordered_json(text).dump(
              -1, ' ', false, nlohmann::ordered_json::error_handler_t::replace)).get<std::string>()
        : text;
    raw_ += chunk;
    finalized_ = final;
    if (structured_auto_) {
        // A partial routing decision is never a call. Cancellation skips finalization.
        if (!final) return {};
        try {
            auto_message_ = chat_parse_auto_tools(raw_, tools_);
            if (!allow_tool_calls_ && !auto_message_.tool_calls.empty())
                throw std::invalid_argument("Tools are disabled for this request");
            return {auto_message_, chat_compute_deltas({}, auto_message_), {}};
        } catch (const std::exception& error) {
            error_ = error.what();
            auto_message_ = {};
            return {{}, {}, error_};
        }
    }
    if (strict_tools_) {
        error_ = validate_tool_blocks(spec_, raw_, final, tools_, allow_tool_calls_);
        if (!error_.empty()) return {{}, {}, error_};
    }
    auto deltas = stream_.update(chunk, final);
    if (!allow_tool_calls_ && !stream_.message().tool_calls.empty()) {
        error_ = "HIP OGA model emitted a tool call even though no tools were enabled for this request.";
        return {{}, {}, error_};
    }
    return {stream_.message(), std::move(deltas), {}};
}
