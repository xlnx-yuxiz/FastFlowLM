// Real-runtime characterization of externally selected token commitment.
// The two commit methods are reported independently. AppendTokens failing an
// EOS transition is evidence against that method, not a failure to run this
// characterization.
#include "AutoModel/hip_oga_sampling.hpp"
#include <ort_genai.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using nlohmann::json;

namespace {

// Allow roughly one bfloat16 ULP around common logit magnitudes while retaining
// an independent greedy-ID equality check.
constexpr double kLogitAbsTolerance = 0.125;
constexpr double kLogitRelativeTolerance = 0.005;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<int> read_eos_ids(const json& config) {
    const auto& eos = config.at("model").at("eos_token_id");
    std::vector<int> ids = eos.is_array() ? eos.get<std::vector<int>>()
                                           : std::vector<int>{eos.get<int>()};
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    require(!ids.empty(), "model config has no EOS token IDs");
    return ids;
}

bool is_eos(int token, const std::vector<int>& eos_ids) {
    return std::find(eos_ids.begin(), eos_ids.end(), token) != eos_ids.end();
}

std::vector<std::int32_t> sequence(OgaGenerator& generator) {
    const auto count = generator.GetSequenceCount(0);
    const auto* data = generator.GetSequenceData(0);
    require(count == 0 || data != nullptr, "OGA returned null data for a nonempty sequence");
    return count == 0 ? std::vector<std::int32_t>{}
                      : std::vector<std::int32_t>(data, data + count);
}

bool appended_exactly_once(const std::vector<std::int32_t>& before,
                           const std::vector<std::int32_t>& after, std::int32_t token) {
    return after.size() == before.size() + 1 &&
           std::equal(before.begin(), before.end(), after.begin()) &&
           after.back() == token;
}

std::vector<std::int32_t> next_tokens(OgaGenerator& generator) {
    const auto values = generator.GetNextTokens();
    return std::vector<std::int32_t>(values.begin(), values.end());
}

json json_tokens(const std::vector<std::int32_t>& values) {
    return json(values);
}

std::unique_ptr<OgaGeneratorParams> make_params(OgaModel& model, int min_length = 0) {
    auto params = OgaGeneratorParams::Create(model);
    params->SetSearchOptionBool("do_sample", false);
    params->SetSearchOption("temperature", 1.0);
    params->SetSearchOption("top_k", 0.0);
    params->SetSearchOption("top_p", 1.0);
    params->SetSearchOption("repetition_penalty", 1.0);
    params->SetSearchOption("random_seed", 42.0);
    params->SetSearchOption("min_length", static_cast<double>(min_length));
    params->SetSearchOption("max_length", 128.0);
    params->SetSearchOption("batch_size", 1.0);
    if (params->GetSearchNumber("no_repeat_ngram_size") != 0)
        params->SetSearchOption("no_repeat_ngram_size", 0.0);
    return params;
}

struct LogitRow {
    std::vector<float> values;
    std::vector<std::int64_t> shape;
};

LogitRow copy_logits(OgaTensor& tensor, int vocab) {
    LogitRow result;
    result.shape = tensor.Shape();
    require(!result.shape.empty() && result.shape.back() == vocab,
            "OGA logits do not have the configured vocabulary width");
    require(std::all_of(result.shape.begin(), result.shape.end() - 1,
                        [](std::int64_t dim) { return dim == 1; }),
            "OGA logits are not one row");
    require(tensor.Data() != nullptr, "OGA logits data is null");
    result.values.resize(static_cast<std::size_t>(vocab));
    if (tensor.Type() == OgaElementType_float32) {
        const auto* data = static_cast<const float*>(tensor.Data());
        std::copy(data, data + vocab, result.values.begin());
    } else if (tensor.Type() == OgaElementType_bfloat16) {
        const auto* data = static_cast<const unsigned char*>(tensor.Data());
        for (int i = 0; i < vocab; ++i) {
            std::uint16_t high = 0;
            std::memcpy(&high, data + static_cast<std::size_t>(i) * sizeof(high), sizeof(high));
            const std::uint32_t bits = static_cast<std::uint32_t>(high) << 16;
            std::memcpy(&result.values[static_cast<std::size_t>(i)], &bits, sizeof(bits));
        }
    } else {
        throw std::runtime_error("probe supports float32 or bfloat16 OGA logits");
    }
    return result;
}

LogitRow current_logits(OgaGenerator& generator, int vocab) {
    auto tensor = generator.GetLogits();
    return copy_logits(*tensor, vocab);
}

struct CurrentRow {
    std::unique_ptr<OgaTensor> tensor;
    LogitRow copied;
};

CurrentRow read_current_row(OgaGenerator& generator, int vocab) {
    CurrentRow row;
    row.tensor = generator.GetLogits();
    row.copied = copy_logits(*row.tensor, vocab);
    return row;
}

std::unique_ptr<OgaTensor> make_logits_tensor(std::vector<float>& values) {
    const std::int64_t shape[] = {1, static_cast<std::int64_t>(values.size())};
    return OgaTensor::Create(values.data(), shape, 2, OgaElementType_float32);
}

int greedy_id(const std::vector<float>& values) {
    require(!values.empty(), "cannot choose from an empty logit row");
    int best_id = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i]) && (best_id < 0 || values[i] > best)) {
            best = values[i];
            best_id = static_cast<int>(i);
        }
    }
    require(best_id >= 0, "logit row has no finite candidate");
    return best_id;
}

json compare_logits(const LogitRow& actual, const LogitRow& reference) {
    const bool shapes_equal = actual.shape == reference.shape;
    json result = {
        {"actual_shape", actual.shape},
        {"reference_shape", reference.shape},
        {"shapes_equal", shapes_equal},
        {"finite_support_equal", false},
        {"max_absolute_error", nullptr},
        {"absolute_tolerance", kLogitAbsTolerance},
        {"relative_tolerance", kLogitRelativeTolerance},
        {"within_tolerance", false},
        {"actual_greedy_id", nullptr},
        {"reference_greedy_id", nullptr},
        {"greedy_ids_equal", false}
    };
    if (actual.values.size() != reference.values.size()) return result;
    bool finite_support_equal = true;
    double maximum_error = 0;
    bool within_tolerance = true;
    for (std::size_t i = 0; i < actual.values.size(); ++i) {
        const float left = actual.values[i];
        const float right = reference.values[i];
        if (std::isfinite(left) != std::isfinite(right)) {
            finite_support_equal = false;
            within_tolerance = false;
            continue;
        }
        if (!std::isfinite(left)) continue;
        const double error = std::abs(static_cast<double>(left) - right);
        maximum_error = std::max(maximum_error, error);
        const double tolerance = kLogitAbsTolerance +
            kLogitRelativeTolerance * std::abs(static_cast<double>(right));
        if (error > tolerance) within_tolerance = false;
    }
    result["finite_support_equal"] = finite_support_equal;
    result["max_absolute_error"] = maximum_error;
    result["within_tolerance"] = shapes_equal && finite_support_equal && within_tolerance;
    try {
        const int actual_id = greedy_id(actual.values);
        const int reference_id = greedy_id(reference.values);
        result["actual_greedy_id"] = actual_id;
        result["reference_greedy_id"] = reference_id;
        result["greedy_ids_equal"] = actual_id == reference_id;
    } catch (...) {
    }
    return result;
}

bool comparison_passes(const json& comparison) {
    return comparison.value("shapes_equal", false) &&
           comparison.value("within_tolerance", false) &&
           comparison.value("greedy_ids_equal", false);
}

json summarize_logits(const LogitRow& row) {
    std::vector<std::pair<float, int>> finite_values;
    finite_values.reserve(row.values.size());
    std::size_t finite_count = 0;
    std::uint64_t hash = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    for (std::size_t i = 0; i < row.values.size(); ++i) {
        const float value = row.values[i];
        if (std::isfinite(value)) {
            ++finite_count;
            finite_values.emplace_back(value, static_cast<int>(i));
        }
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= static_cast<unsigned char>((bits >> (byte * 8)) & 0xff);
            hash *= kFnvPrime;
        }
    }
    std::sort(finite_values.begin(), finite_values.end(),
              [](const auto& left, const auto& right) {
                  if (left.first != right.first) return left.first > right.first;
                  return left.second < right.second;
              });
    json top_logits = json::array();
    const auto count = std::min<std::size_t>(finite_values.size(), 5);
    for (std::size_t i = 0; i < count; ++i)
        top_logits.push_back({{"token_id", finite_values[i].second},
                              {"logit", finite_values[i].first}});
    std::ostringstream digest;
    digest << std::hex << std::setw(16) << std::setfill('0') << hash;
    const int best = greedy_id(row.values);
    return {{"shape", row.shape}, {"finite_count", finite_count},
            {"greedy_id", best}, {"greedy_logit", row.values[static_cast<std::size_t>(best)]},
            {"top_logits", top_logits}, {"fnv1a64_float32_bits", digest.str()}};
}

LogitRow fresh_logits(OgaModel& model, const std::vector<std::int32_t>& tokens, int vocab) {
    auto params = make_params(model);
    auto generator = OgaGenerator::Create(model, *params);
    generator->AppendTokens(tokens.data(), tokens.size());
    return current_logits(*generator, vocab);
}

void append_prompt(OgaGenerator& generator, const std::vector<std::int32_t>& prompt);

std::vector<std::int32_t> tokenize_prompt(OgaModel& model, const std::string& text) {
    auto tokenizer = OgaTokenizer::Create(model);
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(text.c_str(), *sequences);
    require(sequences->Count() == 1, "tokenizer did not return exactly one prompt sequence");
    const auto count = sequences->SequenceCount(0);
    const auto* data = sequences->SequenceData(0);
    require(count != 0 && data != nullptr, "tokenizer returned an empty prompt");
    return std::vector<std::int32_t>(data, data + count);
}

// The one-shot full prefill is retained as a diagnostic only. The T06 runtime
// worklog records stable differences between one-shot and segmented prefill;
// same-segmented fresh and rewind/suffix rows matched exactly. This control
// therefore mirrors the subject's AppendTokens(prompt), then one-token suffix
// segmentation and must run after the subject generator has been destroyed.
std::vector<LogitRow> fresh_segmented_rows(
        OgaModel& model, const std::vector<std::int32_t>& prompt,
        const std::vector<int>& suffix, int vocab) {
    auto params = make_params(model);
    auto generator = OgaGenerator::Create(model, *params);
    append_prompt(*generator, prompt);
    std::vector<LogitRow> rows;
    rows.push_back(current_logits(*generator, vocab));
    for (int token : suffix) {
        const std::int32_t token_id = token;
        generator->AppendTokens(&token_id, 1);
        rows.push_back(current_logits(*generator, vocab));
    }
    return rows;
}

void append_prompt(OgaGenerator& generator, const std::vector<std::int32_t>& prompt) {
    require(!prompt.empty(), "probe prompt must not be empty");
    generator.AppendTokens(prompt.data(), prompt.size());
}

struct SupportResult {
    bool supported = true;
    json reasons = json::array();
};

void fail_support(SupportResult& result, const std::string& reason) {
    result.supported = false;
    result.reasons.push_back(reason);
}

struct NativeGreedyTrace {
    std::vector<int> selected_ids;
    std::vector<int> ordinary_ids;
    std::vector<LogitRow> state_rows; // Prompt row, then one row after each ordinary token.
    json steps = json::array();
    json reasons = json::array();
    int get_logits_api_calls = 0;
    int append_tokens_api_calls = 0;
    int generate_next_token_api_calls = 0;
    int set_logits_api_calls = 0;
    int rewind_to_api_calls = 0;
    bool transitions_valid = true;
};

void fail_trace(NativeGreedyTrace& trace, const std::string& reason) {
    trace.transitions_valid = false;
    trace.reasons.push_back(reason);
}

NativeGreedyTrace run_native_greedy(OgaModel& model, int vocab,
                                    const std::vector<std::int32_t>& prompt,
                                    const std::vector<int>& eos_ids,
                                    int maximum_ordinary_tokens) {
    NativeGreedyTrace trace;
    {
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        append_prompt(*generator, prompt);
        trace.append_tokens_api_calls = 1;
        CurrentRow current = read_current_row(*generator, vocab);
        trace.get_logits_api_calls = 1;
        trace.state_rows.push_back(current.copied);

        while (static_cast<int>(trace.ordinary_ids.size()) < maximum_ordinary_tokens) {
            const auto before = sequence(*generator);
            const int pre_logits_greedy_id = greedy_id(current.copied.values);
            generator->GenerateNextToken();
            ++trace.generate_next_token_api_calls;
            const auto returned = next_tokens(*generator);
            const auto after = sequence(*generator);
            const bool done = generator->IsDone();
            json step = {{"before_sequence", json_tokens(before)},
                         {"get_next_tokens", json_tokens(returned)},
                         {"sequence_after", json_tokens(after)},
                         {"pre_logits_greedy_id", pre_logits_greedy_id},
                         {"is_done", done}};
            if (returned.size() != 1) {
                step["transition_matches"] = false;
                trace.steps.push_back(step);
                fail_trace(trace, "native greedy GenerateNextToken did not return exactly one selected ID");
                break;
            }

            const int token = returned.front();
            const bool eos = is_eos(token, eos_ids);
            const bool selected_matches_greedy = token == pre_logits_greedy_id;
            const bool token_appended_once = appended_exactly_once(before, after, token);
            const bool eos_unchanged = after == before;
            trace.selected_ids.push_back(token);
            step["selected_id"] = token;
            step["is_eos"] = eos;
            step["selected_id_matches_pre_logits_greedy"] = selected_matches_greedy;
            step["ordinary_token_appended_once"] = eos ? json(nullptr) : json(token_appended_once);
            step["eos_left_sequence_unchanged"] = eos ? json(eos_unchanged) : json(nullptr);
            step["transition_matches"] = eos ? (done && eos_unchanged)
                                             : (!done && token_appended_once);
            if (eos) {
                if (!done || !eos_unchanged)
                    fail_trace(trace, "native greedy EOS transition did not mark done without appending EOS");
                if (!selected_matches_greedy)
                    fail_trace(trace, "native greedy EOS ID differs from the preceding logits argmax");
                trace.steps.push_back(step);
                break;
            }

            if (done || !token_appended_once)
                fail_trace(trace, "native greedy ordinary-token transition did not append exactly once and remain active");
            if (!selected_matches_greedy)
                fail_trace(trace, "native greedy selected ID differs from the preceding logits argmax");
            trace.ordinary_ids.push_back(token);
            current = read_current_row(*generator, vocab);
            ++trace.get_logits_api_calls;
            trace.state_rows.push_back(current.copied);
            step["post_logits_greedy_id"] = greedy_id(current.copied.values);
            trace.steps.push_back(step);
        }
    } // Destroy this generator before any repeat or replay generator is created.
    return trace;
}

json summarize_native_trace(const NativeGreedyTrace& trace) {
    json logit_rows = json::array();
    for (std::size_t i = 0; i < trace.state_rows.size(); ++i)
        logit_rows.push_back({{"state_index", i}, {"logits", summarize_logits(trace.state_rows[i])}});
    return {{"selected_ids", trace.selected_ids},
            {"ordinary_ids", trace.ordinary_ids},
            {"state_row_count", trace.state_rows.size()},
            {"logit_rows", logit_rows},
            {"steps", trace.steps},
            {"transition_reasons", trace.reasons},
            {"transitions_valid", trace.transitions_valid},
            {"oga_get_logits_api_calls", trace.get_logits_api_calls},
            {"oga_append_tokens_api_calls", trace.append_tokens_api_calls},
            {"oga_generate_next_token_api_calls", trace.generate_next_token_api_calls},
            {"oga_set_logits_api_calls", trace.set_logits_api_calls},
            {"oga_rewind_to_api_calls", trace.rewind_to_api_calls}};
}

json compare_native_repeats(const NativeGreedyTrace& first,
                            const NativeGreedyTrace& repeat,
                            int required_ordinary_tokens) {
    json report = {{"selected_ids_equal", first.selected_ids == repeat.selected_ids},
                   {"ordinary_ids_equal", first.ordinary_ids == repeat.ordinary_ids},
                   {"first_minimum_ordinary_tokens_met",
                    static_cast<int>(first.ordinary_ids.size()) >= required_ordinary_tokens},
                   {"repeat_minimum_ordinary_tokens_met",
                    static_cast<int>(repeat.ordinary_ids.size()) >= required_ordinary_tokens},
                   {"transition_traces_valid",
                    first.transitions_valid && repeat.transitions_valid},
                   {"state_rows", json::array()},
                   {"all_state_rows_match", false},
                   {"repeatable", false}};

    std::size_t comparable_rows = std::min(first.state_rows.size(), repeat.state_rows.size());
    for (std::size_t step = 0;
         step < std::min(first.selected_ids.size(), repeat.selected_ids.size()); ++step) {
        if (first.selected_ids[step] != repeat.selected_ids[step]) {
            comparable_rows = std::min(comparable_rows, step + 1);
            break;
        }
    }
    bool rows_match = first.state_rows.size() == repeat.state_rows.size();
    for (std::size_t i = 0; i < comparable_rows; ++i) {
        const auto comparison = compare_logits(first.state_rows[i], repeat.state_rows[i]);
        report["state_rows"].push_back({{"state_index", i}, {"comparison", comparison}});
        rows_match = rows_match && comparison_passes(comparison);
    }
    const bool ordinary_count_met =
        static_cast<int>(first.ordinary_ids.size()) >= required_ordinary_tokens &&
        static_cast<int>(repeat.ordinary_ids.size()) >= required_ordinary_tokens;
    const bool tokens_match = first.selected_ids == repeat.selected_ids &&
                              first.ordinary_ids == repeat.ordinary_ids;
    const bool traces_valid = first.transitions_valid && repeat.transitions_valid;
    report["all_state_rows_match"] = rows_match;
    report["repeatable"] = rows_match && tokens_match && ordinary_count_met && traces_valid;
    return report;
}

json replay_native_ids_with_one_finite(OgaModel& model, int vocab,
                                       const std::vector<std::int32_t>& prompt,
                                       const std::vector<int>& eos_ids,
                                       const NativeGreedyTrace& baseline) {
    json report = {{"status", "OBSERVED"}, {"selected_ids", baseline.selected_ids},
                   {"steps", json::array()}, {"all_rows_match_native", true},
                   {"all_selected_ids_returned", true}, {"all_transitions_match", true}};
    int get_logits_calls = 0;
    int append_tokens_calls = 0;
    int generate_calls = 0;
    int set_logits_calls = 0;
    {
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        append_prompt(*generator, prompt);
        ++append_tokens_calls;
        CurrentRow current = read_current_row(*generator, vocab);
        ++get_logits_calls;

        std::vector<std::int32_t> expected_sequence = prompt;
        for (std::size_t i = 0; i < baseline.selected_ids.size(); ++i) {
            const int token = baseline.selected_ids[i];
            json step = {{"selected_id", token}, {"before_sequence", json_tokens(sequence(*generator))}};
            if (i >= baseline.state_rows.size()) {
                step["pre_logits_vs_native"] = nullptr;
                report["all_rows_match_native"] = false;
            } else {
                const auto pre_comparison = compare_logits(current.copied, baseline.state_rows[i]);
                step["pre_logits_vs_native"] = pre_comparison;
                if (!comparison_passes(pre_comparison)) report["all_rows_match_native"] = false;
            }

            std::vector<float> one_finite(static_cast<std::size_t>(vocab),
                                          -std::numeric_limits<float>::infinity());
            one_finite.at(static_cast<std::size_t>(token)) = 0.0f;
            auto replacement = make_logits_tensor(one_finite);
            generator->SetLogits(*replacement);
            ++set_logits_calls;
            generator->GenerateNextToken();
            ++generate_calls;
            const auto returned = next_tokens(*generator);
            const auto after = sequence(*generator);
            const bool eos = is_eos(token, eos_ids);
            const bool done = generator->IsDone();
            const bool returned_selected = returned.size() == 1 && returned.front() == token;
            const bool token_appended_once = appended_exactly_once(expected_sequence, after, token);
            const bool transition_matches = eos ? (done && after == expected_sequence)
                                                : (!done && token_appended_once);
            step["get_next_tokens"] = json_tokens(returned);
            step["returned_selected_id"] = returned_selected;
            step["is_eos"] = eos;
            step["is_done"] = done;
            step["sequence_after"] = json_tokens(after);
            step["ordinary_token_appended_once"] = eos ? json(nullptr) : json(token_appended_once);
            step["transition_matches"] = transition_matches;
            if (!returned_selected) report["all_selected_ids_returned"] = false;
            if (!transition_matches) report["all_transitions_match"] = false;

            if (eos) {
                report["steps"].push_back(step);
                break;
            }

            expected_sequence.push_back(token);
            current = read_current_row(*generator, vocab);
            ++get_logits_calls;
            if (i + 1 >= baseline.state_rows.size()) {
                step["post_logits_vs_native"] = nullptr;
                report["all_rows_match_native"] = false;
            } else {
                const auto post_comparison = compare_logits(current.copied, baseline.state_rows[i + 1]);
                step["post_logits_vs_native"] = post_comparison;
                if (!comparison_passes(post_comparison)) report["all_rows_match_native"] = false;
            }
            report["steps"].push_back(step);
        }
    } // The replay generator is destroyed before the next control is created.

    const bool passed = baseline.transitions_valid && !baseline.selected_ids.empty() &&
                        report["all_rows_match_native"].get<bool>() &&
                        report["all_selected_ids_returned"].get<bool>() &&
                        report["all_transitions_match"].get<bool>();
    report["status"] = passed ? "PASS" : "FAIL";
    report["oga_get_logits_api_calls"] = get_logits_calls;
    report["oga_append_tokens_api_calls"] = append_tokens_calls;
    report["oga_generate_next_token_api_calls"] = generate_calls;
    report["oga_set_logits_api_calls"] = set_logits_calls;
    report["oga_rewind_to_api_calls"] = 0;
    if (!passed) report["error"] = "one-finite replay differs from serial native greedy generation";
    return report;
}

json run_native_equivalence_controls(OgaModel& model, int vocab,
                                    const std::string& prompt_text,
                                    const std::vector<std::int32_t>& prompt,
                                    const std::vector<int>& eos_ids,
                                    SupportResult& support) {
    constexpr int kRequiredOrdinaryTokens = 4;
    constexpr int kMaximumOrdinaryTokens = 8;
    json report = {{"status", "OBSERVED"}, {"prompt_text", prompt_text},
                   {"prompt_ids", json_tokens(prompt)},
                   {"required_ordinary_token_count", kRequiredOrdinaryTokens},
                   {"maximum_ordinary_token_count", kMaximumOrdinaryTokens},
                   {"generator_lifecycle",
                    "native baseline, serial repeat, and one-finite replay use non-overlapping generator lifetimes; only copied IDs and CPU logit rows persist"},
                   {"logit_row_evidence",
                    "full CPU rows are compared in memory; JSON records row shape, finite support, max error, greedy IDs, top five values, and a float32-bit digest"},
                   {"remaining_uncertainty",
                    "the native control qualifies only this tokenized prompt and 4-8 token trace; it does not establish equivalence for arbitrary prompts or other OGA model packages"},
                   {"repeatability", json::object()}, {"one_finite_replay", json::object()}};

    const auto baseline = run_native_greedy(model, vocab, prompt, eos_ids,
                                             kMaximumOrdinaryTokens);
    const auto repeat = run_native_greedy(model, vocab, prompt, eos_ids,
                                           kMaximumOrdinaryTokens);
    report["native_baseline"] = summarize_native_trace(baseline);
    report["fresh_repeat"] = summarize_native_trace(repeat);

    const auto repeatability = compare_native_repeats(baseline, repeat,
                                                       kRequiredOrdinaryTokens);
    report["repeatability"] = repeatability;
    if (!baseline.transitions_valid || !repeat.transitions_valid)
        fail_support(support, "native greedy baseline or its serial repeat had an invalid OGA transition");
    if (static_cast<int>(baseline.ordinary_ids.size()) < kRequiredOrdinaryTokens ||
        static_cast<int>(repeat.ordinary_ids.size()) < kRequiredOrdinaryTokens)
        fail_support(support, "native greedy control produced fewer than four ordinary tokens before EOS");
    if (!repeatability.at("repeatable").get<bool>())
        fail_support(support, "fresh native greedy repeat did not reproduce IDs and logits within the documented tolerance");

    const auto replay = replay_native_ids_with_one_finite(model, vocab, prompt, eos_ids, baseline);
    report["one_finite_replay"] = replay;
    if (replay.at("status") != "PASS")
        fail_support(support, "one-finite replay did not match the independent native greedy trace");

    report["oga_get_logits_api_calls"] =
        baseline.get_logits_api_calls + repeat.get_logits_api_calls +
        replay.value("oga_get_logits_api_calls", 0);
    report["oga_append_tokens_api_calls"] =
        baseline.append_tokens_api_calls + repeat.append_tokens_api_calls +
        replay.value("oga_append_tokens_api_calls", 0);
    report["oga_generate_next_token_api_calls"] =
        baseline.generate_next_token_api_calls + repeat.generate_next_token_api_calls +
        replay.value("oga_generate_next_token_api_calls", 0);
    report["oga_set_logits_api_calls"] =
        replay.value("oga_set_logits_api_calls", 0);
    report["oga_rewind_to_api_calls"] = 0;
    report["support_reasons"] = support.reasons;
    report["status"] = support.supported ? "PASS" : "FAIL";
    return report;
}

json run_one_finite_steps(OgaModel& model, int vocab,
                          const std::vector<std::int32_t>& prompt,
                          const std::vector<int>& selected,
                          const std::vector<int>& eos_ids,
                          SupportResult& support) {
    json report = {{"status", "OBSERVED"}, {"prompt", json_tokens(prompt)},
                   {"selected_ids", selected}, {"steps", json::array()}};
    std::vector<int> committed_ids;
    std::vector<LogitRow> actual_post_rows;
    int subject_get_logits_calls = 0;
    int generate_calls = 0;
    int set_logits_calls = 0;
    {
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        append_prompt(*generator, prompt);
        CurrentRow current = read_current_row(*generator, vocab);
        subject_get_logits_calls = 1;

        for (int token : selected) {
            json step = {{"selected_id", token},
                         {"before_sequence", json_tokens(sequence(*generator))},
                         {"current_logits_shape", current.copied.shape}};
            const auto before = sequence(*generator);
            std::vector<float> one_finite(static_cast<std::size_t>(vocab),
                                          -std::numeric_limits<float>::infinity());
            one_finite.at(static_cast<std::size_t>(token)) = 0.0f;
            auto replacement = make_logits_tensor(one_finite);
            generator->SetLogits(*replacement);
            ++set_logits_calls;
            generator->GenerateNextToken();
            ++generate_calls;

            const auto after = sequence(*generator);
            const auto returned = next_tokens(*generator);
            const bool eos = is_eos(token, eos_ids);
            const bool returned_selected = returned.size() == 1 && returned.front() == token;
            const bool done = generator->IsDone();
            const bool ordinary_append = appended_exactly_once(before, after, token);
            step["get_next_tokens"] = json_tokens(returned);
            step["returned_selected_id"] = returned_selected;
            step["sequence_after"] = json_tokens(after);
            step["sequence_count_delta"] = static_cast<std::int64_t>(after.size()) -
                                            static_cast<std::int64_t>(before.size());
            step["ordinary_token_appended_once"] = eos ? json(nullptr) : json(ordinary_append);
            step["is_eos"] = eos;
            step["is_done"] = done;
            step["eos_transition_matches"] = eos ? json(done) : json(!done);
            if (!returned_selected)
                fail_support(support, "GenerateNextToken GetNextTokens did not return selected ID " +
                                      std::to_string(token));
            if (eos) {
                if (!done) fail_support(support, "GenerateNextToken did not mark EOS " +
                                                 std::to_string(token) + " done");
            } else {
                if (!ordinary_append)
                    fail_support(support, "GenerateNextToken did not append non-EOS ID " +
                                          std::to_string(token) + " exactly once");
                if (done) fail_support(support, "GenerateNextToken marked non-EOS ID " +
                                                std::to_string(token) + " done");
                committed_ids.push_back(token);
                current = read_current_row(*generator, vocab);
                ++subject_get_logits_calls;
                actual_post_rows.push_back(current.copied);
            }
            report["steps"].push_back(step);
            if (eos) break;
        }
    } // Destroy the subject before creating any fresh reference generator.

    const auto segmented_rows = fresh_segmented_rows(model, prompt, committed_ids, vocab);
    std::vector<std::int32_t> expected = prompt;
    for (std::size_t i = 0; i < committed_ids.size(); ++i) {
        expected.push_back(committed_ids[i]);
        const auto one_shot_reference = fresh_logits(model, expected, vocab);
        const auto segmented_comparison = compare_logits(actual_post_rows[i], segmented_rows[i + 1]);
        const auto one_shot_comparison = compare_logits(actual_post_rows[i], one_shot_reference);
        auto& step = report["steps"][i];
        step["next_logits_vs_same_segmented_fresh"] = segmented_comparison;
        step["next_logits_vs_one_shot_prefill_diagnostic"] = one_shot_comparison;
        if (!comparison_passes(segmented_comparison))
            fail_support(support, "GenerateNextToken next logits differ from same-segmented serial fresh state");
    }
    report["subject_get_logits_calls"] = subject_get_logits_calls;
    report["expected_subject_get_logits_calls"] =
        1 + static_cast<int>(committed_ids.size());
    report["same_segmented_fresh_reference_get_logits_calls"] =
        static_cast<int>(segmented_rows.size());
    report["fresh_reference_get_logits_calls"] = static_cast<int>(std::count_if(
        committed_ids.begin(), committed_ids.end(), [&](int token) { return !is_eos(token, eos_ids); }));
    report["oga_get_logits_api_calls"] = subject_get_logits_calls +
        report["same_segmented_fresh_reference_get_logits_calls"].get<int>() +
        report["fresh_reference_get_logits_calls"].get<int>();
    report["oga_append_tokens_api_calls"] = 2 + 2 * static_cast<int>(committed_ids.size());
    report["oga_generate_next_token_api_calls"] = generate_calls;
    report["oga_set_logits_api_calls"] = set_logits_calls;
    report["oga_rewind_to_api_calls"] = 0;
    report["reference_policy"] =
        "same-segmented serial fresh rows gate transition correctness; one-shot full prefill remains diagnostic only";
    report["status"] = support.supported ? "PASS" : "FAIL";
    return report;
}

json run_append_steps(OgaModel& model, int vocab,
                      const std::vector<std::int32_t>& prompt,
                      const std::vector<int>& selected,
                      const std::vector<int>& eos_ids,
                      SupportResult& support) {
    json report = {{"status", "OBSERVED"}, {"prompt", json_tokens(prompt)},
                   {"selected_ids", selected}, {"steps", json::array()}};
    std::vector<int> committed_ids;
    std::vector<LogitRow> actual_post_rows;
    int subject_get_logits_calls = 0;
    int append_calls = 0;
    {
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        append_prompt(*generator, prompt);
        CurrentRow current = read_current_row(*generator, vocab);
        subject_get_logits_calls = 1;
        append_calls = 1;

        for (int token : selected) {
            json step = {{"selected_id", token}, {"before_sequence", json_tokens(sequence(*generator))},
                         {"current_logits_shape", current.copied.shape}};
            const auto before = sequence(*generator);
            const bool before_done = generator->IsDone();
            const std::int32_t token_id = token;
            generator->AppendTokens(&token_id, 1);
            ++append_calls;
            const auto after = sequence(*generator);
            const auto returned = next_tokens(*generator);
            const bool eos = is_eos(token, eos_ids);
            const bool append_once = appended_exactly_once(before, after, token);
            const bool done = generator->IsDone();
            step["get_next_tokens"] = json_tokens(returned);
            step["is_eos"] = eos;
            step["is_done_before"] = before_done;
            step["is_done_after"] = done;
            step["sequence_after"] = json_tokens(after);
            step["sequence_count_delta"] = static_cast<std::int64_t>(after.size()) -
                                            static_cast<std::int64_t>(before.size());
            step["append_once"] = append_once;
            step["selected_id_observed_at_sequence_tail"] = !after.empty() && after.back() == token;
            step["eos_transition_matches"] = eos ? json(done) : json(!done);
            if (!append_once)
                fail_support(support, "AppendTokens did not append selected ID " +
                                      std::to_string(token) + " exactly once");
            if (eos) {
                // EOS identity is established from the requested token and observed
                // sequence/GetNextTokens/IsDone state; AppendTokens is not presumed
                // to behave like GenerateNextToken.
                if (!done)
                    fail_support(support, "AppendTokens did not produce an OGA EOS done transition for " +
                                          std::to_string(token));
                if (returned.size() != 1 || returned.front() != token)
                    fail_support(support, "AppendTokens GetNextTokens did not report EOS " +
                                          std::to_string(token));
                report["steps"].push_back(step);
                break;
            }
            if (done) fail_support(support, "AppendTokens marked non-EOS ID " +
                                            std::to_string(token) + " done");
            committed_ids.push_back(token);
            current = read_current_row(*generator, vocab);
            ++subject_get_logits_calls;
            actual_post_rows.push_back(current.copied);
            report["steps"].push_back(step);
        }
    } // Destroy the subject before creating any fresh reference generator.

    const auto segmented_rows = fresh_segmented_rows(model, prompt, committed_ids, vocab);
    std::vector<std::int32_t> expected = prompt;
    for (std::size_t i = 0; i < committed_ids.size(); ++i) {
        expected.push_back(committed_ids[i]);
        const auto one_shot_reference = fresh_logits(model, expected, vocab);
        const auto segmented_comparison = compare_logits(actual_post_rows[i], segmented_rows[i + 1]);
        const auto one_shot_comparison = compare_logits(actual_post_rows[i], one_shot_reference);
        auto& step = report["steps"][i];
        step["next_logits_vs_same_segmented_fresh"] = segmented_comparison;
        step["next_logits_vs_one_shot_prefill_diagnostic"] = one_shot_comparison;
        if (!comparison_passes(segmented_comparison))
            fail_support(support, "AppendTokens next logits differ from same-segmented serial fresh state");
    }
    report["subject_get_logits_calls"] = subject_get_logits_calls;
    report["expected_subject_get_logits_calls"] =
        1 + static_cast<int>(committed_ids.size());
    report["same_segmented_fresh_reference_get_logits_calls"] =
        static_cast<int>(segmented_rows.size());
    report["fresh_reference_get_logits_calls"] = static_cast<int>(committed_ids.size());
    report["oga_get_logits_api_calls"] = subject_get_logits_calls +
        report["same_segmented_fresh_reference_get_logits_calls"].get<int>() +
        report["fresh_reference_get_logits_calls"].get<int>();
    report["oga_append_tokens_api_calls"] = append_calls +
        static_cast<int>(segmented_rows.size()) + static_cast<int>(committed_ids.size());
    report["oga_generate_next_token_api_calls"] = 0;
    report["oga_set_logits_api_calls"] = 0;
    report["oga_rewind_to_api_calls"] = 0;
    report["reference_policy"] =
        "same-segmented serial fresh rows gate transition correctness; one-shot full prefill remains diagnostic only";
    report["status"] = support.supported ? "PASS" : "UNSUPPORTED";
    report["support_reasons"] = support.reasons;
    return report;
}

json run_one_finite_eos(OgaModel& model, int vocab, int eos,
                        const std::vector<int>& eos_ids,
                        SupportResult& support) {
    json report = {{"method", "one_finite_greedy_generate_next_token"},
                   {"eos_id", eos}, {"status", "OBSERVED"}};
    try {
        require(is_eos(eos, eos_ids), "selected EOS ID is absent from model metadata");
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        const std::vector<std::int32_t> prompt{10};
        append_prompt(*generator, prompt);
        auto current = generator->GetLogits(); // Keep the prepared row alive through SetLogits.
        (void)copy_logits(*current, vocab);
        report["subject_get_logits_calls"] = 1;
        report["oga_get_logits_api_calls"] = 1;
        report["oga_append_tokens_api_calls"] = 1;
        report["oga_generate_next_token_api_calls"] = 1;
        report["oga_set_logits_api_calls"] = 1;
        report["oga_rewind_to_api_calls"] = 0;
        const auto before = sequence(*generator);
        std::vector<float> one_finite(static_cast<std::size_t>(vocab),
                                      -std::numeric_limits<float>::infinity());
        one_finite.at(static_cast<std::size_t>(eos)) = 0.0f;
        auto replacement = make_logits_tensor(one_finite);
        generator->SetLogits(*replacement);
        generator->GenerateNextToken();
        const auto after = sequence(*generator);
        const auto returned = next_tokens(*generator);
        const bool done = generator->IsDone();
        const bool returned_selected = returned.size() == 1 && returned.front() == eos;
        report["get_next_tokens"] = json_tokens(returned);
        report["returned_selected_id"] = returned_selected;
        report["is_done"] = done;
        report["before_sequence"] = json_tokens(before);
        report["after_sequence"] = json_tokens(after);
        report["sequence_count_delta"] = static_cast<std::int64_t>(after.size()) -
                                          static_cast<std::int64_t>(before.size());
        report["eos_present_at_sequence_tail"] = !after.empty() && after.back() == eos;
        const bool sequence_unchanged = after == before;
        report["eos_left_sequence_unchanged"] = sequence_unchanged;
        report["eos_transition_matches"] = done && sequence_unchanged;
        if (!returned_selected) fail_support(support, "GenerateNextToken did not report EOS ID " +
                                                     std::to_string(eos) + " via GetNextTokens");
        if (!done) fail_support(support, "GenerateNextToken did not mark EOS ID " +
                                         std::to_string(eos) + " done");
        if (!sequence_unchanged)
            fail_support(support, "GenerateNextToken appended EOS ID " +
                                 std::to_string(eos) + " to the sequence");
        report["status"] = returned_selected && done && sequence_unchanged ? "PASS" : "FAIL";
    } catch (const std::exception& error) {
        report["status"] = "FAIL";
        report["error"] = error.what();
        fail_support(support, "GenerateNextToken EOS probe threw for ID " +
                              std::to_string(eos) + ": " + error.what());
    }
    return report;
}

json run_append_eos(OgaModel& model, int vocab, int eos,
                    const std::vector<int>& eos_ids,
                    SupportResult& support) {
    json report = {{"method", "append_tokens"}, {"eos_id", eos}, {"status", "OBSERVED"}};
    try {
        require(is_eos(eos, eos_ids), "selected EOS ID is absent from model metadata");
        auto params = make_params(model);
        auto generator = OgaGenerator::Create(model, *params);
        const std::vector<std::int32_t> prompt{10};
        append_prompt(*generator, prompt);
        auto current = generator->GetLogits();
        (void)copy_logits(*current, vocab);
        report["subject_get_logits_calls"] = 1;
        report["oga_get_logits_api_calls"] = 1;
        report["oga_append_tokens_api_calls"] = 2;
        report["oga_generate_next_token_api_calls"] = 0;
        report["oga_set_logits_api_calls"] = 0;
        report["oga_rewind_to_api_calls"] = 0;
        const auto before = sequence(*generator);
        const std::int32_t eos_id = eos;
        generator->AppendTokens(&eos_id, 1);
        const auto after = sequence(*generator);
        const auto returned = next_tokens(*generator);
        const bool done = generator->IsDone();
        const bool appended = appended_exactly_once(before, after, eos);
        const bool returned_selected = returned.size() == 1 && returned.front() == eos;
        report["get_next_tokens"] = json_tokens(returned);
        report["returned_selected_id"] = returned_selected;
        report["is_done"] = done;
        report["before_sequence"] = json_tokens(before);
        report["after_sequence"] = json_tokens(after);
        report["sequence_count_delta"] = static_cast<std::int64_t>(after.size()) -
                                          static_cast<std::int64_t>(before.size());
        report["eos_present_at_sequence_tail"] = !after.empty() && after.back() == eos;
        report["append_once"] = appended;
        report["eos_transition_matches"] = done;
        const bool method_supported = appended && returned_selected && done;
        if (!appended) fail_support(support, "AppendTokens did not append EOS ID " +
                                             std::to_string(eos) + " exactly once");
        if (!returned_selected) fail_support(support, "AppendTokens GetNextTokens did not report EOS ID " +
                                                  std::to_string(eos));
        if (!done) fail_support(support, "AppendTokens did not mark EOS ID " +
                                         std::to_string(eos) + " done");
        report["status"] = method_supported ? "PASS" : "UNSUPPORTED";
    } catch (const std::exception& error) {
        report["status"] = "UNSUPPORTED";
        report["error"] = error.what();
        fail_support(support, "AppendTokens EOS probe threw for ID " +
                              std::to_string(eos) + ": " + error.what());
    }
    return report;
}

json run_min_length_eos_mask(OgaModel& model, int vocab,
                             const std::vector<int>& eos_ids,
                             const json& defaults) {
    json report = {{"status", "OBSERVED"}, {"min_length", 3}, {"prompt", {10}},
                   {"eos_ids", eos_ids}};
    try {
        flm::hip_oga::SamplingState sampling(defaults, vocab);
        sampling.reset_request(json{{"temperature", 0}});
        auto params = make_params(model, 3);
        flm::hip_oga::apply_oga_sampling_options(*params, sampling.effective());
        params->SetSearchOption("min_length", 3.0);
        auto generator = OgaGenerator::Create(model, *params);
        const std::vector<std::int32_t> prompt{10};
        append_prompt(*generator, prompt);
        auto current = generator->GetLogits();
        (void)copy_logits(*current, vocab);
        std::vector<float> scores(static_cast<std::size_t>(vocab),
                                  -std::numeric_limits<float>::infinity());
        scores[11] = 3.0f;
        for (int eos : eos_ids) scores.at(static_cast<std::size_t>(eos)) = 4.0f;
        const auto before = sequence(*generator);
        flm::hip_oga::process_oga_sampling_logits(scores, generator->GetSequenceData(0),
            generator->GetSequenceCount(0), sampling.effective(), *params, eos_ids);
        bool all_eos_masked = true;
        for (int eos : eos_ids) all_eos_masked = all_eos_masked && !std::isfinite(scores[eos]);
        const bool ordinary_candidate_remains = std::isfinite(scores[11]);
        auto replacement = make_logits_tensor(scores);
        generator->SetLogits(*replacement);
        generator->GenerateNextToken();
        const auto returned = next_tokens(*generator);
        const bool selected_non_eos = returned.size() == 1 && returned.front() == 11;
        const bool done = generator->IsDone();
        report["processed_eos_scores_masked"] = all_eos_masked;
        report["ordinary_candidate_remains"] = ordinary_candidate_remains;
        report["get_next_tokens"] = json_tokens(returned);
        report["selected_non_eos"] = selected_non_eos;
        report["is_done"] = done;
        report["sequence_before"] = json_tokens(before);
        const auto after = sequence(*generator);
        report["sequence_after"] = json_tokens(after);
        const bool appended_once = appended_exactly_once(before, after, 11);
        report["ordinary_token_appended_once"] = appended_once;
        report["subject_get_logits_calls"] = 1;
        report["status"] = all_eos_masked && ordinary_candidate_remains &&
                            selected_non_eos && appended_once && !done ? "PASS" : "FAIL";
        report["fresh_reference_get_logits_calls"] = 0;
        report["oga_get_logits_api_calls"] = 1;
        report["oga_append_tokens_api_calls"] = 1;
        report["oga_generate_next_token_api_calls"] = 1;
        report["oga_set_logits_api_calls"] = 1;
        report["oga_rewind_to_api_calls"] = 0;
        if (report["status"] != "PASS")
            report["error"] = "min_length helper did not mask EOS while retaining token 11";
    } catch (const std::exception& error) {
        report["status"] = "FAIL";
        report["error"] = error.what();
    }
    return report;
}

json run_rewind_suffix(OgaModel& model, int vocab) {
    const std::vector<std::int32_t> prompt{10, 10};
    const std::vector<int> suffix{11};
    const std::vector<std::int32_t> expected{10, 10, 11};
    json report = {{"status", "OBSERVED"}, {"checkpoint_prompt", json_tokens(prompt)},
                   {"discarded_selected_id", 10}, {"suffix", suffix}};
    try {
        bool selected = false;
        bool sequence_matches = false;
        bool active_before_rewind = false;
        bool active_after_suffix = false;
        std::vector<std::int32_t> generated;
        std::vector<std::int32_t> after_rewind_append;
        LogitRow actual;
        {
            auto params = make_params(model);
            auto generator = OgaGenerator::Create(model, *params);
            append_prompt(*generator, prompt);
            auto current = generator->GetLogits();
            (void)copy_logits(*current, vocab);
            std::vector<float> one_finite(static_cast<std::size_t>(vocab),
                                          -std::numeric_limits<float>::infinity());
            one_finite[10] = 0.0f;
            auto replacement = make_logits_tensor(one_finite);
            generator->SetLogits(*replacement);
            generator->GenerateNextToken();
            generated = next_tokens(*generator);
            selected = generated.size() == 1 && generated.front() == 10;
            active_before_rewind = !generator->IsDone();
            generator->RewindTo(prompt.size());
            const std::int32_t suffix_id = suffix.front();
            generator->AppendTokens(&suffix_id, 1);
            after_rewind_append = sequence(*generator);
            sequence_matches = after_rewind_append == expected;
            active_after_suffix = !generator->IsDone();
            actual = current_logits(*generator, vocab);
        } // The rewind subject is destroyed before either serial fresh control.

        const auto segmented_rows = fresh_segmented_rows(model, prompt, suffix, vocab);
        const auto one_shot_reference = fresh_logits(model, expected, vocab);
        const json segmented_comparison = compare_logits(actual, segmented_rows.back());
        const json one_shot_comparison = compare_logits(actual, one_shot_reference);
        report["get_next_tokens_before_rewind"] = json_tokens(generated);
        report["selected_discarded_id"] = selected;
        report["is_done_before_rewind"] = !active_before_rewind;
        report["is_done_after_suffix"] = !active_after_suffix;
        report["sequence_after_rewind_append"] = json_tokens(after_rewind_append);
        report["sequence_matches_prompt_plus_suffix"] = sequence_matches;
        report["subject_get_logits_calls"] = 2;
        report["same_segmented_fresh_reference_get_logits_calls"] =
            static_cast<int>(segmented_rows.size());
        report["fresh_reference_get_logits_calls"] = 1;
        report["next_logits_vs_same_segmented_fresh"] = segmented_comparison;
        report["next_logits_vs_one_shot_prefill_diagnostic"] = one_shot_comparison;
        report["reference_policy"] =
            "same-segmented serial fresh row gates rewind/suffix correctness; one-shot full prefill remains diagnostic only";
        report["oga_get_logits_api_calls"] = 2 +
            static_cast<int>(segmented_rows.size()) + 1;
        report["oga_append_tokens_api_calls"] = 2 +
            static_cast<int>(segmented_rows.size()) + 1;
        report["oga_generate_next_token_api_calls"] = 1;
        report["oga_set_logits_api_calls"] = 1;
        report["oga_rewind_to_api_calls"] = 1;
        report["status"] = selected && active_before_rewind && active_after_suffix && sequence_matches &&
            comparison_passes(segmented_comparison) ? "PASS" : "FAIL";
        if (report["status"] != "PASS")
            report["error"] = "rewind plus nonempty suffix differs from same-segmented fresh state";
    } catch (const std::exception& error) {
        report["status"] = "FAIL";
        report["error"] = error.what();
    }
    return report;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_external_sampling <model_dir> <report.json>\n";
        return 2;
    }

    OgaHandle handle;
    json report = {{"status", "FAIL"}, {"model_dir", argv[1]},
                   {"cases", json::object()}};
    try {
        auto model = OgaModel::Create(argv[1]);
        std::ifstream input(std::string(argv[1]) + "/genai_config.json");
        require(static_cast<bool>(input), "could not open genai_config.json");
        const auto config = json::parse(input);
        const int vocab = config.at("model").at("vocab_size").get<int>();
        require(vocab > 20, "fixture vocabulary must include token IDs 10 and 11");
        const auto eos_ids = read_eos_ids(config);
        for (int eos : eos_ids)
            require(eos >= 0 && eos < vocab, "model EOS ID is outside the vocabulary");
        require(!is_eos(10, eos_ids) && !is_eos(11, eos_ids),
                "ordinary controlled token IDs 10 and 11 must not be EOS");

        auto defaults_params = OgaGeneratorParams::Create(*model);
        const auto defaults = flm::hip_oga::read_oga_sampling_options(*defaults_params);
        SupportResult one_finite_support;
        SupportResult append_support;
        SupportResult native_support;
        const std::vector<std::int32_t> prompt{10};
        const std::vector<int> ordinary_ids{10, 11};
        const std::string native_prompt_text =
            "Continue this sentence with several words: The forest path curved beyond the trees and opened into a quiet meadow where";
        const auto native_prompt = tokenize_prompt(*model, native_prompt_text);

        try {
            report["cases"]["native_greedy_equivalence"] =
                run_native_equivalence_controls(*model, vocab, native_prompt_text,
                                                native_prompt, eos_ids, native_support);
        } catch (const std::exception& error) {
            fail_support(native_support,
                         std::string("native greedy equivalence probe threw: ") + error.what());
            report["cases"]["native_greedy_equivalence"] = {
                {"status", "FAIL"}, {"error", error.what()},
                {"support_reasons", native_support.reasons}};
        }

        try {
            report["cases"]["one_finite_greedy_generate_next_token"] =
                run_one_finite_steps(*model, vocab, prompt, ordinary_ids, eos_ids,
                                     one_finite_support);
        } catch (const std::exception& error) {
            fail_support(one_finite_support,
                         std::string("GenerateNextToken ordinary-token probe threw: ") + error.what());
            report["cases"]["one_finite_greedy_generate_next_token"] = {
                {"status", "FAIL"}, {"error", error.what()}};
        }
        try {
            report["cases"]["append_tokens_selected_ids"] =
                run_append_steps(*model, vocab, prompt, ordinary_ids, eos_ids,
                                 append_support);
        } catch (const std::exception& error) {
            fail_support(append_support,
                         std::string("AppendTokens ordinary-token probe threw: ") + error.what());
            report["cases"]["append_tokens_selected_ids"] = {
                {"status", "UNSUPPORTED"}, {"error", error.what()}};
        }

        json one_finite_eos = json::array();
        json append_eos = json::array();
        for (int eos : eos_ids) {
            one_finite_eos.push_back(run_one_finite_eos(*model, vocab, eos, eos_ids,
                                                        one_finite_support));
            append_eos.push_back(run_append_eos(*model, vocab, eos, eos_ids,
                                                append_support));
        }
        report["cases"]["one_finite_greedy_eos"] = one_finite_eos;
        report["cases"]["append_tokens_eos"] = append_eos;
        report["cases"]["min_length_eos_mask"] =
            run_min_length_eos_mask(*model, vocab, eos_ids, defaults);
        report["cases"]["rewind_nonempty_suffix"] = run_rewind_suffix(*model, vocab);

        if (report["cases"]["min_length_eos_mask"].at("status") != "PASS")
            fail_support(one_finite_support, "min_length EOS mask probe failed");
        if (report["cases"]["rewind_nonempty_suffix"].at("status") != "PASS")
            fail_support(one_finite_support, "rewind plus nonempty suffix probe failed");
        const bool controlled_checks_supported = one_finite_support.supported;
        if (!native_support.supported)
            fail_support(one_finite_support, "independent native-equivalence or repeatability control failed");
        const bool candidate_supported = one_finite_support.supported && native_support.supported;
        report["methods"] = {
            {"one_finite_greedy_generate_next_token",
             {{"supported", candidate_supported},
              {"reasons", one_finite_support.reasons}}},
            {"append_tokens_selected_id",
             {{"supported", append_support.supported},
              {"reasons", append_support.reasons}}}
        };
        report["eos_ids"] = eos_ids;
        int get_logits_api_call_count = 0;
        int append_tokens_api_call_count = 0;
        int generate_next_token_api_call_count = 0;
        int set_logits_api_call_count = 0;
        int rewind_to_api_call_count = 0;
        const auto accumulate_counts = [&](const json& value) {
            get_logits_api_call_count += value.value("oga_get_logits_api_calls", 0);
            append_tokens_api_call_count += value.value("oga_append_tokens_api_calls", 0);
            generate_next_token_api_call_count += value.value("oga_generate_next_token_api_calls", 0);
            set_logits_api_call_count += value.value("oga_set_logits_api_calls", 0);
            rewind_to_api_call_count += value.value("oga_rewind_to_api_calls", 0);
        };
        for (const auto& item : report["cases"]) {
            if (item.is_array()) {
                for (const auto& subcase : item) accumulate_counts(subcase);
            } else if (item.is_object()) {
                accumulate_counts(item);
            }
        }
        report["oga_get_logits_api_call_count"] = get_logits_api_call_count;
        report["oga_append_tokens_api_call_count"] = append_tokens_api_call_count;
        report["oga_generate_next_token_api_call_count"] = generate_next_token_api_call_count;
        report["oga_set_logits_api_call_count"] = set_logits_api_call_count;
        report["oga_rewind_to_api_call_count"] = rewind_to_api_call_count;
        report["oga_commit_operation_count"] = append_tokens_api_call_count +
                                                generate_next_token_api_call_count;
        report["oga_commit_operation_count_semantics"] =
            "all AppendTokens and GenerateNextToken calls, including native, repeat, segmented-fresh, and one-shot diagnostic generators; excludes SetLogits and RewindTo";
        report["acceptance_basis"] = {
            {"native_equivalence_and_repeatability",
             report["cases"]["native_greedy_equivalence"].at("status") == "PASS"},
            {"controlled_one_finite_transition_and_rewind_checks", controlled_checks_supported},
            {"append_tokens_route", append_support.supported}};
        report["bounded_steps_note"] =
            "API-call counts include explicitly issued GetLogits calls; the runtime may cache or execute work internally. Native baseline, serial repeat, replay, and fresh reference generators never overlap.";
        report["logit_comparison_tolerance"] = {
            {"absolute", kLogitAbsTolerance}, {"relative", kLogitRelativeTolerance}};
        report["status"] = candidate_supported ? "PASS" : "FAIL";
    } catch (const std::exception& error) {
        report["status"] = "FAIL";
        report["error"] = error.what();
    }

    std::ofstream output(argv[2]);
    if (!output) return 2;
    output << report.dump(2) << '\n';
    std::cout << report.dump(2) << std::endl;
    return report["status"] == "PASS" ? 0 : 1;
}
