/// \file test_media_transaction.cpp
/// \brief No-hardware adapter tests for the extracted HIP OGA media transaction.
#include "AutoModel/hip_oga_prefix_cache.hpp"
#include "AutoModel/hip_oga_media_inputs.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using flm::hip_oga::MediaCacheIdentity;
using flm::hip_oga::MediaTensorIdentity;
using flm::hip_oga::PrefixCacheDecision;
using flm::hip_oga::PrefixCacheDecisionKind;
using flm::hip_oga::PrefixCacheState;
using Clock = std::chrono::steady_clock;

using OgaElementType = int;
constexpr OgaElementType OgaElementType_int32 = 6;
constexpr OgaElementType OgaElementType_int64 = 7;
constexpr OgaElementType OgaElementType_float32 = 1;
constexpr OgaElementType OgaElementType_float16 = 10;
constexpr OgaElementType OgaElementType_bfloat16 = 16;

struct FakeOgaTensor {
    std::vector<std::int64_t> shape;
    OgaElementType type = OgaElementType_int32;
    const void* data = nullptr;

    const std::vector<std::int64_t>& Shape() const { return shape; }
    OgaElementType Type() const { return type; }
    void* Data() const { return const_cast<void*>(data); }
};

struct FakeOgaTensorNames {
    std::vector<std::string> names;

    std::size_t Count() const { return names.size(); }
    const char* Get(std::size_t index) const { return names.at(index).c_str(); }
};

struct OgaNamedTensors {
    std::map<std::string, FakeOgaTensor> tensors;
    std::vector<std::string> names;

    OgaNamedTensors() = default;
    OgaNamedTensors(std::vector<std::int64_t> shape,
                    OgaElementType type,
                    const void* data)
    {
        add("input_ids", std::move(shape), type, data);
    }

    void add(std::string name,
             std::vector<std::int64_t> shape,
             OgaElementType type,
             const void* data) {
        names.push_back(name);
        tensors.emplace(std::move(name), FakeOgaTensor{std::move(shape), type, data});
    }

    std::shared_ptr<FakeOgaTensor> Get(const char* name) {
        const auto tensor = tensors.find(name);
        if (tensor == tensors.end()) {
            throw std::out_of_range(std::string("missing tensor: ") + name);
        }
        return std::make_shared<FakeOgaTensor>(tensor->second);
    }

    std::shared_ptr<FakeOgaTensorNames> GetNames() const {
        return std::make_shared<FakeOgaTensorNames>(FakeOgaTensorNames{names});
    }
};

#include "production_media_tokens.inc"
#include "production_media_tensor.inc"
#include "production_json_reader.inc"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

struct FakeLoadedImages {
    std::vector<std::vector<std::uint8_t>> buffers;
};

struct FakePreparedMedia {
    FakeLoadedImages loaded_images;
    std::shared_ptr<OgaNamedTensors> inputs;
    std::vector<int> tokens;
    std::optional<MediaCacheIdentity> identity;
    nlohmann::json input_contract;
};

class IdentityCaptureHarness {
public:
    bool flat_gemma3_features_ = false;
    bool gemma3_model_ = false;
    int image_token_id_ = 10;
    std::size_t context_length_ = 2048;
    bool media_rewind_qualified_ = true;
    std::string media_processor_signature_ = "processor:test:v1";

    void capture(FakePreparedMedia& prepared, bool capture_identity) {
#include "production_media_prepare.inc"
    }
};

void test_flat_media_preparation_normalizes_before_capture_even_with_cache_off() {
    for (bool capture : {false, true}) {
        std::vector<std::int32_t> ids(520, 10), types(520, 1);
        std::fill(ids.begin()+512, ids.end(), 20);
        std::fill(types.begin()+512, types.end(), 0);
        std::int64_t count = 256;
        std::vector<float> pixels(24, 0.5f);
        FakePreparedMedia prepared;
        prepared.inputs = std::make_shared<OgaNamedTensors>();
        prepared.inputs->add("input_ids", {1,520}, OgaElementType_int32, ids.data());
        prepared.inputs->add("token_type_ids", {1,520}, OgaElementType_int32, types.data());
        prepared.inputs->add("num_image_tokens", {1}, OgaElementType_int64, &count);
        prepared.inputs->add("pixel_values", {2,3,2,2}, OgaElementType_float32, pixels.data());
        prepared.loaded_images.buffers = {{1,2},{3,4}};
        IdentityCaptureHarness harness;
        harness.gemma3_model_ = true;
        harness.flat_gemma3_features_ = true;
        harness.capture(prepared, capture);
        require(count == 512, "flat features must receive the total image token count with cache on or off");
        require(prepared.identity.has_value() == capture, "normalization changed cache eligibility");
        require(prepared.input_contract.at("image_tokens") == 512 &&
                prepared.input_contract.at("input_tokens") == 520,
                "processor diagnostics must describe the validated token layout");
        if (capture) {
            const auto& tensor = prepared.identity->processor_tensors.at(0);
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&count);
            require(tensor.bytes == std::vector<std::uint8_t>(bytes, bytes+sizeof(count)),
                    "cache identity captured the raw count before normalization");
        }
        // The ordinary rank-3 model receives its original per-image count.
        // Identity capture transfers buffer ownership, so a new preparation
        // needs its own freshly loaded image buffers.
        prepared.loaded_images.buffers = {{1,2},{3,4}};
        count = 256;
        harness.flat_gemma3_features_ = false;
        harness.capture(prepared, false);
        require(count == 256, "native Gemma feature count must remain per-image");

        // Invalid processor output must be rejected before count conversion or
        // cache identity capture. Exercise the actual extracted prepare path.
        prepared.identity.reset();
        harness.flat_gemma3_features_ = true;
        ids[0] = 20;
        require_throws([&] { harness.capture(prepared, true); },
                       "mismatched image placeholders must fail preparation");
        require(count == 256 && !prepared.identity.has_value(),
                "invalid token layout changed the tensor count or cache identity");
    }
}

OgaNamedTensors named_media_inputs(const std::vector<std::string>& names,
                                   const std::int32_t* input_ids,
                                   const std::int64_t* num_image_tokens,
                                   const float* pixel_values,
                                   const std::int32_t* token_types) {
    OgaNamedTensors inputs;
    for (const auto& name : names) {
        if (name == "input_ids") {
            inputs.add(name, {1, 3}, OgaElementType_int32, input_ids);
        } else if (name == "num_image_tokens" || name == "num_img_tokens") {
            inputs.add(name, {1}, OgaElementType_int64, num_image_tokens);
        } else if (name == "pixel_values") {
            inputs.add(name, {1, 2}, OgaElementType_float32, pixel_values);
        } else if (name == "token_type_ids") {
            inputs.add(name, {1, 3}, OgaElementType_int32, token_types);
        } else {
            inputs.add(name, {1, 1}, OgaElementType_int32, input_ids);
        }
    }
    return inputs;
}

FakePreparedMedia make_fake_prepared_media(
    const std::vector<std::string>& names,
    const std::int32_t* input_ids,
    const std::int64_t* num_image_tokens,
    const float* pixel_values,
    const std::int32_t* token_types) {
    FakePreparedMedia prepared;
    prepared.inputs = std::make_shared<OgaNamedTensors>(
        named_media_inputs(names, input_ids, num_image_tokens, pixel_values, token_types));
    prepared.tokens = read_oga_token_tensor(*prepared.inputs, "input_ids");
    prepared.loaded_images.buffers = {{0x11, 0x12, 0x13}, {0x21, 0x22}};
    return prepared;
}

std::vector<std::uint8_t> raw_bytes(const void* data, std::size_t size) {
    const auto* first = static_cast<const std::uint8_t*>(data);
    return {first, first + size};
}

void header_print(const char*, const std::string&) {}

namespace time_utils {
std::pair<std::uint64_t, int> duration_ns(Clock::time_point start,
                                         Clock::time_point end) {
    return {static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
            0};
}
}  // namespace time_utils

struct Timer {
    Clock::time_point start() const { return Clock::now(); }
    Clock::time_point stop(std::size_t) const { return Clock::now(); }
};

enum { PREFILL_TIME };
constexpr const char* CANCEL_DETECTED = "cancel_detected";

struct Prepared {
    std::vector<int> tokens;
    std::optional<MediaCacheIdentity> identity;
    std::string snapshot_id;
};

enum class Fault {
    None,
    BeforeRewind,
    AfterRewind,
    PartialAppend,
    AfterAppend,
    FallbackRebuild,
};

const char* fault_name(Fault fault) {
    switch (fault) {
        case Fault::None: return "none";
        case Fault::BeforeRewind: return "before-rewind";
        case Fault::AfterRewind: return "after-rewind";
        case Fault::PartialAppend: return "partial-append";
        case Fault::AfterAppend: return "after-append";
        case Fault::FallbackRebuild: return "fallback-rebuild-fails";
    }
    return "unknown";
}

struct Operation {
    std::string name;
    std::string snapshot_id;
    bool same_prepared_object = false;
};

class Provider {
public:
    std::size_t sequence = 0;
    std::size_t checkpoint = 0;
    std::size_t capacity = 64;
    bool generator_active = false;
    bool cancelled = false;
    bool cancel_after_mutation = false;
    bool fallback_rebuild_failed = false;
    Fault fault = Fault::None;
    bool fault_fired = false;
    bool recovery_requested = false;
    const Prepared* active_prepared = nullptr;
    std::string active_snapshot_id;
    std::vector<Operation> operations;
    std::vector<std::string> begin_snapshot_ids;
    std::vector<bool> begin_used_same_prepared_object;
    std::vector<bool> begin_requested_image_cache;
    std::size_t begin_calls = 0;
    std::size_t checkpoint_calls = 0;
    std::size_t sampling_log_calls = 0;

    void activate(const Prepared& prepared) {
        active_prepared = &prepared;
        active_snapshot_id = prepared.snapshot_id;
    }

    void deactivate() {
        active_prepared = nullptr;
        active_snapshot_id.clear();
    }

    void record(const char* name, const Prepared* supplied_prepared = nullptr) {
        operations.push_back({name,
                              active_snapshot_id,
                              active_prepared != nullptr &&
                                  (supplied_prepared == nullptr ||
                                   supplied_prepared == active_prepared)});
    }

    bool fail_once(Fault expected) {
        if (!fault_fired && fault == expected) {
            fault_fired = true;
            return true;
        }
        return false;
    }

    void mark_mutation() {
        if (cancel_after_mutation) cancelled = true;
    }

    void begin_multimodal_turn(const Prepared& prepared, bool use_image_cache) {
        ++begin_calls;
        begin_snapshot_ids.push_back(prepared.snapshot_id);
        begin_used_same_prepared_object.push_back(&prepared == active_prepared);
        begin_requested_image_cache.push_back(use_image_cache);
        record("begin_multimodal_turn", &prepared);

        if (recovery_requested && fault == Fault::FallbackRebuild) {
            fallback_rebuild_failed = true;
            throw std::runtime_error("injected:fallback-rebuild-fails");
        }
        generator_active = true;
        sequence = prepared.tokens.size();
        checkpoint = sequence;
        mark_mutation();
    }

    void restore_text_checkpoint() {
        require(generator_active, "restore called without an active generator");
        record("restore_text_checkpoint");
        if (fail_once(Fault::BeforeRewind)) {
            throw std::runtime_error("injected:before-rewind");
        }

        sequence = checkpoint;
        record("rewound");
        mark_mutation();
        if (fail_once(Fault::AfterRewind) || fail_once(Fault::FallbackRebuild)) {
            throw std::runtime_error(std::string("injected:") + fault_name(fault));
        }
    }

    void append_text_suffix(const std::vector<int>& suffix) {
        require(generator_active, "append called without an active generator");
        record("append_text_suffix");
        if (suffix.empty()) return;
        if (fault == Fault::PartialAppend && !fault_fired) {
            require(suffix.size() > 1, "partial append fixture must have two suffix tokens");
            ++sequence;
            mark_mutation();
            record("partial_append");
            fault_fired = true;
            throw std::runtime_error("injected:partial-append");
        }

        sequence += suffix.size();
        mark_mutation();
        record("suffix_appended");
        if (fail_once(Fault::AfterAppend)) {
            throw std::runtime_error("injected:after-append");
        }
    }

    int checkpoint_text_prompt() {
        require(generator_active, "checkpoint requested without an active generator");
        ++checkpoint_calls;
        checkpoint = sequence;
        record("checkpoint_text_prompt");
        return static_cast<int>(checkpoint);
    }

    std::size_t generator_capacity() const { return capacity; }

    void log_sampling_settings(bool) {
        ++sampling_log_calls;
        record("log_sampling_settings");
    }

    void clear_context() {
        record("clear_context");
        generator_active = false;
        sequence = 0;
        checkpoint = 0;
        recovery_requested = true;
    }
};

struct MetaInfo {
    bool restore_allowed = true;
    int cached_prompt_tokens = -1;
    std::uint64_t prefill_duration = 0;
    std::string stop_reason;
};

class Harness {
public:
    Provider provider;
    PrefixCacheState prefix_cache_;
    Timer profiler_list[1];
    std::vector<int> token_history;
    std::vector<int> checkpoint_his;
    PrefixCacheDecision cache_decision;
    MetaInfo meta_info;
    std::size_t prompt_tokens = 0;

    void clear_context() {
        provider.clear_context();
        prefix_cache_.reset_checkpoint("context_cleared");
        checkpoint_his.clear();
    }

    bool process(Prepared prepared,
                 bool cache_enabled = true,
                 bool restore_allowed = true) {
        struct ActivePreparedGuard {
            Provider& provider;
            ~ActivePreparedGuard() { provider.deactivate(); }
        } guard{provider};

        provider.activate(prepared);
        meta_info = MetaInfo{};
        meta_info.restore_allowed = restore_allowed;
        prompt_tokens = 0;

        auto* hip_engine = &provider;
        const std::string signature = "test-model;ctx=64;greedy";
        cache_decision = prefix_cache_.evaluate(
            cache_enabled,
            meta_info.restore_allowed,
            false,
            signature,
            prepared.tokens.size(),
            hip_engine->generator_capacity(),
            prepared.tokens,
            prepared.identity ? &*prepared.identity : nullptr);
        auto is_cancelled = [this] { return provider.cancelled; };

#include "production_media_prefill.inc"
        return true;
    }
};

MediaCacheIdentity media_identity(std::size_t token_count, std::uint8_t image_seed = 0x10) {
    std::vector<int> token_types{0, 1, 1, 0};
    if (token_count > token_types.size()) {
        token_types.resize(token_count, 0);
    }
    return MediaCacheIdentity{
        {{image_seed, static_cast<std::uint8_t>(image_seed + 1),
          static_cast<std::uint8_t>(image_seed + 2)},
         {0x30, 0x31}},
        "vision-processor:v1",
        {MediaTensorIdentity{"pixel_values", 1, {1, 2},
                             {0x00, 0x00, 0x80, 0x3f,
                              0x00, 0x00, 0x00, 0x40}}},
        std::move(token_types)};
}

Prepared prepared(std::vector<int> tokens,
                  const std::string& snapshot_id,
                  bool with_identity = true,
                  std::uint8_t image_seed = 0x10) {
    std::optional<MediaCacheIdentity> identity;
    if (with_identity) identity = media_identity(tokens.size(), image_seed);
    return {std::move(tokens), std::move(identity), snapshot_id};
}

const std::vector<int> kPrefixTokens{11, 12, 13, 14};
const std::vector<int> kExtendedTokens{11, 12, 13, 14, 15, 16};

const std::vector<std::string> kOfficialMediaInputNames{
    "input_ids", "num_image_tokens", "pixel_values", "token_type_ids"};

void require_valid_identity_capture(const std::vector<std::string>& names) {
    const std::int32_t input_ids[]{101, 102, 103};
    const std::int64_t image_token_counts[]{2};
    const float pixel_values[]{1.25f, -0.5f};
    const std::int32_t token_types[]{0, 1, 0};
    const std::vector<std::vector<std::uint8_t>> expected_images{
        {0x11, 0x12, 0x13}, {0x21, 0x22}};
    auto prepared = make_fake_prepared_media(names,
                                             input_ids,
                                             image_token_counts,
                                             pixel_values,
                                             token_types);
    IdentityCaptureHarness harness;
    harness.capture(prepared, true);

    require(prepared.identity.has_value(),
            "official media input names did not capture a reusable identity");
    const auto& identity = *prepared.identity;
    require(identity.processor_signature == "processor:test:v1",
            "identity capture lost the processor signature");
    require(identity.images == expected_images,
            "identity capture did not preserve the loaded image byte snapshot");
    require(prepared.loaded_images.buffers.empty(),
            "identity capture did not move the owned image snapshot");
    require(identity.token_types == std::vector<int>({0, 1, 0}),
            "identity capture lost token type alignment");
    require(identity.processor_tensors.size() == 2,
            "identity capture copied the wrong number of fixed tensors");

    const auto& image_counts = identity.processor_tensors[0];
    require(image_counts.name == "num_image_tokens" &&
                image_counts.element_type == static_cast<int>(OgaElementType_int64) &&
                image_counts.shape == std::vector<std::int64_t>({1}) &&
                image_counts.bytes == raw_bytes(image_token_counts, sizeof(image_token_counts)),
            "identity capture did not copy the fixed image-count tensor faithfully");
    const auto& pixels = identity.processor_tensors[1];
    require(pixels.name == "pixel_values" &&
                pixels.element_type == static_cast<int>(OgaElementType_float32) &&
                pixels.shape == std::vector<std::int64_t>({1, 2}) &&
                pixels.bytes == raw_bytes(pixel_values, sizeof(pixel_values)),
            "identity capture did not copy the fixed pixel tensor faithfully");
}

void test_identity_capture_accepts_official_names_in_any_order() {
    require_valid_identity_capture(kOfficialMediaInputNames);
    auto reversed_names = kOfficialMediaInputNames;
    std::reverse(reversed_names.begin(), reversed_names.end());
    require_valid_identity_capture(reversed_names);
}

void test_identity_capture_fails_closed_on_unknown_or_missing_names() {
    const std::int32_t input_ids[]{101, 102, 103};
    const std::int64_t image_token_counts[]{2};
    const float pixel_values[]{1.25f, -0.5f};
    const std::int32_t token_types[]{0, 1, 0};

    auto extra_names = kOfficialMediaInputNames;
    extra_names.push_back("unsupported_processor_input");
    auto with_extra = make_fake_prepared_media(extra_names,
                                               input_ids,
                                               image_token_counts,
                                               pixel_values,
                                               token_types);
    IdentityCaptureHarness extra_harness;
    extra_harness.capture(with_extra, true);
    require(!with_extra.identity,
            "unknown processor input name unexpectedly enabled identity capture");

    auto missing_names = kOfficialMediaInputNames;
    missing_names.erase(std::remove(missing_names.begin(), missing_names.end(),
                                    "num_image_tokens"),
                        missing_names.end());
    auto missing = make_fake_prepared_media(missing_names,
                                            input_ids,
                                            image_token_counts,
                                            pixel_values,
                                            token_types);
    IdentityCaptureHarness missing_harness;
    missing_harness.capture(missing, true);
    require(!missing.identity,
            "missing official image-count tensor unexpectedly enabled identity capture");
    require_throws([&] { (void)missing.inputs->Get("num_image_tokens"); },
                   "fake named tensors accepted a missing requested tensor");

    auto legacy_names = kOfficialMediaInputNames;
    legacy_names[1] = "num_img_tokens";
    auto legacy = make_fake_prepared_media(legacy_names,
                                           input_ids,
                                           image_token_counts,
                                           pixel_values,
                                           token_types);
    IdentityCaptureHarness legacy_harness;
    legacy_harness.capture(legacy, true);
    require(!legacy.identity,
            "legacy image-count tensor name unexpectedly enabled identity capture");
    require_throws([&] { (void)legacy.inputs->Get("num_image_tokens"); },
                   "fake named tensors accepted a missing official image-count tensor");
}

void test_identity_capture_requires_enabled_and_qualified_path() {
    const std::int32_t input_ids[]{101, 102, 103};
    const std::int64_t image_token_counts[]{2};
    const float pixel_values[]{1.25f, -0.5f};
    const std::int32_t token_types[]{0, 1, 0};

    auto disabled = make_fake_prepared_media(kOfficialMediaInputNames,
                                             input_ids,
                                             image_token_counts,
                                             pixel_values,
                                             token_types);
    IdentityCaptureHarness disabled_harness;
    disabled_harness.capture(disabled, false);
    require(!disabled.identity,
            "disabled identity capture retained processor identity");

    auto unqualified = make_fake_prepared_media(kOfficialMediaInputNames,
                                                input_ids,
                                                image_token_counts,
                                                pixel_values,
                                                token_types);
    IdentityCaptureHarness unqualified_harness;
    unqualified_harness.media_rewind_qualified_ = false;
    unqualified_harness.capture(unqualified, true);
    require(!unqualified.identity,
            "unqualified media rewind retained processor identity");
}

void test_oga_token_tensor_preserves_signed_ids_in_int32_and_int64() {
    const std::int32_t int32_ids[]{-1, -200, 23};
    OgaNamedTensors int32_input({1, 3}, OgaElementType_int32, int32_ids);
    require(read_oga_token_tensor(int32_input, "input_ids") ==
                std::vector<int>({-1, -200, 23}),
            "signed int32 image placeholder IDs were not preserved");

    const std::int64_t int64_ids[]{-200, 0, std::numeric_limits<int>::max()};
    OgaNamedTensors int64_input({1, 3}, OgaElementType_int64, int64_ids);
    require(read_oga_token_tensor(int64_input, "input_ids") ==
                std::vector<int>({-200, 0, std::numeric_limits<int>::max()}),
            "int64 token IDs within int range were not preserved");
    int32_input.add("token_type_ids", {1, 3}, OgaElementType_int32, int32_ids);
    require_throws([&] { (void)read_oga_token_tensor(int32_input, "token_type_ids"); },
                   "negative sequence metadata must still be rejected");
}

void test_oga_token_tensor_rejects_int64_values_outside_int_range() {
    const std::int64_t above_max[]{
        static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1};
    OgaNamedTensors high_input({1, 1}, OgaElementType_int64, above_max);
    require_throws([&] { (void)read_oga_token_tensor(high_input, "input_ids"); },
                   "int64 token above int max was accepted");

    const std::int64_t below_min[]{
        static_cast<std::int64_t>(std::numeric_limits<int>::min()) - 1};
    OgaNamedTensors low_input({1, 1}, OgaElementType_int64, below_min);
    require_throws([&] { (void)read_oga_token_tensor(low_input, "input_ids"); },
                   "int64 token below int min was accepted");
}

void test_oga_token_tensor_rejects_malformed_shapes() {
    const std::int32_t ids[]{1, 2, 3};
    for (const auto& shape : std::vector<std::vector<std::int64_t>>{
             {3}, {2, 3}, {}, {1, 0}}) {
        OgaNamedTensors input(shape, OgaElementType_int32, ids);
        require_throws([&] { (void)read_oga_token_tensor(input, "input_ids"); },
                       "malformed token tensor shape was accepted");
    }
}

void test_oga_token_tensor_rejects_unsupported_type_and_null_data() {
    const float float_ids[]{1.0f, 2.0f};
    OgaNamedTensors float_input({1, 2}, OgaElementType_float32, float_ids);
    require_throws([&] { (void)read_oga_token_tensor(float_input, "input_ids"); },
                   "unsupported float token tensor was accepted");

    OgaNamedTensors null_input({1, 2}, OgaElementType_int32, nullptr);
    require_throws([&] { (void)read_oga_token_tensor(null_input, "input_ids"); },
                   "null token tensor data was accepted");
}

void require_fresh(const Harness& harness,
                   const std::string& reason,
                   std::size_t token_count) {
    require(harness.cache_decision.kind == PrefixCacheDecisionKind::Fresh,
            "expected a fresh media transaction decision");
    require(!harness.cache_decision.reuse,
            "fresh media transaction unexpectedly reused the checkpoint");
    require(harness.cache_decision.reason == reason,
            "unexpected fresh reason: " + harness.cache_decision.reason);
    require(harness.cache_decision.cached_tokens == 0,
            "fresh media transaction reported cached tokens");
    require(harness.cache_decision.newly_prefilled_tokens == token_count,
            "fresh media transaction reported wrong prefill accounting");
    require(harness.prompt_tokens == token_count,
            "fresh media transaction reported wrong prompt token count");
    require(harness.meta_info.cached_prompt_tokens == 0,
            "fresh media transaction metadata reported cached tokens");
}

void seed_prefix(Harness& harness) {
    require(harness.process(prepared(kPrefixTokens, "snapshot-prefix")),
            "prefix prefill unexpectedly cancelled");
    require_fresh(harness, "no_checkpoint", kPrefixTokens.size());
    require(harness.prefix_cache_.has_checkpoint(),
            "successful initial media prefill did not commit a checkpoint");
    require(harness.token_history == kPrefixTokens,
            "initial media prefill committed wrong token history");
}

void test_successful_reuse_commits_truthful_accounting() {
    Harness harness;
    seed_prefix(harness);

    const Prepared extension = prepared(kExtendedTokens, "snapshot-extended");
    require(harness.process(extension), "media extension unexpectedly cancelled");
    require(harness.cache_decision.kind == PrefixCacheDecisionKind::Reuse &&
                harness.cache_decision.reuse && harness.cache_decision.reason == "hit",
            "identical ordered media prefix did not report a hit");
    require(harness.cache_decision.cached_tokens == 4 &&
                harness.cache_decision.suffix_start == 4,
            "media hit reported the wrong cached prefix boundary");
    require(harness.cache_decision.newly_prefilled_tokens == 2 &&
                harness.prompt_tokens == 6,
            "media hit reported the wrong suffix accounting");
    require(harness.meta_info.cached_prompt_tokens == 4,
            "media hit metadata reported the wrong cached token count");
    require(harness.provider.sequence == 6 && harness.provider.checkpoint == 6,
            "provider did not append and checkpoint the full extended prompt");
    require(harness.prefix_cache_.checkpoint_size() == 6 &&
                harness.token_history == kExtendedTokens,
            "media hit did not commit the extended prompt state");
    require(harness.provider.operations.size() >= 8 &&
                harness.provider.operations[3].name == "restore_text_checkpoint" &&
                harness.provider.operations[5].name == "append_text_suffix",
            "media hit did not use restore followed by suffix append");
}

void test_changed_media_misses_and_rebuilds_fresh() {
    Harness harness;
    seed_prefix(harness);

    const Prepared changed = prepared(kExtendedTokens, "snapshot-changed-image", true, 0x50);
    require(harness.process(changed), "changed-media rebuild unexpectedly cancelled");
    require_fresh(harness, "media_changed", kExtendedTokens.size());
    require(harness.provider.sequence == kExtendedTokens.size(),
            "changed media did not rebuild the complete prompt");
    require(harness.prefix_cache_.checkpoint_size() == kExtendedTokens.size() &&
                harness.token_history == kExtendedTokens,
            "successful changed-media prefill did not commit fresh state");
    require(harness.provider.begin_requested_image_cache.back(),
            "fresh changed-media transaction lost its prepared processor inputs");
}

void test_missing_identity_keeps_legacy_multimodal_fresh_path() {
    Harness harness;
    seed_prefix(harness);

    const Prepared without_identity =
        prepared(kExtendedTokens, "snapshot-without-identity", false);
    require(harness.process(without_identity),
            "identity-free multimodal prefill unexpectedly cancelled");
    require_fresh(harness, "multimodal_forced_miss", kExtendedTokens.size());
    require(!harness.provider.begin_requested_image_cache.back(),
            "identity-free media request enabled prepared-image caching");
    require(!harness.prefix_cache_.has_checkpoint(),
            "identity-free multimodal request retained a media checkpoint");
}

void test_provider_faults_fallback_once_with_same_prepared_object() {
    const std::vector<Fault> faults{Fault::BeforeRewind,
                                    Fault::AfterRewind,
                                    Fault::PartialAppend,
                                    Fault::AfterAppend};
    for (const Fault fault : faults) {
        Harness harness;
        seed_prefix(harness);
        harness.provider.fault = fault;
        const std::size_t begin_count_before = harness.provider.begin_calls;
        const std::size_t operation_begin = harness.provider.operations.size();
        const Prepared extension = prepared(kExtendedTokens, fault_name(fault));

        require(harness.process(extension),
                std::string(fault_name(fault)) + " fallback unexpectedly failed");
        require_fresh(harness, "provider_exception", kExtendedTokens.size());
        require(harness.provider.begin_calls == begin_count_before + 1,
                "provider exception did not perform exactly one fresh rebuild");
        require(harness.provider.begin_snapshot_ids.back() == extension.snapshot_id &&
                    harness.provider.begin_used_same_prepared_object.back(),
                "fallback did not consume the same prepared media object");
        require(!harness.provider.begin_requested_image_cache.back(),
                "fallback rebuild unexpectedly retained prepared-image cache state");
        require(harness.provider.operations.size() > operation_begin,
                "provider fault produced no recorded transaction operations");
        for (std::size_t i = operation_begin; i < harness.provider.operations.size(); ++i) {
            require(harness.provider.operations[i].snapshot_id == extension.snapshot_id,
                    "fallback operation observed a different prepared snapshot");
        }
        require(harness.prefix_cache_.disabled_for_session(),
                "provider exception did not disable reuse for the session");
        require(!harness.prefix_cache_.has_checkpoint() &&
                    harness.prefix_cache_.checkpoint_size() == 0,
                "provider fallback retained a media checkpoint");
        require(harness.token_history == kExtendedTokens,
                "successful fallback did not commit rebuilt token history");

        const std::size_t next_begin_count = harness.provider.begin_calls;
        require(harness.process(prepared(kExtendedTokens, "snapshot-after-fault")),
                "session-disabled fresh prefill unexpectedly cancelled");
        require_fresh(harness, "session_disabled", kExtendedTokens.size());
        require(harness.provider.begin_calls == next_begin_count + 1,
                "session-disabled request did not rebuild from the full media prompt");
        require(harness.prefix_cache_.disabled_for_session() &&
                    !harness.prefix_cache_.has_checkpoint(),
                "session-disabled request retained a media checkpoint");
    }
}

void test_failed_fallback_leaves_no_generator_checkpoint_or_new_history() {
    Harness harness;
    seed_prefix(harness);
    harness.provider.fault = Fault::FallbackRebuild;
    const std::vector<int> old_history = harness.token_history;
    const std::size_t begin_count_before = harness.provider.begin_calls;
    const Prepared extension = prepared(kExtendedTokens, "snapshot-rebuild-failure");

    bool threw = false;
    try {
        harness.process(extension);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "fallback rebuild failure did not propagate to the caller");
    require(harness.provider.fallback_rebuild_failed,
            "test provider did not inject the fresh rebuild failure");
    require(harness.provider.begin_calls == begin_count_before + 1 &&
                harness.provider.begin_snapshot_ids.back() == extension.snapshot_id &&
                harness.provider.begin_used_same_prepared_object.back(),
            "failed fallback did not attempt one rebuild with the same prepared object");
    require(!harness.provider.generator_active && harness.provider.sequence == 0 &&
                harness.provider.checkpoint == 0,
            "failed fallback retained a generator or provider checkpoint");
    require(!harness.prefix_cache_.has_checkpoint() &&
                harness.prefix_cache_.disabled_for_session(),
            "failed fallback retained a reusable media checkpoint");
    require(harness.token_history == old_history &&
                harness.token_history != kExtendedTokens,
            "failed fallback committed new token history");
    require(harness.prompt_tokens == 0,
            "failed fallback committed a new prompt token count");
}

void test_cancellation_after_provider_mutation_commits_nothing() {
    Harness harness;
    seed_prefix(harness);
    const std::vector<int> old_history = harness.token_history;
    const std::size_t checkpoint_calls_before = harness.provider.checkpoint_calls;
    const std::size_t operation_begin = harness.provider.operations.size();
    const Prepared extension = prepared(kExtendedTokens, "snapshot-cancel-after-rewind");
    harness.provider.cancel_after_mutation = true;

    const bool completed = harness.process(extension);
    require(!completed, "post-mutation cancellation did not stop the transaction");
    require(harness.provider.cancelled,
            "cancellation fixture did not trigger after a provider mutation");
    require(harness.meta_info.stop_reason == CANCEL_DETECTED,
            "post-mutation cancellation did not record its stop reason");
    require(harness.provider.checkpoint_calls == checkpoint_calls_before,
            "cancelled transaction advanced to checkpoint commit");
    require(!harness.provider.generator_active && harness.provider.sequence == 0 &&
                harness.provider.checkpoint == 0,
            "cancelled transaction retained provider state");
    require(!harness.prefix_cache_.has_checkpoint(),
            "cancelled transaction retained a media checkpoint");
    require(harness.token_history == old_history &&
                harness.token_history != kExtendedTokens,
            "cancelled transaction committed new token history");
    require(harness.prompt_tokens == 0 && harness.meta_info.cached_prompt_tokens == -1,
            "cancelled transaction committed token accounting");
    require(harness.provider.operations.size() >= operation_begin + 4 &&
                harness.provider.operations[operation_begin].name == "restore_text_checkpoint" &&
                harness.provider.operations[operation_begin + 2].name == "append_text_suffix",
            "cancellation did not occur after provider mutation");
}

// Returning native defaults on a present but malformed file must fail this
// regression. Both the reader and constructor fragment are production bytes.
void test_media_package_file_errors_do_not_become_native_defaults() {
    namespace fs = std::filesystem;
    struct TemporaryTree {
        fs::path path;
        TemporaryTree() {
            const auto stamp = Clock::now().time_since_epoch().count();
            path = fs::temp_directory_path() / ("flm_media_config_" + std::to_string(stamp));
            require(fs::create_directory(path), "could not create metadata test directory");
        }
        ~TemporaryTree() { std::error_code ignored; fs::remove_all(path, ignored); }
    } temporary;
    const auto config_path = temporary.path;
    const auto file_path = config_path / "flm_hip_oga.json";
    const nlohmann::json effective_config = {{"model", {{"type", "gemma3"}}}};
    auto load = [&] {
        bool flat_gemma3_features_ = false;
        std::string media_processor_signature_;
#include "production_media_config.inc"
        return flat_gemma3_features_;
    };
    auto write = [&](const std::string& contents) {
        std::ofstream output(file_path, std::ios::binary | std::ios::trunc);
        output << contents;
        require(static_cast<bool>(output), "could not write metadata fixture");
    };
    require(!load(), "absent metadata must retain native behavior");
    for (const char* contents : {"null", "{}"}) {
        write(contents);
        require(!load(), "null or empty-object metadata must retain native behavior");
    }
    write(R"({"schema_version":1,"image_features":"gemma3_flat_tokens_v1"})");
    require(load(), "valid on-disk flat feature contract must enable conversion");
    for (const char* contents : {"{broken", "", "[]", "42", "true", "\"gemma3_flat_tokens_v1\"",
                                 R"({"schema_version":2,"image_features":"gemma3_flat_tokens_v1"})"}) {
        write(contents);
        require_throws(load, "invalid on-disk metadata must reject model load, not select native defaults");
    }
    fs::remove(file_path);
    fs::create_directory(file_path);
    require_throws(load, "a directory at the metadata path must reject model load");
}

}  // namespace

int main() {
    try {
        test_media_package_file_errors_do_not_become_native_defaults();
        test_flat_media_preparation_normalizes_before_capture_even_with_cache_off();
        test_identity_capture_accepts_official_names_in_any_order();
        test_identity_capture_fails_closed_on_unknown_or_missing_names();
        test_identity_capture_requires_enabled_and_qualified_path();
        test_oga_token_tensor_preserves_signed_ids_in_int32_and_int64();
        test_oga_token_tensor_rejects_int64_values_outside_int_range();
        test_oga_token_tensor_rejects_malformed_shapes();
        test_oga_token_tensor_rejects_unsupported_type_and_null_data();
        test_successful_reuse_commits_truthful_accounting();
        test_changed_media_misses_and_rebuilds_fresh();
        test_missing_identity_keeps_legacy_multimodal_fresh_path();
        test_provider_faults_fallback_once_with_same_prepared_object();
        test_failed_fallback_leaves_no_generator_checkpoint_or_new_history();
        test_cancellation_after_provider_mutation_commits_nothing();
        std::cout << "PASS: HIP OGA media transaction adapter" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
