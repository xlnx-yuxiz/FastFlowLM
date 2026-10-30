#include "AutoModel/automodel.hpp"
#include "sampling/request.hpp"
#include "test_support.hpp"

#include <deque>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>

namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("lifecycle-test", "flm", [](const BackendContext&)
        -> std::unique_ptr<ModelBackend> {
        throw std::runtime_error("test factory must not be reached");
    });
}
}

namespace {
struct Observation {
    int destructions = 0;
    bool view_cleared = false;
    bool config_alive = false;
    bool npu_alive = false;
};

class EmptyEngine final : public causal_lm {
public:
    buffer<bf16> forward(int) override { return {}; }
    buffer<bf16> prefill(std::vector<int>&, void*) override { return {}; }
    void set_context_length(int) override {}
    void load_weights(Q4NX&) override {}
    void update_max_length(uint32_t) override {}
    void clear_context() override {}
    buffer<bf16> get_k_cache(int, int) override { return {}; }
    buffer<bf16> get_v_cache(int, int) override { return {}; }
    int get_current_context_length() override { return 0; }
    int checkpoint() override { return 0; }
    int restore() override { return 0; }
};

class ObservedBackend final : public flm::backend::ModelBackend {
public:
    ObservedBackend(Observation& result, causal_lm*& view,
                    const std::unique_ptr<LM_Config>& config,
                    const std::unique_ptr<npu_xclbin_manager>& npu)
        : result_(result), view_(view), config_(config), npu_(npu) {}
    ~ObservedBackend() override {
        ++result_.destructions;
        result_.view_cleared = view_ == nullptr;
        result_.config_alive = config_ != nullptr;
        result_.npu_alive = npu_ != nullptr;
    }
    causal_lm& engine() override { return engine_; }
    std::string id() const override { return "flm"; }
private:
    Observation& result_;
    causal_lm*& view_;
    const std::unique_ptr<LM_Config>& config_;
    const std::unique_ptr<npu_xclbin_manager>& npu_;
    EmptyEngine engine_;
};

sampler_config small_sampler_config(int top_k = 1) {
    sampler_config config;
    config.top_k = top_k;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    config.freq_penalty = 0.0f;
    config.pre_penalty = 0.0f;
    config.rep_penalty_window = 8;
    config.freq_penalty_window = 8;
    config.repeat_last_n = 8;
    config.use_optimized_sampling = true;
    config.has_rng_seed = true;
    config.rng_seed = 17;
    return config;
}

class TestModel : public AutoModel {
public:
    std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>) override {
        throw std::logic_error("lifecycle test does not generate");
    }
    bool insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) override {
        throw std::logic_error("lifecycle test does not insert");
    }
    std::string generate_with_prompt(chat_meta_info_t&, lm_uniform_input_t&, int, std::ostream&) override {
        throw std::logic_error("lifecycle test does not generate");
    }
    std::string apply_chat_template(nlohmann::ordered_json&, nlohmann::ordered_json) override {
        throw std::logic_error("lifecycle test does not render templates");
    }
    explicit TestModel(Observation& result, flm_rt::device* device = nullptr,
                       sampler_config sampling_config = small_sampler_config())
        : AutoModel(device) {
        lm_config = std::make_unique<LM_Config>();
        lm_config->_json_config["vocab_size"] = 3;
        // This constructor only stores the device and reserves an empty list.
        // No xclbin is registered and no device memory is allocated.
        npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, device, false);
        backend_ = std::make_unique<ObservedBackend>(result, lm_engine, lm_config, npu);
        lm_engine = &backend_->engine();
        is_model_loaded = true;
        model_path = "previous-model";
        token_history = {1, 2};
        checkpoint_his = {1};
        eos_token_ids = {3};
        buffer_ = "pending output";
        tool_name_ = "pending tool";
        is_in_tool_block_ = true;
        current_mode_ = StreamEventType::REASONING;
        MAX_L = 42;
        last_token = 2;
        total_tokens = 2;
        set_sampler(sampling_config);
    }
    using AutoModel::_release_model_state;
    using AutoModel::_shared_load_backend;
    void set_logical_history(std::vector<int> history) {
        token_history = std::move(history);
    }
    std::vector<int>& logical_history() { return token_history; }
    Sampler& native_sampler() { return *sampler; }
    void begin_turn() { begin_sampling_turn(); }
    int sample(const std::vector<float>& scores) {
        TEST_REQUIRE(scores.size() == lm_config->get<int>("vocab_size"));
        std::vector<bf16> values;
        values.reserve(scores.size());
        for (float score : scores) values.emplace_back(score);
        buffer<bf16> row(values);
        return sampler->sample(row);
    }
    bool state_empty() const {
        return !lm_engine && !backend_ && !npu && !lm_config && !is_model_loaded &&
            model_path.empty() && token_history.empty() && checkpoint_his.empty() &&
            eos_token_ids.empty() && buffer_.empty() && tool_name_.empty() &&
            !is_in_tool_block_ && current_mode_ == StreamEventType::CONTENT &&
            MAX_L == 0 && last_token == -1 && total_tokens == 0;
    }
};

class UnsupportedCommonModel final : public TestModel {
public:
    using TestModel::TestModel;
protected:
    bool supports_common_sampling() const override { return false; }
};

struct LegacySamplerState {
    int top_k;
    float top_p;
    float min_p;
    float temperature;
    float rep_penalty;
    float freq_penalty;
    float pre_penalty;
    size_t rep_penalty_window;
    size_t freq_penalty_window;
    size_t repeat_last_n;
    bool use_optimized_sampling;
    int total_tokens;
    std::deque<int> token_history;
    std::vector<int> counters;
    std::vector<int> token_positions;
    std::unordered_map<int, int> token_counts_sparse;
    std::mt19937_64 rng;
};

LegacySamplerState snapshot(const Sampler& sampler) {
    return {sampler.top_k, sampler.top_p, sampler.min_p, sampler.temperature,
            sampler.rep_penalty, sampler.freq_penalty, sampler.pre_penalty,
            sampler.rep_penalty_window, sampler.freq_penalty_window,
            sampler.repeat_last_n, sampler.use_optimized_sampling,
            sampler.total_tokens, sampler.token_history, sampler.counters,
            sampler.token_positions, sampler.token_counts_sparse, sampler.rng_};
}

bool matches(const Sampler& sampler, const LegacySamplerState& state) {
    return sampler.top_k == state.top_k && sampler.top_p == state.top_p &&
           sampler.min_p == state.min_p && sampler.temperature == state.temperature &&
           sampler.rep_penalty == state.rep_penalty &&
           sampler.freq_penalty == state.freq_penalty &&
           sampler.pre_penalty == state.pre_penalty &&
           sampler.rep_penalty_window == state.rep_penalty_window &&
           sampler.freq_penalty_window == state.freq_penalty_window &&
           sampler.repeat_last_n == state.repeat_last_n &&
           sampler.use_optimized_sampling == state.use_optimized_sampling &&
           sampler.total_tokens == state.total_tokens &&
           sampler.token_history == state.token_history &&
           sampler.counters == state.counters &&
           sampler.token_positions == state.token_positions &&
           sampler.token_counts_sparse == state.token_counts_sparse &&
           sampler.rng_ == state.rng;
}

void require_order(const Observation& result) {
    TEST_REQUIRE(result.destructions == 1);
    TEST_REQUIRE(result.view_cleared);
    TEST_REQUIRE(result.config_alive);
    TEST_REQUIRE(result.npu_alive);
}

void test_explicit_reset() {
    Observation result;
    TestModel model(result);
    TEST_REQUIRE(model.backend_id() == "flm");
    model._release_model_state();
    require_order(result);
    TEST_REQUIRE(model.state_empty());
    TEST_REQUIRE(model.backend_id().empty());
    model._release_model_state();
    TEST_REQUIRE(result.destructions == 1);
}

void test_base_destruction() {
    Observation result;
    std::unique_ptr<AutoModel> model = std::make_unique<TestModel>(result);
    model.reset();
    require_order(result);
}

void test_rejected_selection_keeps_previous_model() {
    Observation result;
    TestModel model(result);
    RequireThrows([&] {
        model._shared_load_backend("unused", {{"details", {{"family", "lifecycle-test"}}}},
                                   16, false, "missing");
    });
    TEST_REQUIRE(result.destructions == 0);
    TEST_REQUIRE(model.backend_id() == "flm");
}

void test_failed_replacement_clears_identity() {
    Observation result;
    flm_rt::device device; // empty handle; does not open a device
    TestModel model(result, &device);
    const auto path = std::filesystem::temp_directory_path() /
        ("flm-lifecycle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(path);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } cleanup{path};
    std::ofstream(path / "config.json") << "{ invalid json";
    RequireThrows([&] {
        model._shared_load_backend(path.string(), {{"details", {{"family", "lifecycle-test"}}}},
                                   16, false, "flm");
    });
    require_order(result);
    TEST_REQUIRE(model.state_empty());
    TEST_REQUIRE(model.backend_id().empty());
}

void test_startup_common_default_survives_backend_request_override() {
    Observation result;
    auto config = small_sampler_config();
    config.rep_penalty = 2.0f;
    TestModel model(result, nullptr, config);
    model.set_sampling_provider("common");
    model.set_logical_history({0});

    model.configure_sampling_request({{"sampler", "backend"}}, true);
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 0.0f}) == 0);

    // Clear only the legacy sampler's continuation ring between simulated HTTP
    // requests. The AutoModel history remains caller-owned and drives common mode.
    model.native_sampler().reset_penalties();
    model.configure_sampling_request(nlohmann::json::object(), true);
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 0.0f}) == 1);
}

void test_common_request_reset_restores_captured_model_defaults() {
    Observation result;
    TestModel model(result);
    model.set_sampling_provider("common");
    model.set_logical_history({0});

    model.configure_sampling_request({{"repetition_penalty", 2.0}}, true);
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 0.0f}) == 1);

    model.configure_sampling_request(nlohmann::json::object(), true);
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 0.0f}) == 0);
}

void test_cli_style_request_update_accumulates_common_options() {
    Observation result;
    TestModel model(result);
    model.set_sampling_provider("common");
    model.set_logical_history({0});

    model.configure_sampling_request({{"repetition_penalty", 2.0}}, true);
    model.configure_sampling_request({{"frequency_penalty", 3.0}}, false);
    model.begin_turn();
    // With both options token 0 scores 10/2 - 3 = 2, below token 1's 7.
    // If the update discarded repetition_penalty, token 0 and token 1 would tie
    // at 7 and the lower ID would win instead.
    TEST_REQUIRE(model.sample({10.0f, 7.0f, 0.0f}) == 1);
}

void test_invalid_common_update_leaves_previous_request_atomic() {
    Observation result;
    TestModel model(result);
    model.set_sampling_provider("common");
    model.set_logical_history({0});
    model.configure_sampling_request({{"repetition_penalty", 2.0}}, true);

    RequireThrows<flm::sampling::SamplingRequestError>([&] {
        model.configure_sampling_request(
            {{"frequency_penalty", -5.0}, {"top_k", 4}}, false);
    });

    model.begin_turn();
    // The rejected negative frequency update would reward token 0 and make it
    // win if it leaked into the active request.
    TEST_REQUIRE(model.sample({10.0f, 7.0f, 0.0f}) == 1);
}

void test_common_seed_zero_replays_after_turn_reset_and_intervening_request() {
    Observation result;
    auto config = small_sampler_config(3);
    TestModel model(result, nullptr, config);
    model.set_sampling_provider("common");
    model.set_logical_history({});
    const std::vector<float> scores{3.0f, 2.0f, 1.0f};

    model.configure_sampling_request({{"seed", 0}}, true);
    model.begin_turn();
    std::vector<int> first;
    for (int i = 0; i < 32; ++i) first.push_back(model.sample(scores));

    model.configure_sampling_request({{"seed", 1234}}, true);
    model.begin_turn();
    for (int i = 0; i < 9; ++i) (void)model.sample(scores);

    model.configure_sampling_request({{"seed", 0}}, true);
    model.begin_turn();
    std::vector<int> replay;
    for (int i = 0; i < 32; ++i) replay.push_back(model.sample(scores));
    TEST_REQUIRE(first == replay);
}

void test_common_request_does_not_mutate_legacy_sampler_state() {
    Observation result;
    auto config = small_sampler_config();
    config.top_p = 0.75f;
    config.min_p = 0.125f;
    config.temperature = 0.8f;
    config.rep_penalty = 1.5f;
    config.freq_penalty = 0.25f;
    config.pre_penalty = 0.125f;
    config.rep_penalty_window = 5;
    config.freq_penalty_window = 6;
    config.repeat_last_n = 7;
    TestModel model(result, nullptr, config);
    model.set_sampling_provider("backend");
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 8.0f}) == 0);
    model.native_sampler().set_seed(0x5eed);
    const auto legacy_before = snapshot(model.native_sampler());

    model.set_logical_history({0, 0});
    model.configure_sampling_request(
        {{"sampler", "common"}, {"do_sample", false}, {"temperature", 1.0},
         {"top_k", 3}, {"top_p", 1.0}, {"min_p", 0.0},
         {"repetition_penalty", 2.0}, {"presence_penalty", 0.5},
         {"frequency_penalty", 0.25}, {"seed", 0}}, true);
    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 8.0f}) == 1);
    TEST_REQUIRE(matches(model.native_sampler(), legacy_before));
}

void test_subclass_without_common_support_rejects_explicit_selection() {
    Observation result;
    auto config = small_sampler_config();
    config.rep_penalty = 2.0f;
    UnsupportedCommonModel model(result, nullptr, config);
    model.set_logical_history({0});

    RequireThrows<flm::sampling::SamplingRequestError>([&] {
        model.set_sampling_provider("common");
    });
    RequireThrows<flm::sampling::SamplingRequestError>([&] {
        model.configure_sampling_request({{"sampler", "common"}}, true);
    });

    model.begin_turn();
    TEST_REQUIRE(model.sample({10.0f, 9.0f, 0.0f}) == 0);
}
}

int main() {
    RunTest(test_explicit_reset, "AutoModel reset releases owner before dependencies");
    RunTest(test_base_destruction, "AutoModel base destructor preserves dependency order");
    RunTest(test_rejected_selection_keeps_previous_model, "invalid selection preserves existing model");
    RunTest(test_failed_replacement_clears_identity, "failed replacement clears backend identity and parser state");
    RunTest(test_startup_common_default_survives_backend_request_override, "startup common provider returns after a backend request override");
    RunTest(test_common_request_reset_restores_captured_model_defaults, "common request reset restores captured sampler defaults");
    RunTest(test_cli_style_request_update_accumulates_common_options, "CLI style common updates accumulate request options");
    RunTest(test_invalid_common_update_leaves_previous_request_atomic, "invalid common request update is atomic");
    RunTest(test_common_seed_zero_replays_after_turn_reset_and_intervening_request, "common seed zero replays after turn reset");
    RunTest(test_common_request_does_not_mutate_legacy_sampler_state, "common requests leave legacy sampler fields and state unchanged");
    RunTest(test_subclass_without_common_support_rejects_explicit_selection, "unsupported subclass rejects common sampling");
    return FailureCount() ? 1 : 0;
}
