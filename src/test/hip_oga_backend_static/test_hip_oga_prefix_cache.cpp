/// \file test_hip_oga_prefix_cache.cpp
/// \brief No-hardware tests for the experimental HIP OGA prefix-cache policy.
#include "AutoModel/hip_oga_prefix_cache.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using flm::hip_oga::PrefixCacheDecision;
using flm::hip_oga::PrefixCacheDecisionKind;
using flm::hip_oga::PrefixCacheState;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void require_fresh(const PrefixCacheDecision& decision,
                   const std::string& reason,
                   std::size_t newly_prefilled_tokens) {
    require(decision.kind == PrefixCacheDecisionKind::Fresh,
            "expected fresh decision for " + reason);
    require(!decision.reuse, "fresh decision must not reuse for " + reason);
    require(decision.reason == reason,
            "unexpected fallback reason: " + decision.reason);
    require(decision.cached_tokens == 0,
            "fresh decision must report zero cached tokens for " + reason);
    require(decision.newly_prefilled_tokens == newly_prefilled_tokens,
            "fresh decision reported wrong prefill count for " + reason);
}

void require_hit(const PrefixCacheDecision& decision,
                 std::size_t cached_tokens,
                 std::size_t newly_prefilled_tokens) {
    require(decision.kind == PrefixCacheDecisionKind::Reuse,
            "expected reuse decision");
    require(decision.reuse, "reuse decision must set reuse=true");
    require(decision.reason == "hit", "reuse decision should report hit");
    require(decision.cached_tokens == cached_tokens,
            "reuse decision reported wrong cached token count");
    require(decision.suffix_start == cached_tokens,
            "reuse decision reported wrong suffix start");
    require(decision.newly_prefilled_tokens == newly_prefilled_tokens,
            "reuse decision reported wrong new prefill count");
}

PrefixCacheDecision decide(PrefixCacheState& state,
                           bool enabled,
                           bool restore_allowed,
                           bool text_only,
                           const std::string& signature,
                           std::size_t required_capacity,
                           std::size_t generator_capacity,
                           const std::vector<int>& tokens) {
    return state.evaluate(enabled,
                          restore_allowed,
                          text_only,
                          signature,
                          required_capacity,
                          generator_capacity,
                          tokens);
}

void test_enabled_same_prefix_hit() {
    PrefixCacheState state;
    state.commit_checkpoint({11, 12, 13}, "model=A;ctx=512;max=8", 512);

    PrefixCacheDecision decision = decide(state,
                                          true,
                                          true,
                                          true,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {11, 12, 13, 14, 15});

    require_hit(decision, 3, 2);
}

void test_disabled_misses_even_when_prompt_cache_hit() {
    PrefixCacheState state;
    state.commit_checkpoint({21, 22, 23}, "model=A;ctx=512;max=8", 512);

    PrefixCacheDecision decision = decide(state,
                                          false,
                                          true,
                                          true,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {21, 22, 23, 24});

    require_fresh(decision, "disabled", 4);
}

void test_divergent_prefix_resets_to_fresh_prompt() {
    PrefixCacheState state;
    state.commit_checkpoint({31, 32, 33}, "model=A;ctx=512;max=8", 512);

    PrefixCacheDecision decision = decide(state,
                                          true,
                                          true,
                                          true,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {31, 32, 99, 34});

    require_fresh(decision, "prefix_diverged", 4);
    require(!state.has_checkpoint(),
            "divergent prompt must drop the stale checkpoint");
}

void test_fully_cached_prompt_uses_fresh_prefill() {
    PrefixCacheState state;
    state.commit_checkpoint({41, 42, 43}, "model=A;ctx=512;max=8", 512);

    PrefixCacheDecision decision = decide(state,
                                          true,
                                          true,
                                          true,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {41, 42, 43});

    require_fresh(decision, "empty_suffix", 3);
    require(decision.attempted, "identical prompt should record attempted reuse");
    require(!state.has_checkpoint(), "identical prompt must rebuild checkpoint");
    require(!state.disabled_for_session(), "empty suffix must not disable the session");

    state.commit_checkpoint({41, 42, 43}, "model=A;ctx=512;max=8", 512);
    const auto extension = decide(state, true, true, true,
        "model=A;ctx=512;max=8", 128, 512, {41, 42, 43, 44});
    require_hit(extension, 3, 1);
}

void test_signature_and_capacity_invalidate() {
    PrefixCacheState state;
    state.commit_checkpoint({51, 52}, "model=A;ctx=512;max=8;temperature=0.8", 64);

    PrefixCacheDecision signature_miss = decide(
        state,
        true,
        true,
        true,
        "model=A;ctx=512;max=8;temperature=0.2",
        64,
        64,
        {51, 52, 53});
    require_fresh(signature_miss, "signature_changed", 3);

    state.commit_checkpoint({51, 52}, "model=A;ctx=512;max=8;temperature=0.2", 64);
    PrefixCacheDecision capacity_miss = decide(
        state,
        true,
        true,
        true,
        "model=A;ctx=512;max=8;temperature=0.2",
        96,
        64,
        {51, 52, 53});
    require_fresh(capacity_miss, "capacity_shortage", 3);
}

void test_provider_exception_disables_reuse_for_session() {
    PrefixCacheState state;
    state.commit_checkpoint({61, 62}, "model=A;ctx=512;max=8", 512);

    state.disable_for_session("RewindTo failed");
    PrefixCacheDecision decision = decide(state,
                                          true,
                                          true,
                                          true,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {61, 62, 63});

    require_fresh(decision, "session_disabled", 3);
    require(state.disabled_for_session(),
            "provider exception must keep reuse disabled for the session");

    state.reset_session();
    state.commit_checkpoint({61, 62}, "model=A;ctx=512;max=8", 512);
    require_hit(decide(state,
                       true,
                       true,
                       true,
                       "model=A;ctx=512;max=8",
                       128,
                       512,
                       {61, 62, 63}),
                2,
                1);
}

void test_vlm_request_forces_miss_and_clears_checkpoint() {
    PrefixCacheState state;
    state.commit_checkpoint({71, 72}, "model=A;ctx=512;max=8", 512);

    PrefixCacheDecision decision = decide(state,
                                          true,
                                          true,
                                          false,
                                          "model=A;ctx=512;max=8",
                                          128,
                                          512,
                                          {71, 72, 73});

    require_fresh(decision, "multimodal_forced_miss", 3);
    require(!state.has_checkpoint(),
            "VLM/image/audio request must invalidate the text checkpoint");
}

void test_env_is_experimental_only() {
    require(!flm::hip_oga::prefix_cache_env_enabled(nullptr),
            "unset env must leave prefix cache disabled");
    require(!flm::hip_oga::prefix_cache_env_enabled("1"),
            "generic truthy env must not enable experimental prefix cache");
    require(!flm::hip_oga::prefix_cache_env_enabled("true"),
            "true env must not enable experimental prefix cache");
    require(flm::hip_oga::prefix_cache_env_enabled("experimental"),
            "experimental env must enable prefix cache");
}

}  // namespace

int main() {
    test_enabled_same_prefix_hit();
    test_disabled_misses_even_when_prompt_cache_hit();
    test_divergent_prefix_resets_to_fresh_prompt();
    test_fully_cached_prompt_uses_fresh_prefill();
    test_signature_and_capacity_invalidate();
    test_provider_exception_disables_reuse_for_session();
    test_vlm_request_forces_miss_and_clears_checkpoint();
    test_env_is_experimental_only();

    std::cout << "PASS: HIP OGA prefix-cache policy" << std::endl;
    return 0;
}
