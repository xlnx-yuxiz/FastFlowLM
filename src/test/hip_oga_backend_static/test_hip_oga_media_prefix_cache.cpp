/// \file test_hip_oga_media_prefix_cache.cpp
/// \brief No-hardware tests for HIP OGA media-aware prefix-cache policy.
#include "AutoModel/hip_oga_prefix_cache.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using flm::hip_oga::MediaCacheIdentity;
using flm::hip_oga::MediaTensorIdentity;
using flm::hip_oga::PrefixCacheDecision;
using flm::hip_oga::PrefixCacheDecisionKind;
using flm::hip_oga::PrefixCacheState;

const std::string kCacheSignature = "model=A;ctx=512;max=8";
const std::vector<int> kCachedTokens{11, 12, 13, 14};
const std::vector<int> kExtendedTokens{11, 12, 13, 14, 15, 16};

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

void require_invalidated(const PrefixCacheState& state,
                         const std::string& reason) {
    require(!state.has_checkpoint(),
            "stale checkpoint survived " + reason);
    require(state.checkpoint_size() == 0,
            "invalidated checkpoint retained tokens for " + reason);
}

MediaCacheIdentity prefix_media_identity() {
    // A 2-float tensor uses ONNX FLOAT type 1, shape [1, 2], and 8 bytes.
    return MediaCacheIdentity{
        {{0x10, 0x11, 0x12}, {0x20, 0x21}},
        "vision-processor:v1",
        {MediaTensorIdentity{"pixel_values", 1, {1, 2},
                             {0x00, 0x00, 0x80, 0x3f,
                              0x00, 0x00, 0x00, 0x40}}},
        {0, 1, 1, 0}};
}

MediaCacheIdentity extended_media_identity() {
    MediaCacheIdentity identity = prefix_media_identity();
    identity.token_types = {0, 1, 1, 0, 0, 0};
    return identity;
}

PrefixCacheDecision evaluate(PrefixCacheState& state,
                             bool enabled,
                             bool restore_allowed,
                             bool text_only,
                             std::string_view signature,
                             std::size_t required_capacity,
                             std::size_t generator_capacity,
                             const std::vector<int>& prompt_tokens,
                             const MediaCacheIdentity* media = nullptr) {
    return state.evaluate(enabled,
                          restore_allowed,
                          text_only,
                          signature,
                          required_capacity,
                          generator_capacity,
                          prompt_tokens,
                          media);
}

void commit_media_prefix(PrefixCacheState& state,
                         const MediaCacheIdentity& identity) {
    state.commit_checkpoint(kCachedTokens, kCacheSignature, 512, identity);
}

PrefixCacheDecision evaluate_media(PrefixCacheState& state,
                                   const std::vector<int>& prompt_tokens,
                                   const MediaCacheIdentity& identity,
                                   bool enabled = true,
                                   bool restore_allowed = true,
                                   std::size_t required_capacity = 128,
                                   std::size_t generator_capacity = 512,
                                   std::string_view signature = kCacheSignature) {
    return evaluate(state,
                    enabled,
                    restore_allowed,
                    false,
                    signature,
                    required_capacity,
                    generator_capacity,
                    prompt_tokens,
                    &identity);
}

// Identical ordered inputs must reuse the saved multimodal KV prefix.
void test_same_ordered_media_prefix_hits() {
    PrefixCacheState state;
    const MediaCacheIdentity cached_media = prefix_media_identity();
    const MediaCacheIdentity extended_media = extended_media_identity();
    commit_media_prefix(state, cached_media);

    const auto decision = evaluate_media(state, kExtendedTokens, extended_media);
    require_hit(decision, 4, 2);
    require(decision.suffix_start == 4,
            "media hit must begin replay at the cached prefix boundary");
}

// Image content or ordering changes must never restore KV for a different visual prompt.
void test_image_changes_invalidate_checkpoint() {
    const MediaCacheIdentity base_media = extended_media_identity();
    std::vector<std::pair<std::string, MediaCacheIdentity>> mutations;

    auto changed_bytes = base_media;
    changed_bytes.images[0][0] ^= 0x01;
    mutations.emplace_back("changed image bytes", std::move(changed_bytes));

    auto swapped_order = base_media;
    std::swap(swapped_order.images[0], swapped_order.images[1]);
    mutations.emplace_back("swapped image order", std::move(swapped_order));

    auto added_image = base_media;
    added_image.images.push_back({0x30, 0x31});
    mutations.emplace_back("added image", std::move(added_image));

    for (const auto& mutation : mutations) {
        PrefixCacheState state;
        commit_media_prefix(state, prefix_media_identity());
        const auto decision = evaluate_media(state, kExtendedTokens, mutation.second);
        require_fresh(decision, "media_changed", kExtendedTokens.size());
        require_invalidated(state, mutation.first);
    }
}

// Preprocessing changes must invalidate cached tensors even when source images match.
void test_preprocessing_changes_invalidate_checkpoint() {
    const MediaCacheIdentity base_media = extended_media_identity();
    std::vector<std::pair<std::string, MediaCacheIdentity>> mutations;

    auto changed_processor = base_media;
    changed_processor.processor_signature = "vision-processor:v2";
    mutations.emplace_back("processor signature", std::move(changed_processor));

    auto changed_name = base_media;
    changed_name.processor_tensors[0].name = "normalized_pixel_values";
    mutations.emplace_back("fixed tensor name", std::move(changed_name));

    auto changed_type = base_media;
    changed_type.processor_tensors[0].element_type = 6;
    mutations.emplace_back("fixed tensor type", std::move(changed_type));

    auto changed_shape = base_media;
    changed_shape.processor_tensors[0].shape = {2, 1};
    mutations.emplace_back("fixed tensor shape", std::move(changed_shape));

    auto changed_tensor_bytes = base_media;
    changed_tensor_bytes.processor_tensors[0].bytes[0] ^= 0x01;
    mutations.emplace_back("fixed tensor bytes", std::move(changed_tensor_bytes));

    for (const auto& mutation : mutations) {
        PrefixCacheState state;
        commit_media_prefix(state, prefix_media_identity());
        const auto decision = evaluate_media(state, kExtendedTokens, mutation.second);
        require_fresh(decision, "preprocessing_changed", kExtendedTokens.size());
        require_invalidated(state, mutation.first);
    }
}

// Token-to-media alignment changes must force a full prefill.
void test_media_layout_changes_invalidate_checkpoint() {
    auto changed_prefix_types = extended_media_identity();
    changed_prefix_types.token_types[2] = 0;

    PrefixCacheState prefix_state;
    commit_media_prefix(prefix_state, prefix_media_identity());
    const auto prefix_decision =
        evaluate_media(prefix_state, kExtendedTokens, changed_prefix_types);
    require_fresh(prefix_decision, "media_layout_changed", kExtendedTokens.size());
    require_invalidated(prefix_state, "changed prefix token types");

    auto changed_suffix_type = extended_media_identity();
    changed_suffix_type.token_types[4] = 1;

    PrefixCacheState suffix_state;
    commit_media_prefix(suffix_state, prefix_media_identity());
    const auto suffix_decision =
        evaluate_media(suffix_state, kExtendedTokens, changed_suffix_type);
    require_fresh(suffix_decision, "media_layout_changed", kExtendedTokens.size());
    require_invalidated(suffix_state, "nonzero suffix token type");
}

// Malformed identities must fail closed rather than accidentally matching cached input.
void test_invalid_media_identities_invalidate_checkpoint() {
    std::vector<std::pair<std::string, MediaCacheIdentity>> malformed;

    auto no_images = extended_media_identity();
    no_images.images.clear();
    malformed.emplace_back("empty image list", std::move(no_images));

    auto empty_image = extended_media_identity();
    empty_image.images[0].clear();
    malformed.emplace_back("empty image payload", std::move(empty_image));

    auto no_processor_signature = extended_media_identity();
    no_processor_signature.processor_signature.clear();
    malformed.emplace_back("empty processor signature", std::move(no_processor_signature));

    auto no_tensors = extended_media_identity();
    no_tensors.processor_tensors.clear();
    malformed.emplace_back("empty fixed tensor list", std::move(no_tensors));

    auto wrong_type_count = extended_media_identity();
    wrong_type_count.token_types.pop_back();
    malformed.emplace_back("wrong token type count", std::move(wrong_type_count));

    for (const auto& candidate : malformed) {
        PrefixCacheState state;
        commit_media_prefix(state, prefix_media_identity());
        const auto decision = evaluate_media(state, kExtendedTokens, candidate.second);
        require_fresh(decision, "invalid_media_identity", kExtendedTokens.size());
        require_invalidated(state, candidate.first);
    }
}

// Switching between text and media cache domains must never reuse the prior domain.
void test_input_domain_changes_invalidate_checkpoint() {
    PrefixCacheState text_state;
    text_state.commit_checkpoint(kCachedTokens, kCacheSignature, 512);
    const MediaCacheIdentity media = extended_media_identity();
    const auto text_to_media = evaluate(text_state,
                                       true,
                                       true,
                                       false,
                                       kCacheSignature,
                                       128,
                                       512,
                                       kExtendedTokens,
                                       &media);
    require_fresh(text_to_media, "input_domain_changed", kExtendedTokens.size());
    require_invalidated(text_state, "text-to-media domain change");

    PrefixCacheState media_state;
    commit_media_prefix(media_state, prefix_media_identity());
    const auto media_to_text = evaluate(media_state,
                                       true,
                                       true,
                                       true,
                                       kCacheSignature,
                                       128,
                                       512,
                                       kExtendedTokens);
    require_fresh(media_to_text, "input_domain_changed", kExtendedTokens.size());
    require_invalidated(media_state, "media-to-text domain change");
}

// A media prompt with no appended tokens retains the existing empty-suffix safety rule.
void test_identical_media_prompt_has_empty_suffix() {
    PrefixCacheState state;
    const MediaCacheIdentity media = prefix_media_identity();
    commit_media_prefix(state, media);

    const auto decision = evaluate_media(state, kCachedTokens, media);
    require_fresh(decision, "empty_suffix", kCachedTokens.size());
    require_invalidated(state, "identical media prompt");
}

// Existing prefix, cache-signature, and capacity failures must keep exact fresh accounting.
void test_existing_fresh_reasons_keep_media_accounting() {
    PrefixCacheState prefix_state;
    commit_media_prefix(prefix_state, prefix_media_identity());
    const MediaCacheIdentity extended_media = extended_media_identity();
    auto divergent_tokens = kExtendedTokens;
    divergent_tokens[2] = 99;
    const auto prefix_decision =
        evaluate_media(prefix_state, divergent_tokens, extended_media);
    require_fresh(prefix_decision, "prefix_diverged", divergent_tokens.size());
    require_invalidated(prefix_state, "changed token prefix");

    PrefixCacheState signature_state;
    commit_media_prefix(signature_state, prefix_media_identity());
    const auto signature_decision = evaluate_media(signature_state,
                                                   kExtendedTokens,
                                                   extended_media,
                                                   true,
                                                   true,
                                                   128,
                                                   512,
                                                   "model=A;ctx=512;max=8;temperature=0.2");
    require_fresh(signature_decision, "signature_changed", kExtendedTokens.size());
    require_invalidated(signature_state, "changed cache signature");

    PrefixCacheState capacity_state;
    commit_media_prefix(capacity_state, prefix_media_identity());
    const auto capacity_decision = evaluate_media(capacity_state,
                                                  kExtendedTokens,
                                                  extended_media,
                                                  true,
                                                  true,
                                                  513,
                                                  512);
    require_fresh(capacity_decision, "capacity_shortage", kExtendedTokens.size());
    require_invalidated(capacity_state, "insufficient generator capacity");
}

// Disabled feature and unavailable checkpoint restore permission must never hit.
void test_disabled_or_restore_disallowed_never_hits() {
    const MediaCacheIdentity media = prefix_media_identity();
    const MediaCacheIdentity extended_media = extended_media_identity();

    PrefixCacheState disabled_state;
    commit_media_prefix(disabled_state, media);
    const auto disabled = evaluate_media(disabled_state,
                                         kExtendedTokens,
                                         extended_media,
                                         false);
    require_fresh(disabled, "disabled", kExtendedTokens.size());

    PrefixCacheState restore_state;
    commit_media_prefix(restore_state, media);
    const auto restore_disallowed = evaluate_media(restore_state,
                                                   kExtendedTokens,
                                                   extended_media,
                                                   true,
                                                   false);
    require_fresh(restore_disallowed,
                  "prompt_cache_miss",
                  kExtendedTokens.size());
}

// The legacy no-identity multimodal call keeps forcing a full prefill.
void test_legacy_multimodal_without_identity_forces_miss() {
    PrefixCacheState state;
    state.commit_checkpoint(kCachedTokens, kCacheSignature, 512);

    const auto decision = evaluate(state,
                                   true,
                                   true,
                                   false,
                                   kCacheSignature,
                                   128,
                                   512,
                                   kExtendedTokens);
    require_fresh(decision,
                  "multimodal_forced_miss",
                  kExtendedTokens.size());
    require_invalidated(state, "legacy multimodal request without identity");
}

// Session disable remains sticky through commits; reset clears state before fresh reuse.
void test_session_disable_and_reset_with_media() {
    PrefixCacheState state;
    state.disable_for_session("rewind failure");
    commit_media_prefix(state, prefix_media_identity());

    const auto disabled =
        evaluate_media(state, kExtendedTokens, extended_media_identity());
    require_fresh(disabled, "session_disabled", kExtendedTokens.size());
    require(state.disabled_for_session(),
            "media commit must not clear the session-disabled flag");

    state.reset_session();
    require(!state.disabled_for_session(),
            "reset_session must re-enable prefix reuse");
    require_invalidated(state, "reset_session");
    const auto no_checkpoint =
        evaluate_media(state, kExtendedTokens, extended_media_identity());
    require_fresh(no_checkpoint, "no_checkpoint", kExtendedTokens.size());

    auto new_media = prefix_media_identity();
    new_media.images[0] = {0x70, 0x71, 0x72};
    commit_media_prefix(state, new_media);
    auto new_extended_media = new_media;
    new_extended_media.token_types = {0, 1, 1, 0, 0, 0};
    require_hit(evaluate_media(state, kExtendedTokens, new_extended_media), 4, 2);
}

// A text commit after clearing media must establish the text domain without stale media identity.
void test_text_commit_after_media_clear_drops_old_identity() {
    PrefixCacheState state;
    commit_media_prefix(state, prefix_media_identity());
    state.reset_checkpoint("replace media checkpoint");
    require(!state.has_checkpoint(),
            "explicit checkpoint clear must remove old media prefix");

    state.commit_checkpoint(kCachedTokens, kCacheSignature, 512, std::nullopt);
    const MediaCacheIdentity media = extended_media_identity();
    const auto media_request =
        evaluate(state,
                 true,
                 true,
                 false,
                 kCacheSignature,
                 128,
                 512,
                 kExtendedTokens,
                 &media);
    require_fresh(media_request, "input_domain_changed", kExtendedTokens.size());
    require_invalidated(state, "text commit after media clear");
}

}  // namespace

int main() {
    test_same_ordered_media_prefix_hits();
    test_image_changes_invalidate_checkpoint();
    test_preprocessing_changes_invalidate_checkpoint();
    test_media_layout_changes_invalidate_checkpoint();
    test_invalid_media_identities_invalidate_checkpoint();
    test_input_domain_changes_invalidate_checkpoint();
    test_identical_media_prompt_has_empty_suffix();
    test_existing_fresh_reasons_keep_media_accounting();
    test_disabled_or_restore_disallowed_never_hits();
    test_legacy_multimodal_without_identity_forces_miss();
    test_session_disable_and_reset_with_media();
    test_text_commit_after_media_clear_drops_old_identity();

    std::cout << "PASS: HIP OGA media prefix-cache policy" << std::endl;
    return 0;
}
