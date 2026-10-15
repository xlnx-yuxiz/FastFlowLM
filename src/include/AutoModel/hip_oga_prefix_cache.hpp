/// \file hip_oga_prefix_cache.hpp
/// \brief No-hardware policy state for experimental HIP OGA prompt KV reuse.
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flm::hip_oga {

enum class PrefixCacheDecisionKind {
    Fresh,
    Reuse,
};

struct PrefixCacheDecision {
    PrefixCacheDecisionKind kind = PrefixCacheDecisionKind::Fresh;
    bool reuse = false;
    bool attempted = false;
    std::string reason = "disabled";
    std::size_t cached_tokens = 0;
    std::size_t suffix_start = 0;
    std::size_t newly_prefilled_tokens = 0;
};

inline bool prefix_cache_env_enabled(const char* value) {
    return value != nullptr && std::string_view(value) == "experimental";
}

class PrefixCacheState {
public:
    PrefixCacheDecision evaluate(bool enabled,
                                 bool restore_allowed,
                                 bool text_only,
                                 std::string_view signature,
                                 std::size_t required_capacity,
                                 std::size_t generator_capacity,
                                 const std::vector<int>& prompt_tokens) {
        PrefixCacheDecision decision;
        decision.newly_prefilled_tokens = prompt_tokens.size();

        if (!enabled) {
            decision.reason = "disabled";
            return decision;
        }
        if (!text_only) {
            reset_checkpoint("multimodal_forced_miss");
            decision.reason = "multimodal_forced_miss";
            return decision;
        }
        if (!restore_allowed) {
            decision.reason = "prompt_cache_miss";
            return decision;
        }

        decision.attempted = true;
        if (disabled_for_session_) {
            decision.reason = "session_disabled";
            return decision;
        }
        if (checkpoint_tokens_.empty()) {
            decision.reason = "no_checkpoint";
            return decision;
        }
        if (signature_ != signature) {
            reset_checkpoint("signature_changed");
            decision.reason = "signature_changed";
            return decision;
        }
        if (generator_capacity < required_capacity) {
            reset_checkpoint("capacity_shortage");
            decision.reason = "capacity_shortage";
            return decision;
        }
        if (prompt_tokens.size() < checkpoint_tokens_.size() ||
            !std::equal(checkpoint_tokens_.begin(),
                        checkpoint_tokens_.end(),
                        prompt_tokens.begin())) {
            reset_checkpoint("prefix_diverged");
            decision.reason = "prefix_diverged";
            return decision;
        }

        decision.kind = PrefixCacheDecisionKind::Reuse;
        decision.reuse = true;
        decision.reason = "hit";
        decision.cached_tokens = checkpoint_tokens_.size();
        decision.suffix_start = checkpoint_tokens_.size();
        decision.newly_prefilled_tokens =
            prompt_tokens.size() - checkpoint_tokens_.size();
        return decision;
    }

    void commit_checkpoint(std::vector<int> prompt_tokens,
                           std::string signature,
                           std::size_t generator_capacity) {
        checkpoint_tokens_ = std::move(prompt_tokens);
        signature_ = std::move(signature);
        generator_capacity_ = generator_capacity;
        last_reason_ = "checkpoint";
    }

    void reset_checkpoint(std::string reason) {
        checkpoint_tokens_.clear();
        signature_.clear();
        generator_capacity_ = 0;
        last_reason_ = std::move(reason);
    }

    void reset_session() {
        disabled_for_session_ = false;
        reset_checkpoint("new_session");
    }

    void disable_for_session(std::string reason) {
        disabled_for_session_ = true;
        reset_checkpoint(std::move(reason));
    }

    bool has_checkpoint() const {
        return !checkpoint_tokens_.empty();
    }

    bool disabled_for_session() const {
        return disabled_for_session_;
    }

    std::size_t checkpoint_size() const {
        return checkpoint_tokens_.size();
    }

    std::size_t generator_capacity() const {
        return generator_capacity_;
    }

    const std::string& last_reason() const {
        return last_reason_;
    }

private:
    std::vector<int> checkpoint_tokens_;
    std::string signature_;
    std::size_t generator_capacity_ = 0;
    bool disabled_for_session_ = false;
    std::string last_reason_ = "new_session";
};

}  // namespace flm::hip_oga
