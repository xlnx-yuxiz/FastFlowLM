// Backend-independent request sampling configuration.
#pragma once

#include "sampling/logits.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace flm::sampling {

class SamplingRequestError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

enum class Provider { Backend, Common };

inline Provider parse_provider(const std::string& name) {
    if (name == "backend") return Provider::Backend;
    if (name == "common") return Provider::Common;
    throw SamplingRequestError("sampler must be backend or common");
}

inline std::string provider_name(Provider provider) {
    switch (provider) {
    case Provider::Backend: return "backend";
    case Provider::Common: return "common";
    }
    throw SamplingRequestError("Invalid sampler provider");
}

inline Provider json_provider(const nlohmann::json& value) {
    if (!value.is_string()) throw SamplingRequestError("sampler must be a string");
    return parse_provider(value.get<std::string>());
}

inline bool is_sampling_field(const std::string& name) {
    return name == "temperature" || name == "top_k" || name == "top_p" ||
           name == "min_p" || name == "repetition_penalty" ||
           name == "presence_penalty" || name == "frequency_penalty" ||
           name == "seed" || name == "do_sample" || name == "sampler";
}

inline int parse_sampling_integer(const std::string& text, const std::string& name, int minimum) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed == text.size() && value >= minimum && value <= std::numeric_limits<int>::max())
            return static_cast<int>(value);
    } catch (const std::invalid_argument&) {
    } catch (const std::out_of_range&) {
    }
    throw SamplingRequestError(name + " must be an int32 integer >= " + std::to_string(minimum));
}

inline int parse_sampling_seed(const std::string& text) {
    return parse_sampling_integer(text, "seed", -1);
}

inline float parse_sampling_float(const std::string& text, const std::string& name) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stof(text, &consumed);
        if (consumed == text.size() && std::isfinite(value)) return value;
    } catch (const std::invalid_argument&) {
    } catch (const std::out_of_range&) {
    }
    throw SamplingRequestError(name + " must be a finite float32 number");
}

inline nlohmann::json normalize_sampling_request(const nlohmann::json& options,
                                                  const nlohmann::json& request) {
    if (!options.is_object() && !options.is_null())
        throw SamplingRequestError("options must be a JSON object");
    if (!request.is_object())
        throw SamplingRequestError("request must be a JSON object");
    nlohmann::json result = nlohmann::json::object();
    for (const auto* source : {&options, &request}) {
        if (source->is_null()) continue;
        for (auto it = source->begin(); it != source->end(); ++it) {
            if (is_sampling_field(it.key())) result[it.key()] = it.value();
        }
    }
    // Validate after merging, so top-level fields override options consistently.
    if (result.contains("sampler")) (void)json_provider(result.at("sampler"));
    return result;
}

inline double sampling_number(const nlohmann::json& value, const std::string& name) {
    if (!value.is_number())
        throw SamplingRequestError(name + " must be a finite number");
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw SamplingRequestError(name + " must be a finite float32 number");
    return number;
}

inline nlohmann::json resolve_sampling_options(const nlohmann::json& defaults,
                                               const nlohmann::json& overrides,
                                               int vocab_size = std::numeric_limits<int>::max()) {
    if (!overrides.is_object())
        throw SamplingRequestError("sampling options must be a JSON object");
    auto effective = defaults;
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        const auto& name = it.key();
        if (!is_sampling_field(name))
            throw SamplingRequestError("Unsupported sampling option: " + name);
        if (name == "sampler") {
            effective[name] = provider_name(json_provider(it.value()));
            continue;
        }
        if (name == "do_sample") {
            if (!it->is_boolean()) throw SamplingRequestError("do_sample must be boolean");
            continue;
        }
        const double value = sampling_number(it.value(), name);
        if (name == "min_p" || name == "presence_penalty" || name == "frequency_penalty") {
            if (name == "min_p" && (value < 0 || value > 1))
                throw SamplingRequestError("min_p must be in [0, 1]");
            if (value != 0 && static_cast<float>(value) == 0)
                throw SamplingRequestError(name + " is too small for float32");
            if (value == 0) effective.erase(name);
            else effective[name] = static_cast<float>(value);
        } else if (name == "top_k" || name == "seed") {
            const double minimum = name == "seed" ? -1 : 0;
            if (std::floor(value) != value || value < minimum || value > std::numeric_limits<int>::max())
                throw SamplingRequestError(name + " must be an int32 integer >= " + std::to_string(static_cast<int>(minimum)));
            effective[name == "seed" ? "random_seed" : name] = static_cast<int>(value);
        } else {
            if ((name == "temperature" && value < 0) ||
                (name == "top_p" && (value < 0 || value > 1)) ||
                (name == "repetition_penalty" && value <= 0))
                throw SamplingRequestError("Invalid " + name + " value");
            if (value > 0 && static_cast<float>(value) == 0)
                throw SamplingRequestError(name + " is too small for float32");
            effective[name] = static_cast<float>(value);
        }
    }
    if (overrides.contains("temperature") || overrides.contains("top_k") || overrides.contains("top_p"))
        effective["do_sample"] = true;
    if (overrides.contains("do_sample")) effective["do_sample"] = overrides.at("do_sample");
    if (effective.at("temperature").get<double>() == 0) {
        effective["do_sample"] = false;
        effective["temperature"] = 1.0; // Numerical processors require a positive internal temperature.
    }
    if (effective.at("top_k").get<int>() > vocab_size)
        throw SamplingRequestError("top_k must not exceed the model vocabulary size");
    return effective;
}

class SamplingState {
public:
    explicit SamplingState(nlohmann::json defaults, int vocab_size = std::numeric_limits<int>::max())
        : defaults_(std::move(defaults)), vocab_size_(vocab_size),
          effective_(resolve_sampling_options(defaults_, overrides_, vocab_size_)) {}

    void reset_request(const nlohmann::json& overrides) {
        auto resolved = resolve_sampling_options(defaults_, overrides, vocab_size_);
        overrides_ = overrides;
        effective_ = std::move(resolved);
    }

    void update(const nlohmann::json& changes) {
        if (!changes.is_object()) throw SamplingRequestError("sampling update must be a JSON object");
        auto candidate = overrides_;
        candidate.update(changes);
        reset_request(candidate);
    }

    const nlohmann::json& effective() const { return effective_; }
    Provider provider() const {
        return effective_.contains("sampler") ? json_provider(effective_.at("sampler")) : default_provider_;
    }
    void set_default_provider(const std::string& name) {
        const auto candidate = parse_provider(name);
        reset_request(nlohmann::json::object());
        default_provider_ = candidate;
    }
    bool requires_fresh_generator() const {
        return effective_.at("do_sample").get<bool>() &&
               effective_.at("random_seed").get<int>() >= 0;
    }

private:
    Provider default_provider_ = Provider::Backend;
    nlohmann::json defaults_;
    int vocab_size_;
    nlohmann::json overrides_ = nlohmann::json::object();
    nlohmann::json effective_;
};

inline LogitsOptions sampling_logits_options(const nlohmann::json& effective) {
    LogitsOptions options;
    options.do_sample = effective.at("do_sample").get<bool>();
    options.temperature = effective.at("temperature").get<double>();
    options.top_k = effective.at("top_k").get<int>();
    options.top_p = effective.at("top_p").get<double>();
    options.repetition_penalty = effective.at("repetition_penalty").get<double>();
    options.min_p = effective.value("min_p", 0.0);
    options.presence_penalty = effective.value("presence_penalty", 0.0);
    options.frequency_penalty = effective.value("frequency_penalty", 0.0);
    return options;
}

// Resolve once per logical request, independently of backend KV lifetime.
inline uint64_t request_seed(const nlohmann::json& effective) {
    const auto& seed = effective.at("random_seed");
    if (seed.is_number_unsigned() || seed.get<int64_t>() >= 0) return seed.get<uint64_t>();
    std::random_device entropy;
    return (static_cast<uint64_t>(entropy()) << 32) ^ static_cast<uint64_t>(entropy());
}

} // namespace flm::sampling
