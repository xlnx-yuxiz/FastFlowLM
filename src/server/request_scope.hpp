#pragma once

#include <utility>
#include <cstdio>
#include <exception>

namespace flm::server {

// Keep request ownership through all early returns and exception paths. The
// cleanup runs while the referenced state is alive. A cleanup failure must not
// replace an in-flight handler exception or terminate an IO worker.
template <class Cleanup>
class RequestScope {
public:
    explicit RequestScope(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
    RequestScope(const RequestScope&) = delete;
    RequestScope& operator=(const RequestScope&) = delete;
    ~RequestScope() noexcept {
        try {
            cleanup_();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Request cleanup failed: %s\n", e.what());
        } catch (...) {
            std::fputs("Request cleanup failed: unknown exception\n", stderr);
        }
    }

private:
    Cleanup cleanup_;
};

} // namespace flm::server
