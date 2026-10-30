#include "request_scope.hpp"
#include <cassert>
#include <stdexcept>

int main() {
    int cleanups = 0;
    const auto early_return = [&] {
        flm::server::RequestScope guard([&] { ++cleanups; });
        return;
    };
    early_return();
    assert(cleanups == 1);
    try {
        flm::server::RequestScope guard([&] { ++cleanups; });
        throw std::runtime_error("handler failure");
    } catch (const std::runtime_error&) {}
    assert(cleanups == 2);
    // A scheduling/cleanup failure must not terminate the server, including
    // while a handler exception is already unwinding.
    try {
        flm::server::RequestScope guard([&] {
            ++cleanups;
            throw std::runtime_error("cleanup failure");
        });
        throw std::runtime_error("handler failure");
    } catch (const std::runtime_error&) {}
    assert(cleanups == 3);
}
