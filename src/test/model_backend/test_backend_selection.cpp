#include "backend_options.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void env(const char* value) {
#ifdef _WIN32
    _putenv_s("FLM_BACKEND", value);
#else
    setenv("FLM_BACKEND", value, 1);
#endif
}
void require(bool condition, const char* name) {
    if (!condition) throw std::runtime_error(name);
}
template<class F> void rejects(F f, const char* name) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(name);
}
}

int main() {
    const char* old_env = std::getenv("FLM_BACKEND");
    const std::string old = old_env ? old_env : "";
    try {
        env("");
        require(flm::backend::normalize_backend("") == "flm", "default uses FLM");
        require(flm::backend::normalize_backend("npu") == "flm", "legacy npu alias");
        require(flm::backend::normalize_backend("flm") == "flm", "flm backend id alias");
        env("hip");
        require(flm::backend::normalize_backend(flm::backend::normalize_backend("npu")) == "flm", "resolved FLM must survive repeated normalization");
        require(flm::backend::normalize_backend("npu") == "flm", "explicit npu wins over env hip");
        require(flm::backend::normalize_backend("flm") == "flm", "explicit flm wins over env hip");
#ifdef FLM_ENABLE_HIP_OGA
        require(flm::backend::normalize_backend("") == "hip", "environment selects hip");
        env("flm");
        require(flm::backend::normalize_backend("hip") == "hip", "explicit hip wins over env flm");
#else
        rejects([] { flm::backend::normalize_backend("hip"); }, "disabled hip was accepted");
#endif
        env("flm");
        require(flm::backend::normalize_backend("") == "flm", "environment flm alias");
        rejects([] { flm::backend::normalize_backend("core"); }, "uncompiled core was accepted");
        rejects([] { flm::backend::normalize_backend("missing"); }, "unknown backend was accepted");
        env(old.c_str());
        std::cout << "PASS backend selection and compatibility aliases\n";
        return 0;
    } catch (const std::exception& error) {
        env(old.c_str());
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
