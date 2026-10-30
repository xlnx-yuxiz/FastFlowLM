/// \file console_utils.hpp
/// \brief Console initialization without device-runtime dependencies.
#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

namespace utils {

#ifdef _WIN32
inline void enable_ansi_on_windows_once() {
    static bool done = false;
    if (done) return;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (!GetConsoleMode(hOut, &mode)) return;
    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, mode);
    done = true;
}
#else
inline void enable_ansi_on_windows_once() {}
#endif

} // namespace utils
