
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_UTILS_HPP
#define SURROGATE_TOOLKIT_UTILS_HPP

#include <cxxabi.h>

inline std::string demangle(std::string mangled) {
    int status = -1;
    const char *m = mangled.c_str();
    char *d = abi::__cxa_demangle(m, nullptr, nullptr, &status);
    if (status == -2) {
        // macOS prepends an additional underscore, which c++filt understands but cxxabi does not
        // (Though NOT when you obtain the mangled name via `typeid(T).name()`)
        // So if the demangling initially fails, we just strip the leading underscore and try again
        m++;
        d = abi::__cxa_demangle(m, nullptr, nullptr, &status);
    }
    if (status != 0) return mangled; // If status != 0, __cxa_demangle DIDN'T malloc, so we don't need to free
    std::string demangled(d);
    free(d);
    return demangled;
}

template <typename T>
inline std::string demangle() {
    return demangle(typeid(T).name());
}

#endif //SURROGATE_TOOLKIT_UTILS_HPP
