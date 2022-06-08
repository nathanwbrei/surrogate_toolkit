
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_ANY_PTR_HPP
#define SURROGATE_TOOLKIT_ANY_PTR_HPP

#include <typeindex>
#include <sstream>

namespace phasm {

class any_ptr {

    void* m_p;
    std::type_index m_t;
    std::string m_typename;

public:
    template <typename T>
    any_ptr(T* p) : m_p(p), m_t(typeid(T)), m_typename(demangle<T>()) {};

    any_ptr() : m_p(nullptr), m_t(typeid(std::nullptr_t)), m_typename("nullptr_t") {};

    any_ptr(const any_ptr& other) : m_p(other.m_p), m_t(other.m_t), m_typename(other.m_typename) {};

    inline void* get() const {
        return m_p;
    }

    inline operator void*() const {
        return m_p;
    }

    template <typename T>
    T* get() const {
        auto tt = std::type_index(typeid(T));
        if (m_p == nullptr) return nullptr;
        if (m_t != tt) {
            std::string other_typename = demangle<T>();
            std::ostringstream oss;
            oss << "any_ptr: Bad cast: expected '" << m_typename << "', got '" << other_typename << "'" << std::endl;
            throw std::runtime_error(oss.str());
        };
        return static_cast<T*>(m_p);
    }
};

}
#endif //SURROGATE_TOOLKIT_ANY_PTR_HPP
