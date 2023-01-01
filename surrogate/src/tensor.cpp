
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "tensor.hpp"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace phasm {

template <typename T>
T* copy_typed(void* source_untyped, size_t length) {
    T* dest = new T[length];
    T* source = static_cast<T*>(source_untyped);
    for (size_t i=0; i<length; ++i) {
        dest[i] = source[i];
    }
    return dest;
}

tensor::tensor(const tensor& other) noexcept {
    m_dtype = other.m_dtype;
    m_length = other.m_length;
    m_shape = other.m_shape;
    switch (m_dtype) {
        case DType::UI8: m_underlying = copy_typed<uint8_t>(other.m_underlying, other.m_length); break;
        case DType::I16: m_underlying = copy_typed<int16_t>(other.m_underlying, other.m_length); break;
        case DType::I32: m_underlying = copy_typed<int32_t>(other.m_underlying, other.m_length); break;
        case DType::I64: m_underlying = copy_typed<int64_t>(other.m_underlying, other.m_length); break;
        case DType::F32: m_underlying = copy_typed<float>(other.m_underlying, other.m_length); break;
        case DType::F64: m_underlying = copy_typed<double>(other.m_underlying, other.m_length); break;
        default: break;
    }
}

tensor& tensor::operator=(const tensor& other) noexcept {
    if (this != &other) return *this;
    m_dtype = other.m_dtype;
    m_length = other.m_length;
    m_shape = other.m_shape;
    m_underlying = other.m_underlying;
    switch (m_dtype) {
        case DType::UI8: m_underlying = copy_typed<uint8_t>(other.m_underlying, other.m_length); break;
        case DType::I16: m_underlying = copy_typed<int16_t>(other.m_underlying, other.m_length); break;
        case DType::I32: m_underlying = copy_typed<int32_t>(other.m_underlying, other.m_length); break;
        case DType::I64: m_underlying = copy_typed<int64_t>(other.m_underlying, other.m_length); break;
        case DType::F32: m_underlying = copy_typed<float>(other.m_underlying, other.m_length); break;
        case DType::F64: m_underlying = copy_typed<double>(other.m_underlying, other.m_length); break;
        default: break;
    }
    return *this;
}
tensor::tensor(tensor &&other) noexcept {
    m_dtype = other.m_dtype;
    m_length = other.m_length;
    m_shape = other.m_shape;
    m_underlying = other.m_underlying;
}

tensor& tensor::operator=(tensor&& other) noexcept {
    if (this != &other) return *this;
    m_dtype = other.m_dtype;
    m_length = other.m_length;
    m_shape = other.m_shape;
    m_underlying = other.m_underlying;
    return *this;
}

tensor::~tensor() {
    switch (m_dtype) {
        case DType::UI8: delete[] static_cast<uint8_t*>(m_underlying); break;
        case DType::I16: delete[] static_cast<int16_t*>(m_underlying); break;
        case DType::I32: delete[] static_cast<int32_t*>(m_underlying); break;
        case DType::I64: delete[] static_cast<int64_t*>(m_underlying); break;
        case DType::F32: delete[] static_cast<float*>(m_underlying); break;
        case DType::F64: delete[] static_cast<double*>(m_underlying); break;
        default: break;
    }
};

template <typename T>
inline bool equals_typed(const tensor& lhs, const tensor& rhs) {
    size_t length = lhs.get_length();
    for (size_t i=0; i<length; ++i) {
        if (lhs.get<T>()[i] != rhs.get<T>()[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline bool fequals_typed(const tensor& lhs, const tensor& rhs) {
    size_t length = lhs.get_length();
    const T* lhs_ptr = lhs.get<T>();
    const T* rhs_ptr = rhs.get<T>();
    for (size_t i=0; i<length; ++i) {
        if (std::abs(lhs_ptr[i]-rhs_ptr[i]) > std::abs(lhs_ptr[i]*std::numeric_limits<T>::epsilon())) {
            return false;
        }
    }
    return true;
}

bool tensor::operator==(const tensor& rhs) const {
    if (m_dtype != rhs.m_dtype) return false;
    if (m_length != rhs.m_length) return false;
    if (m_shape != rhs.m_shape) return false;
    switch (m_dtype) {
        case DType::UI8: return equals_typed<uint8_t>(*this, rhs);
        case DType::I16: return equals_typed<int16_t>(*this, rhs);
        case DType::I32: return equals_typed<int32_t>(*this, rhs);
        case DType::I64: return equals_typed<int64_t>(*this, rhs);
        case DType::F32: return fequals_typed<float>(*this, rhs);
        case DType::F64: return fequals_typed<double>(*this, rhs);
        default: throw std::runtime_error("Tensor has unknown dtype");
    }
}

template <typename T>
inline tensor stack_typed(const std::vector<tensor>& tensors) {

    size_t original_length = tensors[0].get_length();
    size_t stacked_tensor_count = tensors.size();
    size_t stacked_length = original_length * stacked_tensor_count;
    std::vector<int64_t> stacked_shape;
    stacked_shape.push_back(stacked_length);
    for (size_t dim_length : tensors[0].get_shape()) {
        stacked_shape.push_back(dim_length);
    }

    T* buffer = new T[stacked_length];
    for (size_t j = 0; j<stacked_tensor_count; ++j) {
        const T* original_data = tensors[j].get<T>();
        for (size_t i = 0; i<original_length; ++i) {
            buffer[j*original_length + i] = original_data[i];
        }
    }
    return tensor(buffer, stacked_shape);
}

tensor stack(const std::vector<tensor>& tensors) {
    auto stacked_dtype = tensors[0].get_dtype();
    switch (stacked_dtype) {
        case DType::UI8: return stack_typed<uint8_t>(tensors);
        case DType::I16: return stack_typed<int16_t>(tensors);
        case DType::I32: return stack_typed<int32_t>(tensors);
        case DType::I64: return stack_typed<int64_t>(tensors);
        case DType::F32: return stack_typed<float>(tensors);
        case DType::F64: return stack_typed<double>(tensors);
        default:
            throw std::runtime_error("Tensor has unknown dtype");
    }
}

template <typename T>
std::vector<tensor> unstack_typed(const tensor& tensor) {

    size_t combined_length = tensor.get_length();
    size_t split_count = tensor.get_shape()[0];
    assert(combined_length % split_count == 0);
    size_t split_length = combined_length/split_count;
    std::vector<int64_t> split_shape = tensor.get_shape();
    split_shape.erase(split_shape.begin());

    std::vector<phasm::tensor> results;
    const T* original_buffer = tensor.get<T>();
    for (size_t j=0; j<split_count; ++j) {

        T* split_buffer = new T[split_length];
        for (size_t i=0; i<split_length; ++i) {
            split_buffer[i] = original_buffer[j*split_length] + i;
        }
        results.push_back(phasm::tensor(split_buffer, split_shape));
    }
    return results;
}

std::vector<tensor> unstack(const tensor& tensor) {
    auto unstacked_dtype = tensor.get_dtype();
    switch (unstacked_dtype) {
        case DType::UI8: return unstack_typed<uint8_t>(tensor);
        case DType::I16: return unstack_typed<uint8_t>(tensor);
        case DType::I32: return unstack_typed<uint8_t>(tensor);
        case DType::I64: return unstack_typed<uint8_t>(tensor);
        case DType::F32: return unstack_typed<uint8_t>(tensor);
        case DType::F64: return unstack_typed<uint8_t>(tensor);
        default:
            throw std::runtime_error("Tensor has unknown dtype");
    }
}

void print_dtype(std::ostream& os, phasm::DType dtype) {
    switch (dtype) {
        case DType::Undefined: os << "Undefined"; break;
        case DType::UI8: os << "UI8"; break;
        case DType::I16: os << "I16"; break;
        case DType::I32: os << "I32"; break;
        case DType::I64: os << "I64"; break;
        case DType::F32: os << "F32"; break;
        case DType::F64: os << "F64"; break;
    }
}

template <typename T>
inline void print_typed(std::ostream& os, const tensor& t) {
    size_t full_len = t.get_length();
    bool show_full = (full_len < 10);
    size_t max_len = show_full ? full_len: 10;

    const T* data = t.get<T>();
    for (size_t i=0; i<max_len; ++i) {
        os << data[i] << " ";
    }

    if (!show_full) os << "... (" << full_len << " items total)" << std::endl;
}

void tensor::print(std::ostream& os) {
    switch (m_dtype) {
        case DType::UI8: print_typed<uint8_t>(os, *this); break;
        case DType::I16: print_typed<int16_t>(os, *this); break;
        case DType::I32: print_typed<int32_t>(os, *this); break;
        case DType::I64: print_typed<int64_t>(os, *this); break;
        case DType::F32: print_typed<float>(os, *this); break;
        case DType::F64: print_typed<double>(os, *this); break;
        case DType::Undefined: os << "(Tensor data is undefined)"; break;
    }
}


} // namespace phasm

