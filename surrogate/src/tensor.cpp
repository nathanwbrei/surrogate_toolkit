
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "tensor.hpp"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace phasm {

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
    std::vector<size_t> stacked_shape;
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

tensor stack(std::vector<tensor>& tensors) {
    auto stacked_dtype = tensors[0].get_dtype();
    switch (stacked_dtype) {
        case DType::UI8: return stack_typed<uint8_t>(tensors);
        case DType::I16: return stack_typed<uint8_t>(tensors);
        case DType::I32: return stack_typed<uint8_t>(tensors);
        case DType::I64: return stack_typed<uint8_t>(tensors);
        case DType::F32: return stack_typed<uint8_t>(tensors);
        case DType::F64: return stack_typed<uint8_t>(tensors);
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
    std::vector<size_t> split_shape = tensor.get_shape();
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

} // namespace phasm

