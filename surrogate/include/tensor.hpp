
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_TENSOR_HPP
#define SURROGATE_TOOLKIT_TENSOR_HPP

#include <torch/torch.h>

namespace phasm {



enum class DType { Undefined, UI8, I16, I32, I64, F32, F64 };
// We aren't including F16 or BF16. It looks like the PyTorch C++ API doesn't support these even if the Python one does?
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/types.h
// https://pytorch.org/cppdocs/notes/tensor_creation.html
// https://pytorch.org/docs/stable/tensor_attributes.html

template <typename T>
phasm::DType dtype() {
    if (std::is_same_v<T, u_int8_t>) return phasm::DType::UI8;
    if (std::is_same_v<T, int16_t>) return phasm::DType::I16;
    if (std::is_same_v<T, int32_t>) return phasm::DType::I32;
    if (std::is_same_v<T, int64_t>) return phasm::DType::I64;
    if (std::is_same_v<T, float>) return phasm::DType::F32;
    if (std::is_same_v<T, double>) return phasm::DType::F64;
    return phasm::DType::Undefined;
}

inline phasm::DType get_dtype(torch::Dtype t) {
    if (t == torch::kUInt8) return phasm::DType::UI8;
    if (t == torch::kInt16) return phasm::DType::I16;
    if (t == torch::kInt32) return phasm::DType::I32;
    if (t == torch::kInt64) return phasm::DType::I64;
    if (t == torch::kFloat32) return phasm::DType::F32;
    if (t == torch::kFloat64) return phasm::DType::F64;
    return phasm::DType::Undefined;
}



/// Tensor is a lightweight wrapper over torch::Tensor or similar.
/// It supports the following things:
/// 1. It is hashable
/// 2. It has an equals operator that returns a Boolean
/// 3. It can be written and read from a C-style array
///
/// It does _not_ do any of the fun things like slicing or
/// broadcasting. For that you should just use the underlying.
///
/// These are the issues to figure out:
/// 1. How to support TensorFlow and ROOT as additional backends.
///    Options:
///    a. Use preprocessor directives to set the desired
///       backend framework at compile time
///    b. Use PIMPL and load the ML frameworks dynamically
/// 2. How to write back tensors that are internally non-consecutive
///    (i.e. they were created from a slice of a larger tensor)
/// 3. PyTorch at least has a very awkward way of going from primitive
///    types to dtypes and back again (also note that dtypes include
///    some floating point representations that the CPU doesn't understand)
///    The way we are currently handling dtypes is pretty bad
/// 4. Eventually we probably want to support variable length tensors as well
///
class tensor {
    torch::Tensor m_underlying;
    size_t m_length;
    std::vector<int64_t> m_shape;
    DType m_dtype;

public:
    explicit inline tensor(torch::Tensor underlying = {}):
        m_underlying(underlying),
        m_length(underlying.numel()),
        m_dtype(phasm::get_dtype(underlying.dtype().toScalarType()))
    {
        auto dims = underlying.dim();
        for (int64_t d = 0; d<dims; ++d) {
            m_shape.push_back(underlying.size(d));
        }
    };

    template <typename T> explicit tensor(T* consecutive_buffer, size_t length) {
        m_underlying = torch::tensor(at::ArrayRef<T>(consecutive_buffer,length), torch::dtype<T>());
        m_length = length;
        m_shape = {(int64_t)length};
        m_dtype = dtype<T>();
    }

    template <typename T> explicit tensor(T* consecutive_buffer, std::vector<int64_t> shape) {
        size_t numel = 1;
        for (size_t l : shape) {
            numel *= l;
        }
        m_underlying = torch::tensor(at::ArrayRef<T>(consecutive_buffer,numel), torch::dtype<T>());
        m_underlying = m_underlying.reshape(at::ArrayRef(shape.data(), shape.size()));
        m_length = numel;
        m_shape = shape;
        m_dtype = dtype<T>();
    }

    inline torch::Tensor& get_underlying() {  return m_underlying; }
    inline size_t get_length() const { return m_length; }
    inline DType get_dtype() const { return m_dtype; }

    inline bool operator==(const tensor& rhs) const { return this->m_underlying.equal(rhs.m_underlying); }

    template <typename T>
    T* get() {
        return m_underlying.data_ptr<T>();
    }

    template <typename T>
    const T* get() const {
        return m_underlying.data_ptr<T>();
    }
};


tensor stack(std::vector<tensor>&);
std::vector<tensor> unstack(tensor&);
tensor flatten(tensor& t);


inline size_t combineHashes(size_t hash1, size_t hash2) {
    // Not clear why this isn't part of the standard library.
    // Taken from https://en.cppreference.com/w/cpp/utility/hash
    return hash1 ^ (hash2 << 1);
}

template <typename T>
size_t hashTensorOfDType(const phasm::tensor& t) {
    // TODO: This only works on consecutive tensors. Need to figure
    //       out how to iterate over non-consecutive tensors efficiently
    size_t len = t.get_length();
    if (len == 0) return 0;
    std::hash<T> hash;
    const T* ptr = t.get<T>();
    size_t seed = hash(ptr[0]);
    for (size_t i=1; i<len; ++i) {
        seed = combineHashes(seed, hash(ptr[i]));
    }
    return seed;
}

} // namespace phasm


template<>
struct std::hash<phasm::tensor>
{
    std::size_t operator()(phasm::tensor const& t) const noexcept
    {
        switch (t.get_dtype()) {
            case phasm::DType::UI8: return phasm::hashTensorOfDType<uint8_t>(t);
            case phasm::DType::I16: return phasm::hashTensorOfDType<int16_t>(t);
            case phasm::DType::I32: return phasm::hashTensorOfDType<int32_t>(t);
            case phasm::DType::I64: return phasm::hashTensorOfDType<int64_t>(t);
            case phasm::DType::F32: return phasm::hashTensorOfDType<float>(t);
            case phasm::DType::F64: return phasm::hashTensorOfDType<double>(t);
            default: return 0;
        }
    }
};




#endif //SURROGATE_TOOLKIT_TENSOR_HPP
