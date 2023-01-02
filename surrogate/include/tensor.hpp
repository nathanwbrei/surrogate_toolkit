
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_TENSOR_HPP
#define SURROGATE_TOOLKIT_TENSOR_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <ostream>

namespace phasm {


enum class DType { Undefined, UI8, I16, I32, I64, F32, F64 };
// We aren't including F16 or BF16. It looks like the PyTorch C++ API doesn't support these even if the Python one does?
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/types.h
// https://pytorch.org/cppdocs/notes/tensor_creation.html
// https://pytorch.org/docs/stable/tensor_attributes.html

template <typename T>
phasm::DType dtype() {
    if (std::is_same<T, uint8_t>()) return phasm::DType::UI8;
    if (std::is_same<T, int16_t>()) return phasm::DType::I16;
    if (std::is_same<T, int32_t>()) return phasm::DType::I32;
    if (std::is_same<T, int64_t>()) return phasm::DType::I64;
    if (std::is_same<T, float>()) return phasm::DType::F32;
    if (std::is_same<T, double>()) return phasm::DType::F64;
    return phasm::DType::Undefined;
}

void print_dtype(std::ostream& os, phasm::DType dtype);


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

    void* m_data;
    size_t m_length;
    std::vector<int64_t> m_shape;
    DType m_dtype;

public:

    // Construct "Empty" tensor
    tensor() : m_data(nullptr), m_length(0), m_shape({}), m_dtype(DType::Undefined)  {}

    // Construct tensor from buffer
    template <typename T> explicit tensor(T* consecutive_buffer, size_t length) {
        assert(length > 0);
        T* buffer = new T[length];
        for (size_t i=0; i<length; ++i) buffer[i] = consecutive_buffer[i];

        m_data = buffer; // Throw away the type information here!
        m_length = length;
        m_shape = {static_cast<long>(length)};
        // TODO: Clean up this mix of size_t's and int64_t's. Why is it this way?
        //   1. Torch uses _signed_ int64's for indices instead of size_t or ptrdiff_t
        //   2. PIN is very confused about size_t, long, and long long
        m_dtype = dtype<T>();
    }

    // Construct tensor from buffer with shape information, e.g. a _contiguous_ tensor
    template <typename T> explicit tensor(T* consecutive_buffer, const std::vector<int64_t>& shape) {
        // TODO: Normalize shape so that e.g. {5,1,1} => {5}, {1} => {}
        m_length = 1;
        for (size_t l : shape) {
            m_length *= l;
        }
        T* buffer = new T[m_length];
        for (size_t i=0; i<m_length; ++i) buffer[i] = consecutive_buffer[i];
        m_data = buffer;
        m_shape = shape;
        m_dtype = dtype<T>();
    }

    ~tensor();

    tensor(const tensor& other) noexcept;
    tensor& operator=(const tensor& other) noexcept;

    tensor(tensor&& other) noexcept;
    tensor& operator=(tensor&& other) noexcept;

    // inline torch::Tensor& get_underlying() {  return m_underlying; }
    inline size_t get_length() const { return m_length; }
    inline DType get_dtype() const { return m_dtype; }
    inline std::vector<int64_t> get_shape() const { return m_shape; }

    bool operator==(const tensor& rhs) const;

    template <typename T>
    T* get_data() {
        return static_cast<T*>(m_data);
    }

    template <typename T>
    const T* get_data() const {
        return static_cast<T*>(m_data);
    }

    void print(std::ostream& os);

};


phasm::tensor stack(const std::vector<phasm::tensor>&);
std::vector<tensor> unstack(const phasm::tensor&);


inline size_t combineHashes(size_t hash1, size_t hash2) {
    // Not clear why this isn't part of the standard library.
    // Taken from https://en.cppreference.com/w/cpp/utility/hash
    return hash1 ^ (hash2 << 1);
}

template <typename T>
size_t hashTensorOfDType(const phasm::tensor& t) {
    size_t len = t.get_length();
    if (len == 0) return 0;
    std::hash<T> hash;
    const T* ptr = t.get_data<T>();
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
