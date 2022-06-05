
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_TENSOR_HPP
#define SURROGATE_TOOLKIT_TENSOR_HPP

#include <torch/torch.h>

namespace phasm {

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

public:
    explicit tensor(torch::Tensor underlying = {}) :
        m_underlying(std::move(underlying)),
        m_length(underlying.numel())
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
    }

    inline torch::Tensor& get_underlying() {  return m_underlying; }
    inline size_t get_length() const { return m_length; }

    template <typename T>
    T* get() {
        return m_underlying.data_ptr<T>();
    }

    // TODO: Hashable
    // TODO: Boolean equality
    // TODO: Bridge between C types and dtypes
};


tensor stack(std::vector<tensor>&);
std::vector<tensor> unstack(tensor&);
tensor flatten(tensor& t);

} // namespace phasm

#endif //SURROGATE_TOOLKIT_TENSOR_HPP
