
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_OPTICS_H
#define SURROGATE_TOOLKIT_OPTICS_H

#include <torch/torch.h>
// #include <concepts>

// These aren't really optics yet (will they ever be?), but they are operational at least
// The idea is that they form a declarative, composable way of converting arbitrary data
// structures into tensors of floats and back again. The real benefit of this comes when you
// have structures of arrays of arrays of structures or whatever, and you need to turn that
// into a Tensor sanely.

namespace optics {

// This is the abstract base class for an optic.
// Template parameter T means that we can accept a T*, traverse an arbitrary nesting of
// structs, arrays, pointers, unions, etc, and return a Tensor containing the primitives
// at the leaves of the traversal. This tensor describes one actual "input" of the function
// being surrogated.
template <typename T>
struct Optic {
    virtual std::vector<int64_t> shape() = 0; // Torch uses int64_t instead of size_t for its indices and offsets
    virtual torch::Tensor to(T* source) = 0;
    virtual void from(torch::Tensor source, T* dest) = 0;
};

/*
template <typename T>
concept Optic = requires(T t) {
    {t.shape()} -> std::same_as<std::vector<size_t>>;
};
*/
template <typename T>
torch::Dtype choose_dtype_automatically() {
    if (std::is_same_v<T, u_int8_t>) return torch::kUInt8;
    if (std::is_same_v<T, int16_t>) return torch::kI16;
    if (std::is_same_v<T, int32_t>) return torch::kI32;
    if (std::is_same_v<T, int64_t>) return torch::kI64;
    if (std::is_same_v<T, float>) return torch::kF32;
    if (std::is_same_v<T, double>) return torch::kF64;
    return torch::kF32;
}

/// The simplest optic. Converts a single primitive value into a Torch Tensor. Can be used as-is (for primitive inputs)
/// or composed with another optic (e.g. a Lens) to traverse from a locally visible variable to a nested primitive.
/// Primitive ought to be an Iso<T,Tensor>. But (unlike real optics) our optics are constrained to always have a Primitive
/// or PrimitiveArray at the back of the composition chain, so we can throw away the second template parameter.
/// Probably a more rigorous way to do this would be to make Primitive be an Iso<T,Tensor>, so the overall
/// composition chain becomes an Iso<T, Tensor>.
/// TODO: Restrict T to _actual_ primitives
/// Improvements: Add the ability to choose a different dtype for the tensor other than T.

template <typename T>
class Primitive : public Optic<T>{
public:
    Primitive() {}
    std::vector<int64_t> shape() override { return {}; }
    torch::Tensor to(T* source) override {
        return torch::tensor(at::ArrayRef<T>(source,1), torch::dtype<T>());
    }
    void from(torch::Tensor source, T* dest) override {
        *dest = *source.data_ptr<T>();
    }
};


/// The second simplest Optic. We may want to merge these two eventually. This and `Primitive` are the back of the
/// composition chain of optics.
///
/// This does a byte-for-byte copy of a multidimensional C++ array into a Torch tensor. It assumes that the data is
/// contiguous both in the array and in the tensor, and also that we don't have to worry about row-major
/// vs column-major ordering because the neural net won't care as long as we are consistent.
/// Both of these assumptions will eventually break down. Firstly, the Tensor may have been built up from smaller Tensors
/// recursively, and we may somehow take a slice that is not contiguous. (Admittedly this problem is much more likely to
/// manifest on the Array optic than on the PrimitiveArray one.) It may be possible to prove that this is not the case
/// when using these Optics, although it is certainly the case generally.
/// Secondly, we may wish to use the same trained model on inputs which use a different major ordering (e.g.
/// we train a model on a codebase that uses plain arrays, but then reuse it on a different codebase that uses Eigen
/// matrices instead, which are column-major)
template <typename T>
class PrimitiveArray : public Optic<T> {
    const std::vector<int64_t> m_shape;
    // const std::vector<size_t> m_strides;
public:

    explicit PrimitiveArray(std::vector<int64_t> shape)
            : m_shape(std::move(shape)) {};

    std::vector<int64_t> shape() override { return m_shape; }
    torch::Tensor to(T* source) override {
        size_t length = std::accumulate(m_shape.begin(), m_shape.end(), 1ull, [](size_t a, size_t b){return a*b;});
        auto t = torch::tensor(at::ArrayRef<T>(source, length), torch::dtype<T>());
        return t.reshape(at::ArrayRef(m_shape.data(), m_shape.size()));
    }
    void from(torch::Tensor source, T* dest) override {
        size_t length = std::accumulate(m_shape.begin(), m_shape.end(), 1ull, [](size_t a, size_t b){return a*b;});
        T* ptr = source.data_ptr<T>();
        for (size_t i=0; i<length; ++i) {
            dest[i] = ptr[i];
        }
    }
};

/// Pointer is a Lens that performs a pointer dereference
template <typename T>
class Pointer : public Optic<T*> {
    Optic<T>* m_optic;
public:
    Pointer(Optic<T>* optic) : m_optic(optic) {};
    std::vector<int64_t> shape() override { return m_optic->shape(); }
    torch::Tensor to(T* source) override {
        return m_optic->to(source);
    }
    void from(torch::Tensor source, T* dest) override {
        return m_optic->from(source, dest);
    }
};

// Field is an Optic that accepts a struct of type StructT, knows how to extract a field of type FieldT from the StructT,
// and forwards the field to an inner optic of type OpticT that accepts FieldT. The user needs to compose

template <typename StructT, typename FieldT>
class Field : public Optic<StructT> {
    Optic<FieldT>* m_optic;
    std::function<FieldT*(StructT*)> m_accessor;
public:
    Field(Optic<FieldT>* optic, std::function<FieldT*(StructT*)> accessor) : m_optic(optic), m_accessor(accessor) {};
    std::vector<int64_t> shape() { return m_optic->shape(); }
    torch::Tensor to(StructT* source) {
        return m_optic->to(m_accessor(source));
    }
    void from(torch::Tensor source, StructT* dest) {
        return m_optic->from(source, m_accessor(dest));
    }
};

template <typename StructT, typename FieldT>
Field<StructT, FieldT> make_field_lens(Optic<FieldT>* optic, std::function<FieldT*(StructT*)> fn) {
    return Field(optic, fn);
};



} // namespace optics




#endif //SURROGATE_TOOLKIT_OPTICS_H
