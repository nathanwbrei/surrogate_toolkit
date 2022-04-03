
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
    virtual std::vector<size_t> shape() = 0;
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

// TODO: Restrict T to _actual_ primitives
template <typename T>
class Primitive : public Optic<T>{
    torch::Dtype m_dtype;
public:
    Primitive(torch::Dtype dtype=choose_dtype_automatically<T>()) : m_dtype(dtype) {}
    std::vector<size_t> shape() override { return {1}; }
    torch::Tensor to(T* source) override {
        return torch::tensor({*source}, torch::TensorOptions().dtype(m_dtype));
    }
    void from(torch::Tensor source, T* dest) override {
        *dest = *source.data_ptr<T>();
        // TODO: This will throw an exception if the specified dtype isn't compatible with T (instead of converting)
    }
};

template <typename T>
class PrimitiveArray : public Optic<T> {
    const std::vector<size_t> m_shape;
    const std::vector<size_t> m_strides;
public:
    explicit PrimitiveArray(const std::vector<size_t>& shape, const std::vector<size_t>& strides)
    : m_shape(shape), m_strides(strides) {};

    std::vector<size_t> shape() override { return m_shape; }
    torch::Tensor to(T* source) override {
        return torch::tensor({*source, }, torch::TensorOptions().dtype(torch::kFloat32));
    }
    void from(torch::Tensor source, T* dest) override {
        *dest = source.data_ptr<T>();
    }
};

template <typename T>
class Pointer : public Optic<T*> {
    Optic<T>* m_optic;
public:
    Pointer(Optic<T>* optic) : m_optic(optic) {};
    std::vector<size_t> shape() override { return m_optic->shape(); }
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
    std::vector<size_t> shape() { return m_optic->shape(); }
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
