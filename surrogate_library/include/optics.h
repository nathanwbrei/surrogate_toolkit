
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_OPTICS_H
#define SURROGATE_TOOLKIT_OPTICS_H

#include <torch/torch.h>
// #include <concepts>

// These aren't really optics yet (will they ever be?), but they are operational at least
// I'm calling them "Accessors" instead. The idea is that they form a declarative, composable
// way of converting arbitrary data structures into tensors of floats and back again.


// For now we assume everything goes into a Tensor of floats. Eventually we can extend this by adding
// an additional type parameter (just like we did for the {Input,Output}Bindings)
// We might want to have custom logic for converting some primitive datatype into a float and vice versa.
template <typename T>
float convert_to_float(T t) {
    return static_cast<float>(t);
}

template <typename T>
float convert_from_float(float f) {
    return static_cast<T>(f);
}

/// TensorLens<T> goes from a T* to a Tensor and vice versa.
/// This should be a Concept instead?!

namespace optics {

    /*
template <typename T>
concept Optic = requires(T t) {
    {t.shape()} -> std::same_as<std::vector<size_t>>;

};
     */

// TODO: Restrict T to _actual_ primitives
template <typename T>
class Primitive {
public:
    Primitive() {};
    std::vector<size_t> shape() { return {1}; }
    torch::Tensor to(T* source) {
        return torch::tensor({*source}, torch::TensorOptions().dtype(torch::kFloat32));
    }
    void from(torch::Tensor source, T* dest) {
        *dest = *source.data_ptr<T>();
    }
};

template <typename T>
class PrimitiveArray {
    const std::vector<size_t> m_shape;
    const std::vector<size_t> m_strides;
public:
    explicit PrimitiveArray(const std::vector<size_t>& shape, const std::vector<size_t>& strides)
    : m_shape(shape), m_strides(strides) {};

    std::vector<size_t> shape() { return m_shape; }
    torch::Tensor to(T* source) {
        return torch::tensor({*source, }, torch::TensorOptions().dtype(torch::kFloat32));
    }
    void from(torch::Tensor source, T* dest) {
        *dest = source.data_ptr<T>();
    }
};

template <typename T, class OpticT>
class Pointer {
    OpticT m_optic;
public:
    Pointer(OpticT optic) : m_optic(optic) {};
    std::vector<size_t> shape() { return m_optic.shape(); }
    torch::Tensor to(T* source) {
        return m_optic.to(source);
    }
    void from(torch::Tensor source, T* dest) {
        return m_optic.from(source, dest);
    }
};

// Field is an Optic that accepts a struct of type StructT, knows how to extract a field of type FieldT from the StructT,
// and forwards the field to an inner optic of type OpticT that accepts FieldT. The user needs to compose

// TODO: Should be some constraint on OpticT requiring it to accept a FieldT
template <typename StructT, typename FieldT, typename OpticT>
class Field {
    OpticT m_optic;
    std::function<FieldT*(StructT*)> m_accessor;
public:
    Field(OpticT optic, std::function<FieldT*(StructT*)> accessor) : m_optic(optic), m_accessor(accessor) {};
    std::vector<size_t> shape() { return m_optic.shape(); }
    torch::Tensor to(StructT* source) {
        return m_optic.to(m_accessor(source));
    }
    void from(torch::Tensor source, StructT* dest) {
        return m_optic.from(source, m_accessor(dest));
    }
};

template <typename StructT, typename FieldT, typename OpticT>
Field<StructT, FieldT, OpticT> make_field_lens(OpticT optic, std::function<FieldT*(StructT*)> fn) {
    return Field(optic, fn);
};



} // namespace optics




#endif //SURROGATE_TOOLKIT_OPTICS_H
