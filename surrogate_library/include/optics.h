
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_OPTICS_H
#define SURROGATE_TOOLKIT_OPTICS_H

#include <torch/torch.h>
#include "../vacuum_tool/src/utils.hpp"
// #include <concepts>

// These aren't really optics yet (will they ever be?), but they are operational at least
// The idea is that they form a declarative, composable way of converting arbitrary data
// structures into tensors of floats and back again. The real benefit of this comes when you
// have structures of arrays of arrays of structures or whatever, and you need to turn that
// into a Tensor sanely.

namespace optics {

/// Untyped base class for all optics. We need this so that we can traverse the full tree
/// of optics, particularly for fluent interfaces and for transpositions
struct OpticBase {

    OpticBase* parent = nullptr;
    std::vector<OpticBase*> children;
    std::string name;
    std::string produces;
    std::string consumes;
    bool is_leaf = false;

    void unsafe_attach(OpticBase* optic) {
        if (optic->consumes != produces) {
            std::ostringstream ss;
            ss << "Incompatible optics: '" << produces << "' vs '" << optic->consumes << "'";
            throw std::runtime_error(ss.str());
        }
        optic->parent = this;
        children.push_back(optic);
    }

    virtual void use(OpticBase*) {};
};

// This is the abstract base class for an optic.
// Template parameter T means that we can accept a T*, traverse an arbitrary nesting of
// structs, arrays, pointers, unions, etc, and return a Tensor containing the primitives
// at the leaves of the traversal. The roots represent variables we can access at the
// target function's call site; the leaves represent actual model variables, which we
// need to transform to and from tensors.
template <typename T>
struct Optic : public OpticBase {
    virtual std::vector<int64_t> shape() {return {};}; // Torch uses int64_t instead of size_t for its indices and offsets
    virtual torch::Tensor to(T* source) {return {};};
    virtual void from(torch::Tensor source, T* dest) {};
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
    Primitive() {
        OpticBase::consumes = demangle<T>();
        OpticBase::produces = "tensor";
    }
    Primitive(const Primitive& other) = default;

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
    // const std::vector<size_t> m_strides; // We will need these if we want to support reading and writing tensor slices
public:

    explicit PrimitiveArray(std::vector<int64_t> shape) : m_shape(std::move(shape)) {
        OpticBase::consumes = demangle<T>();
        OpticBase::produces = "tensor";
    };
    PrimitiveArray(const PrimitiveArray& other) = default;

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

#if 0
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
#endif

// Field is an Optic that accepts a struct of type StructT, knows how to extract a field of type FieldT from the StructT,
// and forwards the field to an inner optic of type OpticT that accepts FieldT. The user needs to compose

template <typename StructT, typename FieldT>
class Field : public Optic<StructT> {
    Optic<FieldT>* m_optic = nullptr;
    std::function<FieldT*(StructT*)> m_accessor;
public:
    Field(Optic<FieldT>* optic, std::function<FieldT*(StructT*)> accessor) : m_optic(optic), m_accessor(accessor) {
        OpticBase::consumes = demangle<StructT>();
        OpticBase::produces = demangle<FieldT>();
    };
    Field(const Field& other) = default;

    std::vector<int64_t> shape() override { return m_optic->shape(); }
    torch::Tensor to(StructT* source) override {
        return m_optic->to(m_accessor(source));
    }
    void from(torch::Tensor source, StructT* dest) override {
        return m_optic->from(source, m_accessor(dest));
    }
    void attach(Optic<FieldT>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<FieldT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
};

template <typename StructT, typename FieldT>
Field<StructT, FieldT> make_field_lens(Optic<FieldT>* optic, std::function<FieldT*(StructT*)> fn) {
    return Field(optic, fn);
};

template <typename InnerT>
class Array : public Optic<InnerT> {
    Optic<InnerT>* m_optic;
    int64_t m_length;
public:
    Array(Optic<InnerT>* optic, size_t length) : m_optic(optic), m_length(length) {
        OpticBase::consumes = demangle<InnerT>();
        OpticBase::produces = demangle<InnerT>();
    }
    Array(const Array& other) = default;

    std::vector<int64_t> shape() override {
        std::vector<int64_t> result {m_length};
        auto inner_shape = m_optic->shape();
        result.insert(result.end(), inner_shape.begin(), inner_shape.end());
        return result;
    }
    torch::Tensor to(InnerT* source) override {
        std::vector<torch::Tensor> tensors;
        for (int i=0; i<m_length; ++i) {
            tensors.push_back(m_optic->to(source+i));
        }
        return torch::stack(tensors);
    }
    void from(torch::Tensor source, InnerT* dest) override {
        auto unstacked = torch::unbind(source, 0);
        for (int i=0; i<m_length; ++i) {
            m_optic->from(unstacked[i], dest+i);
        }
    }
    void attach(Optic<InnerT>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<InnerT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
};


// For now assume the traversable contains exactly as many elements as expected
template <typename OuterT, typename InnerT, typename IteratorT>
class Traversal : public Optic<OuterT> {
    Optic<InnerT>* m_optic;
    int64_t m_length;

public:
    Traversal(Optic<InnerT>* optic, size_t length) : m_optic(optic), m_length(length) {
        OpticBase::consumes = demangle<OuterT>();
        OpticBase::produces = demangle<InnerT>();
    }
    Traversal(const Traversal& other) = default;
    std::vector<int64_t> shape() override {
        std::vector<int64_t> result {m_length};
        auto inner_shape = m_optic->shape();
        result.insert(result.end(), inner_shape.begin(), inner_shape.end());
        return result;
    }
    torch::Tensor to(OuterT* source) override {
        IteratorT it(source);
        std::vector<torch::Tensor> tensors;

        for (int i=0; i<m_length; ++i) {
            tensors.push_back(m_optic->to(it.Current()));
            it.Next();  // Returns true because container has m_length entries XD
        }
        return torch::stack(tensors);
    }
    void from(torch::Tensor source, OuterT* dest) override {
        auto unstacked = torch::unbind(source, 0);
        IteratorT it(dest);
        for (int i=0; i<m_length; ++i) {
            m_optic->from(unstacked[i], it.Current());
            it.Next();
        }
    }
    void attach(Optic<InnerT>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<InnerT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
};

template<typename OuterT, typename InnerT>
class STLIterator {
    OuterT* underlying;
    typename OuterT::iterator it;
public:
    STLIterator(OuterT* v) : underlying(v), it(std::begin(*v)) {};
    InnerT* Current() {
        return &(*it);
    }
    bool Next() {
        return (++it != std::end(*underlying));
    }
    /*
    void Insert(InnerT&& item) {
        // Problem: insert() doesn't behave consistently across STL containers
        // Problem: I don't think it's safe to use the iterator after the modification has happened
        // Probably the best thing to do is to construct the missing items and append them to the container at the very end
        underlying->insert(it, std::move(item));
    }
    */
};


} // namespace optics




#endif //SURROGATE_TOOLKIT_OPTICS_H
