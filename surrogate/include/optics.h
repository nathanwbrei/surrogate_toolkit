
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_OPTICS_H
#define SURROGATE_TOOLKIT_OPTICS_H

#include "typename.hpp"
#include "any_ptr.hpp"
#include "tensor.hpp"
// #include <concepts>

// These aren't really optics yet (will they ever be?), but they are operational at least
// The idea is that they form a declarative, composable way of converting arbitrary data
// structures into tensors of floats and back again. The real benefit of this comes when you
// have structures of arrays of arrays of structures or whatever, and you need to turn that
// into a Tensor sanely.

namespace phasm {

/// Untyped base class for all optics. We need this so that we can traverse the full tree
/// of optics, particularly for fluent interfaces and for transpositions
struct OpticBase {

    OpticBase* parent = nullptr;
    std::vector<OpticBase*> children;
    std::string name;
    std::string produces;
    std::string consumes;
    bool is_leaf = false;

    virtual std::vector<int64_t> shape() {return {};}; // Torch uses int64_t instead of size_t for its indices and offsets

    void unsafe_attach(OpticBase* optic) {
        if (optic->consumes != produces) {
            std::ostringstream ss;
            ss << "Incompatible optics: '" << produces << "' vs '" << optic->consumes << "'";
            throw std::runtime_error(ss.str());
        }
        optic->parent = this;
        children.push_back(optic);
    }

    virtual tensor unsafe_to(phasm::any_ptr) = 0;
    virtual void unsafe_from(tensor, phasm::any_ptr) = 0;
    virtual void unsafe_use(OpticBase*) {};
    virtual OpticBase* clone() = 0;
};

// This is the abstract base class for an optic.
// Template parameter T means that we can accept a T*, traverse an arbitrary nesting of
// structs, arrays, pointers, unions, etc, and return a Tensor containing the primitives
// at the leaves of the traversal. The roots represent variables we can access at the
// target function's call site; the leaves represent actual model variables, which we
// need to transform to and from tensors.
template <typename T>
struct Optic : public OpticBase {

    virtual tensor to(T* /*source*/) = 0;
    virtual void from(tensor /*source*/, T* /*dest*/) = 0;

    virtual tensor unsafe_to(phasm::any_ptr source) override {
        return to(source.get<T>());
    };
    virtual void unsafe_from(tensor source, phasm::any_ptr dest) override {
        return from(source, dest.get<T>());
    };
};


/// The simplest optic. Converts a single primitive value into a Torch Tensor. Can be used as-is (for primitive inputs)
/// or composed with another optic (e.g. a Lens) to traverse from a locally visible variable to a nested primitive.
/// Primitive ought to be an Iso<T,Tensor>. But (unlike real optics) our optics are constrained to always have a Primitive
/// or PrimitiveArray at the back of the composition chain, so we can throw away the second template parameter.
/// Probably a more rigorous way to do this would be to make TensorIso be an Iso<T,Tensor>, so the overall
/// composition chain becomes an Iso<T, Tensor>.
/// TODO: Restrict T to _actual_ primitives
/// TODO: Add the ability to choose a different dtype for the tensor other than T.
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
class TensorIso : public Optic<T> {
    const std::vector<int64_t> m_shape;
    const size_t m_length;

public:
    explicit TensorIso(std::vector<int64_t> shape = {}) :
        m_shape(std::move(shape)),
        m_length(std::accumulate(m_shape.begin(), m_shape.end(), 1ull, [](size_t a, size_t b){return a*b;}))
    {
        OpticBase::consumes = demangle<T>();
        OpticBase::produces = "tensor";
    };
    TensorIso(const TensorIso& other) = default;

    std::vector<int64_t> shape() override { return m_shape; }

    tensor to(T* source) override {
        return phasm::tensor(source, m_shape);
    }
    void from(tensor source, T* dest) override {
        T* ptr = source.get<T>();
        for (size_t i=0; i<m_length; ++i) {
            dest[i] = ptr[i];
        }
        // TODO: This only works if the tensor is consecutive. Find a clever way to fix this problem
    }
    TensorIso* clone() override {
        return new TensorIso<T>(*this);
    }
};


// Field is an Optic that accepts a struct of type StructT, knows how to extract a field of type FieldT from the StructT,
// and forwards the field to an inner optic of type OpticT that accepts FieldT. The user needs to compose

template <typename StructT, typename FieldT>
class Lens : public Optic<StructT> {
    Optic<FieldT>* m_optic = nullptr;
    std::function<FieldT*(StructT*)> m_accessor;
public:
    Lens(Optic<FieldT>* optic, std::function<FieldT*(StructT*)> accessor) : m_optic(optic), m_accessor(accessor) {
        OpticBase::consumes = demangle<StructT>();
        OpticBase::produces = demangle<FieldT>();
    };
    Lens(const Lens& other) = default;

    std::vector<int64_t> shape() override { return m_optic->shape(); }
    tensor to(StructT* source) override {
        return m_optic->to(m_accessor(source));
    }
    void from(tensor source, StructT* dest) override {
        return m_optic->from(source, m_accessor(dest));
    }
    void attach(Optic<FieldT>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void unsafe_use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<FieldT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
    Lens* clone() override {
        return new Lens<StructT, FieldT>(*this);
    }
};


template <typename T>
class ArrayTraversal : public Optic<T> {
    Optic<T>* m_optic;
    int64_t m_length;
public:
    ArrayTraversal(Optic<T>* optic, size_t length) : m_optic(optic), m_length(length) {
        OpticBase::consumes = demangle<T>();
        OpticBase::produces = demangle<T>();
    }
    ArrayTraversal(const ArrayTraversal& other) = default;

    std::vector<int64_t> shape() override {
        std::vector<int64_t> result {m_length};
        auto inner_shape = m_optic->shape();
        result.insert(result.end(), inner_shape.begin(), inner_shape.end());
        return result;
    }
    tensor to(T* source) override {
        std::vector<tensor> tensors;
        for (int i=0; i<m_length; ++i) {
            tensors.push_back(m_optic->to(source+i));
        }
        return stack(tensors);
    }
    void from(tensor source, T* dest) override {
        auto unstacked = unstack(source);
        for (int i=0; i<m_length; ++i) {
            m_optic->from(unstacked[i], dest+i);
        }
    }
    void attach(Optic<T>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void unsafe_use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<T>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
    ArrayTraversal* clone() override {
        return new ArrayTraversal<T>(*this);
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
    tensor to(OuterT* source) override {
        IteratorT it(source);
        std::vector<tensor> tensors;

        for (int i=0; i<m_length; ++i) {
            tensors.push_back(m_optic->to(it.Current()));
            it.Next();  // Returns true because container has m_length entries XD
        }
        return stack(tensors);
    }
    void from(tensor source, OuterT* dest) override {
        auto unstacked = unstack(source);
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
    void unsafe_use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<InnerT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
    Traversal* clone() override {
        return new Traversal(*this);
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


} // namespace phasm




#endif //SURROGATE_TOOLKIT_OPTICS_H
