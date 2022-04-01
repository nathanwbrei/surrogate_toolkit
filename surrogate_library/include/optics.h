
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_OPTICS_H
#define SURROGATE_TOOLKIT_OPTICS_H

#include <torch/torch.h>

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

/// Abstract base class for accessing a primitive or a fixed-size data structure
struct Accessor {
    virtual std::vector<size_t> get_shape() = 0;
    virtual void fill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) = 0;
    virtual void unfill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) = 0;
};

template <typename T>
class PrimitiveAccessor : public Accessor {
    T* m_pointer;
public:
    explicit PrimitiveAccessor(T* pointer) : m_pointer(pointer) {};
    std::vector<size_t> get_shape() override { return {}; }
    void fill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) override {
        // t[indices] = *m_pointer;
    }
    void unfill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) override {
        // *m_pointer = t[indices]
    }
};

template <typename T>
class ArrayAccessor : public Accessor {
    T* m_pointer;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride; // Handles row-major vs column-major
public:
    explicit ArrayAccessor(T* pointer, size_t length) :
            m_pointer(pointer), m_shape({length}), m_stride({1}) {}

    explicit ArrayAccessor(T* pointer, size_t lengths[2], bool row_major=true) :
            m_pointer(pointer), m_shape({lengths[0], lengths[1]}) {

        if (row_major) {
            m_stride = {1,lengths[0]};
        }
        else {
            m_stride = {lengths[1],1};
        }
        // Will this transpose the matrix on us? Also, does Torch care what order as long as we are consistent?
    }

    explicit ArrayAccessor(T* pointer, std::vector<size_t> shape, std::vector<size_t> stride, bool row_major) :
    m_pointer(pointer), m_shape(shape), m_stride(stride) {}

    std::vector<size_t> get_shape() { return m_shape; }
    void fill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) override {
    }
    void unfill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) override {
    }
};

// TODO: For now we are using virtual functions but this is going to be slow
//       Experiment with making Accessor be a Concept instead of an abstract base class.
//       If we do this, the optimizer could collapse the entire Accessor composite into a pile of for loops
class PointerAccessor : public Accessor {
    Accessor* m_underlying;
public:
    PointerAccessor(Accessor* underlying) : m_underlying(underlying) {};

    std::vector<size_t> get_shape() override { return m_underlying->get_shape(); }
    void fill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) override {
        return m_underlying->fill_tensor(t, indices);
    }
    void unfill_tensor(torch::Tensor& t, const std::vector<size_t>& indices) {
        return m_underlying->unfill_tensor(t, indices);
    }
};

// TODO: We can definitely make this more efficient
template <typename Outer, typename Inner>
class StructAccessor : public Accessor {
    std::function<Inner&(const Outer&)> m_access;
public:
    StructAccessor(std::function<Inner&(const Outer&)> access) : m_access(access) {};
};


// Bundles together a bunch of different Accessors side-by-side in a Tensor.
class ProductAccessor : public Accessor {
    std::vector<Accessor*> underlying;
};


// Handles trees and linked lists
class TraversableAccessor : public Accessor {
};

// Case we haven't figured out yet: Handling sum types: unions, variants, and optionals
// If the inner types are primitives, we can expand sum<a,b,c> into tuple<a,b,c>, pad missing items with zeros.
// But what if they are nested structures?





#endif //SURROGATE_TOOLKIT_OPTICS_H
