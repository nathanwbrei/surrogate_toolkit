
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_BINDING_VISITOR_H
#define SURROGATE_TOOLKIT_BINDING_VISITOR_H

#include <iostream>

/// We want to be able to perform operations across all bindings
/// particularly when we don't know its subtype.

template <typename T> struct InputBindingT;

struct InputBindingVisitor {

    virtual void visit(InputBindingT<float>&) {};
    virtual void visit(InputBindingT<double>&) {};
    virtual void visit(InputBindingT<bool>&) {};
    virtual void visit(InputBindingT<int8_t>&) {};
    virtual void visit(InputBindingT<int16_t>&) {};
    virtual void visit(InputBindingT<int32_t>&) {};
    virtual void visit(InputBindingT<int64_t>&) {};
    virtual void visit(InputBindingT<uint8_t>&) {};
    virtual void visit(InputBindingT<uint16_t>&) {};
    virtual void visit(InputBindingT<uint32_t>&) {};
    virtual void visit(InputBindingT<uint64_t>&) {};
};

template <typename T> struct OutputBindingT;

struct OutputBindingVisitor {

    virtual void visit(OutputBindingT<float>&) {};
    virtual void visit(OutputBindingT<double>&) {};
    virtual void visit(OutputBindingT<bool>&) {};
    virtual void visit(OutputBindingT<int8_t>&) {};
    virtual void visit(OutputBindingT<int16_t>&) {};
    virtual void visit(OutputBindingT<int32_t>&) {};
    virtual void visit(OutputBindingT<int64_t>&) {};
    virtual void visit(OutputBindingT<uint8_t>&) {};
    virtual void visit(OutputBindingT<uint16_t>&) {};
    virtual void visit(OutputBindingT<uint32_t>&) {};
    virtual void visit(OutputBindingT<uint64_t>&) {};
};

#endif //SURROGATE_TOOLKIT_BINDING_VISITOR_H
