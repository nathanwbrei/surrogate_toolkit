
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "parameter.h"
#include "optics.h"

#include "binding_visitor.h"

struct InputBinding {
    virtual void capture() = 0;
    virtual ~InputBinding() = default;
    virtual void accept(InputBindingVisitor& v) = 0;
};

template <typename T>
struct InputBindingT : public InputBinding {

    std::shared_ptr<InputT<T>> parameter;
    T* binding_root;  // This has to be either a global or a stack variable. Possibly the root of a nested data structure.
    optics::Optic<T>* accessor;  // This traverses the data structure at T* to obtain a tensor of primitives


    void capture() override {
        torch::Tensor data = accessor->to(binding_root);
        parameter->captures.push_back(std::move(data));
    }

    void accept(InputBindingVisitor& v) override {
        v.visit(*this);
    }

};


struct OutputBinding {
public:
    virtual void capture() = 0;
    virtual ~OutputBinding() = default;
    virtual void accept(OutputBindingVisitor&) = 0;
};


template <typename T>
struct OutputBindingT : public OutputBinding {
    std::shared_ptr<OutputT<T>> parameter;
    T* binding_root;
    optics::Optic<T>* accessor;  // This traverses the data structure at T* and fills it from a tensor of primitives

    void capture() override {
        torch::Tensor data = accessor->to(binding_root);
        parameter->captures.push_back(std::move(data));
    }

    void accept(OutputBindingVisitor& v) override {
        v.visit(*this);
    }

};


#endif //SURROGATE_TOOLKIT_BINDINGS_H
