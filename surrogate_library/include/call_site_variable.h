
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "model_variable.h"
#include "optics.h"

struct CallSiteVariable {
    std::string name;
    bool is_input = false;
    bool is_output = false;
    virtual torch::Tensor get_tensor() = 0;
    virtual void put_tensor(torch::Tensor) = 0;
    virtual void capture() = 0;
    virtual ~CallSiteVariable() = default;
};

template <typename T>
struct CallSiteVariableT : public CallSiteVariable {

    std::shared_ptr<ModelVariableT<T>> parameter;
    T* binding_root = nullptr;  // This has to be either a global or a stack variable. Possibly the root of a nested data structure.

    torch::Tensor get_tensor() override {
        return parameter->accessor->to(binding_root);
    }

    void put_tensor(torch::Tensor t) override {
        parameter->accessor->from(t, binding_root);
    }

    void capture() override {
        torch::Tensor data = parameter->accessor->to(binding_root);
        parameter->captures.push_back(std::move(data));
    }

};



#endif //SURROGATE_TOOLKIT_BINDINGS_H
