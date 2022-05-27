
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_VARIABLE_H
#define SURROGATE_TOOLKIT_MODEL_VARIABLE_H

#include "range.h"
#include "optics.h"
#include <torch/torch.h>

template <typename T> struct ModelVariableT;

enum class ParameterCategory {
    Continuous, Discrete, Categorical, Text
};

struct ModelVariableVisitor {
    virtual void visit(ModelVariableT<float>&) {};
    virtual void visit(ModelVariableT<double>&) {};
    virtual void visit(ModelVariableT<bool>&) {};
    virtual void visit(ModelVariableT<int8_t>&) {};
    virtual void visit(ModelVariableT<int16_t>&) {};
    virtual void visit(ModelVariableT<int32_t>&) {};
    virtual void visit(ModelVariableT<int64_t>&) {};
    virtual void visit(ModelVariableT<uint8_t>&) {};
    virtual void visit(ModelVariableT<uint16_t>&) {};
    virtual void visit(ModelVariableT<uint32_t>&) {};
    virtual void visit(ModelVariableT<uint64_t>&) {};
};

struct ModelVariable {
    std::string name;
    ParameterCategory category = ParameterCategory::Continuous;
    std::vector<torch::Tensor> captures;
    virtual std::vector<int64_t> shape() = 0;
    virtual ~ModelVariable() = default;
    virtual void accept(ModelVariableVisitor& v) = 0;
};

template <typename T>
struct ModelVariableT : public ModelVariable {
    optics::Optic<T>* accessor = nullptr;
    T* global = nullptr;
    Range<float> range;
    std::vector<int64_t> shape() override {
        if (accessor == nullptr) { throw std::runtime_error("InputT needs an accessor"); }
        return accessor->shape();
    }
    void accept(ModelVariableVisitor& v) override {
        v.visit(*this);
    }
};


#endif //SURROGATE_TOOLKIT_MODEL_VARIABLE_H
