
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_VARIABLE_H
#define SURROGATE_TOOLKIT_MODEL_VARIABLE_H

#include "range.h"
#include "optics.h"
#include <torch/torch.h>

template <typename T> struct ModelVariableT;

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
// N.B. This won't work either because ModelVariable is
// parameterized on the root type (which is unconstrained),
// not on the leaf type (which can only be primitives).

struct ModelVariable {
    std::string name;
    std::vector<torch::Tensor> training_captures;
    torch::Tensor inference_capture;
    virtual std::vector<int64_t> shape() = 0;
    virtual ~ModelVariable() = default;
    virtual void accept(ModelVariableVisitor& v) = 0;
};

template <typename RootT>
struct ModelVariableT : public ModelVariable {
    optics::Optic<RootT>* accessor = nullptr;
    RootT* global = nullptr;
    Range<float> range;
    std::vector<int64_t> shape() override {
        if (accessor == nullptr) { throw std::runtime_error("InputT needs an accessor"); }
        return accessor->shape();
    }
    void accept(ModelVariableVisitor& v) override {
        v.visit(*this);
    }
    void capture_training_data(RootT* binding) {
        torch::Tensor data = accessor->to(binding);
        training_captures.push_back(data);
    }
    void get_inference_data(RootT* binding) {
        inference_capture = accessor->to(binding);
    }
    void put_inference_data(RootT* binding) {
        accessor->from(inference_capture, binding);
    }

};


#endif //SURROGATE_TOOLKIT_MODEL_VARIABLE_H
