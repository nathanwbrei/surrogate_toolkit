
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "model_variable.h"
#include "optics.h"

struct CallSiteVariable {

    std::string name;

    CallSiteVariable() = default;
    virtual ~CallSiteVariable() = default;
    virtual void capture_all_training_data() = 0;
    virtual void get_all_inference_data() = 0;
    virtual void put_all_inference_data() = 0;
};

template <typename T>
struct CallSiteVariableT : public CallSiteVariable {

    std::vector<ModelVariableT<T>*> model_vars;
    T* binding_root = nullptr;

    void capture_all_training_data() override {
        for (auto model_var : model_vars) {
            model_var->capture_training_data(binding_root);
        }
    }

    void get_all_inference_data() override {
        for (auto model_var : model_vars) {
            model_var->get_inference_data(binding_root);
        }
    }

    void put_all_inference_data() override {
        for (auto model_var : model_vars) {
            model_var->put_inference_data(binding_root);
        }
    }

};



#endif //SURROGATE_TOOLKIT_BINDINGS_H
