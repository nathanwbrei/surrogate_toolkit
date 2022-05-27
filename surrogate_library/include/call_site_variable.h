
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "model_variable.h"
#include "optics.h"
#include "any_ptr.hpp"

struct CallSiteVariable {

    std::string name;
    phasm::any_ptr binding;
    std::vector<ModelVariable*> model_vars;


    void capture_all_training_data() {
        for (auto model_var : model_vars) {
            model_var->capture_training_data(binding);
        }
    }

    void get_all_inference_data() {
        for (auto model_var : model_vars) {
            model_var->get_inference_data(binding);
        }
    }

    void put_all_inference_data() {
        for (auto model_var : model_vars) {
            model_var->put_inference_data(binding);
        }
    }
};



#endif //SURROGATE_TOOLKIT_BINDINGS_H
