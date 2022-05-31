
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
    std::vector<std::shared_ptr<ModelVariable>> model_vars;


    inline void capture_all_training_inputs() {
        for (auto model_var : model_vars) {
            if (model_var->is_input) {
                model_var->capture_training_data(binding);
            }
        }
    }

    inline void capture_all_training_outputs() {
        for (auto model_var : model_vars) {
            if (model_var->is_output) {
                model_var->capture_training_data(binding);
            }
        }
    }

    inline void get_all_inference_inputs() {
        for (auto model_var : model_vars) {
            if (model_var->is_input) {
                model_var->get_inference_data(binding);
            }
        }
    }

    inline void put_all_inference_outputs() {
        for (auto model_var : model_vars) {
            if (model_var->is_output) {
                model_var->put_inference_data(binding);
            }
        }
    }
};



#endif //SURROGATE_TOOLKIT_BINDINGS_H
