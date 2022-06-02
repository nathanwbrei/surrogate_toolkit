
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_H
#define SURROGATE_TOOLKIT_SURROGATE_H

#include <set>
#include <utility>
#include <variant>
#include <vector>
#include <functional>

#include "range.h"
#include "model.h"
#include "call_site_variable.h"


namespace phasm {

class Surrogate {
public:
    friend class Model;

    enum class CallMode {
        NotSet, UseOriginal, UseModel, CaptureAndTrain, CaptureAndDump, CaptureAndSummarize
    };

private:
    static inline CallMode s_callmode = CallMode::NotSet;
    std::function<void(void)> original_function;
    std::shared_ptr<Model> model;
public:
    std::vector<std::shared_ptr<CallSiteVariable>> callsite_vars;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> callsite_var_map;

public:
    explicit Surrogate(std::function<void(void)> f, std::shared_ptr<Model> model);

    static void set_call_mode(CallMode callmode) { s_callmode = callmode; }

    template<typename T>
    void bind(std::string param_name, T *slot) {
        auto csv = callsite_var_map.find(param_name);
        if (csv == callsite_var_map.end()) {
            throw std::runtime_error("No such callsite variable specified in model");
        }
        if (csv->second->binding.get<T>() != nullptr) {
            throw std::runtime_error("Callsite variable already set! Possibly a global var");
        }
        csv->second->binding = slot;
    }

    std::shared_ptr<CallSiteVariable> get_binding(size_t index) {
        if (index >= callsite_vars.size()) { throw std::runtime_error("Index out of range for callsite var binding"); }
        return callsite_vars[index];
    }

    std::shared_ptr<CallSiteVariable> get_binding(std::string name) {
        auto pair = callsite_var_map.find(name);
        if (pair == callsite_var_map.end()) { throw std::runtime_error("Invalid input parameter name"); }
        return pair->second;
    }

    template<typename T>
    T get_captured_input(size_t sample_index, size_t parameter_index) {
        auto param = model->get_input(parameter_index);
        torch::Tensor result = param->training_inputs[sample_index];
        return *result.data_ptr<T>();

        // Unpack as single T. This isn't valid when captures is a non zero-dimensional tensor,
        // but this is only used for test cases anyhow and should be removed pretty soon.
        // TODO: Remove get_captured_output completely
    }

    template<typename T>
    T get_captured_output(size_t sample_index, size_t parameter_index) {
        auto param = model->get_output(parameter_index);
        torch::Tensor result = param->training_outputs[sample_index];
        return *result.data_ptr<T>();

        // Unpack as single T. This isn't valid when captures is a non zero-dimensional tensor,
        // but this is only used for test cases anyhow and should be removed pretty soon.
        // TODO: Remove get_captured_output completely
    }

/*
    void bind(const std::vector<void*>& input_pointers, const std::vector<void*>& output_pointers) {
        size_t inputs_size = model->inputs.size();
        size_t outputs_size = model->outputs.size();
        if (inputs_size != input_pointers.size()) {
            throw std::range_error("Wrong size: input_pointers");
        }
	if (outputs_size != output_pointers.size()) {
	    throw std::range_error("Wrong size: output_pointers");
	}
	for (size_t i = 0; i<inputs_size; ++i) {
	    model->inputs[i]->bind(input_pointers[i]);
	}
	for (size_t i = 0; i<outputs_size; ++i) {
	    model->outputs[i]->bind(output_pointers[i]);
	}
    }
*/

    /// call() looks at PHASM_CALL_MODE env var to decide what to do
    void call();

    void call_original() {
        original_function();
    };

    void call_model() {
        model->infer(*this);
    };

    void capture_input_distribution() {
    }

    /// Capturing only needs rvalues. This won't train, but merely update the samples associated
    void call_original_and_capture() {
        for (auto &input: callsite_vars) {
            input->captureAllTrainingInputs();
        }
        original_function();
        for (auto &output: callsite_vars) {
            output->captureAllTrainingOutputs();
        }
        model->captured_rows++;
    }

};

} // namespace phasm


#endif //SURROGATE_TOOLKIT_SURROGATE_H
