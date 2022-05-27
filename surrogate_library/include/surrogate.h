
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


class Surrogate {
public:
    friend class Model;
    enum class CallMode { NotSet, UseOriginal, UseModel, CaptureAndTrain, CaptureAndDump, CaptureAndSummarize };

private:
    static inline CallMode s_callmode = CallMode::NotSet;
    std::function<void(void)> original_function;
    std::shared_ptr<Model> model;
public:
    std::vector<std::shared_ptr<CallSiteVariable>> input_bindings;
    std::vector<std::shared_ptr<CallSiteVariable>> output_bindings;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> input_binding_map;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> output_binding_map;

public:
    explicit Surrogate(std::function<void(void)> f, std::shared_ptr<Model> model);
    static void set_call_mode(CallMode callmode) {s_callmode = callmode;}

    template <typename T>
    void bind_input(std::string param_name, T* slot) {
	auto input = std::make_shared<CallSiteVariableT<T>>();
	input->binding_root = slot;
	input->parameter = model->get_input<T>(param_name);
	input_bindings.push_back(input);
	if (input_binding_map.find(param_name) != input_binding_map.end()) {
	    throw ("Input binding already exists!");
	}
	input_binding_map[param_name] = input;
    }

    template<typename T>
    void bind_output(std::string param_name, T* slot) {
	auto output = std::make_shared<CallSiteVariableT<T>>();
	output->binding_root = slot;
        output->parameter = model->get_output<T>(param_name);
        output_bindings.push_back(output);
        if (output_binding_map.find(param_name) != output_binding_map.end()) {
            throw ("Output binding already exists!");
        }
        output_binding_map[param_name] = output;
    }

    template<typename T>
    void bind_input_output(std::string param_name, T* slot) {
        bind_input<T>(param_name, slot);
        bind_output<T>(param_name, slot);
    }

    template<typename T>
    std::shared_ptr<CallSiteVariableT<T>> get_input_binding(size_t index) {
        if (index >= input_bindings.size()) {
            throw "Index out of range for input binding";
        }
        auto input = input_bindings[index];
        auto downcasted = std::dynamic_pointer_cast<CallSiteVariableT<T>>(input);
        if (downcasted == nullptr) {
            throw "Wrong type for input binding";
        }
        return downcasted;
    }

    template<typename T>
    std::shared_ptr<CallSiteVariableT<T>> get_input_binding(std::string name) {
        auto pair = input_binding_map.find(name);
        if (pair == input_binding_map.end()) {
            throw ("Invalid input parameter name");
        }
        auto downcasted_input = std::dynamic_pointer_cast<CallSiteVariableT<T>>(pair->second);
        if (downcasted_input == nullptr) {
            throw("Wrong type for input parameter");
        }
        return downcasted_input;
    }

    template<typename T>
    std::shared_ptr<CallSiteVariableT<T>> get_output_binding(size_t index) {
        if (index >= output_bindings.size()) {
            throw "Index out of range for output binding";
        }
        auto output = output_bindings[index];
        auto downcasted = std::dynamic_pointer_cast<CallSiteVariableT<T>>(output);
        if (downcasted == nullptr) {
            throw "Wrong type for output binding";
        }
        return downcasted;
    }

    template<typename T>
    std::shared_ptr<CallSiteVariableT<T>> get_output_binding(std::string name) {
        auto pair = output_binding_map.find(name);
        if (pair == output_binding_map.end()) {
            throw ("Invalid output parameter name");
        }
        auto downcasted_output = std::dynamic_pointer_cast<CallSiteVariableT<T>>(pair->second);
        if (downcasted_output == nullptr) {
            throw("Wrong type for input parameter");
        }
        return downcasted_output;
    }

    template <typename T>
    T get_captured_input(size_t sample_index, size_t parameter_index) {
        auto param = model->get_input<T>(parameter_index);
        torch::Tensor result = param->captures[sample_index];
        return *result.data_ptr<T>();

        // Unpack as single T. This isn't valid when captures is a non zero-dimensional tensor,
        // but this is only used for test cases anyhow and should be removed pretty soon.
        // TODO: Remove get_captured_output completely
    }

    template <typename T>
    T get_captured_output(size_t sample_index, size_t parameter_index) {
        auto param = model->get_output<T>(parameter_index);
        torch::Tensor result = param->captures[sample_index];
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
        for (auto& input: input_bindings) {
            input->capture();
        }
        original_function();
        for (auto& output: output_bindings) {
            output->capture();
        }
        model->captured_rows++;
    }

};


#endif //SURROGATE_TOOLKIT_SURROGATE_H
