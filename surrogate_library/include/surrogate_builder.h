
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
#define SURROGATE_TOOLKIT_SURROGATE_BUILDER_H

#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "parameter.h"
#include "range.h"
#include "model.h"


struct Surrogate {

    std::function<void(void)> original_function;
    std::shared_ptr<Model> model;


    explicit Surrogate(std::function<void(void)> f) : original_function(std::move(f)) {
        model = std::make_shared<Model>();
    };

    template <typename T>
    void input(std::string param_name, T* slot, Range<T> range = Range<T>()) {
	auto input = new InputT<T>;
	input->name = param_name;
	input->range = std::move(range);
	input->accessor = [=](){return slot;};
	model->inputs.push_back(std::unique_ptr<Input>(input));
    }

    template<typename T>
    void input(std::string param_name, std::function<T*()> accessor, Range<T> range = Range<T>()) {
        auto input = new InputT<T>;
        input->name = param_name;
        input->range = std::move(range);
        input->accessor = accessor;
        model->inputs.push_back(std::unique_ptr<Input>(input));
    }

    template<typename T>
    void output(std::string param_name, T* slot) {
	auto output = new OutputT<T>;
	output->name = param_name;
	output->getter = [=](){ return *slot; };
	model->outputs.push_back(std::unique_ptr<Output>(output));
    }

    template<typename T>
    void output(std::string param_name, std::function<T()> getter) {
        auto output = new OutputT<T>;
        output->name = param_name;
        output->getter = [=]() { return getter(); };
        model->outputs.push_back(std::unique_ptr<Output>(output));
    }

    template<typename T>
    void input_output(std::string param_name, T* slot, Range<T> range = Range<T>()) {
        input<T>(param_name, slot, range);
        output<T>(param_name, slot);
    }


    template <typename T>
    void setSampleInput(size_t parameter_index, T sample_value) {
        auto* param = dynamic_cast<InputT<T>*>(model->inputs[parameter_index].get());
        param->sample = sample_value;
    }

    template <typename T>
    T getCapturedInput(size_t sample_index, size_t parameter_index) {
        auto* param = dynamic_cast<InputT<T>*>(model->inputs[parameter_index].get());
        return param->captures[sample_index];
    }

    template <typename T>
    T getCapturedOutput(size_t sample_index, size_t parameter_index) {
        auto* param = dynamic_cast<OutputT<T>*>(model->outputs[parameter_index].get());
        return param->captures[sample_index];
    }

    void load_args_into_params() {};

    void train_model_on_samples() {};

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

    void call_original() {
        original_function();
    };

    void call_model() {
        // model.infer();
    };


    void capture_input_distribution() {
	for (auto& input: model->inputs) {
	    input->capture_range();
	}
    }


    /// Capturing only needs rvalues. This won't train, but merely update the samples associated
    void call_original_and_capture() {
        for (auto& input: model->inputs) {
            input->capture_value();
        }
        original_function();
        for (auto& output: model->outputs) {
            output->capture();
        }
    }

    /// From some set of sampled _inputs_ (to be generated algorithmically), call the original function.
    /// This entails writing _into_ args, which means they need to be lvalues
    void call_original_with_sampled_inputs() {
        for (auto& input: model->inputs) {
            input->deploy_sample_value();
            input->capture_value();
        }
	original_function();
	for (auto& output: model->outputs) {
	    output->capture();
	}
    };
};

inline Surrogate make_surrogate(std::function<void()> f) {
    return Surrogate(std::move(f));
}

#endif //SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
