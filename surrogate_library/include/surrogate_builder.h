
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
#define SURROGATE_TOOLKIT_SURROGATE_BUILDER_H

#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "parameter.h"
#include "parameter_range.h"

using namespace parameter;
using namespace ranges;


struct Surrogate {

    std::function<void(void)> original_function;
    std::vector<std::unique_ptr<Input>> inputs;
    std::vector<std::unique_ptr<Output>> outputs;


    explicit Surrogate(std::function<void(void)> f) : original_function(std::move(f)) {};

    template <typename T>
    void input(std::string param_name, T* slot) {
	auto input = new InputT<T>;
	input->name = param_name;
	input->accessor = [=](){return slot;};
	inputs.push_back(std::unique_ptr<Input>(input));
    }

    template<typename T>
    void input(std::string param_name, std::function<T*()> accessor) {
        auto input = new InputT<T>;
        input->name = param_name;
        input->accessor = accessor;
        inputs.push_back(std::unique_ptr<Input>(input));
    }

    template<typename T>
    void output(std::string param_name, T* slot) {
	auto output = new OutputT<T>;
	output->name = param_name;
	output->getter = [=](){ return *slot; };
	outputs.push_back(std::unique_ptr<Output>(output));
    }

    template<typename T>
    void output(std::string param_name, std::function<T()> getter) {
        auto output = new OutputT<T>;
        output->name = param_name;
        output->getter = [=]() { return getter(); };
        outputs.push_back(std::unique_ptr<Output>(output));
    }

    template<typename T>
    void input_output(std::string param_name, T* slot) {
        input<T>(param_name, slot);
        output<T>(param_name, slot);
    }


    template <typename T>
    void setSampleInput(size_t parameter_index, T sample_value) {
        auto* param = dynamic_cast<InputT<T>*>(inputs[parameter_index].get());
        param->sample = sample_value;
    }

    template <typename T>
    T getCapturedInput(size_t sample_index, size_t parameter_index) {
        auto* param = dynamic_cast<InputT<T>*>(inputs[parameter_index].get());
        return param->captures[sample_index];
    }

    template <typename T>
    T getCapturedOutput(size_t sample_index, size_t parameter_index) {
        auto* param = dynamic_cast<OutputT<T>*>(outputs[parameter_index].get());
        return param->captures[sample_index];
    }

    void load_args_into_params() {};

    void train_model_on_samples() {};

    void call_original();

    void call_model();


    /// Capturing only needs rvalues. This won't train, but merely update the samples associated
    void call_original_and_capture() {
        for (auto& input: inputs) {
            input->capture();
        }
        original_function();
        for (auto& output: outputs) {
            output->capture();
        }
    }

    /// From some set of sampled _inputs_ (to be generated algorithmically), call the original function.
    /// This entails writing _into_ args, which means they need to be lvalues
    void call_original_with_sampled_inputs() {
        for (auto& input: inputs) {
            input->deploy_sample();
            input->capture();
        }
	original_function();
	for (auto& output: outputs) {
	    output->capture();
	}
    };

};

inline Surrogate make_surrogate(std::function<void()> f) {
    return Surrogate(std::move(f));
}

#endif //SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
