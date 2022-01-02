
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_CANDIDATE_H
#define SURROGATE_TOOLKIT_CANDIDATE_H


template <typename RetT, typename ...ArgsT>
struct Surrogate {

    struct Input {
        std::string name;
        virtual size_t capture(ArgsT&&... args) = 0;
	virtual bool put_back(size_t index, ArgsT... args) = 0;
	virtual ~Input() = default;
    };

    template <typename T>
    struct InputT : public Input {
	std::vector<T> samples;
	std::function<T*(ArgsT&&... args)> accessor;

	size_t capture(ArgsT&&... args) override {
	    T dest = *accessor(std::forward<ArgsT>(args)...);
	    samples.push_back(dest);
	    return samples.size() - 1;
	}

	bool put_back(size_t index, ArgsT... args) override {
	    if (index >= samples.size()) return false;
	    *accessor(std::forward<ArgsT>(args)...) = samples[index];
	    return true;
	}
    };

    struct Output {
	std::string name;
	virtual size_t capture(RetT ret, ArgsT&&... args) = 0;
	virtual ~Output() = default;
    };

    template <typename T>
    struct OutputT : public Output {
	std::vector<T> samples;
	std::function<T(RetT ret, ArgsT&&... args)> getter;

	size_t capture(RetT ret, ArgsT&&... args) override {
	    T val = getter(ret, std::forward<ArgsT>(args)...);
	    samples.push_back(val);
	    return samples.size() - 1;
	}
    };

    std::function<RetT(ArgsT&&...)> original_function;
    std::vector<std::unique_ptr<Input>> inputs;
    std::vector<std::unique_ptr<Output>> outputs;


    explicit Surrogate(std::function<RetT(ArgsT&&...)> f) : original_function(f) {};

    template<typename T>
    void input(std::string param_name, std::function<T*()> accessor) {
	auto input = new InputT<T>;
	input->name = param_name;
	input->accessor = [=](ArgsT... args) { return accessor(); };
	inputs.push_back(std::unique_ptr<Input>(input));
    }

    template <typename T, int Index>
    void input(std::string param_name) {
        auto input = new InputT<T>;
        input->name = param_name;
        input->accessor = [](ArgsT... args) {
            auto t = std::make_tuple(args...);
            return &(std::get<Index>(t));
        };
        inputs.push_back(std::unique_ptr<Input>(input));
    }

    template<typename T>
    void output(std::string param_name, std::function<T()> getter) {
	auto output = new OutputT<T>;
	output->name = param_name;
	output->getter = [=](RetT, ArgsT...) { return getter(); };
	outputs.push_back(std::unique_ptr<Output>(output));
    }

    template <typename T, int Index>
    void output(std::string param_name) {

	auto output = new OutputT<T>;
	output->name = param_name;
	output->getter = [](RetT, ArgsT... args) {
	    auto t = std::make_tuple(args...);
	    return std::get<Index>(t);
	};
	outputs.push_back(std::unique_ptr<Output>(output));
    }

    template <typename T, int Index>
    void input_output(std::string param_name) {
        input<T,Index>(param_name);
        output<T,Index>(param_name);
    }

    template <typename T>
    void returns(std::string retval_name) {
        auto output = new OutputT<T>;
        output->name = retval_name;
        output->getter = [](RetT ret, ArgsT...) { return ret; };
        outputs.push_back(std::unique_ptr<Output>(output));
    }

    template <typename T>
    T getCapturedInput(size_t sample_index, size_t parameter_index) {
	auto* param = dynamic_cast<InputT<T>*>(inputs[parameter_index].get());
	return param->samples[sample_index];
    }

    template <typename T>
    T getCapturedOutput(size_t sample_index, size_t parameter_index) {
	auto* param = dynamic_cast<OutputT<T>*>(outputs[parameter_index].get());
	return param->samples[sample_index];
    }

    void load_args_into_params() {};

    void train_model_on_samples() {};

    RetT call_model(ArgsT...) {
    }

    RetT call_original(ArgsT&&... args) {
        return original_function(std::forward<ArgsT>(args)...);
    }

    /// Capturing only needs rvalues. This won't train, but merely update the samples associated
    RetT call_original_and_capture_sample(ArgsT&&... args) {
	// // capture input and outputs
	for (auto& input: inputs) {
	    input->capture(std::forward<ArgsT>(args)...);
	}
	RetT ret = original_function(std::forward<ArgsT>(args)...);
	for (auto& output: outputs) {
	    output->capture(ret, std::forward<ArgsT>(args)...);
	}
	return ret;
    }

    /// From some set of sampled _inputs_,
    /// args need to be lvalues because we need a slot we can put our samples in to before calling the function
    void call_original_with_sample_inputs(ArgsT&&...) {};



    /// Calls both the original and model and compares the results.
    /// If model result lies outside the desired accuracy, add input to training data.
    /// If we have enough training data, use it to improve the model.
    RetT operator()(ArgsT&&... args) {
	// Capture inputs
	// Run original
	// Capture outputs
	// Reset input
	// Run model
	// Capture outputs
        return std::apply(original_function, args...);
    }
};


template <typename ReturnT, typename... Args>
using Signature = ReturnT(Args...);

template<typename ReturnT, typename... Args>
Surrogate<ReturnT, Args...> make_surrogate(Signature<ReturnT, Args...> f) {
    return Surrogate<ReturnT, Args...>(std::function<ReturnT(Args...)>(f));
}

#endif //SURROGATE_TOOLKIT_CANDIDATE_H
