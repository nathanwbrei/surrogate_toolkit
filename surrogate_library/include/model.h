
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "parameter.h"
#include <iosfwd>  // Forward decls for std::ostream
#include <memory>

class Surrogate;

class Model { // This is an abstract class
    friend class Surrogate;

    std::vector<std::shared_ptr<Input>> inputs;
    std::vector<std::shared_ptr<Output>> outputs;
    std::map<std::string, std::shared_ptr<Input>> input_map;
    std::map<std::string, std::shared_ptr<Output>> output_map;
    size_t captured_rows = 0;

public:
    template <typename T>
    void input(std::string param_name, Range<float> range = Range<float>()) {
        auto input = std::make_shared<InputT<T>>();
        input->name = param_name;
        input->range = std::move(range);
        inputs.push_back(input);
        if (input_map.find(param_name) != input_map.end()) {
            throw "Input parameter already exists!";
        }
        input_map[param_name] = input;
    }

    template<typename T>
    void output(std::string param_name) {
        auto output = std::make_shared<OutputT<T>>();
        output->name = param_name;
        outputs.push_back(output);
        if (output_map.find(param_name) != output_map.end()) {
            throw "Output parameter already exists!";
        }
        output_map[param_name] = output;
    }

    template<typename T>
    void input_output(std::string param_name, Range<float> range = Range<float>()) {
        input<T>(param_name, range);
        output<T>(param_name);
    }

    template <typename T>
    std::shared_ptr<InputT<T>> get_input(size_t position) {
        if (position >= inputs.size()) {
            throw("Parameter index out of bounds");
        }
        auto input = inputs[position];
        auto downcasted_input = std::dynamic_pointer_cast<InputT<T>>(input);
        if (downcasted_input == nullptr) {
            throw("Wrong type for input parameter");
        }
        return downcasted_input;
    }

    template <typename T>
    std::shared_ptr<InputT<T>> get_input(std::string param_name) {
        auto pair = input_map.find(param_name);
        if (pair == input_map.end()) {
            throw ("Invalid input parameter name");
        }
        auto downcasted_input = std::dynamic_pointer_cast<InputT<T>>(pair->second);
        if (downcasted_input == nullptr) {
            throw("Wrong type for input parameter");
        }
        return downcasted_input;
    }

    template <typename T>
    std::shared_ptr<OutputT<T>> get_output(size_t position) {
        if (position >= outputs.size()) {
            throw("Output parameter index out of bounds");
        }
        auto output = outputs[position];
        auto downcasted_output = std::dynamic_pointer_cast<OutputT<T>>(output);
        if (downcasted_output == nullptr) {
            throw("Wrong type for output parameter");
        }
        return downcasted_output;
    }

    template <typename T>
    std::shared_ptr<OutputT<T>> get_output(std::string param_name) {

        auto pair = output_map.find(param_name);
        if (pair == output_map.end()) {
            throw "Invalid output parameter name";
        }
        auto downcasted_output = std::dynamic_pointer_cast<OutputT<T>>(pair->second);
        if (downcasted_output == nullptr) {
            throw "Wrong type for output parameter";
        }
        return downcasted_output;
    }

    size_t get_capture_count() { return captured_rows; }

    // We want to be able to inherit from this
    virtual ~Model() = default;

    // Train takes all of the captures associated with each parameter
    virtual void train(torch::Tensor) {};

    // Infer takes the sample associated with each parameter
    virtual void infer(Surrogate&) {};

    void dump_captures_to_csv(std::ostream&);

};


#endif //SURROGATE_TOOLKIT_MODEL_H
