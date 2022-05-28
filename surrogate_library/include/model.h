
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "model_variable.h"
#include "call_site_variable.h"
#include <iosfwd>  // Forward decls for std::ostream
#include <memory>

class Surrogate;

/// There should be exactly one Model in your codebase for each unique function that you wish to surrogate.
/// Contrast this with Surrogate. There should be one Surrogate for each call site of that function,
/// and each of these Surrogates delegate to the same underlying model.
class Model {
    friend class Surrogate;

protected:
    std::vector<std::shared_ptr<CallSiteVariable>> callsite_vars;
    std::vector<std::shared_ptr<ModelVariable>> inputs;
    std::vector<std::shared_ptr<ModelVariable>> outputs;
    std::map<std::string, std::shared_ptr<ModelVariable>> input_map;
    std::map<std::string, std::shared_ptr<ModelVariable>> output_map;
    size_t captured_rows = 0;

public:
    Model() = default;

    virtual ~Model() = default; // We want to be able to inherit from this

    // Initialize the underlying neural net once all the inputs and outputs are known
    virtual void initialize() {};

    // Performs tasks such as training or writing to CSV, right before the model gets destroyed
    void finalize();

    size_t get_capture_count() const;

    // Train takes all of the captures associated with each parameter
    virtual void train_from_captures() = 0;

    // Infer takes the sample associated with each parameter
    virtual void infer(Surrogate&) = 0;

    void dump_captures_to_csv(std::ostream&);

    void dump_ranges(std::ostream&);

    virtual void save();

    template <typename T>
    void input(std::string param_name, optics::Optic<T>* accessor=new optics::Primitive<T>(), Range<float> range = Range<float>());

    template<typename T>
    void output(std::string param_name, optics::Optic<T>* accessor=new optics::Primitive<T>());

    template<typename T>
    void input_output(std::string param_name, optics::Optic<T>* accessor=new optics::Primitive<T>(), Range<float> range = Range<float>());

    std::shared_ptr<ModelVariable> get_input(size_t position);

    std::shared_ptr<ModelVariable> get_input(std::string param_name);

    std::shared_ptr<ModelVariable> get_output(size_t position);

    std::shared_ptr<ModelVariable> get_output(std::string param_name);
};



template<typename T>
void Model::input(std::string param_name, optics::Optic<T> *accessor, Range<float> range) {
    auto input = std::make_shared<ModelVariable>();
    input->name = param_name;
    input->accessor = accessor;
    input->range = std::move(range);
    inputs.push_back(input);
    if (input_map.find(param_name) != input_map.end()) {
        throw std::runtime_error("Input parameter already exists!");
    }
    input_map[param_name] = input;
}

template<typename T>
void Model::output(std::string param_name, optics::Optic<T>* accessor) {
    auto output = std::make_shared<ModelVariable>();
    output->name = param_name;
    output->accessor = accessor;
    outputs.push_back(output);
    if (output_map.find(param_name) != output_map.end()) {
        throw std::runtime_error("Output parameter already exists!");
    }
    output_map[param_name] = output;
}

template<typename T>
void Model::input_output(std::string param_name, optics::Optic<T>* accessor=new optics::Primitive<T>(), Range<float> range = Range<float>()) {
    input<T>(param_name, accessor, range);
    output<T>(param_name, accessor);
}

#endif //SURROGATE_TOOLKIT_MODEL_H
