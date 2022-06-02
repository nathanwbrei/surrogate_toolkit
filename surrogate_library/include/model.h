
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "model_variable.h"
#include "call_site_variable.h"
#include <iosfwd>  // Forward decls for std::ostream
#include <memory>

class Surrogate;
namespace phasm::fluent {
class OpticBuilder;
}

/// There should be exactly one Model in your codebase for each unique function that you wish to surrogate.
/// Contrast this with Surrogate. There should be one Surrogate for each call site of that function,
/// and each of these Surrogates delegate to the same underlying model.
class Model {
    friend class Surrogate;

protected:
    std::vector<std::shared_ptr<CallSiteVariable>> callsite_vars;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> callsite_var_map;
    std::vector<std::shared_ptr<ModelVariable>> inputs;
    std::vector<std::shared_ptr<ModelVariable>> outputs;
    std::map<std::string, std::shared_ptr<ModelVariable>> input_map;
    std::map<std::string, std::shared_ptr<ModelVariable>> output_map;
    size_t captured_rows = 0;

public:
    Model() = default;
    Model(const phasm::fluent::OpticBuilder& b);

    virtual ~Model() = default; // We want to be able to inherit from this

    // Initialize the underlying neural net once all the inputs and outputs are known
    virtual void initialize() {};

    // Performs tasks such as training or writing to CSV, right before the model gets destroyed
    void finalize();

    size_t get_capture_count() const;

    // Train takes all of the captures associated with each parameter
    virtual void train_from_captures() {};

    // Infer takes the sample associated with each parameter
    virtual void infer(Surrogate&) {};

    void dump_captures_to_csv(std::ostream&);

    void dump_ranges(std::ostream&);

    virtual void save();

    template <typename T>
    void add_input(std::string param_name);

    template<typename T>
    void add_output(std::string param_name);

    template<typename T>
    void add_input_output(std::string param_name);

    template <typename T>
    void add_input(std::string call_site_var_name, optics::Optic<T>* accessor, std::string model_var_name);

    template <typename T>
    void add_output(std::string call_site_var_name, optics::Optic<T>* accessor, std::string model_var_name);

    template <typename T>
    void add_input_output(std::string call_site_var_name, optics::Optic<T>* accessor, std::string model_var_name);


    std::shared_ptr<ModelVariable> get_input(size_t position);

    std::shared_ptr<ModelVariable> get_input(std::string param_name);

    std::shared_ptr<ModelVariable> get_output(size_t position);

    std::shared_ptr<ModelVariable> get_output(std::string param_name);
};

template<typename T>
void Model::add_input(std::string call_site_var_name) {
    add_input(call_site_var_name, new optics::Primitive<T>, call_site_var_name);
}

template<typename T>
void Model::add_output(std::string call_site_var_name) {
    add_output(call_site_var_name, new optics::Primitive<T>, call_site_var_name);
}

template<typename T>
void Model::add_input_output(std::string call_site_var_name) {
    add_input_output(call_site_var_name, new optics::Primitive<T>, call_site_var_name);
}

template<typename T>
void Model::add_input(std::string call_site_var_name, optics::Optic<T> *accessor, std::string model_var_name) {
    auto input = std::make_shared<ModelVariable>();
    input->name = model_var_name;
    input->is_input = true;
    input->accessor = accessor;
    inputs.push_back(input);
    if (input_map.find(model_var_name) != input_map.end()) {
        throw std::runtime_error("Input parameter already exists!");
    }
    input_map[model_var_name] = input;

    std::shared_ptr<CallSiteVariable> csv = nullptr;
    auto pair = callsite_var_map.find(call_site_var_name);
    if (pair == callsite_var_map.end()) {
        csv = std::make_shared<CallSiteVariable>();
        csv->name = call_site_var_name;
        csv->binding = phasm::any_ptr((T*)nullptr);
        callsite_var_map[call_site_var_name] = csv;
        callsite_vars.push_back(csv);
    }
    else {
        csv = pair->second;
    }
    csv->model_vars.push_back(input);
}

template<typename T>
void Model::add_output(std::string call_site_var_name, optics::Optic<T>* accessor, std::string model_var_name) {
    auto output = std::make_shared<ModelVariable>();
    output->name = model_var_name;
    output->is_output = true;
    output->accessor = accessor;
    outputs.push_back(output);
    if (output_map.find(model_var_name) != output_map.end()) {
        throw std::runtime_error("Output parameter already exists!");
    }
    output_map[model_var_name] = output;

    std::shared_ptr<CallSiteVariable> csv = nullptr;
    auto pair = callsite_var_map.find(call_site_var_name);
    if (pair == callsite_var_map.end()) {
        csv = std::make_shared<CallSiteVariable>();
        csv->name = call_site_var_name;
        csv->binding = phasm::any_ptr((T*)nullptr);
        callsite_var_map[call_site_var_name] = csv;
        callsite_vars.push_back(csv);
    }
    else {
        csv = pair->second;
    }
    csv->model_vars.push_back(output);
}

template<typename T>
void Model::add_input_output(std::string call_site_var_name, optics::Optic<T>* accessor, std::string model_var_name) {
    add_input<T>(call_site_var_name, accessor, model_var_name);
    add_output<T>(call_site_var_name, accessor, model_var_name);
}

#endif //SURROGATE_TOOLKIT_MODEL_H
