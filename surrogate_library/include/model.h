
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "model_variable.h"
#include "call_site_variable.h"
#include <iosfwd>  // Forward decls for std::ostream
#include <memory>

namespace phasm {

class OpticBuilder;

class Surrogate;

/// There should be exactly one Model in your codebase for each unique function that you wish to surrogate.
/// Contrast this with Surrogate. There should be one Surrogate for each call site of that function,
/// and each of these Surrogates delegate to the same underlying model.
class Model {
    friend class Surrogate;

protected:
    std::vector<std::shared_ptr<CallSiteVariable>> m_unbound_callsite_vars;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> m_unbound_callsite_var_map;
    std::vector<std::shared_ptr<ModelVariable>> m_model_vars;
    std::map<std::string, std::shared_ptr<ModelVariable>> m_model_var_map;

    // The following two are just for convenience
    std::vector<std::shared_ptr<ModelVariable>> m_inputs;
    std::vector<std::shared_ptr<ModelVariable>> m_outputs;
    size_t m_captured_rows = 0;

public:
    Model() = default;
    explicit Model(const OpticBuilder &b);
    virtual ~Model() = default; // We want to be able to inherit from this

    // Initialize the underlying neural net once all the inputs and outputs are known
    virtual void initialize() {};

    // Performs tasks such as training or writing to CSV, right before the model gets destroyed
    void finalize();

    // The total number of training samples we have accumulated so far
    size_t get_capture_count() const;

    std::shared_ptr<ModelVariable> get_model_var(size_t position);

    std::shared_ptr<ModelVariable> get_model_var(std::string param_name);

    virtual void train_from_captures() {};

    virtual void infer(std::vector<std::shared_ptr<CallSiteVariable>>&) {};

    void dump_captures_to_csv(std::ostream &);

    void dump_ranges(std::ostream &);

    virtual void save();

    template<typename T>
    void add_var(std::string param_name, Direction dir);

    template<typename T>
    void add_var(std::string call_site_var_name, Optic <T> *accessor, std::string model_var_name, Direction dir);

};


// --------------------
// Template definitions
// --------------------

template<typename T>
void Model::add_var(std::string param_name, Direction dir) {
    add_var(param_name, new Primitive<T>, param_name, dir);
}


template<typename T>
void Model::add_var(std::string call_site_var_name, Optic<T> *accessor, std::string model_var_name, Direction dir) {
    auto mv = std::make_shared<ModelVariable>();
    mv->name = model_var_name;
    mv->is_input = (dir == Direction::Input) || (dir == Direction::InputOutput);
    mv->is_output = (dir == Direction::Output) || (dir == Direction::InputOutput);
    mv->accessor = accessor;
    if (m_model_var_map.find(model_var_name) != m_model_var_map.end()) {
        throw std::runtime_error("Model variable already exists!");
    }
    m_model_vars.push_back(mv);
    m_model_var_map[model_var_name] = mv;
    if (mv->is_input) m_inputs.push_back(mv);
    if (mv->is_output) m_outputs.push_back(mv);

    std::shared_ptr<CallSiteVariable> csv = nullptr;
    auto pair = m_unbound_callsite_var_map.find(call_site_var_name);
    if (pair == m_unbound_callsite_var_map.end()) {
        csv = std::make_shared<CallSiteVariable>();
        csv->name = call_site_var_name;
        csv->binding = phasm::any_ptr((T *) nullptr);
        m_unbound_callsite_var_map[call_site_var_name] = csv;
        m_unbound_callsite_vars.push_back(csv);
    } else {
        csv = pair->second;
    }
    csv->model_vars.push_back(mv);
}


} // namespace phasm
#endif //SURROGATE_TOOLKIT_MODEL_H
