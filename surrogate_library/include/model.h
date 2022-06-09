
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "model_variable.h"
#include "surrogate.h"

namespace phasm {


/// There should be exactly one Model in your codebase for each unique function that you wish to surrogate.
/// Contrast this with Surrogate. There should be one Surrogate for each call site of that function,
/// and each of these Surrogates delegate to the same underlying model.
class Model {
    friend class Surrogate;

protected:
    std::vector<std::shared_ptr<ModelVariable>> m_model_vars;
    size_t m_captured_rows = 0;

    // The following are just for convenience
    std::vector<std::shared_ptr<ModelVariable>> m_inputs;
    std::vector<std::shared_ptr<ModelVariable>> m_outputs;
    std::map<std::string, std::shared_ptr<ModelVariable>> m_model_var_map;

public:
    Model() = default;
    virtual ~Model() = default; // We want to be able to inherit from this

    /// Surrogate calls set_model_vars() for us before calling initialize(). This way,
    /// the model can configure itself to adjust to the input and output sizes.
    void add_model_vars(const std::vector<std::shared_ptr<ModelVariable>>& model_vars) {
        for (auto m : model_vars) {
            m_model_vars.push_back(m);
            m_model_var_map[m->name] = m;
            if (m->is_input) m_inputs.push_back(m);
            if (m->is_output) m_outputs.push_back(m);
        }
    }

    // Performs tasks such as training or writing to CSV, right before the model gets destroyed.
    void finalize(CallMode callmode);

    // The total number of training samples we have accumulated so far
    size_t get_capture_count() const;

    std::shared_ptr<ModelVariable> get_model_var(size_t position);

    std::shared_ptr<ModelVariable> get_model_var(std::string param_name);

    void dump_captures_to_csv(std::ostream &);

    void dump_ranges(std::ostream &);


    // Initialize the underlying neural net once all the inputs and outputs are known
    virtual void initialize() {};

    virtual void train_from_captures() {};

    virtual bool infer() { return false; };

    virtual void save();
};



} // namespace phasm
#endif //SURROGATE_TOOLKIT_MODEL_H
