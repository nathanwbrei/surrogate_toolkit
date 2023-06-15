
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_H
#define SURROGATE_TOOLKIT_SURROGATE_H

#include <vector>
#include "call_site_variable.h"

namespace phasm {

class Model;
enum class CallMode {
    NotSet, UseOriginal, UseModel, DumpTrainingData, DumpValidationData, TrainModel, DumpInputSummary
};

inline std::ostream& operator<<(std::ostream& os, CallMode cm) {
    switch (cm) {
        case CallMode::NotSet: os << "NotSet"; break;
        case CallMode::UseOriginal: os << "UseOriginal"; break;
        case CallMode::UseModel: os << "UseModel"; break;
        case CallMode::DumpTrainingData: os << "DumpTrainingData"; break;
        case CallMode::DumpValidationData: os << "DumpValidationData"; break;
        case CallMode::TrainModel: os << "TrainModel"; break;
        case CallMode::DumpInputSummary: os << "DumpInputSummary"; break;
    }
}

class Surrogate {
public:
    friend class Model;

private:
    CallMode m_callmode = CallMode::NotSet;
    std::function<void(void)> m_original_function;
    std::shared_ptr<Model> m_model;
    std::vector<std::shared_ptr<CallSiteVariable>> m_callsite_vars;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> m_callsite_var_map;

public:

    Surrogate() = default;
    ~Surrogate();

    // ------------------------------------------------------------------------
    // Main API: This is how users are supposed to interact with a Surrogate
    // ------------------------------------------------------------------------

    template<typename T>
    Surrogate& bind_callsite_var(std::string param_name, T *slot);

    Surrogate& bind_all_callsite_vars(void* head...);

    inline Surrogate& bind_original_function(std::function<void(void)> f) { m_original_function = std::move(f); return *this;};

    void call();
    void call_model();
    void call_original();
    void call_original_and_capture();
    void call_model_and_capture();
    void capture_input_range();

    // ------------------------------------------------------------------------
    // Configuration: These are meant to be called by the SurrogateBuilder
    // ------------------------------------------------------------------------

    inline Surrogate& set_callmode(CallMode callmode) { m_callmode = callmode; return *this; };
    inline Surrogate& set_model(const std::shared_ptr<Model>& model) { m_model = model; return *this; };
    Surrogate& add_callsite_vars(const std::vector<std::shared_ptr<CallSiteVariable>> &vars);

    // ------------------------------------------------------------------------
    // Inspection: These are meant to be used for debugging and testing
    // ------------------------------------------------------------------------

    inline std::shared_ptr<Model> get_model() { return m_model; }
    std::shared_ptr<CallSiteVariable> get_callsite_var(size_t index);
    std::shared_ptr<CallSiteVariable> get_callsite_var(std::string name);

    std::vector<std::shared_ptr<ModelVariable>> get_model_vars() {
        std::vector<std::shared_ptr<ModelVariable>> results;
        for (auto csv : m_callsite_vars) {
            for (auto mv : csv->model_vars) {
                results.push_back(mv);
            }
        }
        return results;
    }

};


// --------------------------
// Free function declarations
// --------------------------

CallMode get_call_mode_from_envvar();
void print_help_screen();


// --------------------
// Template definitions
// --------------------

template<typename T>
Surrogate& Surrogate::bind_callsite_var(std::string param_name, T *slot) {
    auto csv = m_callsite_var_map.find(param_name);
    if (csv == m_callsite_var_map.end()) {
        throw std::runtime_error("No such callsite variable specified in model");
    }
    if (csv->second->binding.get() != nullptr) {
        throw std::runtime_error("Callsite variable already set! Possibly a global var");
    }
    csv->second->binding = slot;
    return *this;
}


} // namespace phasm


#endif //SURROGATE_TOOLKIT_SURROGATE_H
