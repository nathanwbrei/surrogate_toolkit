
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_H
#define SURROGATE_TOOLKIT_SURROGATE_H

#include <vector>
#include "call_site_variable.h"

namespace phasm {

class Model;
class OpticBuilder;

class Surrogate {
public:
    friend class Model;

    enum class CallMode {
        NotSet, UseOriginal, UseModel, CaptureAndTrain, CaptureAndDump, CaptureAndSummarize
    };

private:
    static inline CallMode s_callmode = CallMode::NotSet;
    std::function<void(void)> m_original_function;
    std::shared_ptr<Model> m_model;
    std::vector<std::shared_ptr<CallSiteVariable>> m_bound_callsite_vars;
    std::map<std::string, std::shared_ptr<CallSiteVariable>> m_bound_callsite_var_map;

public:

    Surrogate();

    void add_vars(const OpticBuilder& b);

    template<typename T>
    void add_var(std::string param_name, Direction dir);

    template<typename T>
    void add_var(std::string call_site_var_name, Optic <T> *accessor, std::string model_var_name, Direction dir);



    template<typename T>
    Surrogate& bind(std::string param_name, T *slot);

    // inline Surrogate& set_call_mode(CallMode callmode) { m_callmode = callmode; return *this;}
    Surrogate& set_model(const std::shared_ptr<Model>& model);
    inline std::shared_ptr<Model> get_model() { return m_model; }
    Surrogate& bind_locals_to_model(void* head...);
    inline Surrogate& bind_locals_to_original_function(std::function<void(void)> f) {m_original_function = std::move(f); return *this;};


    // These are mainly for testing purposes

    std::shared_ptr<CallSiteVariable> get_binding(size_t index);
    std::shared_ptr<CallSiteVariable> get_binding(std::string name);

    std::vector<std::shared_ptr<ModelVariable>> get_model_vars() {
        std::vector<std::shared_ptr<ModelVariable>> results;
        for (auto csv : m_bound_callsite_vars) {
            for (auto mv : csv->model_vars) {
                results.push_back(mv);
            }
        }
        return results;
    }


    static void set_call_mode(CallMode callmode);

    void call();
    void call_original();
    void call_original_and_capture();
    void capture_input_distribution();
    void call_model();
};


// --------------------------
// Free function declarations
// --------------------------

Surrogate::CallMode get_call_mode_from_envvar();
void print_help_screen();


// --------------------
// Template definitions
// --------------------

template<typename T>
Surrogate& Surrogate::bind(std::string param_name, T *slot) {
    auto csv = m_bound_callsite_var_map.find(param_name);
    if (csv == m_bound_callsite_var_map.end()) {
        throw std::runtime_error("No such callsite variable specified in model");
    }
    if (csv->second->binding.get() != nullptr) {
        throw std::runtime_error("Callsite variable already set! Possibly a global var");
    }
    csv->second->binding = slot;
    return *this;
}

// --------------------
// Template definitions
// --------------------

template<typename T>
void Surrogate::add_var(std::string param_name, Direction dir) {
    add_var(param_name, new Primitive<T>, param_name, dir);
}


template<typename T>
void Surrogate::add_var(std::string call_site_var_name, Optic<T> *accessor, std::string model_var_name, Direction dir) {
    auto mv = std::make_shared<ModelVariable>();
    mv->name = model_var_name;
    mv->is_input = (dir == Direction::IN) || (dir == Direction::INOUT);
    mv->is_output = (dir == Direction::OUT) || (dir == Direction::INOUT);
    mv->accessor = accessor;

    std::shared_ptr<CallSiteVariable> csv = nullptr;
    auto pair = m_bound_callsite_var_map.find(call_site_var_name);
    if (pair == m_bound_callsite_var_map.end()) {
        csv = std::make_shared<CallSiteVariable>(call_site_var_name, make_any<T>());
        m_bound_callsite_var_map[call_site_var_name] = csv;
        m_bound_callsite_vars.push_back(csv);
    } else {
        csv = pair->second;
    }
    csv->model_vars.push_back(mv);
}


} // namespace phasm


#endif //SURROGATE_TOOLKIT_SURROGATE_H
