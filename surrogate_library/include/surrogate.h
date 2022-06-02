
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_H
#define SURROGATE_TOOLKIT_SURROGATE_H

#include <vector>
#include "call_site_variable.h"


namespace phasm {

class Model;

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
    explicit Surrogate(std::function<void(void)> f, std::shared_ptr<Model> model);

    template<typename T>
    void bind(std::string param_name, T *slot);

    std::shared_ptr<CallSiteVariable> get_binding(size_t index);
    std::shared_ptr<CallSiteVariable> get_binding(std::string name);

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
void Surrogate::bind(std::string param_name, T *slot) {
    auto csv = m_bound_callsite_var_map.find(param_name);
    if (csv == m_bound_callsite_var_map.end()) {
        throw std::runtime_error("No such callsite variable specified in model");
    }
    if (csv->second->binding.get<T>() != nullptr) {
        throw std::runtime_error("Callsite variable already set! Possibly a global var");
    }
    csv->second->binding = slot;
}

} // namespace phasm


#endif //SURROGATE_TOOLKIT_SURROGATE_H
