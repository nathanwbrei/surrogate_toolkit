
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "surrogate.h"
#include <iostream>
#include "model.h"

#include "surrogate_builder.h"

namespace phasm {


Surrogate::~Surrogate() {
    // TODO: If model is shared, maybe check reference count?
    if (m_model != nullptr) m_model->finalize(m_callmode);
}


/// Binds all local variables in one call with minimal overhead. This is a more efficient and concise
/// replacement for repeated calls to Surrogate::bind("varname", v*). However, it is much more error prone.
/// The user has to provide the pointers in the same order that the corresponding CallSiteVariables were added.
/// Furthermore, there is no type safety, unlike in bind<T>, not even at runtime.
/// Potential improvements: It is probably possible to regain the type safety by using a variadic template instead.
Surrogate& Surrogate::bind_locals_to_model(void* head, ...) {
    va_list args;
    va_start(args, head);
    m_callsite_vars[0]->binding.unsafe_set(head);
    size_t len = m_callsite_vars.size();
    for (size_t i=1; i<len; ++i) {
        m_callsite_vars[i]->binding.unsafe_set(va_arg(args, void*));
    }
    va_end(args);
    return *this;
};

std::shared_ptr<CallSiteVariable> Surrogate::get_callsite_var(size_t index) {
    if (index >= m_callsite_vars.size()) { throw std::runtime_error("Index out of range for callsite var binding"); }
    return m_callsite_vars[index];
}


std::shared_ptr<CallSiteVariable> Surrogate::get_callsite_var(std::string name) {
    auto pair = m_callsite_var_map.find(name);
    if (pair == m_callsite_var_map.end()) { throw std::runtime_error("Invalid input parameter name"); }
    return pair->second;
}




void Surrogate::call() {
    switch (m_callmode) {
        case CallMode::UseModel:
            call_model();
            break;
        case CallMode::UseOriginal:
            call_original();
            break;
        case CallMode::CaptureAndTrain:
        case CallMode::CaptureAndDump:
            call_original_and_capture();
            break;
        case CallMode::CaptureAndSummarize:
            capture_input_range();
            break;
        case CallMode::NotSet:
        default:
            print_help_screen();
            exit(-1);
    }
}


void Surrogate::call_original() {
    m_original_function();
}


void Surrogate::call_original_and_capture() {
    for (auto &input: m_callsite_vars) {
        input->captureAllTrainingInputs();
    }
    m_original_function();
    for (auto &output: m_callsite_vars) {
        output->captureAllTrainingOutputs();
    }
    m_model->m_captured_rows++;
}


void Surrogate::capture_input_range() {

}


void Surrogate::call_model() {
    for (const std::shared_ptr<CallSiteVariable>& v : m_callsite_vars) {
        v->captureAllInferenceInputs();
    }
    bool result = m_model->infer();
    if (result) {
        for (const std::shared_ptr<CallSiteVariable>& v : m_callsite_vars) {
            v->publishAllInferenceOutputs();
        }
    }
    else {
        // TODO: Currently we do nothing, but we most likely want to call_original,
        //       and we may even wish to capture the results. Do we ignore the
        //       results, dump them, or use them for training? Unclear at the moment.
        //call_original_and_capture();
    }
}


CallMode get_call_mode_from_envvar() {
    char *callmode_str = std::getenv("PHASM_CALL_MODE");
    if (callmode_str == nullptr) return CallMode::NotSet;
    if (strcmp(callmode_str, "UseModel") == 0) return CallMode::UseModel;
    if (strcmp(callmode_str, "UseOriginal") == 0) return CallMode::UseOriginal;
    if (strcmp(callmode_str, "CaptureAndTrain") == 0) return CallMode::CaptureAndTrain;
    if (strcmp(callmode_str, "CaptureAndDump") == 0) return CallMode::CaptureAndDump;
    if (strcmp(callmode_str, "CaptureAndSummarize") == 0) return CallMode::CaptureAndSummarize;
    return CallMode::NotSet;
}


void print_help_screen() {
    std::cout << std::endl;
    std::cout << "PHASM doesn't know what you want to do." << std::endl;
    std::cout << "Please specify a call mode using the PHASM_CALL_MODE environment variable." << std::endl;
    std::cout << "Valid options are:  " << std::endl;
    std::cout << "    UseModel             Use the surrogate model with its current training parameters" << std::endl;
    std::cout << "    UseOriginal          Do NOT use the surrogate model" << std::endl;
    std::cout
            << "    CaptureAndTrain      Call the original function, capture all inputs and outputs, and use them to train the model"
            << std::endl;
    std::cout
            << "    CaptureAndDump       Call the original function, capture all inputs and outputs, and dump them to CSV"
            << std::endl;
    std::cout
            << "    CaptureAndSummarize  Call the original function, track the ranges of all inputs and outputs, and dump them to file"
            << std::endl;
    std::cout << std::endl;
}

Surrogate& Surrogate::add_callsite_vars(const std::vector<std::shared_ptr<CallSiteVariable>> &vars) {
    for (auto csv : vars) {
        m_callsite_vars.push_back(csv);
        m_callsite_var_map[csv->name] = csv;
    }
    return *this;
}

} // namespace phasm