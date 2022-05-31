
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include "surrogate.h"
#include <cstdlib>

void hello_from_surrogate_library() {
    std::cout << "Hello from surrogate library" << std::endl;
}

Surrogate::CallMode get_call_mode_from_envvar() {
    char* callmode_str = std::getenv("PHASM_CALL_MODE");
    if (callmode_str == nullptr) return Surrogate::CallMode::NotSet;
    if (strcmp(callmode_str, "UseModel") == 0) return Surrogate::CallMode::UseModel;
    if (strcmp(callmode_str, "UseOriginal") == 0) return Surrogate::CallMode::UseOriginal;
    if (strcmp(callmode_str, "CaptureAndTrain") == 0) return Surrogate::CallMode::CaptureAndTrain;
    if (strcmp(callmode_str, "CaptureAndDump") == 0) return Surrogate::CallMode::CaptureAndDump;
    if (strcmp(callmode_str, "CaptureAndSummarize") == 0) return Surrogate::CallMode::CaptureAndSummarize;
    return Surrogate::CallMode::NotSet;
}

void print_help_screen() {
    std::cout << std::endl;
    std::cout << "PHASM doesn't know what you want to do." << std::endl;
    std::cout << "Please specify a call mode using the PHASM_CALL_MODE environment variable." << std::endl;
    std::cout << "Valid options are:  " << std::endl;
    std::cout << "    UseModel             Use the surrogate model with its current training parameters" << std::endl;
    std::cout << "    UseOriginal          Do NOT use the surrogate model" << std::endl;
    std::cout << "    CaptureAndTrain      Call the original function, capture all inputs and outputs, and use them to train the model" << std::endl;
    std::cout << "    CaptureAndDump       Call the original function, capture all inputs and outputs, and dump them to CSV" << std::endl;
    std::cout << "    CaptureAndSummarize  Call the original function, track the ranges of all inputs and outputs, and dump them to file" << std::endl;
    std::cout << std::endl;
}

Surrogate::Surrogate(std::function<void(void)> f, std::shared_ptr<Model> model)
    : original_function(std::move(f)), model(model) {

    if (s_callmode == CallMode::NotSet) {
        s_callmode = get_call_mode_from_envvar();
    }

    // Copy over expected callsite vars from model so that we can validate the bindings
    callsite_vars = model->callsite_vars;
    callsite_var_map = model->callsite_var_map;
};

void Surrogate::call() {
    switch (s_callmode) {
        case CallMode::UseModel: call_model(); break;
        case CallMode::UseOriginal: call_original(); break;
        case CallMode::CaptureAndTrain:
        case CallMode::CaptureAndDump: call_original_and_capture(); break;
        case CallMode::CaptureAndSummarize: capture_input_distribution(); break;
        case CallMode::NotSet:
        default:
            print_help_screen();
            exit(-1);
    }
    if (s_callmode == CallMode::UseModel) {
        call_model();
    }
}