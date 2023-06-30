
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <julia_model.h>
#include <julia_modelvars.h>
#include <iostream>


int64_t phasm_modelvars_count(void* model) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_count()" << std::endl;
    return m->get_model_var_count();
}

const char* phasm_modelvars_getname(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_getname()" << std::endl;
    return m->get_model_var(index)->name.c_str();
}

bool phasm_modelvars_isinput(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_isinput()" << std::endl;
    return m->get_model_var(index)->is_input;
}

bool phasm_modelvars_isoutput(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_isoutput()" << std::endl;
    return m->get_model_var(index)->is_output;
}

double* phasm_modelvars_inputdata(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_inputdata()" << std::endl;
    return m->get_model_var(index)->inference_input.get_data<double>();
}

double* phasm_modelvars_outputdata(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    std::cout << "Inside phasm_modelvars_outputdata()" << std::endl;
    return m->get_model_var(index)->inference_output.get_data<double>();
}
