
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <julia_model.h>
#include <julia_modelvars.h>
#include <iostream>


int64_t phasm_modelvars_count(void* model) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    return m->get_model_var_count();
}

const char* phasm_modelvars_getname(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    return m->get_model_var(index)->name.c_str();
}

bool phasm_modelvars_isinput(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    return m->get_model_var(index)->is_input;
}

bool phasm_modelvars_isoutput(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    return m->get_model_var(index)->is_output;
}

double* phasm_modelvars_getinputdata(void* model, int64_t index) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    return m->get_model_var(index)->inference_input.get_data<double>();
}

void phasm_modelvars_setoutputdata(void* model, int64_t index, double* data, size_t length) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    auto mv = m->get_model_var(index);
    mv->inference_output = phasm::tensor(data, length);
}

void phasm_modelvars_setoutputdata2(void* model, int64_t index, double* data, int64_t* shape, size_t dims) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    auto mv = m->get_model_var(index);
    std::vector<int64_t> shapev;
    for (size_t i=0; i<dims; ++i) {
        shapev.push_back(shape[i]);
    }
    mv->inference_output = phasm::tensor(data, shapev);
}

