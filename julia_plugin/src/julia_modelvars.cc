
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

void phasm_modelvars_getinputdata(void* model, int64_t index, phasm::DType* dtype, void** data, const int64_t** shape, size_t* ndims) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    auto& t = m->get_model_var(index)->inference_input;
    *data = t.get_data<double>();
    *shape = t.get_shape().data();
    *ndims = t.get_shape().size();
    *dtype = t.get_dtype();
}

void phasm_modelvars_setoutputdata(void* model, int64_t index, phasm::DType dtype, void* data, size_t length) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    auto mv = m->get_model_var(index);
    switch (dtype) {
        case phasm::DType::UI8: 
            mv->inference_output = phasm::tensor((uint8_t*)data, length); break;
        case phasm::DType::I16:
            mv->inference_output = phasm::tensor((int16_t*)data, length); break;
        case phasm::DType::I32: 
            mv->inference_output = phasm::tensor((int32_t*)data, length); break;
        case phasm::DType::I64:
            mv->inference_output = phasm::tensor((int64_t*)data, length); break;
        case phasm::DType::F32:
            mv->inference_output = phasm::tensor((float*)data, length); break;
        case phasm::DType::F64:
            mv->inference_output = phasm::tensor((double*)data, length); break;
        default:
            throw std::runtime_error("Invalid DType!");
    }
}

void phasm_modelvars_setoutputdata2(void* model, int64_t index, phasm::DType dtype, void* data, int64_t* shape, size_t dims) {
    auto m = static_cast<phasm::JuliaModel*>(model);
    auto mv = m->get_model_var(index);
    std::vector<int64_t> shapev;
    for (size_t i=0; i<dims; ++i) {
        shapev.push_back(shape[i]);
    }
    switch (dtype) {
        case phasm::DType::UI8: 
            mv->inference_output = phasm::tensor((uint8_t*)data, shapev); break;
        case phasm::DType::I16:
            mv->inference_output = phasm::tensor((int16_t*)data, shapev); break;
        case phasm::DType::I32: 
            mv->inference_output = phasm::tensor((int32_t*)data, shapev); break;
        case phasm::DType::I64:
            mv->inference_output = phasm::tensor((int64_t*)data, shapev); break;
        case phasm::DType::F32:
            mv->inference_output = phasm::tensor((float*)data, shapev); break;
        case phasm::DType::F64:
            mv->inference_output = phasm::tensor((double*)data, shapev); break;
        default:
            throw std::runtime_error("Invalid DType!");
    }
}

