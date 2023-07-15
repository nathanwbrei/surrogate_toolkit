
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "model.h"
#include "surrogate.h"
#include <fstream>
#include <iostream>

namespace phasm {


size_t Model::get_capture_count() const { return m_captured_rows; }

std::shared_ptr<ModelVariable> Model::get_model_var(size_t position) {
    if (position >= m_model_vars.size()) { throw std::runtime_error("Parameter index out of bounds"); }
    return m_model_vars[position];
}

std::shared_ptr<ModelVariable> Model::get_model_var(std::string param_name) {
    auto pair = m_model_var_map.find(param_name);
    if (pair == m_model_var_map.end()) { throw std::runtime_error("Invalid input parameter name"); }
    return pair->second;
}


/// finalize() performs additional work we are supposed to do before the program exits but while we still have a complete Model.
/// This work is specified by Surrogate::s_callmode, and most notably includes
/// training the model on all data captured during the program's run. Anyone subclassing Model should call
/// finalize() as the first statement in their destructor, as otherwise Surrogate::call will be broken for that Model.
/// We cannot call this from ~Model because it won't get called until the child destructor has already been called
/// and the virtual call to train_from_captures is no longer valid.
/// TODO: All this moves to Surrogate. Model keeps a reference count. Last surrogate turns out the lights.
void Model::finalize(CallMode callmode) {

    std::cout << "PHASM: Starting model shutdown" << std::endl;
    switch (callmode) {
        case CallMode::TrainModel:
            std::cout << "PHASM: Training model from captures" << std::endl;
            train_from_captures();
            break;
        case CallMode::DumpTrainingData: {
            std::cout << "PHASM: Dumping training data to ./training_captures.csv" << std::endl;
            std::ofstream outfile("training_captures.csv");
            dump_captures_to_csv(outfile);
            break;
        }
        case CallMode::DumpValidationData: {
            std::cout << "PHASM: Dumping validation data to ./validation_captures.csv" << std::endl;
            std::ofstream outfile("validation_captures.csv");
            dump_captures_to_csv(outfile);
            break;
        }
        case CallMode::DumpInputSummary:
            std::cout << "PHASM: Dumping input summary (Note: this is a no-op for now)" << std::endl;
        default:
            break;
    }
    std::cout << "PHASM: Finished model shutdown" << std::endl;
}


void Model::dump_captures_to_csv(std::ostream &os) {
    // print column header
    for (auto input: m_inputs) {
        int length = 1;
        for (int dim: input->shape()) length *= dim;
        if (length == 1) {
            os << input->name << ", ";
        } else {
            for (int i = 0; i < length; ++i) {
                os << input->name << "[" << i << "], ";
            }
        }
    }
    for (size_t i = 0; i < m_outputs.size(); ++i) {
        int length = 1;
        for (int dim: m_outputs[i]->shape()) length *= dim;
        if (length == 1) {
            os << m_outputs[i]->name;
            if (i < (m_outputs.size() - 1)) os << ", ";
        } else {
            for (int j = 0; j < length; ++j) {
                os << m_outputs[i]->name << "[" << j << "]";
                bool last_col = (i == m_outputs.size() - 1) && (j == length - 1);
                if (!last_col) os << ", ";
            }
        }
    }
    os << std::endl;

    // print body
    for (size_t i = 0; i < m_captured_rows; ++i) {
        for (size_t j = 0; j < m_inputs.size(); ++j) {
            auto t = m_inputs[j]->training_inputs[i];
            for (size_t k = 0; k < t.get_length(); ++k) {
                switch (t.get_dtype()) {
                    case DType::UI8: os << *(t.get_data<uint8_t>() + k); break;
                    case DType::I16: os << *(t.get_data<int16_t>() + k); break;
                    case DType::I32: os << *(t.get_data<int32_t>() + k); break;
                    case DType::I64: os << *(t.get_data<int64_t>() + k); break;
                    case DType::F32: os << *(t.get_data<float>() + k); break;
                    case DType::F64: os << *(t.get_data<double>() + k); break;
                    default: os << "?, "; break;
                }
                os << ", ";
            }
        }
        for (size_t j = 0; j < m_outputs.size(); ++j) {
            auto t = m_outputs[j]->training_outputs[i];
            for (size_t k = 0; k < t.get_length(); ++k) {
                switch (t.get_dtype()) {
                    case DType::UI8: os << *(t.get_data<uint8_t>() + k); break;
                    case DType::I16: os << *(t.get_data<int16_t>() + k); break;
                    case DType::I32: os << *(t.get_data<int32_t>() + k); break;
                    case DType::I64: os << *(t.get_data<int64_t>() + k); break;
                    case DType::F32: os << *(t.get_data<float>() + k); break;
                    case DType::F64: os << *(t.get_data<double>() + k); break;
                    default: os << "?"; break;
                }
                bool last_col = (j == m_outputs.size() - 1) && (k == t.get_length() - 1);
                if (!last_col) os << ", ";
            }
        }
        os << std::endl;
    }
}

void Model::dump_ranges(std::ostream &) {
    // for (auto i : inputs) {
    // }
}


} // namespace phasm