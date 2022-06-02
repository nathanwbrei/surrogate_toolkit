
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "model.h"
#include "surrogate.h"
#include <iostream>
#include "fluent.h"

namespace phasm {

Model::Model(const OpticBuilder &b) {
    for (std::shared_ptr<CallSiteVariable> &csv: b.get_callsite_vars()) {
        callsite_vars.push_back(csv);
        callsite_var_map[csv->name] = csv;
    }

    for (std::shared_ptr<ModelVariable> &mv: b.get_model_vars()) {
        // Note: this (and everything hereafter) assumes that model vars are duplicated when they are used as
        // both an input and an output. If you set both is_input=true and is_output=true on one ModelVariable,
        // everything will break. This is likely to change, though.
        if (mv->is_input) {
            inputs.push_back(mv);
            input_map[mv->name] = mv;
        }
        if (mv->is_output) {
            outputs.push_back(mv);
            output_map[mv->name] = mv;
        }
    }
}

size_t Model::get_capture_count() const { return captured_rows; }

std::shared_ptr<ModelVariable> Model::get_input(size_t position) {
    if (position >= inputs.size()) { throw std::runtime_error("Parameter index out of bounds"); }
    return inputs[position];
}

std::shared_ptr<ModelVariable> Model::get_input(std::string param_name) {
    auto pair = input_map.find(param_name);
    if (pair == input_map.end()) { throw std::runtime_error("Invalid input parameter name"); }
    return pair->second;
}

std::shared_ptr<ModelVariable> Model::get_output(size_t position) {
    if (position >= outputs.size()) { throw std::runtime_error("Output parameter index out of bounds"); }
    return outputs[position];
}

std::shared_ptr<ModelVariable> Model::get_output(std::string param_name) {
    auto pair = output_map.find(param_name);
    if (pair == output_map.end()) { throw std::runtime_error("Invalid output parameter name"); }
    return pair->second;
}

/// finalize() performs additional work we are supposed to do before the program exits but while we still have a complete Model.
/// This work is specified by Surrogate::s_callmode, and most notably includes
/// training the model on all data captured during the program's run. Anyone subclassing Model should call
/// finalize() as the first statement in their destructor, as otherwise Surrogate::call will be broken for that Model.
/// We cannot call this from ~Model because it won't get called until the child destructor has already been called
/// and the virtual call to train_from_captures is no longer valid.
void Model::finalize() {

    switch (Surrogate::s_callmode) {
        case Surrogate::CallMode::CaptureAndTrain:
            train_from_captures();
            break;
        case Surrogate::CallMode::CaptureAndDump:
            dump_captures_to_csv(std::cout);
            break;
        case Surrogate::CallMode::CaptureAndSummarize:
        default:
            break;
    }
}


void Model::dump_captures_to_csv(std::ostream &os) {
    // print column header
    for (auto input: inputs) {
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
    for (size_t i = 0; i < outputs.size(); ++i) {
        int length = 1;
        for (int dim: outputs[i]->shape()) length *= dim;
        if (length == 1) {
            os << outputs[i]->name;
            if (i < (outputs.size() - 1)) os << ", ";
        } else {
            for (int j = 0; j < length; ++j) {
                os << outputs[i]->name << "[" << j << "]";
                bool last_col = (i == outputs.size() - 1) && (j == length - 1);
                if (!last_col) os << ", ";
            }
        }
    }
    os << std::endl;

    // print body
    for (size_t i = 0; i < captured_rows; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto t = inputs[j]->training_captures[i].flatten(0, -1);
            for (int k = 0; k < t.numel(); ++k) {
                os << t[k].item().toFloat() << ", ";
            }
        }
        for (size_t j = 0; j < outputs.size(); ++j) {
            auto t = outputs[j]->training_captures[i].flatten(0, -1);
            for (int k = 0; k < t.numel(); ++k) {
                os << t[k].item().toFloat();
                bool last_col = (j == outputs.size() - 1) && (k == t.numel() - 1);
                if (!last_col) os << ", ";
            }
        }
        os << std::endl;
    }
    os << std::endl;
}

void Model::save() {
    std::ofstream outfile("captures.csv");
    dump_captures_to_csv(outfile);
    outfile.close();
}

void Model::dump_ranges(std::ostream &) {
    // for (auto i : inputs) {
    // }
}


} // namespace phasm