
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "model.h"
#include "surrogate.h"
#include <iostream>


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


void Model::dump_captures_to_csv(std::ostream& os) {
    // print column header
    for (auto input : inputs) {
        int length = 1;
        for (int dim : input->shape()) length *= dim;
        if (length == 1) {
            os << input->name << ", ";
        }
        else {
            for (int i=0; i<length; ++i) {
                os << input->name << "[" << i << "], ";
            }
        }
    }
    for (size_t i=0; i<outputs.size(); ++i) {
        int length = 1;
        for (int dim : outputs[i]->shape()) length *= dim;
        if (length == 1) {
            os << outputs[i]->name;
            if (i < (outputs.size()-1)) os << ", ";
        }
        else {
            for (int j=0; j<length; ++j) {
                os << outputs[i]->name << "[" << j << "]";
                bool last_col = (i == outputs.size()-1) && (j==length-1);
                if (!last_col) os << ", ";
            }
        }
    }
    os << std::endl;

    // print body
    for (size_t i=0; i<captured_rows; ++i) {
        for (size_t j=0; j<inputs.size(); ++j) {
            auto t = inputs[j]->training_captures[i].flatten(0, -1);
            for (int k=0; k<t.numel(); ++k) {
                os << t[k].item().toFloat() << ", ";
            }
        }
        for (size_t j=0; j<outputs.size(); ++j) {
            auto t = outputs[j]->training_captures[i].flatten(0, -1);
            for (int k=0; k<t.numel(); ++k) {
                os << t[k].item().toFloat();
                bool last_col = (j == outputs.size()-1) && (k==t.numel()-1);
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

void Model::dump_ranges(std::ostream&) {
    // for (auto i : inputs) {
    // }
}
