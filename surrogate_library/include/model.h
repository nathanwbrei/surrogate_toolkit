
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_MODEL_H
#define SURROGATE_TOOLKIT_MODEL_H

#include "parameter.h"


struct Model { // This is an abstract class

    // We want to be able to inherit from this
    virtual ~Model() = default;

    // Train takes all of the captures associated with each parameter
    void train(std::vector<std::unique_ptr<Input>>& inputs, std::vector<std::unique_ptr<Output>>& outputs) = delete;

    // Infer takes the sample associated with each parameter
    void infer(std::vector<std::unique_ptr<Input>>& inputs, std::vector<std::unique_ptr<Output>>& outputs) = delete;

};

struct MemorizingModel : public Model {
    void write_to_file() {

    }

};


#endif //SURROGATE_TOOLKIT_MODEL_H
