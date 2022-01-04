
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PYTORCH_MODEL_H
#define SURROGATE_TOOLKIT_PYTORCH_MODEL_H

#include "model.h"
#include <torch/torch.h>

struct PyTorchModel : public Model {

    PyTorchModel() {
	torch::Tensor tensor = torch::zeros({2, 2});
	std::cout << tensor << std::endl;
    }

    // Train takes all of the captures associated with each parameter
    void train(std::vector<std::unique_ptr<Input>>& inputs, std::vector<std::unique_ptr<Output>>& outputs) {

    }

    // Infer takes the sample associated with each parameter
    void infer(std::vector<std::unique_ptr<Input>>& inputs, std::vector<std::unique_ptr<Output>>& outputs) {

    }

};


#endif //SURROGATE_TOOLKIT_PYTORCH_MODEL_H
