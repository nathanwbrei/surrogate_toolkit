
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PYTORCH_MODEL_H
#define SURROGATE_TOOLKIT_PYTORCH_MODEL_H

#include "model.h"
#include <torch/torch.h>

struct Net : torch::nn::Module {

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

};

struct PyTorchModel : public Model {

    std::shared_ptr<Net> net;

    PyTorchModel() {
        // Create a new Net.
        auto net = std::make_shared<Net>();

        // Create a multi-threaded data loader for the MNIST dataset.
        auto data_loader = torch::data::make_data_loader(
                torch::data::datasets::MNIST("./data").map(
                        torch::data::transforms::Stack<>()),
                64);

    }

    torch::Tensor preprocess(std::vector<std::unique_ptr<Input>>& inputs) {
        return torch::ones({2,2});
    }


    // Train takes all of the captures associated with each parameter
    void train(torch::Tensor batch) override {
        auto dataset = torch::data::datasets::MNIST("./mnist")
                .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                .map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader(std::move(dataset));

        // Instantiate an SGD optimization algorithm to update our Net's parameters.
        torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

        for (size_t epoch = 1; epoch <= 10; ++epoch) {
            size_t batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader) {
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                torch::Tensor prediction = net->forward(batch.data);
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                              << " | Loss: " << loss.item<float>() << std::endl;
                    // Serialize your model periodically as a checkpoint.
                    torch::save(net, "net.pt");
                }
            }
        }

    }

    // Infer takes the sample associated with each parameter
    void infer(torch::Tensor batch) {

        // (*net)()

    }

};


#endif //SURROGATE_TOOLKIT_PYTORCH_MODEL_H
