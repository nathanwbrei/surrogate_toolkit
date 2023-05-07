
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "torchscript_model.h"
#include "torch_tensor_utils.h"
#include <cassert>

namespace phasm {

TorchscriptModel::TorchscriptModel(std::string filename) {
    try {
        m_module = torch::jit::load(filename);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model. Abort...\n";
        // TODO： what if having problem loading the model?
        assert(false);  // manually exit
        return;
    }
    std::cout << "Loading pytorch pt model at [[" << filename <<"]] succeed.\n\n";
}

TorchscriptModel::~TorchscriptModel() {
}

void TorchscriptModel::initialize() {
}

at::Tensor TorchscriptModel::forward(std::vector<torch::jit::IValue> inputs) {
    return m_module.forward(inputs).toTensor();
}

bool TorchscriptModel::infer() {

    std::vector<torch::jit::IValue> inputs;
    for (const auto &input_model_var: m_inputs) {
        inputs.push_back(to_torch_tensor(input_model_var->inference_input));
    }

    auto output = m_module.forward(inputs).toTensor();
    std::vector<torch::Tensor> output_tensors = split_and_unflatten_outputs(output, m_output_lengths, m_output_shapes);

    size_t i = 0;
    for (const auto &output_model_var: m_outputs) {
        output_model_var->inference_output = to_phasm_tensor(output_tensors[i++]);
    }
    return true;
}

void TorchscriptModel::train_from_captures() {

    std::cout
            << "Training a TorchScript model from within C++ is temporarily disabled. Please train from Python for now"
            << std::endl;
    // Temporarily disable training the torchscript module
    /*
    Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(m_module.parameters(), 0.01);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    // For now each batch contains a single sample

    for (size_t i=0; i<get_capture_count(); ++i) {
        std::vector<torch::Tensor> sample_inputs;
        for (auto input : inputs) {
            sample_inputs.push_back(input->captures[i]);
        }
        auto sample_input = flatten_and_join(std::move(sample_inputs));

        std::vector<torch::Tensor> sample_outputs;
        for (auto output : outputs) {
            sample_outputs.push_back(output->captures[i]);
        }
        auto sample_output = flatten_and_join(std::move(sample_outputs));

        batches.push_back({sample_input, sample_output});
    }


    // Train on each batch
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (const auto& batch: batches) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = m_network->forward(batch.first);
            // Compute a loss value to judge the prediction of our model.
            // std::cout << "prediction" << std::endl << prediction.dtype() << std::endl;
            // std::cout << "actual" << std::endl << batch.second.dtype() << std::endl;
            torch::Tensor loss = torch::mse_loss(prediction, batch.second);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(m_network, "net.pt");
            }
        }
    }
    */
}


} // namespace phasm
