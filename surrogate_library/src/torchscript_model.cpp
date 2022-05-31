
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "torchscript_model.h"

TorchscriptModel::TorchscriptModel(std::string filename) {
    m_module = torch::jit::load(filename);
}

TorchscriptModel::~TorchscriptModel() {
    finalize();
}

void TorchscriptModel::initialize() {
}


void TorchscriptModel::infer(Surrogate &s) {

    std::vector<torch::Tensor> input_tensors;

    for (const std::shared_ptr<CallSiteVariable>& csv : s.callsite_vars) {
        csv->get_all_inference_inputs();
    }
    for (const auto& input_model_var : inputs) {
        input_tensors.push_back(input_model_var->inference_capture);
    }

    // This all assumes a single Tensor of floats as input and output
    torch::Tensor input = flatten_and_join(input_tensors);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = m_module.forward(inputs).toTensor();

    std::vector<torch::Tensor> output_tensors = split_and_unflatten_outputs(output);

    size_t i = 0;
    for (const auto& output_model_var : outputs) {
        output_model_var->inference_capture = input_tensors[i++];
    }
    for (const auto& output_callsite_var : s.callsite_vars) {
        output_callsite_var->put_all_inference_outputs();
    }
}

void TorchscriptModel::train_from_captures() {

    std::cout << "Training a TorchScript model from within C++ is temporarily disabled. Please train from Python for now" << std::endl;
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

torch::Tensor TorchscriptModel::flatten_and_join(std::vector<torch::Tensor> inputs) {
    for (auto& input : inputs) {
        input = input.flatten(0, -1).toType(c10::ScalarType::Float);
    }
    auto result = torch::cat(inputs);
    return result;
}

std::vector<torch::Tensor> TorchscriptModel::split_and_unflatten_outputs(torch::Tensor output) const {
    std::vector<torch::Tensor> outputs;
    int64_t start = 0;
    for (size_t i=0; i<m_output_lengths.size(); ++i) {
        torch::Tensor o = output.slice(0, start, start+m_output_lengths[i]).reshape(m_output_shapes[i]);
        outputs.push_back(o);
        start += m_output_lengths[i];
    }
    return outputs;
}


