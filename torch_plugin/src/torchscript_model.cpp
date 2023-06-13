
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "torchscript_model.h"
#include "torch_tensor_utils.h"

namespace phasm {

TorchscriptModel::TorchscriptModel(std::string filename) {
    try {
        m_module = torch::jit::load(filename);
    }
    catch (const c10::Error &e) {
        std::cerr << "PHASM: FATAL ERROR: Exception loading TorchScript file" << std::endl;
        std::cerr << "  Filename is '" << m_filename << "'" << std::endl;
        std::cerr << "  Exception contains: " << std::endl;
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    std::cerr << "PHASM: Loaded TorchScript model '" << filename << "'" << std::endl;
}

TorchscriptModel::~TorchscriptModel() {
}

void TorchscriptModel::initialize() {
}

torch::jit::script::Module& TorchscriptModel::get_module() {
    return m_module;
}

bool TorchscriptModel::infer() {

    if (m_combine_tensors) {
        std::vector<torch::Tensor> input_tensors;

        for (const auto &input_model_var: m_inputs) {
            input_tensors.push_back(to_torch_tensor(input_model_var->inference_input));
        }

        // This all assumes a single Tensor of floats as input and output
        torch::Tensor input = flatten_and_join(input_tensors);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = m_module.forward(inputs).toTensor();

        std::vector<torch::Tensor> output_tensors = split_and_unflatten_outputs(output, m_output_lengths, m_output_shapes);

        size_t i = 0;
        for (const auto &output_model_var: m_outputs) {
            output_model_var->inference_output = to_phasm_tensor(input_tensors[i++]);
        }
    }
    else {
        std::vector<torch::jit::IValue> inputs;
        for (const auto &input_model_var: m_inputs) {
            inputs.push_back(to_torch_tensor(input_model_var->inference_input));
        }

        auto output = m_module.forward(inputs);
        if (output.isTensor()) {

            if (m_outputs.size() == 1) {
                m_outputs[0]->inference_output = to_phasm_tensor(output.toTensor());
            }
            else {
                std::cerr << "PHASM: FATAL ERROR: Torchscript model outputs a single tensor when multiple expected" << std::endl;
                std::cerr << "  Surrogate expects " << m_outputs.size() << std::endl;
                std::cerr << "  Filename is '" << m_filename << "'" << std::endl;
                exit(1);
            }
        }
        else if (output.isTuple()) {
            auto tuple = output.toTuple();
            if (tuple->size() != m_outputs.size()) {
                std::cerr << "PHASM: FATAL ERROR: Torchscript model output tuple size mismatch" << std::endl;
                std::cerr << "  Surrogate expects " << m_outputs.size() << std::endl;
                std::cerr << "  PT file provides " << tuple->size() << std::endl;
                std::cerr << "  Filename is '" << m_filename << "'" << std::endl;
                exit(1);
            }
            size_t i = 0;
            for (const auto &output_model_var: m_outputs) {
                output_model_var->inference_output = to_phasm_tensor(tuple->elements()[i++].toTensor());
            }
        }
        else {
            // TODO: We could probably accept the case of output.isTensorList
            std::cerr << "PHASM: FATAL ERROR: Torchscript model has invalid output type for forward()" << std::endl;
            std::cerr << "  The model's forward() method must return either a tensor or a tuple of tensors." << std::endl;
            std::cerr << "  Filename is '" << m_filename << "'" << std::endl;
            exit(1);
        }
    }

    // TODO: Figure out how to extract UQ info from model so that we can return false when appropriate
    return true;
}

void TorchscriptModel::train_from_captures() {

    std::cerr << "PHASM: FATAL ERROR: Training a TorchScript model from C++ is temporarily disabled. Please train from Python for now" << std::endl;
    exit(1);
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
