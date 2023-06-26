//
// Created by Xinxin Mei on 12/2/22.
//

/**
 * This example is trying to load a DNN model (*.pt) originally trained and saved with PyTorch.
 * It is based on the libtorch tutorial at https://pytorch.org/tutorials/advanced/cpp_export.html
 *
 * Usage: ./phasm-example-loading-pt <path/to/lstm-model.pt>
 * The test model is taken from https://github.com/cissieAB/gluex-tracking-pytorch-lstm.
 * CUDA device is required. The test input tensor is of dimension (1256, 7, 6).
 *
 * The LSTM model definition： https://github.com/cissieAB/gluex-tracking-pytorch-lstm/blob/main/python/utils.py#L70
 * Ifarm location: /work/epsci/shared_pkg/lstm_model.pt
 **/

#include <iostream>
#include <memory>

#include <torch/torch.h>

#include "torch_utils.h"
#include "torchscript_model.h"

# define BATCH_SIZE 2048
# define SEQ_LENGTH 7  // For the test lstm_model.pt model only

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: phasm-example-loading-pt <path-to-exported-pt-module>\n";
        exit(-1);
    }

    phasm::get_libtorch_version();

    bool has_gpu = phasm::has_cuda_device();
    torch::Device device= torch::kCPU;
    if (has_gpu) {
        std::cout << "Use CUDA device 0 to load module!\n" << std::endl;
        device = torch::kCUDA;
    }

    /* Load module to the assigned device */
    phasm::TorchscriptModel model = phasm::TorchscriptModel(argv[1], true, device);

    /* Get the input layer information and claim the input based on device */
    std::vector<int64_t> first_layer_shape = model.GetFirstLayerShape();

    // Caculate the input dimension, based on first layer weight matrix shape.
    // TODO (@xmei): below method is for RNN layer only. Different for nn.linear and nn.conv.
    std::vector<int64_t> input_shape;
    input_shape.push_back(BATCH_SIZE);
    input_shape.push_back(SEQ_LENGTH);
    std::copy(first_layer_shape.begin() + 1, first_layer_shape.end(), std::back_inserter(input_shape));
    std::cout << "Input dimension: " << input_shape << std::endl;

    /** Test feed-forward computation with an input tensor **/
    //The input must be of type std::vector.
    std::vector<torch::jit::IValue> input;
    input.push_back(torch::randn(input_shape, device));

    at::Tensor output = model.get_module().forward(input).toTensor();
    std::cout << "Output sizes: " << output.sizes() << std::endl;
    std::cout << "Output.device().type(): " << output.device().type() << std::endl;
    std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    return 0;
}
