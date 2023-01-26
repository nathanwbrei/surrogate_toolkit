//
// Created by Xinxin Mei on 12/2/22.
//

/**
 * This example is trying to load a DNN model (*.pt) originally trained and saved in Python with C++.
 * It is based on the libtorch tutorial at https://pytorch.org/tutorials/advanced/cpp_export.html
 *
 * Usage: ./phasm-example-loading-pt <path/to/lstm-model.pt>
 * The test model is taken from https://github.com/cissieAB/gluex-tracking-pytorch-lstm.
 * CUDA device is required. The test input tensor is of dimension (1256, 7, 6).
 */

#include "torch_cuda_utils.h"
#include "torchscript_model.h"

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: phasm-example-loading-pt <path-to-exported-script-module>\n";
        return -1;
    }

    if (not phasm::has_cuda_device) {
        std::cout << "CUDA device is required for this example!\n Exit..." << std::endl;
        return -1;
    }

    std::cout << "Run model on CUDA device 0. \n" << std::endl;
    phasm::get_current_cuda_device_info();
    phasm::get_libtorch_version();

    // set the device string
    auto device_str = torch::kCUDA;
    torch::Device device(device_str);

    std::string pt_name_str = argv[1];
    try {
        phasm::TorchscriptModel cuda_model = phasm::TorchscriptModel(pt_name_str);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    std::cout << "Loading gluex-tracking-lstm pytorch pt model.................... succeed\n\n";

    /** Test feed-forward computation with an input tensor **/
    //the input must be of type std::vector.	
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(torch::ones({1256, 7, 6}, device));  // lstm input dimension
//
//    at::Tensor output = cuda_module.forward(inputs).toTensor();
//    std::cout << "output sizes: " << output.sizes() << std::endl;
//    std::cout << "output.device().type(): " << output.device().type() << std::endl;
//    std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    return 0;
}

