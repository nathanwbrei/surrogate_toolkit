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

#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>  // for cuda device properties
#include <torch/torch.h>

#include <iostream>
#include <memory>

void print_cuda_device_info() {
    cudaDeviceProp *cuda_prop = at::cuda::getCurrentDeviceProperties();
    std::cout << "  CUDA device name: " << cuda_prop->name << std::endl;
    std::cout << "  CUDA compute capacity: "
              << cuda_prop->major << "." << cuda_prop->minor << std::endl;
    std::cout << "  LibTorch version: "
              << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "."
              << TORCH_VERSION_PATCH << std::endl;
    std::cout << std::endl;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: phasm-example-loading-pt <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    std::cout << "Loading gluex-tracking-lstm pt model.................... succeed\n\n";

    /** Test feed-forward computation with an input tensor **/
    std::cout << "Run model on CUDA device 0." << std::endl;
    auto cuda_available = torch::cuda::is_available();
    if (not cuda_available) {
        std::cout << "CUDA device is required to do the computation!" << std::endl;
        return -1;
    }
    print_cuda_device_info();
    auto device_str = torch::kCUDA;
    torch::Device device(device_str);

    //the input must be of type std::vector.	
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1256, 7, 6}, device));  // lstm input dimension

    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "output sizes: " << output.sizes() << std::endl;
    std::cout << "output.device().type(): " << output.device().type() << std::endl;
    std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    return 0;
}

