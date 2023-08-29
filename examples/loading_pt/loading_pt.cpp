//
// Created by Xinxin Mei on 12/2/22.
//

/**
 * This example is trying to load a DNN model (*.pt) originally trained and saved with PyTorch.
 * It is based on the libtorch tutorial at https://pytorch.org/tutorials/advanced/cpp_export.html
 *
 * Usage: ./phasm-example-loading-pt <path/to/lstm-model.pt>
 * The test model is taken from https://github.com/cissieAB/gluex-tracking-pytorch-lstm
 * Ifarm location: /work/epsci/shared_pkg/lstm_model.pt
 **/

#include "torchscript_model.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <cmath>

# define MIN_BATCH_SIZE 64
# define MAX_BATCH_SIZE 16384
# define DEMO_BATCH_SIZE 2048
# define DEMO_SEQ_LENGTH 7  // For the test lstm_model.pt model only

struct benchRes {
    int64_t batch_size;
    double milliseconds;
};

/// @brief Measure the elaspsed time of (loading module + input initilization + single round inference).
/// @param input_shape the dimension of the input shape.
/// @return duration of the process in milliseconds (10e-3 s).
double benchmarkKernel(const std::vector<int64_t>& input_shape, bool use_gpu, std::string model_path) {
    torch::Device bench_device = use_gpu ? torch::kCUDA : torch::kCPU;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<torch::jit::IValue> input;
    input.push_back(torch::randn(input_shape, bench_device));

    phasm::TorchscriptModel bench_model = phasm::TorchscriptModel(model_path, false, bench_device);
    at::Tensor output = bench_model.get_module().forward(input).toTensor();

    // Operate on the first element of the output to make sure inference execution is completed.
    // std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << '\n';
    float firstElement = output[0][0].item<float>();
    firstElement += 1.0;

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

    return double(t_duration) / 1000.0;  // return in milliseconds
}

void benchmarkWrapper(const std::vector<int64_t>& first_layer_shape, bool use_gpu, const std::string& pt_path) {
    std::string device_type = use_gpu? "GPU":"CPU";
    std::cout << "\n\n########################################\n";
    std::cout << "Benchmarking on " << device_type << std::endl;
    std::cout << "########################################\n";
    std::vector<benchRes> resVec;
    std::vector<int64_t> input_shape;
    for (int64_t batch_size = MIN_BATCH_SIZE; batch_size <= MAX_BATCH_SIZE; batch_size *= 2) {
        input_shape.clear();
        input_shape.push_back(batch_size);
        input_shape.push_back(DEMO_SEQ_LENGTH);
        std::copy(first_layer_shape.begin() + 1, first_layer_shape.end(), std::back_inserter(input_shape));

        benchRes curElement;
        curElement.batch_size = batch_size;
        curElement.milliseconds = benchmarkKernel(input_shape, use_gpu, pt_path);
        resVec.push_back(curElement);
    }

    // Print the bechmarking result to a txt file
    std::string fileName = use_gpu ? "lstm-bench-gpuRes.txt": "lstm-bench-cpuRes.txt";

    std::ofstream outputFile(fileName);
    // Check if the file was opened successfully
    if (outputFile.is_open()) {
        std::cout << "Batch_size, milliseconds" << std::endl;
        outputFile << "Batch_size, milliseconds" << std::endl;
        // Write the vector's contents to the file
        for (const auto& element : resVec) {
            std::cout << element.batch_size << ", " << element.milliseconds << std::endl;
            outputFile << element.batch_size << ", " << element.milliseconds << std::endl;
        }
        outputFile.close();
        std::cout << "Benchmark results written to '" << fileName << "'.\n\n" << std::endl;
    } else {
        std::cerr << "Unable to open the file." << std::endl;
    }

}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: phasm-example-loading-pt <path-to-exported-pt-module>\n";
        exit(-1);
    }

    phasm::get_libtorch_version();

    /** FIRST RD: demo functionality on CPU */
    torch::Device device = torch::kCPU;

    /* Load module to the assigned device */
    phasm::TorchscriptModel model = phasm::TorchscriptModel(argv[1], true, device);

    /* Get the input layer information and claim the input based on device */
    std::vector<int64_t> first_layer_shape = model.GetFirstLayerShape();

    // Caculate the input dimension, based on first layer's weight matrix shape.
    // TODO (@xmei): below method is for RNN layer only. Different for nn.linear and nn.conv.
    std::vector<int64_t> input_shape;
    input_shape.push_back(DEMO_BATCH_SIZE);
    input_shape.push_back(DEMO_SEQ_LENGTH);
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

    /** BENCHMARKING ON CPU*/
    benchmarkWrapper(first_layer_shape, false, argv[1]);

    /** BENCHMARKING ON GPU*/
    bool has_gpu = phasm::has_cuda_device();
    if (!has_gpu) {
        std::cout << "No CUDA devices available. Exit...\n\n";
        return 0;
    }
    benchmarkWrapper(first_layer_shape, true, argv[1]);

    return 0;
}
