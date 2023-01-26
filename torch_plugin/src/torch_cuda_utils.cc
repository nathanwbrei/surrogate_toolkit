//
// Created by xmei on 1/26/23.
//

#include "torch_cuda_utils.h"

namespace phasm {

    bool has_cuda_device() {
        auto cuda_available = torch::cuda::is_available();
        if (not cuda_available) {
            return false;
        } else {
            return true;
        }
    }

    void get_current_cuda_device_info() {
        cudaDeviceProp *cuda_prop = at::cuda::getCurrentDeviceProperties();
        std::cout << "CUDA device name: " << cuda_prop->name << std::endl;
        std::cout << "CUDA compute capacity: "
                  << cuda_prop->major << "." << cuda_prop->minor << std::endl;
        std::cout << std::endl;
    }

    void get_libtorch_version() {
        std::cout << "LibTorch version: "
                  << TORCH_VERSION_MAJOR << "."
                  << TORCH_VERSION_MINOR << "."
                  << TORCH_VERSION_PATCH << std::endl;
        std::cout << std::endl;
    }
}
