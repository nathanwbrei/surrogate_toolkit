//
// Created by xmei on 1/26/23.
//

#include "torch_cuda_utils.h"

namespace phasm {

    void get_current_cuda_device_info() {
        cudaDeviceProp *cuda_prop = at::cuda::getCurrentDeviceProperties();
        std::cout << "CUDA device name: " << cuda_prop->name << std::endl;
        std::cout << "CUDA compute capacity: "
                  << cuda_prop->major << "." << cuda_prop->minor << std::endl;
        std::cout << std::endl;
    }
}
