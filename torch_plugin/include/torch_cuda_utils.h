//
// Created by xmei on 1/26/23.
//

#include <ATen/cuda/CUDAContext.h>  // for cuda device properties
#include <torch/torch.h>

#ifndef PHASM_TORCH_CUDA_UTILS_H
#define PHASM_TORCH_CUDA_UTILS_H

namespace phasm {

    /**
     * Ask the node whether it has a CUDA device
     * @return whether the device has CUDA device or not
     */
    bool has_cuda_device();

    /**
     * Print current device information: the device name and compute capacity.
     */
    void get_current_cuda_device_info();

    /**
     * Print the libtorch version.
     */
    void get_libtorch_version();
}

#endif //PHASM_TORCH_CUDA_UTILS_H
