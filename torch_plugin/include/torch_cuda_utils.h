//
// Created by xmei on 1/26/23.
//

#include <ATen/cuda/CUDAContext.h>  // for cuda device properties
#include <torch/torch.h>

#ifndef PHASM_TORCH_CUDA_UTILS_H
#define PHASM_TORCH_CUDA_UTILS_H

namespace phasm {

    /**
     * Print current device information: the device name and compute capacity.
     */
    void get_current_cuda_device_info();
}

#endif //PHASM_TORCH_CUDA_UTILS_H
