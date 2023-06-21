//
// Created by xmei on 6/21/23.
//

#include <torch/torch.h>

#ifndef PHASM_TORCH_UTILS_H
#define PHASM_TORCH_UTILS_H

namespace phasm {

    /**
     * Ask the node whether it has a CUDA device.
     * @return whether the device has CUDA device or not.
     */
    bool has_cuda_device();

    /**
     * Print the libtorch version.
     */
    void get_libtorch_version();
}

#endif //PHASM_TORCH_UTILS_H
