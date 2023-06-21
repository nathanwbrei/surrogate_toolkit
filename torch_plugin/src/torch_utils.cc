//
// Created by xmei on 6/21/23.
//

#include "torch_utils.h"

namespace phasm {

    bool has_cuda_device() {
        return torch::cuda::is_available();
    }

    void get_libtorch_version() {
        std::cout << "LibTorch version: "
                  << TORCH_VERSION_MAJOR << "."
                  << TORCH_VERSION_MINOR << "."
                  << TORCH_VERSION_PATCH << std::endl;
        std::cout << std::endl;
    }
}
