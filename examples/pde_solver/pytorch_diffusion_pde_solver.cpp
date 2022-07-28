
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <iomanip>
#include <model.h>
#include <surrogate_builder.h>
#include "feedforward_model.h"
#include "torchscript_model.h"

phasm::TorchscriptModel PDE_Solver = phasm::TorchscriptModel("~/phasm/python/pytorch_diffusion_pde_solver.cpp");

phasm::Surrogate s_surrogate = phasm::SurrogateBuilder()
    .set_model(std::make_shared<phasm::TorchscriptModel>(phasm::TorchscriptModel("~/phasm/python/pytorch_diffusion_pde_solver.cpp")))

int main() {
    
}
