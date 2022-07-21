
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <iomanip>
#include <model.h>
#include <surrogate_builder.h>
#include "feedforward_model.h"
#include "torchscript_model.h"


phasm::TorchscriptModel model = phasm::TorchscriptModel("~/home/USER/phasm/python/tutorial.pt"); // update the PATH depending on the user


phasm::Surrogate s_surrogate = phasm::SurrogateBuilder()
        .set_model(std::make_shared<phasm::TorchscriptModel>(phasm::TorchscriptModel("~/home/coho905/phasm/python/tutorial.pt")))
        .local_primitive<double>("x", phasm::IN)
        .local_primitive<double>("y", phasm::IN)
        .local_primitive<double>("z", phasm::IN)
        .local_primitive<double>("result", phasm::OUT)
        .finish();






int main(int argc, char** argv) {

    std::cout<< "testing the implementation";
    return 0;
}
