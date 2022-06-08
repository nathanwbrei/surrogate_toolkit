
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "feedforward_model.h"
#include "surrogate.h"
#include "fluent.h"


std::shared_ptr<phasm::FeedForwardModel> s_model = nullptr;

struct ToyMagFieldMap {

    void getField(double x, double y, double z, double& Bx, double& By, double& Bz) {
        using namespace phasm;
        if (s_model == nullptr) {
            // TODO: std::call_once or similar
            OpticBuilder builder;
            builder.local_primitive<double>("x", IN)
                   .local_primitive<double>("y", IN)
                   .local_primitive<double>("z", IN)
                   .local_primitive<double>("Bx", OUT)
                   .local_primitive<double>("By", OUT)
                   .local_primitive<double>("Bz", OUT);

            s_model = std::make_shared<FeedForwardModel>();
            s_model->add_vars(builder);
            s_model->initialize(); // TODO: If we forget this, everything crashes when we try to train
        }
        std::cout << "Binding &Bz=" << &Bz << std::endl;
        // Because Bz is a reference, &Bz changes depending on the caller, so we have to rebind on every call!
        // This includes both the lambda and the CallSiteVariables.
        // In theory we don't have to re-copy the CallSiteVariables over from the model every time, but
        // that is an optimization that can wait until the next time we rejigger the whole domain model.

        auto surrogate = phasm::Surrogate([&](){ return this->getFieldOriginal(x,y,z,Bx,By,Bz);}, s_model);
        surrogate.bind_all_locals(&x, &y, &z, &Bx, &By, &Bz);
        surrogate.call();
        // TODO: Fluent interface for surrogate!?
    }
    void getFieldOriginal(double x, double y, double z, double& Bx, double& By, double& Bz) {
        Bx = 2;
        By = x + 3;
        Bz = x * x;
    }
};

TEST_CASE("Toy magnetic field map") {

    // TODO: Can we make this any less awkward?
    phasm::Surrogate::set_call_mode(phasm::Surrogate::CallMode::CaptureAndDump);

    ToyMagFieldMap tmfm;
    double Bx, By, Bz;

    tmfm.getField(1, 2, 3, Bx, By, Bz);
    REQUIRE(Bx == 2);
    REQUIRE(By == 4);
    REQUIRE(Bz == 1);

    tmfm.getField(2, 2, 2, Bx, By, Bz);
    REQUIRE(Bx == 2);
    REQUIRE(By == 5);
    REQUIRE(Bz == 4);

    double Bz2;

    tmfm.getField(7, 2, 2, Bx, By, Bz2);
    REQUIRE(Bz == 4);
    REQUIRE(Bz2 == 49);

    s_model->dump_captures_to_csv(std::cout);

    REQUIRE(s_model->get_capture_count() == 3);
    REQUIRE(s_model->get_model_var("x")->training_inputs[0].get<double>()[0] == 1);
    REQUIRE(s_model->get_model_var("Bz")->training_outputs[1].get<double>()[0] == 4);
}