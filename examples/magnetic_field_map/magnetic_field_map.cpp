

// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>

#include <JANA/JApplication.h>
#include <JANA/Calibrations/JCalibrationManager.h>
#include "DMagneticFieldMapFineMesh.h"
#include "surrogate_builder.h"
#include "torchscript_model.h"
#include "feedforward_model.h"
#include "JCalibrationGenerator_TestFixture.hpp"


int main() {

    using phasm::SurrogateBuilder, phasm::Direction, phasm::CallMode;
    // auto model = std::make_shared<phasm::TorchscriptModel>("model.pt");
    auto model = std::make_shared<phasm::FeedForwardModel>();

    phasm::Surrogate surrogate = SurrogateBuilder()
            .set_model(model)
            .set_callmode(CallMode::CaptureAndDump)
            .local_primitive<double>("x", Direction::IN)
            .local_primitive<double>("y", Direction::IN)
            .local_primitive<double>("z", Direction::IN)
            .local_primitive<double>("Bx", Direction::OUT)
            .local_primitive<double>("By", Direction::OUT)
            .local_primitive<double>("Bz", Direction::OUT)
            .finish();

    std::map<std::string, std::string> magnet_calib_data;
    magnet_calib_data["URL_base"] = "https://halldweb.jlab.org/resources";
    magnet_calib_data["path"] = "Magnets/Solenoid/solenoid_1350A_poisson_20160222";
    magnet_calib_data["md5"] = "a96263c5e2f3936241b3eadb85e6559f";
    JCalibration_TestFixture calib;
    calib.SetCalib("Magnets/Solenoid/solenoid_1350A_poisson_20160222", magnet_calib_data);
    auto calib_man = std::make_shared<JCalibrationManager>();
    calib_man->AddCalibrationGenerator(new JCalibrationGenerator_TestFixture(calib));
    japp = new JApplication;
    japp->ProvideService(calib_man);

    DMagneticFieldMapFineMesh mfmfm(japp, 1, "Magnets/Solenoid/solenoid_1350A_poisson_20160222");

    double x, y, z, bx, by, bz;

    surrogate.bind_original_function([&]() { mfmfm.GetField(x, y, z, bx, by, bz); });
    surrogate.bind_all_callsite_vars(&x, &y, &z, &bx, &by, &bz);

    for (x = -20.0; x < 20.0; x+=4.0) {
        for (y = -20.0; y < 20.0; y+=4.0) {
            for (z = 0.0; z < 100.0; z+=2.0) {
                // mfmfm.GetField(x,y,z,bx,by,bz);
                surrogate.call();
                std::cout << "x=" << x << ", y=" << y << ", z=" << z << ", Bx=" << bx << ", By=" << by << ", Bz=" << bz << std::endl;
            }
        }
    }
}

