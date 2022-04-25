

// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>

#include <JANA/JApplication.h>
#include <JANA/Calibrations/JCalibrationManager.h>
#include <JANA/Calibrations/JCalibrationGeneratorCCDB.h>
#include "DMagneticFieldMapFineMesh.h"
#include "surrogate.h"
#include "torchscript_model.h"

int main() {


    auto model = std::make_shared<TorchscriptModel>("model.pt");
    model->input("x", new optics::Primitive<double>());
    model->input("y", new optics::Primitive<double>());
    model->input("z", new optics::Primitive<double>());
    model->output("Bx", new optics::Primitive<double>());
    model->output("By", new optics::Primitive<double>());
    model->output("Bz", new optics::Primitive<double>());
    model->initialize();

    Surrogate::set_call_mode(Surrogate::CallMode::CaptureAndDump);

    japp = new JApplication;
    auto calib_man = std::make_shared<JCalibrationManager>();
    // calib_man->AddCalibrationGenerator(new JCalibrationGeneratorCCDB);
    // japp->SetParameterValue("jana:calib_url", "mysql://ccdb_user@hallddb.jlab.org/ccdb");
    japp->SetParameterValue("jana:calib_url", "file:///cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources");
    japp->SetParameterValue("jana:resource_dir", "/cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources");
    japp->ProvideService(calib_man);
    // DMagneticFieldMapFineMesh mfmfm(japp, 1, "Magnets/Solenoid/solenoid_1350A_poisson_20160222");
    DMagneticFieldMapFineMesh mfmfm(japp, "/cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources/Magnets/Solenoid/finemeshes/solenoid_1350A_poisson_20160222.evio");

    double x, y, z, bx, by, bz;

    Surrogate surrogate([&](){ mfmfm.GetField(x,y,z,bx,by,bz); }, model);
    surrogate.bind_input("x", &x);
    surrogate.bind_input("y", &y);
    surrogate.bind_input("z", &z);
    surrogate.bind_output("Bx", &bx);
    surrogate.bind_output("By", &by);
    surrogate.bind_output("Bz", &bz);

    for (x = 0.0; x < 3.0; x+=.5) {
        for (y = 0.0; y < 3.0; y+=0.5) {
            for (z = 0.0; z < 3.0; z+=0.5) {
                surrogate.call();
            }
        }
    }
}

