

// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>

#include <JANA/JApplication.h>
#include <JANA/Calibrations/JCalibrationManager.h>
#include <JANA/Calibrations/JCalibrationGeneratorCCDB.h>
#include "DMagneticFieldMapFineMesh.h"

int main() {
    japp = new JApplication;
    auto calib_man = std::make_shared<JCalibrationManager>();
    // calib_man->AddCalibrationGenerator(new JCalibrationGeneratorCCDB);
    // japp->SetParameterValue("jana:calib_url", "mysql://ccdb_user@hallddb.jlab.org/ccdb");
    japp->SetParameterValue("jana:calib_url", "file:///cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources");
    japp->SetParameterValue("jana:resource_dir", "/cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources");
    japp->ProvideService(calib_man);
    // DMagneticFieldMapFineMesh mfmfm(japp, 1, "Magnets/Solenoid/solenoid_1350A_poisson_20160222");
    DMagneticFieldMapFineMesh mfmfm(japp, "/cvmfs/oasis.opensciencegrid.org/gluex/group/halld/www/halldweb/html/resources/Magnets/Solenoid/finemeshes/solenoid_1350A_poisson_20160222.evio");

    double bx, by, bz;
    mfmfm.GetField(0.0, 0.0, 0.0, bx, by, bz);
    std::cout << bx << ", " << by << ", " << bz << std::endl;

    mfmfm.GetField(1.0, 1.0, 1.0, bx, by, bz);
    std::cout << bx << ", " << by << ", " << bz << std::endl;

    mfmfm.GetField(0.001, 0.001, 0.001, bx, by, bz);
    std::cout << bx << ", " << by << ", " << bz << std::endl;
}

