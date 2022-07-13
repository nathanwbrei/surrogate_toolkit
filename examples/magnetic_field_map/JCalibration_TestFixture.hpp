
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_J_CALIBRATION_TEST_FIXTURE_HPP
#define SURROGATE_TOOLKIT_J_CALIBRATION_TEST_FIXTURE_HPP

#include <JANA/Calibrations/JCalibration.h>

class JCalibration_TestFixture : public JCalibration {

    std::map<std::string, std::map<std::string, std::string>> m_mss;
    std::map<std::string, std::vector<std::string>> m_vs;
    std::map<std::string, std::vector<std::map<std::string, std::string>>> m_vmss;
    std::map<std::string, std::vector<std::vector<std::string>>> m_vvs;

public:

    JCalibration_TestFixture();
    JCalibration_TestFixture(const JCalibration_TestFixture& other) = default;

    void GetListOfNamepaths(std::vector<std::string> &namepaths) override;

    bool GetCalib(std::string namepath, std::map<std::string, std::string> &svals, uint64_t event_number=0) override;
    bool GetCalib(std::string namepath, std::vector<std::string> &svals, uint64_t event_number=0) override;
    bool GetCalib(std::string namepath, std::vector<std::map<std::string, std::string> > &svals, uint64_t event_number=0) override;
    bool GetCalib(std::string namepath, std::vector<std::vector<std::string>> &svals, uint64_t event_number=0) override;

    void SetCalib(std::string namepath, std::map<std::string, std::string> &svals);
    void SetCalib(std::string namepath, std::vector<std::string> &svals);
    void SetCalib(std::string namepath, std::vector<map<std::string, std::string>> &svals);
    void SetCalib(std::string namepath, std::vector<std::vector<std::string>> &svals);

};


#endif //SURROGATE_TOOLKIT_J_CALIBRATION_TEST_FIXTURE_HPP
