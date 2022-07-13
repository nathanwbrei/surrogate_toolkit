
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "JCalibration_TestFixture.hpp"


JCalibration_TestFixture::JCalibration_TestFixture() : JCalibration("", 0, "") { };

void JCalibration_TestFixture::GetListOfNamepaths(vector<std::string> &namepaths) {
    for (auto& pair: m_vs) {
        namepaths.push_back(pair.first);
    }
    for (auto& pair: m_mss) {
        namepaths.push_back(pair.first);
    }
    for (auto& pair: m_vmss) {
        namepaths.push_back(pair.first);
    }
    for (auto& pair: m_vvs) {
        namepaths.push_back(pair.first);
    }
}

bool JCalibration_TestFixture::GetCalib(std::string namepath, map<std::string, std::string> &svals, uint64_t) {
    auto item = m_mss.find(namepath);
    if (item == m_mss.end()) return false;
    svals = item->second;
    return true;
}

bool JCalibration_TestFixture::GetCalib(std::string namepath, vector<std::string> &svals, uint64_t) {
    auto item = m_vs.find(namepath);
    if (item == m_vs.end()) return false;
    svals = item->second;
    return true;
}

bool JCalibration_TestFixture::GetCalib(std::string namepath, vector<std::map<std::string, std::string>> &svals,
                                        uint64_t) {
    auto item = m_vmss.find(namepath);
    if (item == m_vmss.end()) return false;
    svals = item->second;
    return true;
}

bool JCalibration_TestFixture::GetCalib(std::string namepath, vector<std::vector<std::string>> &svals,
                                        uint64_t) {
    auto item = m_vvs.find(namepath);
    if (item == m_vvs.end()) return false;
    svals = item->second;
    return true;
}

void JCalibration_TestFixture::SetCalib(std::string namepath, map<std::string, std::string> &svals) {
    m_mss[namepath] = svals;
}

void JCalibration_TestFixture::SetCalib(std::string namepath, vector<std::string> &svals) {
    m_vs[namepath] = svals;
}

void JCalibration_TestFixture::SetCalib(std::string namepath, vector<map<std::string, std::string>> &svals) {
    m_vmss[namepath] = svals;
}

void JCalibration_TestFixture::SetCalib(std::string namepath, vector<std::vector<std::string>> &svals) {
    m_vvs[namepath] = svals;
}


