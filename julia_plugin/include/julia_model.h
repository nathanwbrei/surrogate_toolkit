
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#pragma once
#include <string>
#include <model.h>


namespace phasm {


class JuliaModel : public phasm::Model {

    std::string m_filepath;

public:
    JuliaModel(std::string filepath) : m_filepath(filepath) {}

    void initialize() override;

    void train_from_captures() override;

    bool infer() override;

};

} // namespace phasm
