
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#pragma once
#include <memory>

namespace phasm {

struct Plugin {
    virtual ~Plugin() = default;
    virtual void print_hello() = 0;
    // Eventually get_model(std::string modeltype, std::string modelname)
};

using PluginGetter = Plugin*();

/// Each plugin is a shared library which is required to provide the following:
/// extern "C" { std::unique_ptr<Plugin> load_plugin(); }

} // namespace phasm


