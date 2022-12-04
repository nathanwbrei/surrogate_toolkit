
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#pragma once
#include <memory>

namespace phasm {

struct Plugin {

    virtual ~Plugin() = default;
    // We don't use the filename as the plugin name anymore because normalizing the filename is more trouble than it's worth.
    virtual std::string get_name() = 0;
    virtual void print_hello() = 0;
    // Eventually make_model(std::string modeltype, std::string modelname)
    // Eventually make_tensor(...)
};

using PluginGetter = Plugin*();

/// Each plugin is a shared library which is required to provide the following:
/// extern "C" { std::unique_ptr<Plugin> get_plugin(); }

} // namespace phasm


