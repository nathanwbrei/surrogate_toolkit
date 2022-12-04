
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "plugin_loader.h"

TEST_CASE("PluginLoader basics") {
    phasm::PluginLoader pl;
    pl.add_plugin("phasm-torch-plugin");
    pl.attach_plugins();
    auto* plugin = pl.get_plugin("phasm-torch-plugin");
    plugin->print_hello();
}

