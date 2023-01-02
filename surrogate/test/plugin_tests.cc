
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "plugin_loader.h"

TEST_CASE("PluginLoader basics") {
    phasm::PluginLoader pl;
    auto* plugin = pl.get_or_load_plugin("phasm-torch-plugin");
    REQUIRE(plugin->get_name() == "phasm-torch-plugin");
}

