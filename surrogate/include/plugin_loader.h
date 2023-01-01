
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#pragma once
#include <string>
#include <vector>
#include <map>

#include "plugin.h"

namespace phasm {

class PluginLoader {

public:
    PluginLoader();
    ~PluginLoader();
    void add_plugin_path(std::string path);
    Plugin* get_or_load_plugin(const std::string& plugin_name);

    static PluginLoader& get_singleton();

private:
    std::string find_plugin(const std::string& short_plugin_name);
    Plugin* load_plugin(const std::string& exact_plugin_name);

    std::vector<std::string> m_plugin_paths;
    std::map<std::string, Plugin*> m_loaded_plugins;
};


} // namespace phasm
