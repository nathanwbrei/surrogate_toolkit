
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

    void add_plugin(std::string plugin_name);

    void add_plugin_path(std::string path);

    void attach_plugins();

    Plugin* attach_plugin(std::string plugin_name);

    Plugin* get_plugin(const std::string& plugin_name);

private:
    std::string m_plugin_paths_str;
    std::string m_plugin_names_str;
    std::vector<std::string> m_plugins_to_include;
    std::vector<std::string> m_plugins_to_exclude;
    std::vector<std::string> m_plugin_paths;
    std::map<std::string, std::pair<Plugin*, void *>> m_loaded_plugins;

    // bool m_verbose = false;
    // JLogger m_logger;
};

extern PluginLoader g_plugin_loader;

} // namespace phasm
