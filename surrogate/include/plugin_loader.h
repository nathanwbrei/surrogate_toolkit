
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_PLUGIN_LOADER_H
#define SURROGATE_TOOLKIT_PLUGIN_LOADER_H

#include <string>
#include <vector>
#include <map>

class PluginLoader {

public:
    PluginLoader();
    ~PluginLoader();
    void add_plugin(std::string plugin_name);
    void add_plugin_path(std::string path);
    void attach_plugins();
    void attach_plugin(std::string plugin_name);

private:
    std::string m_plugin_paths_str;
    std::string m_plugin_names_str;
    std::vector<std::string> m_plugins_to_include;
    std::vector<std::string> m_plugins_to_exclude;
    std::vector<std::string> m_plugin_paths;
    std::map<std::string, void*> m_sohandles; // key=plugin name  val=dlopen handle

    bool m_verbose = false;
    // JLogger m_logger;
};

#endif //SURROGATE_TOOLKIT_PLUGIN_LOADER_H
