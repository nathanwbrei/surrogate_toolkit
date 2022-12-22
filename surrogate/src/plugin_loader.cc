
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "plugin_loader.h"

#include <dlfcn.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include <sstream>

namespace phasm {

PluginLoader g_plugin_loader;


PluginLoader::PluginLoader() {

    // Obtain the search paths from colon-separated PHASM_PLUGIN_PATH environment variable
    char* envvar = std::getenv("PHASM_PLUGIN_PATH");
    if (envvar != nullptr) {
        std::stringstream envvar_ss(envvar);
        std::string path;
        while (getline(envvar_ss, path, ':')) add_plugin_path(path);
    }

    // Check in the current directory as well, but give this the lowest priority
    add_plugin_path(".");
}


PluginLoader::~PluginLoader() {
    // Loop over open plugin handles.
    // Call FinalizePlugin if it has one and close it in all cases.
    // typedef void FinalizePlugin_t();
    for (auto& p: m_loaded_plugins) {
        auto soname = p.first;
        auto handle = p.second.second;
        /*
        FinalizePlugin_t *finalize_proc = (FinalizePlugin_t *) dlsym(handle, "FinalizePlugin");
        if (finalize_proc) {
            std::cout << "Finalizing plugin \"" << soname << "\"" << std::endl;
            (*finalize_proc)();
        }
        */
        // Close plugin handle
        dlclose(handle);
    }
}


void PluginLoader::add_plugin_path(std::string path) {

    /// Add a path to the directories searched for plugins. This should not include the plugin name itself.
    /// If the path is already in the list, it will not be added twice.
    ///
    /// Generally, users will set the path via the PHASM_PLUGIN_PATH environment variable and
    /// won't need to call this method. However, this may be called if it needs to be done programmatically.
    ///
    /// @param path directory to search for plugins.

    for (std::string &n: m_plugin_paths) {
        if (n == path) { return; }
    }
    m_plugin_paths.push_back(path);
}


Plugin* PluginLoader::get_or_load_plugin(const std::string& plugin_name) {

    /// If the plugin has already been loaded, this will return the cached version.
    /// Otherwise it will load the plugin, cache it for future use, and then return it.
    /// Note that the PluginLoader retains ownership of all Plugin objects.

    // Search cached plugins
    auto it = m_loaded_plugins.find(plugin_name);
    if (it != m_loaded_plugins.end()) {
        // We found it in cache
        return m_loaded_plugins[plugin_name].first;
    }

    // Search for file in m_plugin_paths
    std::string exact_plugin_name = find_plugin(plugin_name); // Will except if not found

    // Load plugin from library file
    Plugin* plugin = load_plugin(exact_plugin_name); // Will except if not found
    if (plugin == nullptr) {
        throw std::runtime_error("Unable to attach plugin!");
    }


    return plugin;
}

std::string PluginLoader::find_plugin(const std::string& short_plugin_name) {

    for (std::string path: m_plugin_paths) {
        std::string fullpath = path + "/" + short_plugin_name + ".so";
        if (access(fullpath.c_str(), F_OK) != -1) {
            std::cout << "Checking '" << fullpath << "' ... Found!" << std::endl;
            return fullpath;
        }
        else {
            std::cout << "Checking '" << fullpath << "' ... Not found" << std::endl;
        }
    }

    // If we didn't find the plugin, then complain and quit
    std::stringstream oss;
    oss << "Couldn't find plugin '" << short_plugin_name << "' on $PHASM_PLUGIN_PATH" << std::endl;

    std::cout << oss.str();
    throw std::runtime_error(oss.str());
}

Plugin* PluginLoader::load_plugin(const std::string& exact_plugin_name) {

    // Open shared object
    void *handle = dlopen(exact_plugin_name.c_str(), RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
    if (!handle) {
        std::cout << dlerror() << std::endl;
        return nullptr;
    }

    phasm::PluginGetter *get_plugin = (phasm::PluginGetter *) dlsym(handle, "get_plugin");

    if (get_plugin) {
        std::cout << "Initializing plugin \"" << exact_plugin_name << "\"" << std::endl;
        Plugin* plugin = (*get_plugin)();

        m_loaded_plugins[plugin->get_name()] = {plugin, handle};
        // It's a little bit weird to cache the plugin here, but otherwise we lose the handle.
        // The alternative is to return std::pair<Plugin*, void*>, or to put the handle inside
        // the Plugin object. I'm leaning towards the latter.

        return plugin;
    } else {
        dlclose(handle);
        std::cout << "Plugin \"" << exact_plugin_name << "\" does not have a get_plugin() function. Ignoring." << std::endl;
        return nullptr;
    }
}


} //namespace phasm
