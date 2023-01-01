
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "plugin_loader.h"

#include <dlfcn.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include <sstream>



namespace phasm {

PluginLoader& PluginLoader::get_singleton() {
    /// We allow Surrogates to be declared as static globals. This is because the surrogate needs to
    /// have a lifetime that matches or exceeds that of the function being surrogated. However, this
    /// poses problems with the plugin loader, which should also be a global or static for the same reasons.
    /// C++ won't get the static initialization order right, so the constructor for a global Surrogate
    /// will see a zero-initialized global PluginLoader whose constructor hasn't been called yet.
    /// We remediate this situation by forcing the PluginLoader ctor to be called via an accessor function.
    /// Note: I'm not convinced we won't run into other problems stemming from C++'s static initialization order fiasco
    ///       so we may end up revisiting this approach.
    static PluginLoader g_plugin_loader;
    return g_plugin_loader;
}

PluginLoader::PluginLoader() {

    std::cout << "PHASM: Instantiating plugin loader" << std::endl;
    // Obtain the search paths from colon-separated PHASM_PLUGIN_PATH environment variable
    char* envvar = std::getenv("PHASM_PLUGIN_PATH");
    if (envvar != nullptr) {
        std::stringstream envvar_ss(envvar);
        std::string path;
        while (getline(envvar_ss, path, ':')) add_plugin_path(path);
    }

    // Check in the current directory as well, but give this the lowest priority
#define XSTR(a) STR(a)
#define STR(a) #a
#define ADD_PLUGIN_PATH( x ) add_plugin_path(XSTR(x));
    ADD_PLUGIN_PATH(PHASM_PLUGIN_DIR);
#undef ADD_PLUGIN_PATH
}


PluginLoader::~PluginLoader() {
    // Loop over open plugin handles.
    // Call FinalizePlugin if it has one and close it in all cases.
    // typedef void FinalizePlugin_t();
    for (auto& p: m_loaded_plugins) {
        auto soname = p.first;
        auto handle = p.second->dl_handle;
        std::cout << "PHASM: Closing plugin '" << p.first << "'" << std::endl;

        // For now the Plugin is static, so DO NOT DELETE.
        dlclose(handle);

        // It is tempting to give ownership of the handle to the Plugin so that the Plugin
        // dlcloses() itself automatically. However, the handle exists potentially before
        // the Plugin is created, and is required to live _at_least_ as long as the Plugin
        // but possibly longer. So really, if dl were object-oriented, Handle should own Plugin.
        // Thus PluginLoader owns the handle. The handle "owns" the Plugin in the sense that the
        // Plugin is static so it gets disposed when the handle gets dlclosed().

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
    std::cout << "PHASM: Adding plugin search path: '" << path << "'" << std::endl;
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
        return m_loaded_plugins[plugin_name];
    }

    // Search for file in m_plugin_paths
    std::string exact_plugin_name = find_plugin(plugin_name); // Will except if not found

    // Load plugin from library file
    Plugin* plugin = load_plugin(exact_plugin_name); // Will except if not found
    if (plugin == nullptr) {
        throw std::runtime_error("Unable to attach plugin!");
    }

    m_loaded_plugins[plugin->get_name()] = plugin;
    return plugin;
}

std::string PluginLoader::find_plugin(const std::string& short_plugin_name) {

    for (std::string path: m_plugin_paths) {
        std::string fullpath = path + "/" + short_plugin_name + ".so";
        if (access(fullpath.c_str(), F_OK) != -1) {
            std::cout << "PHASM: Checking '" << fullpath << "' ... Found!" << std::endl;
            return fullpath;
        }
        else {
            std::cout << "PHASM: Checking '" << fullpath << "' ... Not found" << std::endl;
        }
    }

    // If we didn't find the plugin, then complain and quit
    std::stringstream oss;
    oss << "PHASM: Couldn't find plugin '" << short_plugin_name << "' on $PHASM_PLUGIN_PATH=";
    for (auto path : m_plugin_paths) {
        oss << "'" << path << "',";
    }
    oss << std::endl;

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
        std::cout << "PHASM: Opening plugin '" << exact_plugin_name << "'" << std::endl;
        Plugin* plugin = (*get_plugin)();
        plugin->dl_handle = handle;
        return plugin;
    } else {
        dlclose(handle);
        std::cout << "PHASM: Plugin '" << exact_plugin_name << "' does not have a get_plugin() function. Ignoring." << std::endl;
        return nullptr;
    }
}


} //namespace phasm
