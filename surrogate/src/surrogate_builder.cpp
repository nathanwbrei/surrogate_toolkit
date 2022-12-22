
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "surrogate_builder.h"
#include "plugin_loader.h"

namespace phasm {

SurrogateBuilder& SurrogateBuilder::set_model(std::string plugin_name, std::string model_name) {
    m_model = g_plugin_loader.get_or_load_plugin(plugin_name)->make_model(model_name);
    return *this;
}


std::vector<std::shared_ptr<CallSiteVariable>> SurrogateBuilder::get_callsite_vars() const {
    return m_csvs;
}

std::vector<std::shared_ptr<ModelVariable>> SurrogateBuilder::get_model_vars() const {
    std::vector<std::shared_ptr<ModelVariable>> results;
    for (const auto &csv: m_csvs) {
        for (const auto &mv: csv->model_vars) {
            results.push_back(mv);
        }
    }
    return results;
}


/// Given a leaf in the optics tree, create a new chain of optics which
/// represents the path from the root node to the leaf. This way we can build the
/// tensor for a specific model variable instead of building all tensors at once.
OpticBase *cloneOpticsFromLeafToRoot(OpticBase *leaf) {
    OpticBase *current = leaf->clone();
    OpticBase *parent = leaf->parent;
    while (parent != nullptr) {
        auto *new_parent = parent->clone();
        current->parent = new_parent;
        new_parent->unsafe_use(current);
        new_parent->children.clear();
        new_parent->children.push_back(current);
        current = new_parent;
        parent = current->parent;
    }
    return current;
}


void SurrogateBuilder::printOptic(OpticBase *optic, int level) {

    for (int i = 0; i < level; ++i) {
        std::cout << "    ";
    }
    if (!optic->name.empty()) {
        std::cout << optic->name << ": ";
    }
    if (optic->consumes.empty()) {
        std::cout << optic->produces << std::endl;
    } else if (optic->produces.empty()) {
        std::cout << optic->consumes << std::endl;
    } else {
        std::cout << optic->consumes << "->" << optic->produces << std::endl;
    }
    for (auto child: optic->children) {
        printOptic(child, level + 1);
    }
}

void SurrogateBuilder::printOpticsTree() {
    for (const auto &csv: m_csvs) {
        for (auto o: csv->optics_tree) {
            printOptic(o, 0);
        }
    }
}

void SurrogateBuilder::printModelVars() {
    auto vars = get_model_vars();
    for (const auto &p: vars) {
        std::cout << p->name << ":" << std::endl;
        printOptic(p->accessor, 1);
    }
}

}