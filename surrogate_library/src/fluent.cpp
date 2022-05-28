
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "fluent.h"

using namespace phasm::fluent;


void Builder::printOptic(OpticBase* optic, int level) {

    for (int i=0; i<level; ++i) {
        std::cout << "    ";
    }
    if (!optic->name.empty()) {
        std::cout << optic->name << ": ";
    }
    if (optic->consumes.empty()) {
        std::cout << optic->produces << std::endl;
    }
    else if (optic->produces.empty()) {
        std::cout << optic->consumes << std::endl;
    }
    else {
        std::cout << optic->consumes << "->" << optic->produces << std::endl;
    }
    for (auto child : optic->children) {
        printOptic(child, level+1);
    }
}

/// Given a leaf in the optics tree, create a new chain of optics which
/// represents the path from the root node to the leaf. This way we can build the
/// tensor for a specific model variable instead of building all tensors at once.
optics::OpticBase* Builder::createOpticPathFromLeafToRoot(OpticBase* leaf) {
    OpticBase* current = new OpticBase(*leaf);
    OpticBase* parent = leaf->parent;
    while (parent != nullptr) {
        auto* new_parent = new OpticBase(*parent);
        current->parent = new_parent;
        new_parent->unsafe_use(current);
        new_parent->children.clear();
        new_parent->children.push_back(current);

        current = new_parent;
        parent = current->parent;
    }
    return current;
}

void Builder::print() {
    for (auto g : globals) {
        printOptic(g, 0);
    }
    for (auto l : locals) {
        printOptic(l, 0);
    }
}

std::map<std::string, OpticBase*> Builder::getModelVars() {
    std::map<std::string, OpticBase*> results;
    std::queue<OpticBase*> q;
    for (auto g : globals) {
        q.push(g);
    }
    for (auto l : locals) {
        q.push(l);
    }
    while (!q.empty()) {
        OpticBase* o = q.front();
        if (o->is_leaf) {
            results[o->name] = createOpticPathFromLeafToRoot(o);
        }
        else {
            for (OpticBase* c : o->children) {
                q.push(c);
            }
        }
        q.pop();
    }
    return results;
}

void Builder::printModelVars() {
    auto vars = getModelVars();
    for (auto p : vars) {
        std::cout << p.first << ":" << std::endl;
        printOptic(p.second, 1);
    }
}
