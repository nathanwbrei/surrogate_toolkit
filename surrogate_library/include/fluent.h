
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_FLUENT_H
#define SURROGATE_TOOLKIT_FLUENT_H

#include <optics.h>

namespace phasm::fluent {
using namespace optics;

template <typename T>
struct Root : Optic<T> {
    std::string binding_name;
    Optic<T>* optic;
};

struct Builder;

template <typename HeadT, typename... RestTs>
struct Cursor {
    OpticBase* focus = nullptr;
    Builder* builder = nullptr;

    Cursor(OpticBase* focus, Builder* builder) : focus(focus), builder(builder) {}


    Cursor<HeadT, RestTs...> primitive(std::string name) {
        auto child = new Primitive<HeadT>();
        child->name = name;
        child->is_leaf = true;
        focus->unsafe_attach(child);
        return *this;
    }

    Cursor<HeadT, RestTs...> primitives(std::string name, std::vector<int64_t>&& shape) {
        auto child = new PrimitiveArray<HeadT>(std::move(shape));
        child->name = name;
        child->is_leaf = true;
        focus->unsafe_attach(child);
        return *this;
    }

    template <typename T>
    Cursor<T, HeadT, RestTs...> accessor(std::function<T*(HeadT*)> lambda) {
        auto child = new Field<HeadT, T>(nullptr, lambda);
        focus->unsafe_attach(child);
        return Cursor<T, HeadT, RestTs...>(child, builder);
    }

    Cursor<HeadT, RestTs...> array(size_t size) {
        auto child = new Array<HeadT>(nullptr, size);
        focus->unsafe_attach(child);
        return Cursor<HeadT, RestTs...>(child, builder);
    }

    Cursor<RestTs...> end() {
        return Cursor<RestTs...>(focus->parent, builder);
    }
};


template <typename HeadT>
struct Cursor<HeadT> {
    OpticBase* focus = nullptr;
    Builder* builder = nullptr;

    Cursor(OpticBase* focus, Builder* builder) : focus(focus), builder(builder) {}

    Cursor<HeadT> primitive(std::string name) {
        auto child = new Primitive<HeadT>();
        child->name = name;
        child->is_leaf = true;
        focus->unsafe_attach(child);
        return *this;
    }

    Cursor<HeadT> primitives(std::string name, std::vector<int64_t>&& shape) {
        auto child = new PrimitiveArray<HeadT>(std::move(shape));
        child->name = name;
        child->is_leaf = true;
        focus->unsafe_attach(child);
        return *this;
    }

    template <typename T>
    Cursor<T, HeadT> accessor(std::function<T*(HeadT*)> lambda) {
        auto child = new Field<HeadT, T>(nullptr, lambda);
        focus->unsafe_attach(child);
        return Cursor<T, HeadT>(child, builder);
    }

    Cursor<HeadT> array(size_t size) {
        auto child = new Array<HeadT>(nullptr, size);
        focus->unsafe_attach(child);
        return Cursor<HeadT>(child, builder);
    }

    template <typename T>
    Cursor<T, HeadT> custom(OpticBase* child) {
        focus->unsafe_attach(child);
        return Cursor<T, HeadT>(child, builder);
    }

    Builder& end() {
        return *builder;
    }
};



struct Builder {
    std::vector<OpticBase*> locals;
    std::vector<OpticBase*> globals;
    std::map<std::string, OpticBase*> model_vars;

    template <typename T>
    Cursor<T> local(std::string name) {
        auto r = new Root<T>;
        r->name = name;
        r->binding_name = name;
        r->produces = demangle<T>();
        locals.push_back(r);
        return Cursor<T>(r, this);
    }

    template <typename T>
    Cursor<T> global(std::string name, T* var) {
        auto r = new Root<T>;
        r->name = name;
        r->binding_name = name;
        r->produces = demangle<T>();
        globals.push_back(r);
        return Cursor<T>(r, this);
    }

private:
    void printOptic(OpticBase* optic, int level) {
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
    static inline OpticBase* createOpticPathFromLeafToRoot(OpticBase* leaf) {
        OpticBase* current = new OpticBase(*leaf);
        OpticBase* parent = leaf->parent;
        while (parent != nullptr) {
            auto* new_parent = new OpticBase(*parent);
            current->parent = new_parent;
            new_parent->use(current);
            new_parent->children.clear();
            new_parent->children.push_back(current);

            current = new_parent;
            parent = current->parent;
        }
        return current;
    }



public:
    void print() {
        for (auto g : globals) {
            printOptic(g, 0);
        }
        for (auto l : locals) {
            printOptic(l, 0);
        }
    }

    inline std::map<std::string, OpticBase*> getModelVars() {
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

    inline void printModelVars() {
        auto vars = getModelVars();
        for (auto p : vars) {
            std::cout << p.first << ":" << std::endl;
            printOptic(p.second, 1);
        }
    }

};


} // namespace phasm::fluent

#endif //SURROGATE_TOOLKIT_FLUENT_H
