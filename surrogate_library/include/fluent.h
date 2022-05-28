
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_FLUENT_H
#define SURROGATE_TOOLKIT_FLUENT_H

#include <optics.h>

namespace phasm::fluent {
using namespace optics;


template <typename HeadT, typename ...RestTs>
struct Cursor;

struct Builder {
    std::vector<OpticBase*> locals;
    std::vector<OpticBase*> globals;
    std::map<std::string, OpticBase*> model_vars;
    std::map<std::string, OpticBase*> getModelVars();
    void print();
    void printModelVars();

    template <typename T>
    Cursor<T> local(std::string name);

    template <typename T>
    Cursor<T> global(std::string name, T*);

private:
    void printOptic(OpticBase* optic, int level);
    static OpticBase* createOpticPathFromLeafToRoot(OpticBase* leaf);
};


template <typename HeadT>
struct Cursor<HeadT> {
    OpticBase* focus = nullptr;
    Builder* builder = nullptr;

    Cursor(OpticBase* focus, Builder* builder);
    Cursor<HeadT> primitive(std::string name);
    Cursor<HeadT> primitives(std::string name, std::vector<int64_t>&& shape);
    Cursor<HeadT> array(size_t size);
    Builder& end();

    template <typename T>
    Cursor<T, HeadT> accessor(std::function<T*(HeadT*)> lambda);
};


template <typename HeadT, typename... RestTs>
struct Cursor {
    OpticBase* focus = nullptr;
    Builder* builder = nullptr;

    Cursor(OpticBase* focus, Builder* builder);
    Cursor<HeadT, RestTs...> primitive(std::string name);
    Cursor<HeadT, RestTs...> primitives(std::string name, std::vector<int64_t>&& shape);
    Cursor<HeadT, RestTs...> array(size_t size);
    Cursor<RestTs...> end();

    template <typename T>
    Cursor<T, HeadT, RestTs...> accessor(std::function<T*(HeadT*)> lambda);
};


template <typename T>
struct Root : Optic<T> {
    Optic<T>* optic;
    T* global = nullptr;
    Root() {
        OpticBase::consumes = "nothing";
        OpticBase::produces = demangle<T>();
    }
};


// ------------------------------------------------------
// Template member function definitions for Builder
// ------------------------------------------------------

template <typename T>
Cursor<T> Builder::local(std::string name) {
    auto r = new Root<T>;
    r->name = name;
    r->produces = demangle<T>();
    locals.push_back(r);
    return Cursor<T>(r, this);
}

template <typename T>
Cursor<T> Builder::global(std::string name, T*) {
    auto r = new Root<T>;
    r->name = name;
    r->produces = demangle<T>();
    globals.push_back(r);
    return Cursor<T>(r, this);
}


// ------------------------------------------------------
// Template member function definitions for Cursor<HeadT>
// ------------------------------------------------------

template<typename HeadT>
Cursor<HeadT>::Cursor(OpticBase *focus, Builder *builder) : focus(focus), builder(builder) {}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitive(std::string name) {
    auto child = new Primitive<HeadT>();
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);
    return *this;
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitives(std::string name, std::vector<int64_t> &&shape) {
    auto child = new PrimitiveArray<HeadT>(std::move(shape));
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);
    return *this;
}

template<typename HeadT>
template<typename T>
Cursor<T, HeadT> Cursor<HeadT>::accessor(std::function<T *(HeadT *)> lambda) {
    auto child = new Field<HeadT, T>(nullptr, lambda);
    focus->unsafe_attach(child);
    return Cursor<T, HeadT>(child, builder);
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::array(size_t size) {
    auto child = new Array<HeadT>(nullptr, size);
    focus->unsafe_attach(child);
    return Cursor<HeadT>(child, builder);
}

template<typename HeadT>
Builder &Cursor<HeadT>::end() {
    return *builder;
}


// -----------------------------------------------------------------
// Template member function definitions for Cursor<HeadT, RestTs...>
// -----------------------------------------------------------------

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...>::Cursor(OpticBase* focus, Builder* builder) : focus(focus), builder(builder) {}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitive(std::string name) {
    auto child = new Primitive<HeadT>();
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);
    return *this;
}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitives(std::string name, std::vector<int64_t>&& shape) {
    auto child = new PrimitiveArray<HeadT>(std::move(shape));
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);
    return *this;
}

template <typename HeadT, typename... RestTs>
template <typename T>
Cursor<T, HeadT, RestTs...> Cursor<HeadT, RestTs...>::accessor(std::function<T*(HeadT*)> lambda) {
    auto child = new Field<HeadT, T>(nullptr, lambda);
    focus->unsafe_attach(child);
    return Cursor<T, HeadT, RestTs...>(child, builder);
}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::array(size_t size) {
    auto child = new Array<HeadT>(nullptr, size);
    focus->unsafe_attach(child);
    return Cursor<HeadT, RestTs...>(child, builder);
}

template <typename HeadT, typename... RestTs>
Cursor<RestTs...> Cursor<HeadT, RestTs...>::end() {
    return Cursor<RestTs...>(focus->parent, builder);
};


} // namespace phasm::fluent

#endif //SURROGATE_TOOLKIT_FLUENT_H
