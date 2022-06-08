
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_FLUENT_H
#define SURROGATE_TOOLKIT_FLUENT_H

#include <optics.h>
#include <call_site_variable.h>
#include <model_variable.h>


namespace phasm {



template <typename HeadT, typename ...RestTs>
struct Cursor;

/// OpticBuilder gives us a type-safe and intuitive way to represent the tree of accessors
/// There are two representations that make sense:
/// 1. A tree, where the root is a CallSiteVariable, the branches are Optics, and the leaves are ModelVariables.
/// 2. A collection of root-to-leaf paths, each 'belonging' to its corresponding ModelVariable.

/// Right now, PHASM uses (2) internally because it is simple. However, when we add support for non-default-constructible
/// objects, we will likely need (1) because we cannot always hydrate objects piecemeal.

class OpticBuilder {

    std::vector<std::shared_ptr<CallSiteVariable>> m_csvs;

public:
    template <typename T>
    Cursor<T> local(std::string name);

    template <typename T>
    Cursor<T> global(std::string name, T*);

    std::vector<std::shared_ptr<CallSiteVariable>> get_callsite_vars() const;
    std::vector<std::shared_ptr<ModelVariable>> get_model_vars() const;
    void printOpticsTree();
    void printModelVars();

private:
    void printOptic(OpticBase* optic, int level);
};


template <typename HeadT>
struct Cursor<HeadT> {
    std::shared_ptr<CallSiteVariable> current_callsite_var;
    OpticBuilder* builder = nullptr;

    Cursor(OpticBase* o, std::shared_ptr<CallSiteVariable> csv, OpticBuilder* builder);
    Cursor<HeadT> primitive(std::string name, Direction dir=Direction::Input);
    Cursor<HeadT> primitives(std::string name, std::vector<int64_t>&& shape, Direction dir=Direction::Input);
    Cursor<HeadT> array(size_t size);
    OpticBuilder& end();

    template <typename T>
    Cursor<T, HeadT> accessor(std::function<T*(HeadT*)> lambda);
};


template <typename HeadT, typename... RestTs>
struct Cursor {
    OpticBase* focus = nullptr;
    std::shared_ptr<CallSiteVariable> current_callsite_var;
    OpticBuilder* builder = nullptr;

    Cursor(OpticBase* focus, std::shared_ptr<CallSiteVariable> callsite_var, OpticBuilder* builder);
    Cursor<HeadT, RestTs...> primitive(std::string name, Direction dir=Direction::Input);
    Cursor<HeadT, RestTs...> primitives(std::string name, std::vector<int64_t>&& shape, Direction dir=Direction::Input);
    Cursor<HeadT, RestTs...> array(size_t size);
    Cursor<RestTs...> end();

    template <typename T>
    Cursor<T, HeadT, RestTs...> accessor(std::function<T*(HeadT*)> lambda);
};

OpticBase* cloneOpticsFromLeafToRoot(OpticBase* leaf);


// ------------------------------------------------------
// Template member function definitions for Builder
// ------------------------------------------------------

template <typename T>
Cursor<T> OpticBuilder::local(std::string name) {

    auto csv = std::make_shared<CallSiteVariable>(std::move(name), make_any<T>());
    m_csvs.push_back(csv);
    return Cursor<T>(nullptr, csv, this);
}

template <typename T>
Cursor<T> OpticBuilder::global(std::string name, T* tp) {
    auto csv = std::make_shared<CallSiteVariable>(std::move(name), make_any<T>(tp));
    m_csvs.push_back(csv);
    return Cursor<T>(nullptr, csv, this);
}


// ------------------------------------------------------
// Template member function definitions for Cursor<HeadT>
// ------------------------------------------------------

template<typename HeadT>
Cursor<HeadT>::Cursor(OpticBase*, std::shared_ptr<CallSiteVariable> c, OpticBuilder *b) : current_callsite_var(c), builder(b) {}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitive(std::string name, Direction dir) {
    auto child = new Primitive<HeadT>();
    child->name = name;
    child->is_leaf = true;
    current_callsite_var->optics_tree.push_back(child);

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = child;
    mv->is_input = (dir == Direction::Input) || (dir == Direction::InputOutput);
    mv->is_output = (dir == Direction::Output) || (dir == Direction::InputOutput);
    current_callsite_var->model_vars.push_back(mv);
    return *this;
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitives(std::string name, std::vector<int64_t> &&shape, Direction dir) {
    auto child = new PrimitiveArray<HeadT>(std::move(shape));
    child->name = name;
    child->is_leaf = true;
    current_callsite_var->optics_tree.push_back(child);

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = child;
    mv->is_input = (dir == Direction::Input) || (dir == Direction::InputOutput);
    mv->is_output = (dir == Direction::Output) || (dir == Direction::InputOutput);
    current_callsite_var->model_vars.push_back(mv);
    return *this;
}

template<typename HeadT>
template<typename T>
Cursor<T, HeadT> Cursor<HeadT>::accessor(std::function<T *(HeadT *)> lambda) {
    auto child = new Field<HeadT, T>(nullptr, lambda);
    current_callsite_var->optics_tree.push_back(child);
    return Cursor<T, HeadT>(child, current_callsite_var, builder);
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::array(size_t size) {
    auto child = new Array<HeadT>(nullptr, size);
    current_callsite_var->optics_tree.push_back(child);
    return Cursor<HeadT>(child, builder);
}

template<typename HeadT>
OpticBuilder &Cursor<HeadT>::end() {
    return *builder;
}


// -----------------------------------------------------------------
// Template member function definitions for Cursor<HeadT, RestTs...>
// -----------------------------------------------------------------

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...>::Cursor(OpticBase* focus, std::shared_ptr<CallSiteVariable> callsite_var, OpticBuilder* builder)
: focus(focus), current_callsite_var(callsite_var), builder(builder) {}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitive(std::string name, Direction dir) {
    auto child = new Primitive<HeadT>();
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = cloneOpticsFromLeafToRoot(child);
    mv->is_input = (dir == Direction::Input) || (dir == Direction::InputOutput);
    mv->is_output = (dir == Direction::Output) || (dir == Direction::InputOutput);
    current_callsite_var->model_vars.push_back(mv);
    return *this;
}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitives(std::string name, std::vector<int64_t>&& shape, Direction dir) {
    auto child = new PrimitiveArray<HeadT>(std::move(shape));
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = cloneOpticsFromLeafToRoot(child);
    mv->is_input = (dir == Direction::Input) || (dir == Direction::InputOutput);
    mv->is_output = (dir == Direction::Output) || (dir == Direction::InputOutput);
    current_callsite_var->model_vars.push_back(mv);
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
    return Cursor<RestTs...>(focus->parent, current_callsite_var, builder);
};


} // namespace phasm

#endif //SURROGATE_TOOLKIT_FLUENT_H
