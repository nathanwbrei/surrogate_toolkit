
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
#define SURROGATE_TOOLKIT_SURROGATE_BUILDER_H

#include <model.h>
#include <surrogate.h>

namespace phasm {


template <typename HeadT, typename ...RestTs>
struct Cursor;

/// OpticBuilder gives us a type-safe and intuitive way to represent the tree of accessors
/// There are two representations that make sense:
/// 1. A tree, where the root is a CallSiteVariable, the branches are Optics, and the leaves are ModelVariables.
/// 2. A collection of root-to-leaf paths, each 'belonging' to its corresponding ModelVariable.

/// Right now, PHASM uses (2) internally because it is simple. However, when we add support for non-default-constructible
/// objects, we will likely need (1) because we cannot always hydrate objects piecemeal.

inline CallMode g_callmode = get_call_mode_from_envvar();

class SurrogateBuilder {

    std::vector<std::shared_ptr<CallSiteVariable>> m_csvs;
    std::shared_ptr<Model> m_model;
    CallMode m_callmode = CallMode::NotSet;

public:
    inline SurrogateBuilder& set_model(std::shared_ptr<Model> model, bool enable_tensor_combining=false) { m_model = model; m_model->enable_tensor_combining(enable_tensor_combining); return *this; }
    inline SurrogateBuilder& set_callmode(CallMode callmode) { m_callmode = callmode; return *this; }

    SurrogateBuilder& set_model(std::string plugin_name, std::string model_name, bool enable_tensor_combining=false);

    template <typename T>
    Cursor<T> local(std::string name);

    template <typename T>
    Cursor<T> global(std::string name, T*);

    template <typename T>
    SurrogateBuilder& local_primitive(std::string name, Direction dir, std::vector<int64_t>&& shape = {1});

    template <typename T>
    SurrogateBuilder& global_primitive(std::string name, T* binding, Direction dir, std::vector<int64_t>&& shape = {1});

    std::vector<std::shared_ptr<CallSiteVariable>> get_callsite_vars() const;
    std::vector<std::shared_ptr<ModelVariable>> get_model_vars() const;

    Surrogate finish() const;
    void printOpticsTree();
    void printModelVars();

private:
    void printOptic(OpticBase* optic, int level);
};


template <typename HeadT>
struct Cursor<HeadT> {
    std::shared_ptr<CallSiteVariable> current_callsite_var;
    SurrogateBuilder* builder = nullptr;

    Cursor(OpticBase* o, std::shared_ptr<CallSiteVariable> csv, SurrogateBuilder* builder);
    Cursor<HeadT> primitive(std::string name, Direction dir=Direction::IN, DType dtype= default_dtype<HeadT>());
    Cursor<HeadT> primitives(std::string name, std::vector<int64_t>&& shape, Direction dir=Direction::IN, DType dtype= default_dtype<HeadT>());
    Cursor<HeadT> array(size_t size);
    SurrogateBuilder& end();

    template <typename T>
    Cursor<T, HeadT> accessor(std::function<T*(HeadT*)> lambda);

    template <typename T>
    Cursor<T, HeadT> accessor(std::function<T(HeadT*)> getter, std::function<void(HeadT*,T)> setter);
};


template <typename HeadT, typename... RestTs>
struct Cursor {
    OpticBase* focus = nullptr;
    std::shared_ptr<CallSiteVariable> current_callsite_var;
    SurrogateBuilder* builder = nullptr;

    Cursor(OpticBase* focus, std::shared_ptr<CallSiteVariable> callsite_var, SurrogateBuilder* builder);
    Cursor<HeadT, RestTs...> primitive(std::string name, Direction dir=Direction::IN, DType dtype= default_dtype<HeadT>());
    Cursor<HeadT, RestTs...> primitives(std::string name, std::vector<int64_t>&& shape, Direction dir=Direction::IN, DType dtype= default_dtype<HeadT>());
    Cursor<HeadT, RestTs...> array(size_t size);
    Cursor<RestTs...> end();

    template <typename T>
    Cursor<T, HeadT, RestTs...> accessor(std::function<T*(HeadT*)> lambda);

    template <typename T>
    Cursor<T, HeadT, RestTs...> accessor(std::function<T(HeadT*)> getter, std::function<void(HeadT*,T)> setter);
};

OpticBase* cloneOpticsFromLeafToRoot(OpticBase* leaf);


// ------------------------------------------------------
// Template member function definitions for Builder
// ------------------------------------------------------

template <typename T>
Cursor<T> SurrogateBuilder::local(std::string name) {

    auto csv = std::make_shared<CallSiteVariable>(std::move(name), make_any<T>());
    m_csvs.push_back(csv);
    return Cursor<T>(nullptr, csv, this);
}

template <typename T>
Cursor<T> SurrogateBuilder::global(std::string name, T* tp) {
    auto csv = std::make_shared<CallSiteVariable>(std::move(name), make_any<T>(tp));
    m_csvs.push_back(csv);
    return Cursor<T>(nullptr, csv, this);
}

template <typename T>
SurrogateBuilder& SurrogateBuilder::local_primitive(std::string name, Direction dir, std::vector<int64_t>&& shape) {
    auto cursor = local<T>(name).primitives(name, std::move(shape), dir);
    return *this;
}

template <typename T>
SurrogateBuilder& SurrogateBuilder::global_primitive(std::string name, T* tp, Direction dir, std::vector<int64_t>&& shape) {
    global<T>(name, tp).primitives(name, dir, std::move(shape));
    return *this;
}

// ------------------------------------------------------
// Template member function definitions for Cursor<HeadT>
// ------------------------------------------------------

template<typename HeadT>
Cursor<HeadT>::Cursor(OpticBase*, std::shared_ptr<CallSiteVariable> c, SurrogateBuilder *b) : current_callsite_var(c), builder(b) {}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitive(std::string name, Direction dir, DType dtype) {
    return primitives(name, {}, dir, dtype);
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::primitives(std::string name, std::vector<int64_t> &&shape, Direction dir, DType dtype) {
    auto child = new TensorIso<HeadT>(std::move(shape), dtype);
    child->name = name;
    child->is_leaf = true;
    current_callsite_var->optics_tree.push_back(child->clone());

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = child;
    mv->is_input = (dir == Direction::IN) || (dir == Direction::INOUT);
    mv->is_output = (dir == Direction::OUT) || (dir == Direction::INOUT);
    current_callsite_var->model_vars.push_back(mv);
    return *this;
}

template<typename HeadT>
template<typename T>
Cursor<T, HeadT> Cursor<HeadT>::accessor(std::function<T *(HeadT *)> lambda) {
    auto child = new Lens<HeadT, T>(nullptr, lambda);
    current_callsite_var->optics_tree.push_back(child);
    return Cursor<T, HeadT>(child, current_callsite_var, builder);
}

template<typename HeadT>
template<typename T>
Cursor<T, HeadT> Cursor<HeadT>::accessor(std::function<T(HeadT*)> getter, std::function<void(HeadT*,T)> setter) {
    auto child = new ValueLens<HeadT, T>(nullptr, getter, setter);
    current_callsite_var->optics_tree.push_back(child);
    return Cursor<T, HeadT>(child, current_callsite_var, builder);
}

template<typename HeadT>
Cursor<HeadT> Cursor<HeadT>::array(size_t size) {
    auto child = new ArrayTraversal<HeadT>(nullptr, size);
    current_callsite_var->optics_tree.push_back(child);
    return Cursor<HeadT>(child, builder);
}

template<typename HeadT>
SurrogateBuilder &Cursor<HeadT>::end() {
    return *builder;
}


// -----------------------------------------------------------------
// Template member function definitions for Cursor<HeadT, RestTs...>
// -----------------------------------------------------------------

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...>::Cursor(OpticBase* focus, std::shared_ptr<CallSiteVariable> callsite_var, SurrogateBuilder* builder)
: focus(focus), current_callsite_var(callsite_var), builder(builder) {}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitive(std::string name, Direction dir, DType dtype) {
    return primitives(name, {}, dir, dtype);
}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::primitives(std::string name, std::vector<int64_t>&& shape, Direction dir, DType dtype) {
    auto child = new TensorIso<HeadT>(std::move(shape), dtype);
    child->name = name;
    child->is_leaf = true;
    focus->unsafe_attach(child);

    auto mv = std::make_shared<ModelVariable>();
    mv->name = name;
    mv->accessor = cloneOpticsFromLeafToRoot(child);
    mv->is_input = (dir == Direction::IN) || (dir == Direction::INOUT);
    mv->is_output = (dir == Direction::OUT) || (dir == Direction::INOUT);
    current_callsite_var->model_vars.push_back(mv);
    return *this;
}

template <typename HeadT, typename... RestTs>
template <typename T>
Cursor<T, HeadT, RestTs...> Cursor<HeadT, RestTs...>::accessor(std::function<T*(HeadT*)> lambda) {
    auto child = new Lens<HeadT, T>(nullptr, lambda);
    focus->unsafe_attach(child);
    return Cursor<T, HeadT, RestTs...>(child, builder);
}

template <typename HeadT, typename... RestTs>
template <typename T>
Cursor<T, HeadT, RestTs...> Cursor<HeadT, RestTs...>::accessor(std::function<T(HeadT*)> getter, std::function<void(HeadT*,T)> setter) {
    auto child = new ValueLens<HeadT, T>(nullptr, getter, setter);
    focus->unsafe_attach(child);
    return Cursor<T, HeadT, RestTs...>(child, builder);
}

template <typename HeadT, typename... RestTs>
Cursor<HeadT, RestTs...> Cursor<HeadT, RestTs...>::array(size_t size) {
    auto child = new ArrayTraversal<HeadT>(nullptr, size);
    focus->unsafe_attach(child);
    return Cursor<HeadT, RestTs...>(child, builder);
}

template <typename HeadT, typename... RestTs>
Cursor<RestTs...> Cursor<HeadT, RestTs...>::end() {
    return Cursor<RestTs...>(focus->parent, current_callsite_var, builder);
};


} // namespace phasm

#endif //SURROGATE_TOOLKIT_SURROGATE_BUILDER_H
