
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "parameter.h"

struct InputBinding {
    virtual void capture() = 0;
    virtual void deploy_sample() = 0;
    virtual ~InputBinding() = default;
};

template <typename T>
struct InputBindingT : public InputBinding {

    std::shared_ptr<InputT<T>> parameter;
    T* slot; // TODO: Replace with accessor or Lens or something when the time comes
    T sample;

    void capture() override {
        parameter->captures.push_back(*slot);
    }

    void deploy_sample() override {
        *slot = sample;
    }

};


struct OutputBinding {
public:
    virtual void capture() = 0;
    virtual ~OutputBinding() = default;
};


template <typename T>
struct OutputBindingT : public OutputBinding {
    std::shared_ptr<OutputT<T>> parameter;
    T* slot;

    void capture() override {
        parameter->captures.push_back(*slot);
    }

};


#endif //SURROGATE_TOOLKIT_BINDINGS_H
