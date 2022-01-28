
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SAMPLER_H
#define SURROGATE_TOOLKIT_SAMPLER_H

#include <binding.h>

struct Sampler {

    /// Sampler::next() updates its binding(s) with the next sample in the sequence.
    /// Returns true if there are _more_ samples remaining. E.g.:
    /// `while (sampler.next()) surrogate.call_and_capture();`
    virtual bool next() = 0;

};



template <typename T>
struct GridSampler : public Sampler {

    T current_sample;
    T initial_sample;
    T final_sample;
    T step_size;
    T* slot;

    GridSampler(InputBindingT<T>& binding, size_t nsteps = 100) {
        auto param = binding.parameter;
        initial_sample = param->range.lower_bound_inclusive;
        final_sample = param->range.upper_bound_inclusive;
        current_sample = initial_sample;
        step_size = (final_sample - current_sample) / nsteps;
        if (step_size < 1) step_size = 1;
        slot = binding.slot;
    }

    bool next() override {
        *slot = current_sample;
        if (current_sample >= final_sample) {
            current_sample = initial_sample;
            return false;
        }
        else {
            current_sample += step_size;
            return true;
        }
    }
};


template <typename T>
class CartesianProductSampler : public Sampler {
    std::vector<Sampler> m_samplers;

    CartesianProductSampler(std::vector<Sampler> samplers)
    : m_samplers(std::move(samplers)) {
    }

    bool next() override {
        for(int i=m_samplers.size()-1; i>=0; --i) {
            bool result = m_samplers[i].next();
            if (result) return true;
        }
        return false;
    }
};

template <typename T>
struct FiniteSetSampler : public Sampler {
    std::vector<T> samples;
    size_t sample_index = 0;
    T* slot;

    FiniteSetSampler(InputBindingT<T>& binding) {
        slot = binding.slot;
        auto& s = binding.parameter->range.items;
        samples.insert(samples.end(), s.begin(), s.end());
    }

    bool next() override {
        *slot = samples[sample_index++];
        if (sample_index >= samples.size()) {
            sample_index = 0;
            return false;
        }
        return true;
    }
};



struct InputBindingVisitorChild : public InputBindingVisitor {

    template<typename T> inline void visitReal(InputBindingT<T>& t) {
        std::cout << "Visiting some kind of real number" << std::endl;
    }

    template<typename T> inline void visitInt(InputBindingT<T>& t) {
        std::cout << "Visiting some kind of int" << std::endl;
    }

    void visit(InputBindingT<double>& t) override { visitReal<double>(t); }
    void visit(InputBindingT<float>& t) override { visitReal<float>(t); }
    void visit(InputBindingT<int64_t>& t) override { visitInt<int64_t>(t); }
    void visit(InputBindingT<int32_t>& t) override { visitInt<int32_t>(t); }
    void visit(InputBindingT<int16_t>& t) override { visitInt<int16_t>(t); }
    void visit(InputBindingT<int8_t>& t) override { visitInt<int8_t>(t); }
    void visit(InputBindingT<uint64_t>& t) override { visitInt<uint64_t>(t); }
    void visit(InputBindingT<uint32_t>& t) override { visitInt<uint32_t>(t); }
    void visit(InputBindingT<uint16_t>& t) override { visitInt<uint16_t>(t); }
    void visit(InputBindingT<uint8_t>& t) override { visitInt<uint8_t>(t); }
};



#endif //SURROGATE_TOOLKIT_SAMPLER_H
