
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SAMPLER_H
#define SURROGATE_TOOLKIT_SAMPLER_H

#include <call_site_variable.h>

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

    GridSampler(CallSiteVariableT<T>& cs, ModelVariableT<T>* var, size_t nsteps = 100) {
        initial_sample = var->range.lower_bound_inclusive;
        final_sample = var->range.upper_bound_inclusive;
        current_sample = initial_sample;
        step_size = (final_sample - current_sample) / nsteps;
        if (step_size < 1) step_size = 1;
        slot = cs.binding_root;
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

    FiniteSetSampler(CallSiteVariableT<T>& binding) {
        slot = binding.binding_root;
        auto& s = binding.model_vars[0]->range.items;
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



struct MyVisitor : public ModelVariableVisitor {

    template<typename T> inline void visitReal(ModelVariableT<T>&) {
        std::cout << "Visiting some kind of real number" << std::endl;
    }

    template<typename T> inline void visitInt(ModelVariableT<T>&) {
        std::cout << "Visiting some kind of int" << std::endl;
    }

    void visit(ModelVariableT<double>& t) override { visitReal<double>(t); }
    void visit(ModelVariableT<float>& t) override { visitReal<float>(t); }
    void visit(ModelVariableT<int64_t>& t) override { visitInt<int64_t>(t); }
    void visit(ModelVariableT<int32_t>& t) override { visitInt<int32_t>(t); }
    void visit(ModelVariableT<int16_t>& t) override { visitInt<int16_t>(t); }
    void visit(ModelVariableT<int8_t>& t) override { visitInt<int8_t>(t); }
    void visit(ModelVariableT<uint64_t>& t) override { visitInt<uint64_t>(t); }
    void visit(ModelVariableT<uint32_t>& t) override { visitInt<uint32_t>(t); }
    void visit(ModelVariableT<uint16_t>& t) override { visitInt<uint16_t>(t); }
    void visit(ModelVariableT<uint8_t>& t) override { visitInt<uint8_t>(t); }
};



#endif //SURROGATE_TOOLKIT_SAMPLER_H
