
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SAMPLER_H
#define SURROGATE_TOOLKIT_SAMPLER_H

#include <call_site_variable.h>

namespace phasm {

struct Sampler {

    /// Sampler::next() updates its binding(s) with the next sample in the sequence.
    /// Returns true if there are _more_ samples remaining. E.g.:
    /// `while (sampler.next()) surrogate.call_and_capture();`
    virtual bool next() = 0;

};


template<typename T>
struct GridSampler : public Sampler {

    T current_sample;
    T initial_sample;
    T final_sample;
    T step_size;
    T *slot;

    GridSampler(CallSiteVariable &cs, std::shared_ptr<ModelVariable> var, size_t nsteps = 100) {
        initial_sample = var->range.lower_bound_inclusive;
        final_sample = var->range.upper_bound_inclusive;
        current_sample = initial_sample;
        step_size = (final_sample - current_sample) / nsteps;
        if (step_size < 1) step_size = 1;
        slot = cs.binding.get<T>();
    }

    bool next() override {
        *slot = current_sample;
        if (current_sample >= final_sample) {
            current_sample = initial_sample;
            return false;
        } else {
            current_sample += step_size;
            return true;
        }
    }
};


template<typename T>
class CartesianProductSampler : public Sampler {
    std::vector<Sampler> m_samplers;

    CartesianProductSampler(std::vector<Sampler> samplers)
            : m_samplers(std::move(samplers)) {
    }

    bool next() override {
        for (int i = m_samplers.size() - 1; i >= 0; --i) {
            bool result = m_samplers[i].next();
            if (result) return true;
        }
        return false;
    }
};

template<typename T>
struct FiniteSetSampler : public Sampler {
    std::vector<T> samples;
    size_t sample_index = 0;
    T *slot;

    FiniteSetSampler(CallSiteVariable &binding) {
        slot = binding.binding.get<T>();
        auto &s = binding.model_vars[0]->range.items;
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

} // namespace phasm



#endif //SURROGATE_TOOLKIT_SAMPLER_H
