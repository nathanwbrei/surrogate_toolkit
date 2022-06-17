
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SAMPLER_H
#define SURROGATE_TOOLKIT_SAMPLER_H

#include <tensor.hpp>

namespace phasm {

struct Sampler {

    /// Sampler::next() updates its binding(s) with the next sample in the sequence.
    /// Returns true if there are _more_ samples remaining. E.g.:
    /// `while (sampler.next()) surrogate.call_and_capture();`
    virtual bool next() = 0;
};

class GridSampler : public Sampler {

    tensor current_sample;
    tensor initial_sample;
    tensor final_sample;
    tensor sample_delta;
    // int64_t sample_count;
    std::shared_ptr<ModelVariable> model_var;

public:

    GridSampler(std::shared_ptr<ModelVariable> var, int nsteps = 100) {
        model_var = var;
        initial_sample = var->range.lower_bound_inclusive.value();
        final_sample = var->range.upper_bound_inclusive.value();
        current_sample = initial_sample;
        sample_delta = tensor((final_sample.get_underlying() - current_sample.get_underlying()) / nsteps);
    }

    bool next() override {
        model_var->training_inputs.push_back(current_sample);
        if ((current_sample.get_underlying() >= final_sample.get_underlying()).all().item<bool>()) {
            current_sample = initial_sample;
            return false;
        } else {
            current_sample.get_underlying() += sample_delta.get_underlying();
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

struct FiniteSetSampler : public Sampler {
    std::vector<tensor> samples;
    std::shared_ptr<ModelVariable> model_var;
    size_t sample_index = 0;

    FiniteSetSampler(std::shared_ptr<ModelVariable> mv) {
        model_var = mv;
        auto &s = mv->range.items;
        samples.insert(samples.end(), s.begin(), s.end());
    }

    bool next() override {
        if (sample_index >= samples.size()) {
            sample_index = 0;
            return false;
        }
        return true;
    }
};

} // namespace phasm



#endif //SURROGATE_TOOLKIT_SAMPLER_H
