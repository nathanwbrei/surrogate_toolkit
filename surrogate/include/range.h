
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_RANGE_H
#define SURROGATE_TOOLKIT_RANGE_H

#include <unordered_set>
#include <torch/torch.h>
#include <map>
#include <vector>
#include <limits>
#include <ostream>
#include <experimental/optional>

#include "tensor.hpp"

namespace phasm {

enum class RangeType {
    Unknown, Skip, Interval, FiniteSet
};

struct Range {
    RangeType rangeType;

    std::experimental::optional<tensor> lower_bound_inclusive;
    std::experimental::optional<tensor> upper_bound_inclusive;
    std::unordered_set<tensor> items;

    static inline size_t max_samples_in_finite_set = 100;
    size_t samples_count = 0;

    Range() : rangeType(RangeType::Skip) {};

    Range(std::unordered_set<tensor> items) : rangeType(RangeType::FiniteSet), items(std::move(items)) {}

    Range(tensor lower, tensor upper) : rangeType(RangeType::Interval), lower_bound_inclusive(lower),
                              upper_bound_inclusive(upper) {}

    bool contains(const tensor& t) {
        if (rangeType == RangeType::FiniteSet) {
            return (items.find(t) != items.end());
        }
        else if (rangeType == RangeType::Interval){
            return false;
            // TODO: Re-enable
            // return ((t >= lower_bound_inclusive.value()).all() &&
            //         (t <= upper_bound_inclusive.value()).all());
        }
        else {
            return true;
        }
    }

    void capture(tensor t) {
        // TODO: Re-enable
        // if (t < lower_bound_inclusive) lower_bound_inclusive = t;
        // if (t > upper_bound_inclusive) upper_bound_inclusive = t;
        if (rangeType == RangeType::FiniteSet && samples_count++ <= max_samples_in_finite_set) {
            items.insert(t);
        }
    }

    void report(std::ostream &os) {
        os << "Min = " << std::endl;
        // lower_bound_inclusive->get_underlying().print();
        os << "Max = " << std::endl;
        // upper_bound_inclusive->get_underlying().print();
    }
};

} // namespace phasm

#endif //SURROGATE_TOOLKIT_RANGE_H
