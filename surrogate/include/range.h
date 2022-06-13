
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_RANGE_H
#define SURROGATE_TOOLKIT_RANGE_H

#include <set>
#include <map>
#include <vector>
#include <limits>
#include <ostream>

namespace phasm {

enum class RangeType {
    Interval, FiniteSet
};

template<typename T>
struct Range {
    RangeType rangeType;
    std::set<T> items;
    T lower_bound_inclusive = std::numeric_limits<T>::lowest();
    T upper_bound_inclusive = std::numeric_limits<T>::max();

    // For range-finding
    size_t remaining_captures = 10000;
    std::map<T, size_t> distribution;
    T bucket_count = 50;

    Range() : rangeType(RangeType::Interval) {};

    Range(std::set<T> items) : rangeType(RangeType::FiniteSet), items(std::move(items)) {}

    Range(T lower, T upper) : rangeType(RangeType::Interval), lower_bound_inclusive(lower),
                              upper_bound_inclusive(upper) {}

    bool contains(T t) {
        if (rangeType == RangeType::FiniteSet) {
            return (items.find(t) != items.end());
        } else {
            return (t >= lower_bound_inclusive) && (t <= upper_bound_inclusive);
        }
    }

    void capture(T t) {
        if (t < lower_bound_inclusive) lower_bound_inclusive = t;
        if (t > upper_bound_inclusive) upper_bound_inclusive = t;
        if (rangeType == RangeType::FiniteSet) {
            items.insert(t);
        }
        if (remaining_captures > 0) {
            remaining_captures -= 1;
            distribution[t] += 1;
        }
    }

    std::vector<size_t> make_histogram() {
        T bucket_size = (upper_bound_inclusive - lower_bound_inclusive) / bucket_count;
        std::vector<size_t> hist(bucket_count, 0);
        for (auto pair: distribution) {
            size_t bucket = (pair.first - lower_bound_inclusive) / bucket_size;
            hist[bucket] += pair.second;
        }
        return hist;
    }

    void report(std::ostream &os) {
        os << "Min = " << lower_bound_inclusive << std::endl;
        os << "Max = " << upper_bound_inclusive << std::endl;
        os << "Distribution = " << std::endl;
        if (distribution.size() < bucket_count) {
            for (auto &pair: distribution) {
                os << pair.first << ": " << pair.second << std::endl;
            }
            os << std::endl;
        } else {
            auto hist = make_histogram();
            T bucket_size = (upper_bound_inclusive - lower_bound_inclusive) / bucket_count;
            T interval_start = lower_bound_inclusive;
            T interval_end = interval_start + bucket_size;
            for (int bucket = 0; bucket < bucket_count; ++bucket) {
                os << interval_start << "..." << interval_end << ": " << hist[bucket] << std::endl;
                interval_start = interval_end;
                interval_end += bucket_size;
            }
        }
    }
};

} // namespace phasm

#endif //SURROGATE_TOOLKIT_RANGE_H
