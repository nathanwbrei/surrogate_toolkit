
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_PARAMETER_RANGE_H
#define SURROGATE_TOOLKIT_PARAMETER_RANGE_H

namespace ranges {

struct NoRange {};

template <typename T> struct FiniteSet {
    std::set<T> items;
};

template <typename T> struct Histogram {
    T lower_bound;
    T upper_bound;
    T bucket_size;
    std::vector<uint32_t> values;
    std::set<T> outliers;

    Histogram(T lower_bound, T upper_bound, T bucket_size)
    : lower_bound(lower_bound), upper_bound(upper_bound), bucket_size(bucket_size) {

        size_t bucket_count = (upper_bound - lower_bound) / bucket_size;
        values.resize(bucket_count, 0);
    }
};

template <typename T> struct Interval {
    T lower_bound_inclusive = std::numeric_limits<T>::max();
    T upper_bound_inclusive = std::numeric_limits<T>::min();
};

/// When exploring
template <typename T> struct Capture : public Histogram<T>, public Interval<T>, public FiniteSet<T> {
};

template <typename T>
using ParameterRange = std::variant<NoRange, FiniteSet<T>, Histogram<T>, Interval<T>, Capture<T>>;


template <typename T>
void capture(NoRange&, const T& t) {}

template <typename T>
void capture(FiniteSet<T>& fs, const T& t) {
    fs.items.insert(t);
}

template <typename T>
void capture(Interval<T>& i, const T& t) {
    if (t < i.lower_bound_inclusive) {
        i.lower_bound_inclusive = t;
    }
    else if (t > i.upper_bound_inclusive) {
        i.upper_bound_inclusive = t;
    }
}

template <typename T>
void capture(Histogram<T> h, const T& t) {
    if (t < h.lower_bound || t > h.upper_bound) {
        h.outliers.insert(t);
    }
    else {
        size_t bucket = (t - h.lower_bound) / h.bucket_size;

    }
}


} // ranges


#endif //SURROGATE_TOOLKIT_PARAMETER_RANGE_H
