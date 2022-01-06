
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_RANGE_H
#define SURROGATE_TOOLKIT_RANGE_H

#include <set>
#include <map>
#include <limits>


template <typename T>
struct Range {
    virtual bool contains(T) = 0;
    virtual ~Range() = default;
};

template <typename T>
struct FiniteSet : public Range<T> {
    std::set<T> items;
    FiniteSet(std::set<T> items) : items(items) {};
    bool contains(T t) override {
        return (items.find(t) != items.end());
    }
};


template <typename T>
struct Interval : public Range<T> {
    T lower_bound_inclusive = std::numeric_limits<T>::lowest();
    T upper_bound_inclusive = std::numeric_limits<T>::max();

    Interval() {};
    Interval(T lower, T upper) : lower_bound_inclusive(lower), upper_bound_inclusive(upper) {}
    bool contains(T t) override {
        return (t = lower_bound_inclusive) && (t <= upper_bound_inclusive);
    }
};


template <typename T>
struct RangeFinder {
    size_t remaining_captures;
    size_t bucket_count = 50;
    T lower_bound_inclusive = std::numeric_limits<T>::lowest();
    T upper_bound_inclusive = std::numeric_limits<T>::max();
    std::map<T, size_t> counts;

    RangeFinder(size_t max_captures = 10000) : remaining_captures(max_captures) {};

    void capture(T t) {
	if (t < lower_bound_inclusive) lower_bound_inclusive = t;
	if (t > upper_bound_inclusive) upper_bound_inclusive = t;
	if (remaining_captures > 0) {
	    remaining_captures -= 1;
	    counts[t] += 1;
	}
    }

    std::vector<size_t> make_histogram() {
	T bucket_size = (upper_bound_inclusive - lower_bound_inclusive) / bucket_count;
	std::vector<size_t> hist(bucket_count, 0);
	for (auto pair : counts) {
	    size_t bucket = (pair.first - lower_bound_inclusive) / bucket_size;
	    hist[bucket] += pair.second;
	}
	return hist;
    }

    void report(std::ostream &os) {
	os << "Min = " << lower_bound_inclusive << std::endl;
	os << "Max = " << upper_bound_inclusive << std::endl;
	os << "Distribution = " << upper_bound_inclusive << std::endl;
	if (counts.size() < bucket_count) {
	    for (auto &pair : counts) {
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


#endif //SURROGATE_TOOLKIT_RANGE_H
