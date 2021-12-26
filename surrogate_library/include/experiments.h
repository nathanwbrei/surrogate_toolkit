
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_EXPERIMENTS_H
#define SURROGATE_TOOLKIT_EXPERIMENTS_H

/// These are small scale experiments to build up familiarity with C++20 template metaprogramming.
/// The problem with template metaprogramming is that the pattern language gets lost in the very nonobvious
/// implementation details. So we build up the patterns in isolation first.

namespace experiments {

template <typename ReturnT, typename ...ArgsT>
class WrappedFunction {
    std::function<ReturnT(ArgsT...)> m_original;
public:
    WrappedFunction(std::function<ReturnT(ArgsT...)> f) : m_original(f) {};

    ReturnT operator()(ArgsT&&... args) {
        std::cout << "Wrapping original function" << std::endl;
        return m_original(std::forward<ArgsT>(args)...);
    }
};



template <typename ReturnT, typename ...ArgsT>
class MemoizedFunction {
    std::function<ReturnT(ArgsT...)> m_original;
    std::map<std::tuple<ArgsT...>, ReturnT> m_lookup_table;

public:
    bool was_last_call_memoized = false;

    MemoizedFunction(std::function<ReturnT(ArgsT...)> f) : m_original(f) {};

    ReturnT operator()(ArgsT... args) {
        std::tuple<ArgsT...> inputs(args...);
        auto pair = m_lookup_table.find(inputs);
        if (pair == m_lookup_table.end()) {
            was_last_call_memoized = false;
            std::cout << "Memoizing original function" << std::endl;
            ReturnT output = m_original(args...);
            m_lookup_table.insert({inputs, output});
            return output;
        }
        std::cout << "Original function already memoized" << std::endl;
        was_last_call_memoized = true;
        return pair->second;
    }
};


template <typename RetT, typename ...ArgsT>
class CurriedFunction {
    std::function<RetT(ArgsT...)> m_original;
    std::tuple<ArgsT...> m_inputs;

public:
    CurriedFunction(std::function<RetT(ArgsT...)> f, ArgsT... args) : m_original(f) {
        m_inputs = std::tuple<ArgsT...>(args...);
    };

    RetT operator()() {
        return std::apply(m_original, m_inputs);
    }
};


}

#endif //SURROGATE_TOOLKIT_EXPERIMENTS_H
