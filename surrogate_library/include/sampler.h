
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_SAMPLER_H
#define SURROGATE_TOOLKIT_SAMPLER_H

struct InputBindingVisitorChild : public InputBindingVisitor {

    template<typename T> inline void visitReal(InputBinding<T>& t) {
        std::cout << "Visiting some kind of real number" << std::endl;
    }

    template<typename T> inline void visitInt(InputBinding<T>& t) {
        std::cout << "Visiting some kind of int" << std::endl;
    }

    void visit(InputBinding<double>& t) override { visitReal<double>(t); }
    void visit(InputBinding<float>& t) override { visitReal<float>(t); }
    void visit(InputBinding<int64_t>& t) override { visitInt<int64_t>(t); }
    void visit(InputBinding<int32_t>& t) override { visitInt<int32_t>(t); }
    void visit(InputBinding<int16_t>& t) override { visitInt<int16_t>(t); }
    void visit(InputBinding<int8_t>& t) override { visitInt<int8_t>(t); }
    void visit(InputBinding<uint64_t>& t) override { visitInt<uint64_t>(t); }
    void visit(InputBinding<uint32_t>& t) override { visitInt<uint32_t>(t); }
    void visit(InputBinding<uint16_t>& t) override { visitInt<uint16_t>(t); }
    void visit(InputBinding<uint8_t>& t) override { visitInt<uint8_t>(t); }
};


class Sampler {

    InputBindingVisitor m_visitor;
    std::shared_ptr<Surrogate> m_surrogate;

    Sampler(std::shared_ptr<Surrogate> s) : m_surrogate(s) {
    }

    void set_each_input_to_min() {
        for (auto input : s.)
    }

};


class GridSampler {

};


#endif //SURROGATE_TOOLKIT_SAMPLER_H
