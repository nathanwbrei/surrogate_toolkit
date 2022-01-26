// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include <iostream>

template <typename T>
struct InputBinding;

struct InputBindingVisitor {

    virtual void visit(InputBinding<float>&) {};
    virtual void visit(InputBinding<double>&) {};
    virtual void visit(InputBinding<bool>&) {};
    virtual void visit(InputBinding<int8_t>&) {};
    virtual void visit(InputBinding<int16_t>&) {};
    virtual void visit(InputBinding<int32_t>&) {};
    virtual void visit(InputBinding<int64_t>&) {};
    virtual void visit(InputBinding<uint8_t>&) {};
    virtual void visit(InputBinding<uint16_t>&) {};
    virtual void visit(InputBinding<uint32_t>&) {};
    virtual void visit(InputBinding<uint64_t>&) {};
};

template <typename T>
struct InputBinding {
    T t;
    void accept(InputBindingVisitor& v) {
        v.visit(*this);
    }
};

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


TEST_CASE("VisitorTests") {
    InputBinding<int> b;
    InputBinding<double> d;
    InputBindingVisitorChild vc;
    // auto lambda = []<typename T>(InputBinding<T>& ib) {return ib.t;};
    b.accept(vc);
    d.accept(vc);
}

