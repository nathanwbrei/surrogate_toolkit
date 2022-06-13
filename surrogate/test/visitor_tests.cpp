// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include <iostream>

namespace phasm::tests::visitor_tests {


template <typename T>
struct CallSiteVariable;

struct CallSiteVariableVisitor {

    virtual void visit(CallSiteVariable<float>&) {};
    virtual void visit(CallSiteVariable<double>&) {};
    virtual void visit(CallSiteVariable<bool>&) {};
    virtual void visit(CallSiteVariable<int8_t>&) {};
    virtual void visit(CallSiteVariable<int16_t>&) {};
    virtual void visit(CallSiteVariable<int32_t>&) {};
    virtual void visit(CallSiteVariable<int64_t>&) {};
    virtual void visit(CallSiteVariable<uint8_t>&) {};
    virtual void visit(CallSiteVariable<uint16_t>&) {};
    virtual void visit(CallSiteVariable<uint32_t>&) {};
    virtual void visit(CallSiteVariable<uint64_t>&) {};
};

template <typename T>
struct CallSiteVariable {
    T t;
    void accept(CallSiteVariableVisitor& v) {
        v.visit(*this);
    }
};

struct MyVisitor : public CallSiteVariableVisitor {

    template<typename T> inline void visitReal(CallSiteVariable<T>& t) {
        std::cout << "Visiting some kind of real number" << std::endl;
    }

    template<typename T> inline void visitInt(CallSiteVariable<T>& t) {
        std::cout << "Visiting some kind of int" << std::endl;
    }

    void visit(CallSiteVariable<double>& t) override { visitReal<double>(t); }
    void visit(CallSiteVariable<float>& t) override { visitReal<float>(t); }
    void visit(CallSiteVariable<int64_t>& t) override { visitInt<int64_t>(t); }
    void visit(CallSiteVariable<int32_t>& t) override { visitInt<int32_t>(t); }
    void visit(CallSiteVariable<int16_t>& t) override { visitInt<int16_t>(t); }
    void visit(CallSiteVariable<int8_t>& t) override { visitInt<int8_t>(t); }
    void visit(CallSiteVariable<uint64_t>& t) override { visitInt<uint64_t>(t); }
    void visit(CallSiteVariable<uint32_t>& t) override { visitInt<uint32_t>(t); }
    void visit(CallSiteVariable<uint16_t>& t) override { visitInt<uint16_t>(t); }
    void visit(CallSiteVariable<uint8_t>& t) override { visitInt<uint8_t>(t); }
};


TEST_CASE("VisitorTests") {
    CallSiteVariable<int> b;
    CallSiteVariable<double> d;
    MyVisitor vc;
    // auto lambda = []<typename T>(InputBinding<T>& ib) {return ib.t;};
    b.accept(vc);
    d.accept(vc);
}

} // namespace phasm::tests::visitor_tests
