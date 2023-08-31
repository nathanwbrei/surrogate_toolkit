
#include <catch.hpp>
#include <iostream>
#include "optics.h"

using namespace phasm;
namespace phasm::test::optics_oop_tests {

class Point {
    double m_x = 0.0;
    double m_y = 0.0;
public:
    double getX() const { return m_x; }
    double getY() const { return m_y; }
    void setX(double x) { m_x = x; }
    void setY(double y) { m_y = y; }
};

Point translate(const Point& initial, double dx, double dy) {
    Point result;
    result.setX(initial.getX() + dx);
    result.setY(initial.getY() + dy);
    return result;
}

template <typename ClassT, typename FieldT>
class Lens2 : public Optic<ClassT> {
    Optic<FieldT>* m_optic = nullptr;
    std::function<FieldT(ClassT*)> m_getter;
    std::function<void(ClassT*, FieldT)> m_setter;
public:
    Lens2(Optic<FieldT>* optic, 
         std::function<FieldT(ClassT*)> getter,
         std::function<void(ClassT*,FieldT)> setter)
      : m_optic(optic), m_getter(getter), m_setter(setter) {
        OpticBase::consumes = demangle<ClassT>();
        OpticBase::produces = demangle<FieldT>();
    };
    Lens2(const Lens2& other) = default;

    std::vector<int64_t> shape() override { return m_optic->shape(); }
    tensor to(ClassT* source) override {
        FieldT val = m_getter(source); 
        return m_optic->to(&val);
    }
    void from(tensor source, ClassT* dest) override {
        FieldT val;
        // Fill data from tensor into temporary FieldT that lives on the stack
        m_optic->from(source, &val);
        // Then call setter on dest object
        m_setter(dest, val);
    }
    void attach(Optic<FieldT>* optic) {
        OpticBase::unsafe_attach(optic);
        m_optic = optic;
    }
    void unsafe_use(OpticBase* optic) override {
        auto downcasted = dynamic_cast<Optic<FieldT>*>(optic);
        if (downcasted == nullptr) {
            throw std::runtime_error("Incompatible optic!");
        }
        m_optic = downcasted;
    }
    Lens2* clone() override {
        return new Lens2<ClassT, FieldT>(*this);
    }
};


TEST_CASE("Surrogate API with OOP") {
    Point initial;
    initial.setX(22.0);
    initial.setY(49.0);

    auto primitive_lens = TensorIso<double>();
    auto getY = [](Point* p) { return p->getY(); };
    auto setY = [](Point* p, double y) { p->setY(y); };
    auto val_lens = Lens2<Point, double>(&primitive_lens, getY, setY);

    // Attempt to read from the Point
    auto t = val_lens.to(&initial);
    double *tp = t.get_data<double>();
    REQUIRE(*tp == (double) 49.0);

    // Attempt to write to the Point
    *tp = 7.6;
    val_lens.from(t, &initial);

    // Attempt to read from the Point into a fresh tensor
    auto tt = val_lens.to(&initial);
    double *ttp = tt.get_data<double>();
    REQUIRE(*ttp == (double) 7.6);
}


} // namespace
