
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


TEST_CASE("Surrogate API with OOP") {
    Point initial;
    initial.setX(22.0);
    initial.setY(49.0);

    auto primitive_lens = TensorIso<double>();
    auto val_lens = ValueLens<Point, double>(&primitive_lens, &Point::getY, &Point::setY);

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
