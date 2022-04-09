
#include <iostream>

/// The purpose of vacuum_target is to test the vacuum tool on a variety of functions, escalating in complexity.

/// 1: Test our ability to recognize pure functions of primitives
double target1(int x, double y) {
    return x + y;
}

void test_target1() {
    int x1 = 22;
    double y1 = 0.0;
    std::cout << "test_target1: Input x1 = " << x1 << std::endl;
    std::cout << "test_target1: Input y1 = " << y1 << std::endl;
    double retval = target1(x1, y1);
    std::cout << "test_target1: Retval = " << retval << std::endl;
}

/// 2: Test our ability to recognize pure functions of primitives and also primitive globals.
double z2 = 100.0;
double target2(int x, double y) {
    return x + y + z2;
}

void test_target2() {
    int x2 = 22;
    double y2 = 0.0;
    std::cout << "test_target2: Input x2 = " << x2 << std::endl;
    std::cout << "test_target2: Input y2 = " << y2 << std::endl;
    std::cout << "test_target2: Expect a READ from z2 at " << &z2 << std::endl;
    double retval = target2(x2, y2);
    std::cout << "test_target2: Retval = " << retval << std::endl;
}

/// 3: Test our ability to recognize pure functions which ignore one or more args.
double target3(int x, __attribute__((unused)) double y) {
    return x * 2;
}

void test_target3() {
    int x3 = 22;
    double y3 = 0.0;
    std::cout << "test_target3: Input x3 = " << x3 << std::endl;
    std::cout << "test_target3: Input y3 = " << y3 << std::endl;
    std::cout << "test_target3: Expect a READ from z2 at " << &z2 << std::endl;
    double retval = target3(x3, y3);
    std::cout << "test_target3: Retval = " << retval << std::endl;
}

/// 4: Test our ability to recognize when a global is read via a nested function call
double z4 = 97.2;
double target4_helper(int y) {
    return y + z4;
}
double target4(int x) {
    return target4_helper(x) - 97;
}

void test_target4() {
    std::cout << "test_target4: Expect a READ from z4 at " << &z4 << std::endl;
    int x4 = 22;
    std::cout << "test_target4: Input x4 = " << x4 << std::endl;
    double retval = target4(x4);
    std::cout << "test_target4: Retval = " << retval << std::endl;
}

/// 5: Test our ability to recognize when a global is written to
int w5 = 22;
int z5 = 0;
int target5(int x, int y) {
    void* dest;
    asm("movq %%rbp,%0" : "=r"(dest));
    std::cout << "target5: $rbp = " << dest << ", &x = " << &x << ", &y = " << &y << std::endl;
    z5 = 1;
    return w5 + x * y;
}

void test_target5() {
    void* dest;
    asm("movq %%rbp,%0" : "=r"(dest));
    std::cout << "test_target5: $rbp = " << dest << std::endl;
    std::cout << "test_target5: Expect a WRITE to z5 at " << &z5 << std::endl;
    std::cout << "test_target5: Expect a READ from w5 at " << &w5 << std::endl;
    int x5 = 3;
    int y5 = x5*2;
    std::cout << "test_target5: Input x5 = " << x5 << std::endl;
    std::cout << "test_target5: Input y5 = " << y5 << std::endl;
    int retval = target5(x5,y5);
    std::cout << "test_target5: Output z5 = " << z5 << std::endl;
    std::cout << "test_target5: Retval = " << retval << std::endl;
}

int main(int argc, char** argv) {
    test_target1();
    test_target2();
    test_target3();
    test_target4();
    test_target5();
}

