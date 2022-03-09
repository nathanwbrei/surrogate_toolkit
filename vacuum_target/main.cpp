
#include <iostream>


/// The purpose of vacuum_target is to test the vacuum tool on a variety of functions, escalating in complexity.


/// 1: Test our ability to recognize pure functions of primitives
double target1(int x, double y) {
    return x + y;
}

/// 2: Test our ability to recognize pure functions of primitives and also primitive globals.
double z2 = 100.0;
double target2(int x, double y) {
    return x + y + z2;
}

/// 3: Test our ability to recognize pure functions which ignore one or more args.
double target3(int x, __attribute__((unused)) double y) {
    return x * 2;
}

/// 4: Test our ability to recognize when a global is read via a nested function call
double z4 = 97.2;
double target4_helper(int y) {
    return y + z4;
}
double target4(int x) {
    return target4_helper(x) - 97;
}

/// 5: Test our ability to recognize when a global is written to
int w5 = 22;
int z5 = 0;
int target5(int x, int y) {
    z5 = 1;
    return w5 + x * y;
}


int main() {
    std::cout << "target1(22, 1.0) -> " << target1(22, 1.0) << std::endl;
    std::cout << "target2(22, 1.0) -> " << target2(22, 1.0) << std::endl;
    std::cout << "target3(22, 1.0) -> " << target3(22, 1.0) << std::endl;
    std::cout << "target4(22) -> " << target4(22) << std::endl;

    // The print statements are here to prevent the optimizer from eliminating all of this
    std::cout << "&w5=" << &w5 << std::endl;
    std::cout << "&z5=" << &z5 << std::endl;
    int x5 = 3;
    int y5 = x5*2;
    target5(x5,y5);
    std::cout << "w=" << w5 << std::endl;
    std::cout << "z=" << z5 << std::endl;
}

