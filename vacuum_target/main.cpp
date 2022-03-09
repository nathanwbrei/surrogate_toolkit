
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

int main() {
    std::cout << "target1(22, 1.0) -> " << target1(22, 1.0) << std::endl;
    std::cout << "target2(22, 1.0) -> " << target2(22, 1.0) << std::endl;
    std::cout << "target3(22, 1.0) -> " << target3(22, 1.0) << std::endl;
    std::cout << "target4(22) -> " << target4(22) << std::endl;
}

