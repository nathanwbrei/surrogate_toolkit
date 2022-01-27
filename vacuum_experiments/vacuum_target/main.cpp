
#include <iostream>


double z = 100.0;

double target(int x, double y) {
    return x + y + z;
}

int main() {
    std::cout << target(22, 1.0) << std::endl;
}

