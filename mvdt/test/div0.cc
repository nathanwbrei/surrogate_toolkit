
#include <iostream>


int f(int b) {
    int a, c;
    switch (b) {
        case 1: a = b / 0; break;
        case 4: c = b - 4; a = b/c; break;
    }
    return a;
}

int g(int x) {
    return x / 0;
}

int main(int argc, char* argv[]) {
    std::cout << f(argc) << std::endl;
    std::cout << g(argc) << std::endl;
}