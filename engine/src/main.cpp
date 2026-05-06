#include "value.h"
#include <iostream>

// a*b + c = 10*5 + 4 = 54
int main() {
    auto a = Value::create(10);
    auto b = Value::create(5);
    auto c = Value::create(4);

    std::cout << a*b + c << std::endl;
    return 0;
};