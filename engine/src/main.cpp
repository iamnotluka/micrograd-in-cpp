#include "value.h"
#include <iostream>

Value::Value(double data) : data_(data) {}

// a*b + c = 10*5 + 4 = 54
int main() {
    Value a(10);
    Value b(5);
    Value c(4);

    std::cout << (a*b + c).data() << std::endl;
    return 0;
};