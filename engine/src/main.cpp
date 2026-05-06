#include "value.h"
#include <iostream>

Value::Value(double data) : data_(data) {}

int main() {
    Value a(10);
    Value b(5);
    std::cout << (a + b).data() << std::endl;
    return 0;
};