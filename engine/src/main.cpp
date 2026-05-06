#include "value.h"
#include <iostream>

Value::Value(double data) : data_(data) {}

double Value::data() const {
    return data_;
}

int main() {
    Value v(10);
    std::cout << v.data() << std::endl;
    return 0;
};