#pragma once
#include <iostream>

class Value {
    public:
        Value(double data);

        double data() const {
           return data_;
        }

        Value operator+(const Value& other) const {
            return Value(data_ + other.data_);
        }

        Value operator*(const Value& other) const {
            return Value(data_ * other.data_);
        }

    private:
        double data_;
};

std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(" << value.data() << ")";
    return os;
}
