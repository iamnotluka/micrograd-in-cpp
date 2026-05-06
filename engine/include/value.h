#pragma once

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