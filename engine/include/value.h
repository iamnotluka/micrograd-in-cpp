#pragma once
#include <iostream>
#include <memory>
#include <vector>

class Value {
    public:
        Value(double data) : data_(data) {}

        static std::shared_ptr<Value> create(double data) {
            return std::make_shared<Value>(data);
        };

        static std::shared_ptr<Value> create(double data, const std::vector<std::shared_ptr<Value>>& children) {
            std::shared_ptr<Value> new_value = std::make_shared<Value>(data);
            new_value->children_ = children;
            return new_value;
        };

        double data() const {
           return data_;
        }

    private:
        double data_;
        std::vector<std::shared_ptr<Value>> children_;
};

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& value) {
    os << "Value(" << value->data() << ")";
    return os;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() + b->data(), {a, b});
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() * b->data(), {a, b});
}