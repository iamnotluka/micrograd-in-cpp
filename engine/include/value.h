#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Value {
    public:
        Value(double data) : data_(data) {}

        static std::shared_ptr<Value> create(double data, char label) {
            auto new_value = std::make_shared<Value>(data);
            new_value->label_ = label;
            new_value->grad_ = 0;
            return new_value;
        };

        static std::shared_ptr<Value> create(double data, const std::vector<std::shared_ptr<Value>>& prev, char op) {
            std::shared_ptr<Value> new_value = std::make_shared<Value>(data);
            new_value->prev_ = prev;
            new_value->op_ = op;
            new_value->grad_ = 0;
            return new_value;
        };

        double data() const {
           return data_;
        }

        char op() const {
            return op_;
        }

        char label() const {
            return label_;
        }

        void set_label(char label) {
            label_ = label;
        }

        std::vector<std::shared_ptr<Value>> prev() const {
            return prev_;
        }

        double grad() const {
            return grad_;
        }

        void set_grad(double grad) {
            grad_ = grad;
        }

    private:
        double data_;
        std::vector<std::shared_ptr<Value>> prev_;
        char op_;
        char label_;
        double grad_;
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() + b->data(), {a, b}, '+');
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() * b->data(), {a, b}, '*');
}

inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& value) {
    os << "Value(" << value->data() << ", op: " << value->op() << ", ";
    for (int i = 0; i < value->prev().size(); i++) {
        os << "Value(" << value->prev()[i]->data() << ")";
        if (i < value->prev().size() - 1) os << ", ";
    }
    os << "]";
    return os;
}
