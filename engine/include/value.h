#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Value {
    public:
        Value(double data, std::string label)
            : data_(data), label_(std::move(label)), grad_(0) {}

        Value(double data, std::vector<std::shared_ptr<Value>> prev, std::string op)
            : data_(data), prev_(std::move(prev)), op_(std::move(op)), grad_(0) {}

        static std::shared_ptr<Value> create(double data, std::string label) {
            return std::make_shared<Value>(data, std::move(label));
        }

        static std::shared_ptr<Value> create(double data, std::vector<std::shared_ptr<Value>> prev, std::string op) {
            return std::make_shared<Value>(data, std::move(prev), std::move(op));
        }

        double data() const {
            return data_;
        }

        std::string op() const {
            return op_;
        }

        std::string label() const {
            return label_;
        }

        void set_label(std::string label) {
            label_ = std::move(label);
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

        inline std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a) {       
            double x = a->data();
            double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
            return Value::create(t, {a}, "tanh");
        }

    private:
        double data_;
        std::vector<std::shared_ptr<Value>> prev_;
        std::string op_;
        std::string label_;
        double grad_;
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() + b->data(), {a, b}, "+");
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() - b->data(), {a, b}, "-");
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() / b->data(), {a, b}, "/");
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return Value::create(a->data() * b->data(), {a, b}, "*");
}

inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& value) {
    os << "Value(" << value->data() << ", op: " << value->op() << ", ";
    for (size_t i = 0; i < value->prev().size(); i++) {
        os << "Value(" << value->prev()[i]->data() << ")";
        if (i < value->prev().size() - 1) os << ", ";
    }
    os << "]";
    return os;
}
