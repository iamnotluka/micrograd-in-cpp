#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <set>
#include <cmath>

class Value {
    public:
        Value(double data)
            : data_(data), grad_(0), local_backward_([](){}) {}

        Value(double data, std::string label)
            : data_(data), label_(std::move(label)), grad_(0), local_backward_([](){}) {}

        Value(double data, std::vector<std::shared_ptr<Value>> prev, std::string op)
            : data_(data), prev_(std::move(prev)), op_(std::move(op)), grad_(0), local_backward_([](){}) {}

        static std::shared_ptr<Value> create(double data) {
            return std::make_shared<Value>(data);
        }

        static std::shared_ptr<Value> create(double data, std::string label) {
            return std::make_shared<Value>(data, std::move(label));
        }

        static std::shared_ptr<Value> create(double data, std::vector<std::shared_ptr<Value>> prev, std::string op) {
            return std::make_shared<Value>(data, std::move(prev), std::move(op));
        }

        double data() const { return data_; }

        void set_data(double data ) { data_ = data; }

        std::string op() const { return op_; }

        std::string label() const { return label_; }

        void set_label(std::string label) { label_ = std::move(label); }

        std::vector<std::shared_ptr<Value>> prev() const { return prev_; }

        double grad() const { return grad_; }

        void set_grad(double grad) { grad_ = grad; }

        void set_local_backward(std::function<void()> local_backward) { local_backward_ = std::move(local_backward); }

        void local_backward() const { local_backward_(); };

        void backward() {
            std::vector<std::shared_ptr<Value>> topo;
            std::set<Value*> visited;

            std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& node) {
                if (visited.find(node.get()) != visited.end()) return;

                visited.insert(node.get());

                for (const auto& child : node->prev_) {
                    build_topo(child);
                }

                topo.push_back(node);
            };

            build_topo(std::shared_ptr<Value>(this, [](Value*){}));
            grad_ = 1.0;

            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                (*it) -> local_backward_();
            };
        }

        inline std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a) {       
            double x = a->data();
            double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);

            auto new_value = Value::create(t, {a}, "tanh");
            Value* out = new_value.get();
            new_value->set_local_backward([a, t, out]() {
                a->set_grad(a->grad() + out->grad() * (1-t*t));
            });

            return new_value;
        }

        inline std::shared_ptr<Value> exp(const std::shared_ptr<Value>& a) {
            double x = a->data();
            double e = std::exp(x);

            auto new_value = Value::create(e, {a}, "exp");
            Value* out = new_value.get();
            new_value->set_local_backward([a, e, out]() {
                a->set_grad(a->grad() + out->grad() * e);
            });

            return new_value;
        }

        inline std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, double n) {
            auto new_value = Value::create(std::pow(a->data(), n), {a}, "pow");
            Value* out = new_value.get();
            new_value->set_local_backward([a, n, out]() {
                a->set_grad(a->grad() + out->grad() * n * std::pow(a->data(), n - 1));
            });
            return new_value;
        }

    private:
        double data_;
        std::vector<std::shared_ptr<Value>> prev_;
        std::string op_;
        std::string label_;
        double grad_;
        std::function<void()> local_backward_;
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto new_value = Value::create(a->data() + b->data(), {a, b}, "+");
    Value* out = new_value.get();
    new_value->set_local_backward([a, b, out]() {
        a->set_grad(a->grad() + out->grad());
        b->set_grad(b->grad() + out->grad());
    });
    return new_value;
}

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const double b) {      
    return a + Value::create(b);                                                            
}                                                                                               
                                                                                            
inline std::shared_ptr<Value> operator+(const double a, const std::shared_ptr<Value>& b) {      
    return Value::create(a) + b;                                                          
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto new_value = Value::create(a->data() - b->data(), {a, b}, "-");
    Value* out = new_value.get();
    new_value->set_local_backward([a, b, out]() {
        a->set_grad(a->grad() + out->grad());
        b->set_grad(b->grad() - out->grad());
    });
    return new_value;
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const double b) {
    return a - Value::create(b);
}

inline std::shared_ptr<Value> operator-(const double a, const std::shared_ptr<Value>& b) {
    return Value::create(a) - b;
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto new_value = Value::create(a->data() / b->data(), {a, b}, "/");
    Value* out = new_value.get();
    new_value->set_local_backward([a, b, out]() {
        b->set_grad(b->grad() + out->grad() * (-a->data() / (b->data() * b->data())));
        a->set_grad(a->grad() + out->grad() * (1 / b->data()));
    });
    return new_value;
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const double b) {
    return a / Value::create(b);
}

inline std::shared_ptr<Value> operator/(const double a, const std::shared_ptr<Value>& b) {
    return Value::create(a) / b;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto new_value = Value::create(a->data() * b->data(), {a, b}, "*");
    Value* out = new_value.get();
    new_value->set_local_backward([a, b, out]() {
        a->set_grad(a->grad() + out->grad() * b->data());
        b->set_grad(b->grad() + out->grad() * a->data());
    });
    return new_value;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const double b) {
    return a * Value::create(b);
}

inline std::shared_ptr<Value> operator*(const double a, const std::shared_ptr<Value>& b) {
    return Value::create(a) * b;
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
