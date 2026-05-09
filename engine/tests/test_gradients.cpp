#include "value.h"
#include "loss.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <functional>
#include <memory>

const double H = 1e-6;
const double TOLERANCE = 1e-4;

double numerical_gradient(std::function<double()> f, std::shared_ptr<Value> wrt) {
    double original = wrt->data();

    wrt->set_data(original + H);
    double f_plus = f();

    wrt->set_data(original - H);
    double f_minus = f();

    wrt->set_data(original);
    return (f_plus - f_minus) / (2 * H);
}

void check_gradient(const std::string& test_name, double analytical, double numerical) {
    double diff = std::abs(analytical - numerical);
    std::cout << std::fixed << std::setprecision(6);
    std::string status = diff > TOLERANCE ? "FAIL" : "PASS";
    std::cout << status << "  " << test_name << std::endl;
    std::cout << "        expected: " << numerical << std::endl;
    std::cout << "             got: " << analytical << std::endl;
    if (diff > TOLERANCE) {
        std::cout << "            diff: " << diff << std::endl;
        assert(false);
    }
    std::cout << std::endl;
}

void test_addition() {
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);

    auto c = a + b;
    c->backward();

    check_gradient("add d/da", a->grad(), numerical_gradient([&]() { return (a + b)->data(); }, a));
    check_gradient("add d/db", b->grad(), numerical_gradient([&]() { return (a + b)->data(); }, b));
}

void test_multiplication() {
    auto a = Value::create(2.0);
    auto b = Value::create(-3.0);

    auto c = a * b;
    c->backward();

    check_gradient("mul d/da", a->grad(), numerical_gradient([&]() { return (a * b)->data(); }, a));
    check_gradient("mul d/db", b->grad(), numerical_gradient([&]() { return (a * b)->data(); }, b));
}

void test_subtraction() {
    auto a = Value::create(5.0);
    auto b = Value::create(3.0);

    auto c = a - b;
    c->backward();

    check_gradient("sub d/da", a->grad(), numerical_gradient([&]() { return (a - b)->data(); }, a));
    check_gradient("sub d/db", b->grad(), numerical_gradient([&]() { return (a - b)->data(); }, b));
}

void test_division() {
    auto a = Value::create(6.0);
    auto b = Value::create(4.0);

    auto c = a / b;
    c->backward();

    check_gradient("div d/da", a->grad(), numerical_gradient([&]() { return (a / b)->data(); }, a));
    check_gradient("div d/db", b->grad(), numerical_gradient([&]() { return (a / b)->data(); }, b));
}

void test_pow() {
    auto a = Value::create(3.0);

    auto c = a->pow(a, 2.0);
    c->backward();

    check_gradient("pow d/da", a->grad(), numerical_gradient([&]() { return a->pow(a, 2.0)->data(); }, a));
}

void test_tanh() {
    auto a = Value::create(0.5);

    auto c = a->tanh(a);
    c->backward();

    check_gradient("tanh d/da", a->grad(), numerical_gradient([&]() { return a->tanh(a)->data(); }, a));
}

void test_exp() {
    auto a = Value::create(1.0);

    auto c = a->exp(a);
    c->backward();

    check_gradient("exp d/da", a->grad(), numerical_gradient([&]() { return a->exp(a)->data(); }, a));
}

void test_softmax_cross_entropy() {
    auto a = Value::create(1.2);
    auto b = Value::create(-0.7);
    auto c = Value::create(0.3);
    std::vector<std::shared_ptr<Value>> logits = {a, b, c};

    auto loss = softmax_cross_entropy_loss(logits, 2);
    loss->backward();

    check_gradient(
        "softmax cross entropy d/da",
        a->grad(),
        numerical_gradient([&]() { return softmax_cross_entropy_loss(logits, 2)->data(); }, a)
    );
    check_gradient(
        "softmax cross entropy d/db",
        b->grad(),
        numerical_gradient([&]() { return softmax_cross_entropy_loss(logits, 2)->data(); }, b)
    );
    check_gradient(
        "softmax cross entropy d/dc",
        c->grad(),
        numerical_gradient([&]() { return softmax_cross_entropy_loss(logits, 2)->data(); }, c)
    );
}

void test_composite() {
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);
    auto expr = [&]() { return ((a * b) + a)->tanh((a * b) + a); };

    auto c = expr();
    c->backward();

    check_gradient("composite d/da", a->grad(), numerical_gradient([&]() { return expr()->data(); }, a));

    a->set_grad(0.0);
    b->set_grad(0.0);
    c = expr();
    c->backward();

    check_gradient("composite d/db", b->grad(), numerical_gradient([&]() { return expr()->data(); }, b));
}

void test_values_do_not_self_reference() {
    std::weak_ptr<Value> weak_result;

    {
        auto a = Value::create(2.0);
        auto b = Value::create(3.0);
        auto c = (a * b) + a;
        weak_result = c;
        c->backward();
    }

    bool was_destroyed = weak_result.expired();
    std::cout << (was_destroyed ? "PASS" : "FAIL") << "  values do not self-reference" << std::endl;
    if (!was_destroyed) {
        assert(false);
    }
    std::cout << std::endl;
}

int main() {
    test_addition();
    test_multiplication();
    test_subtraction();
    test_division();
    test_pow();
    test_tanh();
    test_exp();
    test_softmax_cross_entropy();
    test_composite();
    test_values_do_not_self_reference();

    std::cout << "\nAll gradient checks passed." << std::endl;
    return 0;
}
