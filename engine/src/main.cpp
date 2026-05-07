#include "value.h"
#include "graph.h"
#include <iostream>
#include <fstream>

int main() {
    auto x1 = Value::create(2.0, "x1");
    auto x2 = Value::create(0.0, "x2");

    auto w1 = Value::create(-3, "w1");
    auto w2 = Value::create(1, "w2");

    auto b = Value::create(6.8813735870195432, "b");

    auto x1w1 = x1*w1; x1w1->set_label("x1*w1");
    auto x2w2 = x2*w2; x2w2->set_label("x2*w2");
    auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2->set_label("x1*w1 + x2*w2");

    auto n = x1w1x2w2 + b; n->set_label("n");

    auto o = n->tanh(n); o->set_label("o");
    o->set_grad(1);
    o->backward();
    n->backward();
    x1w1x2w2->backward();
    x1w1->backward();
    x2w2->backward();

    std::ofstream file("graph.dot");
    file << create_dot_graph_string(o);
    file.close();

    return 0;
}