#include "value.h"
#include "graph.h"
#include <iostream>
#include <fstream>

int main() {
    auto a = Value::create(-3, 'a');
    auto b = Value::create(2, 'b');
    auto c = Value::create(10, 'c');

    auto e = a*b;
    e->set_label('e');

    auto d = e + c;
    d->set_label('d');

    auto f = Value::create(-2, 'f');
    auto L = d*f; L->set_label('L');

    std::cout << L << std::endl;

    std::ofstream file("graph.dot");
    file << create_dot_graph_string(L);
    file.close();

    return 0;
}