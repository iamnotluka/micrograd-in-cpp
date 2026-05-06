#include "value.h"
#include "graph.h"
#include <iostream>
#include <fstream>

int main() {
    auto a = Value::create(10);
    auto b = Value::create(5);
    auto c = Value::create(4);

    auto result = a*b + c;
    std::cout << result << std::endl;

    std::ofstream file("graph.dot");
    file << create_dot_graph_string(result);
    file.close();

    return 0;
}