#include "value.h"
#include "graph.h"
#include <mlp.h>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<std::shared_ptr<Value>> x = {Value::create(2.0), Value::create(3.0)};
    MLP mlp(2, std::vector<int>{4, 4, 1});

    auto out = mlp(x);

    std::ofstream file("graph.dot");
    file << create_dot_graph_string(out);
    file.close();

    return 0;
}