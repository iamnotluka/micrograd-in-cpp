#include "value.h"
#include "graph.h"
#include <layer.h>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<std::shared_ptr<Value>> x = {Value::create(2.0), Value::create(3.0)};     
    Layer l(2, 3);                                                                          
    auto out = l(x);
    std::ofstream file("graph.dot");
    file << create_dot_graph_string(out);
    file.close();

    return 0;
}