#include "value.h"
#include "graph.h"
#include <neuron.h>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<std::shared_ptr<Value>> x = {Value::create(2.0), Value::create(3.0)};     
    Neuron n(2);                                                                          
    auto out = n(x);
    std::ofstream file("graph.dot");
    file << create_dot_graph_string(out);
    file.close();

    return 0;
}