#pragma once
#include "value.h"
#include <unordered_map>

int build_dot_graph(const std::shared_ptr<Value>& node, int& node_counter, std::string& dot_graph_string, std::unordered_map<Value*, int>& visited);

std::string create_dot_graph_string(const std::shared_ptr<Value>& root);

int build_dot_graph(const std::shared_ptr<Value>& node, int& node_counter, std::string& dot_graph_string, std::unordered_map<Value*, int>& visited) {
    auto it = visited.find(node.get());
    if (it != visited.end()) {
        return it->second;
    }

    int node_id = node_counter++;
    visited[node.get()] = node_id;
    std::string node_name = std::string("node") + std::to_string(node_id);

    std::string label_str = "";
    if (!node->label().empty()) {
        label_str += node->label();
        label_str += " | ";
    }
    dot_graph_string += std::string("    \"") + node_name + "\" [label=\"" + label_str + "data " + std::to_string(node->data()) + " | grad " + std::to_string(node->grad()) + "\", shape=record];\n";

    if (!node->prev().empty()) {
        std::string op_name = std::string("op") + std::to_string(node_id);
        std::string op_label = "";
        op_label += node->op();

        dot_graph_string += std::string("    \"") + op_name + "\" [label=\"" + op_label + "\", shape=ellipse];\n";
        dot_graph_string += std::string("    \"") + op_name + "\" -> \"" + node_name + "\";\n";

        for (const auto& child : node->prev()) {
            int child_id = build_dot_graph(child, node_counter, dot_graph_string, visited);
            std::string child_name = std::string("node") + std::to_string(child_id);
            dot_graph_string += std::string("    \"") + child_name + "\" -> \"" + op_name + "\";\n";
        }
    }

    return node_id;
}

std::string create_dot_graph_string(const std::shared_ptr<Value>& root) {
    std::string dot_graph_string = "digraph {\n    rankdir=LR;\n";
    int node_counter = 0;
    std::unordered_map<Value*, int> visited;
    build_dot_graph(root, node_counter, dot_graph_string, visited);
    dot_graph_string += "}\n";

    return dot_graph_string;
}

std::string create_dot_graph_string(const std::vector<std::shared_ptr<Value>>& roots) {
    std::string dot_graph_string = "digraph {\n    rankdir=LR;\n";
    int node_counter = 0;
    std::unordered_map<Value*, int> visited;
    for (const auto& root : roots) {
        build_dot_graph(root, node_counter, dot_graph_string, visited);
    }
    dot_graph_string += "}\n";

    return dot_graph_string;
}