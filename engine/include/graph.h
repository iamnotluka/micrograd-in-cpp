  #pragma once                                                                                                                                         
  #include "value.h"
                                                                                                                                                       
  int build_dot_graph(const std::shared_ptr<Value>& node, int& node_counter, std::string& dot_graph_string);                                           
                                                                                                                                                     
  std::string create_dot_graph_string(const std::shared_ptr<Value>& root);