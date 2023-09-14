
#pragma once
#include <complex>
#include <memory>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

struct Flamegraph {
  struct Node {
    std::string symbol;
    uint64_t own_sample_count = 0;
    uint64_t total_sample_count = 0;
    bool exclude = false;
    int priority = 0;
    std::vector<std::unique_ptr<Node>> children;
  };

  std::unique_ptr<Node> root = std::make_unique<Node>();

  void add(std::vector<std::string> symbol, uint64_t sample_count);
  void add(std::string line);
  void filter(std::string eventloop_symbol, float threshold_percent, float column_percent);
  void print(std::ostream &os=std::cout);
  void write(std::ostream &os=std::cout);

  Flamegraph() = default;
  Flamegraph(std::string filename);
};
