
#pragma once
#include <complex>
#include <stdlib.h>
#include <string>
#include <vector>

struct Flamegraph {
  struct Node {
    std::string symbol;
    uint64_t own_sample_count;
    uint64_t total_sample_count;
    std::vector<std::unique_ptr<Node>> children;
  };
  std::unique_ptr<Node> root;

  void add(std::vector<std::string> symbol, uint64_t sample_count);
  void add(std::string line);

  Flamegraph() = default;
  Flamegraph(std::string filename);
};



