
#pragma once
#include <complex>
#include <memory>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>

struct Flamegraph {

  enum class FilterResult {Include, ExcludeKernelFn, ExcludeOutsideEvtLoop, ExcludeTiny, ExcludeMemoryBound, ExcludeTower};

  struct Node {
    std::string symbol;
    std::vector<std::unique_ptr<Node>> children;
    uint64_t own_sample_count = 0;
    uint64_t total_sample_count = 0;
    float eventloop_fraction = 0.0;
    FilterResult filterResult = FilterResult::Include;
  };

  std::string eventloop_symbol = ""; // Empty string means "none, aka use whole program as event loop"
  std::unique_ptr<Node> root = std::make_unique<Node>();

  Flamegraph() = default;
  Flamegraph(std::string filename);

  void add(std::vector<std::string> symbol, uint64_t sample_count);
  void add(std::string line);
  void filter(const std::string& eventloop_symbol, float threshold_percent, float tower_percent);
  void print(bool all=true, std::ostream &os=std::cout);
  void write(bool all=true, std::ostream &os=std::cout);

  std::vector<std::string> getSurrogateCandidates();
  void printSurrogateCandidates(std::ostream& os=std::cout);

  std::map<std::string, std::tuple<uint8_t,uint8_t,uint8_t>> buildColorPalette();
  void writeColorPalette(std::ostream& os=std::cout);

  
};
