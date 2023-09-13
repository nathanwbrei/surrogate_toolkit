#include "flamegraph.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

Flamegraph::Flamegraph(std::string filename) {}

std::vector<std::string> split(std::string line, char delimiter) {

  std::vector<std::string> tokens;
  std::istringstream iss(line);
  std::string token;

  while (std::getline(iss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

void Flamegraph::add(std::vector<std::string> stacktrace,
                     uint64_t sample_count) {
  Node *current_node = root.get();
  current_node->total_sample_count += sample_count;
  // The root node total sample count is used as the denominator for every
  // percentage

  for (auto &current_symbol : stacktrace) {

    for (const std::unique_ptr<Node> &node : current_node->children) {
      if (node->symbol == current_symbol) {
        // Symbol found
        node->total_sample_count += sample_count;
        current_node = node.get();
        break;
      }
    }
    // Symbol not found, so we create a node for it
    auto created_node = std::make_unique<Node>();
    created_node->symbol = current_symbol;
    created_node->own_sample_count = 0; // In case this is not a leaf node
    created_node->total_sample_count = sample_count;
    Node *next_node = created_node.get();
    current_node->children.push_back(std::move(created_node));
    current_node = next_node;
  }
  // current_node points to the leaf node. This is the only time we set
  // own_sample_count
  current_node->own_sample_count = sample_count;
}

void Flamegraph::add(std::string line) {

  std::vector<std::string> halvsies = split(line, ' ');
  assert(halvsies.size() == 2);
  assert(halvsies[1].size() == 1);
  std::vector<std::string> stacktrace = split(halvsies[0], ';');
  uint64_t sample_count;
  sample_count << (halvsies[1][0]);
  add(stacktrace, sample_count);
}
