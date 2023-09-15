#include "flamegraph.hpp"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <vector>
#include <iostream>

Flamegraph::Flamegraph(std::string filename) {
  std::ifstream fs (filename);
  std::string line; 
  while (std::getline(fs, line)) {
    add(line);
  }
}

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

    bool node_found = false;
    for (const std::unique_ptr<Node> &node : current_node->children) {
      if (node->symbol == current_symbol) {
        // Symbol found
        node->total_sample_count += sample_count;
        current_node = node.get();
        node_found = true;
      }
    }
    if (! node_found) {
      // Symbol not found, so we create a node for it
      auto created_node = std::make_unique<Node>();
      created_node->symbol = current_symbol;
      created_node->own_sample_count = 0; // In case this is not a leaf node
      created_node->total_sample_count = sample_count;
      Node *next_node = created_node.get();
      current_node->children.push_back(std::move(created_node));
      current_node = next_node;
    }
  }

  // current_node points to the leaf node. This is the only time we set
  // own_sample_count
  current_node->own_sample_count = sample_count;
}

void Flamegraph::add(std::string line) {

  size_t pos = line.find_last_of(' ');
  std::string stacktrace_str = line.substr(0, pos);
  std::string sample_count_str = line.substr(pos+1);
  std::vector<std::string> stacktrace = split(stacktrace_str, ';');
  uint64_t sample_count;
  std::stringstream ss(sample_count_str);
  ss >> sample_count;
  add(stacktrace, sample_count);
}

void printNode(std::ostream& os, Flamegraph::Node* node, int level, bool all) {
    if (all || node->filterResult == Flamegraph::FilterResult::Include)  {
        for (int i=0; i<level; ++i) {
            os << "  ";
        }
        os << node->symbol << " [" << node->total_sample_count << ", " << node->own_sample_count << "]" << std::endl;
    }
    for (const auto& child : node->children) {
        printNode(os, child.get(), level+1, all);
    }
}

void Flamegraph::print(std::ostream& os, bool all) {
    for (const auto& child : root->children) {
        printNode(os, child.get(), 0, all);
    }
}

void writeNode(std::ostream& os, Flamegraph::Node* node, std::string symbol_prefix, bool all) {

    if (symbol_prefix.empty()) {
        symbol_prefix = node->symbol;
    }
    else {
        symbol_prefix = symbol_prefix + ";" + node->symbol;
    }
    if (all || node->filterResult == Flamegraph::FilterResult::Include) {
        if (node->own_sample_count != 0) {
            os << symbol_prefix << " " << node->own_sample_count << std::endl;
        }
    }
    for (const auto& child : node->children) {
        writeNode(os, child.get(), symbol_prefix, all);
    }
}

void Flamegraph::write(std::ostream& os, bool all) {
    for (const auto& child : root->children) {
        writeNode(os, child.get(), "", all);
    }
}


