#include "flamegraph.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>
#include <iostream>
#include <cstring>

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

std::string stringifyFilterResult(Flamegraph::FilterResult fr) {
    switch (fr) {
        case Flamegraph::FilterResult::Include: return "Include";
        case Flamegraph::FilterResult::ExcludeOutsideEvtLoop: return "ExcludeOutsideEvtLoop";
        case Flamegraph::FilterResult::ExcludeTiny: return "ExcludeTiny";
        case Flamegraph::FilterResult::ExcludeTower: return "ExcludeTower";
        case Flamegraph::FilterResult::ExcludeMemoryBound: return "ExcludeMemoryBound";
        case Flamegraph::FilterResult::ExcludeKernelFn: return "ExcludeKernelFn";
        default: return "Corrupted FilterResult!!!";
    }
}

void printNode(std::ostream& os, Flamegraph::Node* node, int level, bool all) {
    if (all || node->filterResult == Flamegraph::FilterResult::Include)  {
        for (int i=0; i<level; ++i) {
            os << "  ";
        }
        os << node->symbol << " [" << node->total_sample_count << ", " << node->own_sample_count << ", " << node->eventloop_fraction << ", "<< stringifyFilterResult(node->filterResult) << "]" << std::endl;
    }
    for (const auto& child : node->children) {
        printNode(os, child.get(), level+1, all);
    }
}

void Flamegraph::print(bool all, std::ostream& os) {
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

void Flamegraph::write(bool all, std::ostream& os) {
    for (const auto& child : root->children) {
        writeNode(os, child.get(), "", all);
    }
}

void filter_eventloop(Flamegraph::Node* node, const std::string& eventloop_symbol) {
    if (node->symbol != eventloop_symbol) {
        node->filterResult = Flamegraph::FilterResult::ExcludeOutsideEvtLoop;
        for (const auto& child : node->children) {
            filter_eventloop(child.get(), eventloop_symbol);
        }
    }
    // If we found the eventloop symbol, the recursion terminates so that the eventloop and 
    // its children stay Included. 
}

void filter_kernelfns(Flamegraph::Node* node) {
    // I just want str.ends_with()
    auto pos = node->symbol.size()-4;
    if (strcmp(node->symbol.data() + pos, "_[k]") == 0) {
        if (node->filterResult == Flamegraph::FilterResult::Include) {
            node->filterResult = Flamegraph::FilterResult::ExcludeKernelFn;
        }
    }
    // TODO: Somthing similar for JIT and inline functions
    for (const auto& child : node->children) {
        filter_kernelfns(child.get());
    }
}

void filter_tinies(Flamegraph::Node* node, float tower_fraction) {
    if (node->eventloop_fraction <= tower_fraction) {
        if (node->filterResult == Flamegraph::FilterResult::Include) {
            // There might be multiple reasons to exclude, so we just go with the first reason 
            // We then order our filters from "most reliable" to "least reliable"
            node->filterResult = Flamegraph::FilterResult::ExcludeTiny;
        }
    }
    for (const auto& child : node->children) {
        filter_tinies(child.get(), tower_fraction);
    }
}

void filter_towers(Flamegraph::Node* node, float tower_fraction) {
    // Determine if this node is the base of a tower
    for (const auto& child : node->children) {
        if ((child->eventloop_fraction / node->eventloop_fraction) >= tower_fraction) {
            if (node->filterResult == Flamegraph::FilterResult::Include) {
                node->filterResult = Flamegraph::FilterResult::ExcludeTower;
            }
        }
        filter_towers(child.get(), tower_fraction);
    }

    // Recurse over all children
    for (const auto& child : node->children) {
        filter_towers(child.get(), tower_fraction);
    }
}

void score(Flamegraph::Node* node, const std::string& eventloop_symbol, uint64_t eventloop_total_samples) {
    if (node->symbol == eventloop_symbol) {
        eventloop_total_samples = node->total_sample_count;
        node->eventloop_fraction = 1.0;
    }
    else if (eventloop_total_samples != 0) {
        // eventloop_total_samples == 0   =>  Not inside the event loop yet
        node->eventloop_fraction = (float) node->total_sample_count/eventloop_total_samples;

    }
    for (const auto& child : node->children) {
        score(child.get(), eventloop_symbol, eventloop_total_samples);
    }
    // Recurse over all nodes because we don't know where the eventloop will appear (it might even appear multiple times!)
}


void Flamegraph::filter(const std::string& eventloop_symbol, float tiny_fraction, float tower_fraction) {
    this->eventloop_symbol = eventloop_symbol; // Store this so we don't need to thread it through all the filters
    score(root.get(), this->eventloop_symbol, 0);
    filter_eventloop(root.get(), eventloop_symbol);
    filter_kernelfns(root.get());
    filter_tinies(root.get(), tiny_fraction);
    filter_towers(root.get(), tower_fraction);
}


void buildColorPalette(Flamegraph::Node* node, std::map<std::string, std::tuple<uint8_t,uint8_t,uint8_t>>& palette) {

    auto it = palette.find(node->symbol);
    if (it == palette.end()) {
        if (node->filterResult == Flamegraph::FilterResult::Include) {
            palette.insert({node->symbol, {255,100,0}}); // Orange
        }
        else {
            palette.insert({node->symbol, {200,200,200}}); // Gray
        }
    }
    else {
        // Include overrides previous exclude. Why? Highlight candidate functions that are ALSO used outside event loop
        if (node->filterResult == Flamegraph::FilterResult::Include) {
            it->second = {255, 100, 0}; // Orange
        }
    }
    for (const auto& child : node->children) {
        buildColorPalette(child.get(), palette);
    }
}

std::map<std::string, std::tuple<uint8_t,uint8_t,uint8_t>> Flamegraph::buildColorPalette() {

    std::map<std::string, std::tuple<uint8_t,uint8_t,uint8_t>> palette;
    ::buildColorPalette(root.get(), palette);
    return palette;
}

void Flamegraph::writeColorPalette(std::ostream& os) {
    auto palette = buildColorPalette();
    for (auto pair : palette) {
        os << pair.first << "->rgb(" << (int)std::get<0>(pair.second) << "," << (int)std::get<1>(pair.second) << "," << (int)std::get<2>(pair.second) << ")" << std::endl;
    }
}




