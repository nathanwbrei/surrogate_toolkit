
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "../vacuum_tool/src/utils.hpp"
#include <iostream>


struct Node {
    Node* parent = nullptr;
    std::vector<Node*> children;
    std::string consumes;
    std::string produces;

    void print(int level) {
        for (int i=0; i<level; ++i) {
            std::cout << "    ";
        }
        if (consumes.empty()) {
            std::cout << produces << std::endl;
        }
        else if (produces.empty()) {
            std::cout << consumes << std::endl;
        }
        else {
            std::cout << consumes << "->" << produces << std::endl;
        }
        for (auto child : children) {
            child->print(level+1);
        }
    }
};

template <typename T>
struct Root : public Node {
    Root() {
        produces = demangle(typeid(T).name());
    };
};

template <typename S, typename T>
struct Branch : public Node {
    Branch() {
        consumes = demangle(typeid(S).name());
        produces = demangle(typeid(T).name());
    }
};

template <typename T>
struct Leaf : public Node {
    Leaf() {
        consumes = demangle(typeid(T).name());
    }
};

template <typename T, typename CursorT>
struct CursorHelper {
    CursorT* cursor;
    CursorHelper(CursorT* cursor) : cursor(cursor){};
};

struct TreeBuilder;

template <typename HeadT, typename... RestTs>
struct TreeCursor : public CursorHelper<HeadT, TreeCursor<HeadT, RestTs...>> {
    Node* focus;
    // We actually know the type of this thing.
    // If RestTs = empty, this is a Root<HeadT>
    // Otherwise          this is a Branch<HeadT, head(RestTs)
    TreeBuilder* builder = nullptr;

    TreeCursor(Node* focus, TreeBuilder* builder) : CursorHelper<HeadT, TreeCursor<HeadT, RestTs...>>(this), focus(focus), builder(builder) {}

    template <typename T>
    TreeCursor<T, HeadT, RestTs...> attachBranch(Branch<HeadT, T>* branch) {
        focus->children.push_back(branch);
        branch->parent = focus;
        return TreeCursor<T, HeadT, RestTs...>(branch, builder);
    }

    TreeCursor attachLeaf(Leaf<HeadT>* leaf) {
        focus->children.push_back(leaf);
        leaf->parent = focus;
        return *this;
    }

    TreeCursor<RestTs...> end() {
        return TreeCursor<RestTs...>(focus->parent, builder);
    }

};


template <typename S>
struct TreeCursor<S> : public CursorHelper<S, TreeCursor<S>> {
    Node* focus;
    TreeBuilder* builder = nullptr;

    TreeCursor(Node* focus, TreeBuilder* builder) : CursorHelper<S, TreeCursor<S>>(this), focus(focus), builder(builder) {}

    template <typename T>
    TreeCursor<T, S> attachBranch(Branch<S, T>* branch) {
        focus->children.push_back(branch);
        branch->parent = focus;
        return TreeCursor<T, S>(branch, builder);
    }

    TreeCursor attachLeaf(Leaf<S>* leaf) {
        focus->children.push_back(leaf);
        leaf->parent = focus;
        return *this;
    }

    TreeBuilder& end() {
        return *builder;
    }


};

struct TreeBuilder {
    std::vector<Node*> roots;
    template <typename T>
    TreeCursor<T> attachRoot(Root<T>* root) {
        roots.push_back(root);
        return TreeCursor<T>(root, this);
    }

    TreeBuilder& doMoreBuilderStuff() {
        return *this;
    };

    void print() {
        for (auto root : roots) {
            root->print(0);
        }
    }
};

class Silly {};

template <typename CursorT>
struct CursorHelper<Silly, CursorT> {
    CursorT* cursor;
    CursorHelper(CursorT* cursor) : cursor(cursor){};
    CursorT& doSomethingSpecificWithSilly() {return *cursor;};
};

TEST_CASE("Typed fluent 'LEGO' interface") {
    TreeBuilder builder;
    builder
    .attachRoot(new Root<int>)
        .attachLeaf(new Leaf<int>)
        .attachBranch(new Branch<int, double>)
            .attachLeaf(new Leaf<double>)
            .attachLeaf(new Leaf<double>)
            .end()
        .attachBranch(new Branch<int, Silly>)
        .end()
    .end()
    .attachRoot(new Root<Silly>)
        .doSomethingSpecificWithSilly()
        .attachLeaf(new Leaf<Silly>)
        .end()
    .doMoreBuilderStuff();

    builder.print();
}


template <typename... ParentT>
struct BuilderA;

template <typename ParentT>
struct BuilderB {
    ParentT* parent;
    BuilderB(ParentT* parent) : parent(parent) {};
    BuilderA<BuilderB<ParentT>> openBuilderA() {
        return BuilderA(this);
    }
    BuilderB<BuilderB<ParentT>> openBuilderB() {
        return BuilderB(this);
    }
    BuilderB& doBStuff() {
        return *this;
    }
    ParentT& close() {
        return *parent;
    }
};

template <typename ParentT>
struct BuilderA<ParentT> {
    ParentT* parent;

    BuilderA(ParentT* parent) : parent(parent) {};

    ParentT& close() {
        return *parent;
    }

    BuilderA<BuilderA<ParentT>> openBuilderA() {
        return BuilderA<BuilderA<ParentT>>(this);
    }
    BuilderB<BuilderA<ParentT>> openBuilderB() {
        return BuilderB(this);
    }
    BuilderA& doAStuff() {
        return *this;
    }
};

template <>
struct BuilderA<> {
    BuilderA() = default;

    BuilderA<BuilderA<>> openBuilderA() {
        return BuilderA<BuilderA<>>(this);
    }
    BuilderB<BuilderA<>> openBuilderB() {
        return BuilderB(this);
    }
    BuilderA& doAStuff() {
        return *this;
    }

};

TEST_CASE("Recursive builders") {
    BuilderA builder;
    builder
        .doAStuff()
        .openBuilderA()
            .openBuilderB()
                .doBStuff()
                .close()
            .doAStuff()
            .openBuilderA()
                .doAStuff()
                .openBuilderA()
                .close()
            .close()
        .close();
};





