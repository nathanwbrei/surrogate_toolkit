
Structs we can always decompose into one param per field:
`{a,b,c} --> (a,b,c)`

And we can collapse nested structs into a single struct:
`{a,b,c,{d,e}} --> {a,b,c,d,e}`

Lists we cannot do either. However, we can convert array of struct into struct of array:
`[{a}] --> {[a]}`

Next we can "sort" an arbitrary nesting of lists and structs into a nested struct whose fields are all either primitives or nested lists:
`[{[[{{d}}]]}] -->  {{{[[[d]]]}}}`

And then collapse the outer structs so that each field is associated with a nested list of variable length and nesting level. (In the trivial case, this includes primitives).
`{{{[[[d]]]}}} --> ([[[d]]])`

A nested list with variable length and nesting level is exactly a ragged tensor.  Thus, every parameter is associated with a ragged tensor, as opposed to a primitive.

This is nice and simple to model and to attach to a ML model. However, the user is stuck with the unenviable task of converting arbitrary structured data into a structure of nested arrays.

Maybe we can help them. Maybe they provide iterators for each list level and getters for each struct level. Maybe we use XPath or Lens style abstractions to do all the dirty work of packing the tensor. Here's the first thing that pops into my head:

```c++
struct T { int a; std::vector<B> bs;};
struct B { std::vector<double> c; };

// accessed via

void access_list()

auto t = surrogate.struct_input<T>();
t.primitive<A>("a", [](T& t){return t.a;});

auto lb = t.list<B>("bs", [](T& t){ return {std::begin(t.bs), std::end(t.bs)};}); 
// TODO: How do we write back to this?

// etc


                                       .list<T, B>("b", [](T t){return t.b})
									   
```

Capturing this should be straightforward, sampling I can imagine being very fragile attempting to write these. 

