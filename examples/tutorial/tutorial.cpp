
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>
#include <surrogate_builder.h>
#include <feedforward_model.h>

/* This tutorial describes the key features of libphasm. These allow the user to introduce a neural net surrogate model
 * for an arbitrary function, capture training data, create new training data via sampling, train the model in
 * place, etc. Although the PHASM project also includes performance analysis and memory tracing functionality, they will
 * not be covered here.
 */

/* Here is the simplest possible use of libphasm. Given some arbitrary (in reality quite expensive) function,
 * we'd like to create a _surrogate_ which mimics it to a desired order of accuracy. For illustration purposes we'll
 * use a simple polynomial.
 * */

double f(double x, double y, double z) {
    return 3*x*x + 2*y + z;
}

/* We write our PHASM wrapper for f like so. f_wrapper is a drop-in replacement for f,
 * which
 * */

double f_wrapper(double x, double y, double z) {
    using namespace phasm;
    static Surrogate surrogate = SurrogateBuilder()                    // [1]
            .set_model(std::make_shared<FeedForwardModel>())           // [2]
            .local_primitive<double>("x", IN)                          // [3]
            .local_primitive<double>("y", IN)
            .local_primitive<double>("z", IN)
            .local_primitive<double>("returns", OUT)
            .finish();                                                 // [4]

    double result = 0.0;

    surrogate.bind_original_function([&](){ result = f(x,y,z); })      // [5]
             .bind_all_callsite_vars(&x, &y, &z)                       // [6]
             .call_original_and_capture();                             // [7]

    return result;
}

/* Let's unpack this piece by piece.
 *
 * [1] PHASM's main abstraction for a surrogate model is called `Surrogate`. Because we are
 *     wrapping the original function and always calling the wrapper, we only need one Surrogate
 *     object for f_wrapper, and it needs to last for the lifetime of the program,
 *     so we make it static. To configure a Surrogate, use the `SurrogateBuilder`, which provides
 *     a fluent interface.
 *
 * [2] The first thing we need to tell the SurrogateBuilder is which model to use. Here we are
 *     using FeedForwardModel, which gives us the simplest possible neural net architecture using
 *     PyTorch as the backend. You can configure the number and size of the hidden layers using
 *     optional constructor arguments; the model automatically sizes its input and output layers
 *     to match the size of the input and output data. Generally, however, we expect users
 *     to design a model inside JupyterLab and export it using TorchScript. PHASM conveniently
 *     provides `TorchScriptModel` for this purpose. You can also write your own Model subclasses.
 *
 * [3] The next thing the Surrogate needs to know about is all of its inputs and outputs. Here we
 *     declare that the Surrogate should accept as inputs 3 doubles from the local scope, and
 *     as outputs one double. So far this is comparable other surrogate modelling toolkits, but
 *     just keep reading!
 *
 * [4] Once we've passed the SurrogateBuilder all the information it needs, we call `finish()`,
 *     which tells it to construct the Surrogate and negotiate the input and output data sizes with
 *     the Model. We've configured the Surrogate! (Note that because we declared the Surrogate to be
 *     static, we bypassed the lazy initialization/double-checked locking problem, so the initialization
 *     is thread-safe.)
 *
 * [5] For our Surrogate to call the original function f, we need to _bind_ the input and output parameters to f.
 *     We pass the Surrogate a lambda function with type signature `void(void)`. Note that we have to declare a
 *     variable on the stack to use as the return value. If you are wondering why we do this on every call as
 *     opposed to once during startup, it is because the input parameters might be references, which would otherwise
 *     capture incorrectly.
 *
 * [6] For our Surrogate to use the Model, we need to bind the input and output parameters to the model. This is
 *     fundamentally more interesting, because the Model expects tensors containing primitives, not arbitrary C/C++
 *     types. For this simple case, all we have to do is call `bind_all_callsite_vars()` passing in _pointers_ to
 *     the parameters in the order they were declared to the SurrogateBuilder.
 *
 * [7] Finally, we can use our Surrogate. The code shown here will call the original function, capture both the
 *     inputs and outputs as tensors, and save them. We can later dump the data to CSV or use it to train the model
 *     directly.
 * */

// Global variables
// Arrays of structs of structs of arrays of data
//
// Phasm call mode





int main() {
    return 0;
}