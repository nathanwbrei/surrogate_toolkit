
# Playbook

These are the bottlenecks we know about so far:

- Memory transfer onto GPU
- Memory transfer off GPU
- Data cache situation on GPU?
- Instruction cache situation on GPU?
- Cheap floating point ops such as multiply/add  (the best possible bottleneck!)
- Expensive floating point ops such as sqrt
- Weird ops such as gather/scatter

We have two goals: 

1. Determine the values for these bottlenecks for a given GPU design. Ideally we can do this 
   by just reading the documentation, but perhaps not. At any rate, this is work that somebody 
   has surely already done; we just have to find it.

2. Determine which bottlenecks constrain the execution of our code. Ideally we can figure 
   this out using perf, totalview, vtune, or similar. The case where our code is a basic neural
   net is going to be substantially simpler than the general case. 
   
## Approach

The simplest starting place is with a multilayered perceptron. The independent variables are the 
   amount of memory movement in, and out, and the number of layers and neurons. The dependent variables
   are the latency and throughput. We can generate the input data randomly. If all we 
   are doing is inference, we don't need to train the weights at all. If we want to realistically simulate
   training a NN, we can find a function with a variable input size and use the surrogate toolkit
   to turn it into a NN and sample the input space. 


   
   