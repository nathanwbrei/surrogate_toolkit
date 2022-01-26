# GPU Perf Tester

The goal of this subproject is to perform experiments that trigger GPU bottlenecks.
Once we find the bottlenecks, we can figure out how to deduce the presence of these 
bottlenecks using profiling tools, well in advance of attempting to port the code to CUDA.
Everything we learn about deducing the GPU bottlenecks goes into a playbook document, which 
lives at `docs/playbook.md`.

This is conceptually very similar to JANA's JTest. 

The main bottleneck is expected to be memory movement.