
Let's start with a particularly simple example: a magnetic field map.
Let's pretend our magnetic field map is just a function with this signature:

```c++
void getB(double x, double y, double z, double* Bx, double* By, double* Bz);
```

Its inputs, `x,y,z`, are spatial coordinates, and its outputs `Bx,By,Bz`, are 
the components of the magnetic field vector at that location. Maybe this function 
works by evaluating a complicated polynomial model derived from the magnet 
geometry. Or maybe it reads a file containing a grid of values and interpolates
a value from the closest grid points. Or maybe it just returns `Bx=By=0; Bz=2.0`. 
As long as `getB` always returns the same outputs given the same inputs, we don't 
care.

We suddenly care about this very much when we are trying to run this in real-time, or
on parallel hardware such as GPUs or FPGAs. Different representations have different
memory and compute complexity, and we need a representation that has a good mechanical
sympathy with our hardware and our data rates. Ideally we could transform our existing
code from an inefficient representation to a more convenient one without having to 
rewrite everything from scratch.

What PHASM does is assist us in creating a neural net representation of a piece of existing code. 
