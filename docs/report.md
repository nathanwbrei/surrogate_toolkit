# PHASM Report: Heat Diffusion PDE
Created by: Dhruv Bejugam
Last Updated: 07/22/22
Summary: Report about Jefferson Lab internship and working on the PHASM project

---
##  Objective
The overall goal of PHASM (Parallel Hardware and Surrogate Models) is to get older C++ Jefferson Lab code to run on parallel hardware by replacing the intensive parts of existing code with surrogate neural network models. Our internship group was tasked with fixing a missing piece of PHASM: the lack of effective ODE and PDE neural network solvers. In order to fix this missing piece of PHASM, each intern was tasked with a separate assignment. My assignment was creating a Python heat diffusion PDE neural network solver and implementing it into C++ using PHASM. 

##  Methodology
Before creating the neural network, the first step to understand this task was researching and understanding heat diffusion partial equations. This included looking at concepts like boundary conditions, initial conditions, and the diffusion of heat over time. Then, I starting researching PINNs and PDE solvers through research papers, previous code implementations, and PyTorch libraries. During my research, I was able to find a PyTorch library, Neurodiffeq, which was optimized for PDE neural network solvers. After getting familiar with Neurodiffeq, I was able to create my first PDE solver modeling the solution of a heat diffusion equation. From there, I experimented with the network created and developed two other PDE solver neural networks using only PyTorch (non-heat diffusion equations). Because of the lack of time I had, I wasn't able to make the most intensive PDE solvers I could, but I took into account how much time I had and relied heavily on previous research. With the 3 neural networks I created, I continued to examine the accuracy and solutions of the neural networks by comparing the analytical solution and the neural network approximation, and also inspecting loss and residual. From here, I looked for ways to improving my models by making efficiency and accuracy changes. After optimizing my models, I starting integrating my Neurodiffeq heat diffusion model into C++ using the PHASM toolkit, TorchScript, and Git. 

## Code Developed
 ****All code is extensively explained through comments in repository\****
 
****Python programs can be run by accessing 'python' folder in repository and C++ program can be run by building PHASM\****

- Neurodiffeq Laplace Equation Approximator
	- Using PyTorch based package (Neurodiffeq) to approximate 2D heat diffusion equation
	- Takes in (dirichlet) boundary conditions, initial conditions, and equation on a 1 x 1 grid
	- Plots solution to equation in 3D view
	- Path: [phasm/python/neurodiffeq_PDE_Approximator.ipynb](https://github.com/nathanwbrei/phasm/blob/main/python/neurodiffeq_PDE_Approximator.ipynb)
- PyTorch Burgers' Equation Approximator
	- Using PyTorch to approximate Burgers' Equation
	- Takes in file with training points (initial, boundary, and collocation points)
	- Uses architecture implemented from "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)"
	- Plots solution to equation at different times
	- Path: [phasm/python/PyTorch_PDE_Approximator_1.ipynb](https://github.com/nathanwbrei/phasm/blob/main/python/PyTorch_PDE_Approximator_1.ipynb)
- PyTorch Inviscid Burgers' Equation Approximator
	- Using PyTorch to approximate inviscid Burgers' equation with *u* as constant
	- Creates sample training points (initial, boundary, and collocation points)
	- Uses architecture implemented from "[Solving Partial Differential Equations with Neural Networks](https://arxiv.org/abs/1912.04737)"
	- Plots solution to equation at different times
	- Path: [phasm/python/PyTorch_PDE_Approximator_2.ipynb](https://github.com/nathanwbrei/phasm/blob/main/python/PyTorch_PDE_Approximator_2.ipynb)
- C++ PDE Solver Implementation 
	- Using TorchScript to implement Neurodiffeq neural network into PHASM C++ repository
	- Still needs more work done using PHASM toolkit 
	- Path: *not yet pushed to PHASM\*
 

##  Results
By the end of the internship I created 3 accurate, efficient PDE solvers. The loss for the models is very low. The loss and/or the residual is shown for the models in the python notebooks. While two of these models are not currently heat diffusion PDE approximators, they are successful and can be used for modeling heat diffusion by changing parameters. I was able to successfully exported my model out of Python and import it into the C++ PHASM repository. However, I did not get as far as I wanted with using some of the PHASM commands.

##  Conclusion
### Future Steps:
- Changing two PyTorch PDE solvers for purely heat diffusion partial differential equations
- Continuing with C++ PDE solver implementation
- Experimenting with PDE solvers in C++

