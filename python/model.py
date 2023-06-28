import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # checking device to use CUDA
N = nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )  # Custom Architecture
A = 0.
nnf = lambda x: A + x * N(x)  # neural network function
f = lambda x: .5 + .5*torch.exp(x.clone().detach().requires_grad_(True)) * (torch.sin(x.clone().detach().requires_grad_(True))- torch.cos(x.clone().detach().requires_grad_(True)))  #actual v_x function

def loss(x):  #loss function

    x.requires_grad = True
    outputs = nnf(x)  # outputs from the neural network
    calculated_gradients = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                        create_graph=True)[0]   # gradients of predicted function to compare to the actual 

    return  torch.mean( abs( calculated_gradients - f(x) ))   # absolute value of error

lowerbound=1  #lower bound
upperbound = 3  #upper bound
points = 2000  # number of collaction points being randomly selected
x = torch.Tensor(np.linspace(lowerbound, upperbound, points)[:, None])  # linear space to test on
optimizer = torch.optim.LBFGS(N.parameters())     #used to optimize the different weights 
def closure():   # function for the optimization
    optimizer.zero_grad()
    l = loss(x)  
    l.backward()  #backpropogation to optimize
    print(l)  
    return l

for i in range(9):  
    optimizer.step(closure)


linearspace = np.linspace(lowerbound, upperbound, points)[:, None]   # 

with torch.no_grad():
    pv = nnf(torch.Tensor(linearspace)).numpy()  #predicted values using the neural network 
av = .5 + .5*torch.exp(torch.Tensor(linearspace)) * (torch.sin(torch.Tensor(linearspace))- torch.cos(torch.Tensor(linearspace)))  #actual values


#plot
fig, ax = plt.subplots(dpi=100)
ax.plot(linearspace, av, label='True S')
ax.plot(linearspace, pv, '--', label='NN S')
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
plt.legend(loc='Best')
# saving the model
traced_script_module = torch.jit.trace(N, torch.Tensor(linearspace))
traced_script_module.save("model.pt")
