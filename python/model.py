# First PINN, some of it is tutorial+Doc code
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # checking device to use CUDA
N = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Softmax(),
            nn.Linear(32, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
        )  # Custom Architecture
A = 0.
nnf = lambda x: A + x * N(x)  # neural network function
f = lambda x, Psi: .5 + .5*torch.exp(x.clone().detach().requires_grad_(True)) * (torch.sin(x.clone().detach().requires_grad_(True))- torch.cos(x.clone().detach().requires_grad_(True)))

def loss(x):  #loss function

    x.requires_grad = True
    outputs = nnf(x)  # outputs from the neural network
    Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                        create_graph=True)[0]   # Predicted function

    return  torch.mean( ( Psi_t_x - f(x, outputs) )  ** 2)   # mean squared error

lowerbound=1  #lower bound
upperbound = 3  #upper bound
points = 2000  #collaction points
x = torch.Tensor(np.linspace(lowerbound, upperbound, points)[:, None])  # linear space to test on
optimizer = torch.optim.LBFGS(N.parameters())
def closure():   #optimization function
    optimizer.zero_grad()
    l = loss(x)  # loss
    l.backward()  #backpropogation
    print(l)  #print loss
    return l

for i in range(16):
    optimizer.step(closure)


linearspace = np.linspace(lowerbound, upperbound, points)[:, None]

with torch.no_grad():
    py = nnf(torch.Tensor(linearspace)).numpy()  #predicted y
ay = .5 + .5*torch.exp(torch.Tensor(linearspace)) * (torch.sin(torch.Tensor(linearspace))- torch.cos(torch.Tensor(linearspace)))


#plot
fig, ax = plt.subplots(dpi=100)
ax.plot(linearspace, ay, label='True S')
ax.plot(linearspace, py, '--', label='NN S')
ax.set_xlabel('$x$')
ax.set_ylabel('$Psi(x)$')
plt.legend(loc='Best')

traced_script_module = torch.jit.trace(N, torch.Tensor(linearspace))
traced_script_module.save("model.pt")