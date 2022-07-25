
# Colin Wolfe Write-Up
SRGS 2022 - PHASM Project


## Tutorial PyTorch Model
Located in the phasm/python/tutorial.py

This is a neural network that was built to approximate the polynomical 3x^2 + 2y + z.
This NN was trained on just 2 data points, and it's main goal was to be used to practice implementing PyTorch neural networks
into PHASM surrogate models.  
It takes in a 1 X 3 Tensor of doubles and spits out a 1 X 1 Tensor of doubles that serve as its prediction.
The model currently runs through a 1000 epochs of training.

The architecture of the model is fairly simple and defined in the class Neural_Network()
```

class Neural_Network(nn.Module):   #defining the neural network class over the next few lines with a custom architecture, weights, defining the forward propogation, activation function, and the backpropogration
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):   #determining the initial weights and bias for the function
        self.z = torch.matmul(X, self.W1) # 3 X 3 
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o
        
    def sigmoid(self, s):   #activation function
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):  # gradient of the activation function for error calculation
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):  #back propogration for optimization
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):  #training the model
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):  #saving the weights to be accessed later
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        
    def predict(self):  # printing out the predictions
        print ("Predicted data based on trained weights: ")  # printing the predicted data
        print ("Input (scaled): \n" + str(xPredicted)) 
        print ("Output: \n" + str(self.forward(xPredicted)))
```

You can change the number of inputs by updating a few things.   

```
X =  torch.tensor(([18,3,6], [15,13,18], [14,22,8]), dtype=torch.float) # 3 X 2 tensor
y = torch.tensor(([984], [719], [640]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([6,12,20]), dtype=torch.float) # 1 X 3 tensor
```
Change the values of X, Y, and xPredicted to make them match the size of the inputs and outputs you need.

The other place to update is in the Neural_Network class
```
class Neural_Network(nn.Module):   #defining the neural network class over the next few lines with a custom architecture, weights, defining the forward propogation, activation function, and the backpropogration
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3
```
Change the self.inputSize and the self.outputSize to make sure that it is properly lined up with the data being provided to it.


This model could be used as a good introduction to PyTorch and for implementing PyTorch models into PHASM.

Future Steps:
```
1.) Training the Model on more data
2.) Testing the Model on more data
```


## The ODE Charged Particle  (phasm/python/model.py)
The PyTorch neural network is attempting to model the x position of a charged spherical particle moving through an external field.  The equations are defined in the code on line 20.


The Model uses an architecture of 2 hidden layer, 3 activation functions, takes in a sequence of t as the input, and prints out a sequence of predicted Xs as an output
```
N = nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )  
```
I then train the model on collocation points between one and three to approximate the true solution.  After that I optimze using a LBFG function and define my loss as the difference of the gradient of the predicted value minus the actual gradient (as one does in PINNs).  I run 9 epochs of optimization.
```
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
```

I then plot the actual solution vs the predicted solution using MatPlotLib
```
fig, ax = plt.subplots(dpi=100)
ax.plot(linearspace, av, label='True S')
ax.plot(linearspace, pv, '--', label='NN S')
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
plt.legend(loc='Best')
```

Lastly I save the model as a TorchScript file using the Torchvision library
```
traced_script_module = torch.jit.trace(N, torch.Tensor(linearspace))
traced_script_module.save("model.pt")  # saved as model.pt in the python folder
```

Future Steps:
```
1.) Develop the model so that it takes in doubles instead of sequences
2.) Train on a CSV file instead of collaction points to decrease computing power
3.) Create an implementation plan for them
```


## Scripting Changes

### Changes to the download_deps.sh file

Made it so once an error occurs the program stops and gives a descriptive error of what went wrong.

Updated code (Lines 3-8):

```
set -e

# Record the last command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo the error message given before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

```

Checks to see if the deps directory already exists, if so the download does not continue

Updated Code (Lines 21-28):

```
deps="deps/"
if [ -d "$deps" ]; then
  echo "$deps has been found."
  echo "download not continuing."
  exit
else
  echo "deps has not been found"
fi
```


Changing the paths for a few downloads.  I updated the Libtorch unzip path and the Intel PIN tar path

Updated Code (lines 51 and 62):

```
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip  # LINE 51, updated the path since the download path didn't line up with the unzip before

mv pin-3.22-98547-g7a303a835-gcc-linux.tar.gz pin  #LINE 62, updated the file being moved as the original file being moved didn't exist
```

## Changes to PHASM files


### memtrace_pin_frontend.cpp  (memtrace_pin/memtrace_pin_frontend.cpp)
In the memtrace_pin_frontend.cpp file, I made a few changes regarding the printf functions on line 58, 64, 70, and 121.  

Original Code:

```
VOID record_malloc_first_argument(ADDRINT size, VOID* ip) {
    if (in_target_routine) {
        printf("cm %p Malloc request of size %llu\n", ip, size);  # LINE 58
    }
}

VOID record_malloc_return(ADDRINT addr, VOID* ip) {
    if (in_target_routine) {
        printf("rm %p: Malloc returned %llx\n", ip, addr);  # LINE 60
    }
}

VOID record_free_first_argument(ADDRINT addr, VOID* ip) {
    if (in_target_routine) {
        printf("cf %p: Freeing %llx\n", ip, addr);  # LINE 70
    }
};

if (rtn_name == KnobTargetFunction.Value()) {
        printf("Instrumenting %s (%llu)\n", rtn_name.c_str(), current_routine);
        target_function_found = true;  # LINE 121

```

New Code:

```
VOID record_malloc_first_argument(ADDRINT size, VOID* ip) {
    if (in_target_routine) {
        printf("cm %p Malloc request of size %lu\n", ip, size);  # LINE 58
    }
}

VOID record_malloc_return(ADDRINT addr, VOID* ip) {
    if (in_target_routine) {
        printf("rm %p: Malloc returned %lx\n", ip, addr);  # LINE 64
    }
}

VOID record_free_first_argument(ADDRINT addr, VOID* ip) {
    if (in_target_routine) {
        printf("cf %p: Freeing %lx\n", ip, addr);  # LINE 70
    }
};

if (rtn_name == KnobTargetFunction.Value()) {
    printf("Instrumenting %s (%lu)\n", rtn_name.c_str(), current_routine);   #LINE 121
    target_function_found = true;
```

By making these changes, I solved errors regarding the type of int being stored.  By changing the long long to a long, the file no longer presented any problems during installation.


### Tensor.hpp   (phasm/surrogate/include/tensor.hpp)

On lines 23-30  I updated the function being called from is_same_v to is_same because that would better suit the comparison of the template to the array reference.  I also added parenthesis to the end of all the comparison types.

Updated code:

```
    # I updated the is_same for all of these lines, and the () after the closing > for the comparison
    if (std::is_same<T, u_int8_t>()) return phasm::DType::UI8;   
    if (std::is_same<T, int16_t>()) return phasm::DType::I16;
    if (std::is_same<T, int32_t>()) return phasm::DType::I32;
    if (std::is_same<T, int64_t>()) return phasm::DType::I64;
    if (std::is_same<T, float>()) return phasm::DType::F32;
    if (std::is_same<T, double>()) return phasm::DType::F64;
```

In this file I also updated line 98 by adding in the type for the ArrayRef variable.

Updated code:

```
# made it ArrayRef<int64_t> instead of just ArrayRef
m_underlying = m_underlying.reshape(at::ArrayRef<int64_t>(shape.data(), shape.size()));   
```

### Range.h  (phasm/surrogate/include/range.h)

I updated a library that was not available for people without C++ 17 (and maybe not for Linux users either??)

I fixed the std::optional by making it std::experimental/optional

SIDENOTE:  This may not work on MacOS or if the C++ version is too high.  If an error occurs during download then revert back to the original.

Updated Code:  (Updated lines 15, 28, 29)

```
#include <experimental/optional>  #LINE 15  made it experimental/optional instead of just optional

std::experimental::optional<tensor> lower_bound_inclusive;   # LINE 28
std::experimental::optional<tensor> upper_bound_inclusive;   # LINE 29

# Made it std::experimental/optional instead of std::optional

```

## Implementation

The main part of the phasm tutorial implementation is in the surrogate builder.  The application works in a similar way to the other PHASM examples, the only difference is the way the Surrogate Model is created.

```
phasm::Surrogate s_surrogate = phasm::SurrogateBuilder()
        .set_model(std::make_shared<phasm::TorchscriptModel>(phasm::TorchscriptModel("~/home/USER/phasm/python/tutorial.pt"))) //update the PATH depending on the user
        .local_primitive<double>("x", phasm::IN)
        .local_primitive<double>("y", phasm::IN)
        .local_primitive<double>("z", phasm::IN)
        .local_primitive<double>("result", phasm::OUT)
        .finish();


```
Making the the TorchscriptModel shared allows for it to be passed to the .set_model function.  I will push the tutorial.pt file to the repo so that way people are not required to run tutorial.py.  This is also helpful for people who do not have to install pytorch and python onto their computer.

Future Steps:
```
1.)  Create and end to end running tutorial version of the PyTorch model working in PHASM
2.)  Create another example using the ODE Solver
```



