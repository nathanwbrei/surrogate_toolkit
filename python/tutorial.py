
import torch
import torch.nn as nn
import torchvision

X = torch.tensor(([18,3,6], [15,13,18], [14,22,8]), dtype=torch.float) # 3 X 3 tensor to train the data on, super small dataset
y = torch.tensor(([984], [719], [640]), dtype=torch.float) # 3 X 1 tensor  to label the correct answer for the training (formula is 3x^2 + 2y + z)
xPredicted = torch.tensor(([6,12,20]), dtype=torch.float) # 1 X 3 tensor to test the model



# scale units
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)

X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)
y = y / 100

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


NN = Neural_Network()
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))).detach().item()))  # mean sum loss
    NN.train(X, y)  # training the model
NN.saveWeights(NN)   # saving the weights and bias so they can be accessed later
NN.predict()   #prediction

inputs = torch.tensor(([9, .5, 1]), dtype=torch.float)   # sample of a tensor sent to the model
traced_script_module = torch.jit.trace(NN, inputs)   #tracing it
traced_script_module.save("tutorial.pt")  #saving it
