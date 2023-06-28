import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import csv

# read the data
def get_device():
    device = 'cpu'
    return device
device = get_device()
df_all = pd.read_csv('magfieldmap3.csv', dtype=float).dropna()
print(df_all)

df_test = df_all.iloc[0:100, :]
df_train = df_all.iloc[100:, :]

# seperate labels from data
train_labels = df_train.iloc[:, 3:] #Bx, By, Bz
train_images = df_train.iloc[:, 0:3] #x, y, z
test_labels = df_test.iloc[:, 3:]
test_images = df_test.iloc[:, 0:3]




train_images = torch.from_numpy(train_images.to_numpy()).float()
train_labels = torch.squeeze(torch.from_numpy(train_labels.to_numpy()).float())
test_images = torch.from_numpy(test_images.to_numpy()).float()
test_labels = torch.squeeze(torch.from_numpy(test_labels.to_numpy()).float())



#graphing

#z, bz
z = np.array(df_all[' z'])
bz = np.array(df_all[' Bz'])

plt.scatter(z, bz, c='r')

plt.xlabel('z')
plt.ylabel('bz')
plt.title('graph')
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.grid(True)
plt.ylim(bottom=-1.0)


plt.show()

#x, bx
x = np.array(df_all['x'])
bx = np.array(df_all[' Bx'])

plt.scatter(x, bx, c='g')

plt.xlabel('x')
plt.ylabel('bx')
plt.title('graph')
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.ylim(bottom=-1.0)
plt.grid(True)
plt.show()

#y, by
y = np.array(df_all[' y'])
by = np.array(df_all[' By'])

plt.scatter(y, by, c='b')

plt.xlabel('y')
plt.ylabel('by')
plt.title('graph')
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.ylim(bottom=-1.0)
plt.grid(True)
plt.show()



class Mag_net(nn.Module):

  #2 hidden layers
  

    def __init__(self,input_features=3,hidden1=6, hidden2=10,out_features=3):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,out_features)

        

    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x





model = Mag_net()
# loss function (Mean Squared Error Loss)
loss_function = nn.MSELoss()   
# optimization
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)  
ec = []
l = []
x = 0
epochs=250

final_losses=[]
#training
for i in range(epochs):

    i= i+1
   
    y_pred=model.forward(train_images)

    loss=loss_function(y_pred,train_labels)

    final_losses.append(loss)
    x+=1
    ec.append(x)
    l.append(loss.item())
   
    #loss printed every 10 epochs
    if i % 10 == 0:
      



        print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



x = ec
bx = l

plt.plot(x, bx, c='r')

plt.xlabel('x')
plt.ylabel('bx')
plt.title('LOSS over 200 epochs')
plt.rcParams["figure.figsize"] = [7.00, 7.00]

plt.grid(True)
plt.show()

def predict(data):
    predict_data = data
  
    #converting to float
    if type(predict_data[0]) == int:
      predict_data[0] = float(predict_data[0])
    if type(predict_data[1]) == int:
      predict_data[1] = float(predict_data[1])
    if type(predict_data[2]) == int:
      predict_data[2] = float(predict_data[2])
    #printing input values
    print("INPUT")
    print()
    print("x: " + str(predict_data[0]))
    print("y: " + str(predict_data[1]))
    print("z: " + str(predict_data[2]))
    print()
    #convert to tensor
    predict_data_tensor = torch.tensor(predict_data)

    prediction_value    = model(predict_data_tensor)
    
    #converting to np array
    prediction = prediction_value.cpu().detach().numpy()
    #printing the prediction
    print("PREDICTED OUTPUT")
    print()
    print("Bx: " + str(prediction[0]))
    print("By:" + str(prediction[1])) 
    print("Bz: " + str(prediction[2]) ) 

    return prediction_value

input = [8.38032, 3.27613, 129.795]
predict(input)
z2 = []
bz2 = []
for i in test_images:
  out = predict(i)
  input = i.cpu().detach().numpy()
  bzp = out.cpu().detach().numpy()
  z2.append(input[2])
  bz2.append(bzp[2])

v1 = z2
v2 = bz2
test_la2 = df_test.iloc[:, 3:]
test_im2 = df_test.iloc[:, 0:3]


z3 = np.array(test_im2[' z'])
bz3 = np.array(test_la2[' Bz'])

plt.plot(z3, bz3, c='r', label = "actual")
plt.plot(v1, v2, c='g', label = "predicted")
plt.legend(loc="upper left")

plt.xlabel('z')
plt.ylabel('bz')
plt.title('Actual vs Predicted Z and Bz through model')
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.grid(True)
plt.ylim(bottom=0.8)
plt.ylim(top=2.2)
