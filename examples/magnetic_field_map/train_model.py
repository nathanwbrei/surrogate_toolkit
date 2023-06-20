import torch
import numpy as np
import pandas as pd
import sys
from torch_mlp import TorchMLP
from sklearn.model_selection import train_test_split

import matplotlib
# This weirdness lets us generate a plot when we don't have a GUI (e.g. in containers and ssh hosts)
try:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    havedisplay = True
except:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    havedisplay = False


plt.rcParams.update({'font.size':20})

training_data_filename = "training_captures.csv"
if len(sys.argv) > 1:
    training_data_filename = sys.argv[1]
print("Loading training data from '" + training_data_filename + "'")
df = pd.read_csv(training_data_filename)

features = df[['x',' y',' z']].values
targets = df[[' Bx',' By',' Bz']].values
X_train, X_test, Y_train, Y_test = train_test_split(features,targets,test_size=0.1)

print("...done")
print(" ")

print("Setting up MLP...")

# Define a model config:
model_cfg = {
    'n_inputs': 3,
    'n_outputs': 3,
    'architecture': [30,30,30],
    'activations': ['relu']*3,
    'dropouts': [0.0]*3,
    'output_activation': 'linear',
    'n_epochs': 10000,
    'batch_size': 256,
    'n_mon_epoch': 1000,
    'n_read_epoch': 100,
    'learning_rate': 1e-4
}

mfield_mlp = TorchMLP(model_cfg,"cpu")

print("...done")
print(" ")

print("Training MLP...")


loss_dict = mfield_mlp.fit(
    x=torch.as_tensor(X_train,dtype=torch.float32),
    y=torch.as_tensor(Y_train,dtype=torch.float32),
    x_test=torch.as_tensor(X_test,dtype=torch.float32),
    y_test=torch.as_tensor(Y_test,dtype=torch.float32)
)

print("...done")
print(" ")
print("Checking model performance...")

# Check the training history:

figt,axt = plt.subplots(figsize=(10,8))

axt.plot(loss_dict['loss'],linewidth=3.0,label='Training')
axt.plot(loss_dict['val_loss'],linewidth=3.0,label='Validation')
axt.grid(True)
axt.legend(fontsize=15)
axt.set_xlabel('Epochs')
axt.set_ylabel('Loss')


# Check residuals:
with torch.no_grad():
    residuals = targets - mfield_mlp.model(torch.as_tensor(features,dtype=torch.float32)).numpy()

    figr, axr = plt.subplots(1,3,figsize=(15,8),sharey=True)

    axr[0].hist(residuals[:,0],100,histtype='step',color='k',linewidth=3.0)
    axr[0].set_xlabel('Residual Bx')
    axr[0].set_ylabel('Entries')
    axr[0].grid(True)
    axr[0].set_xlim(-0.2,0.2)

    axr[1].hist(residuals[:,1],100,histtype='step',color='k',linewidth=3.0)
    axr[1].set_xlabel('Residual By')
    axr[1].grid(True)
    axr[1].set_xlim(-0.2,0.2)

    axr[2].hist(residuals[:,2],100,histtype='step',color='k',linewidth=3.0)
    axr[2].set_xlabel('Residual Bz')
    axr[2].grid(True)
    axr[2].set_xlim(-0.2,0.2)


plt.show()
plt.savefig('model_performance.png')

print("...done")
print(" ")

print("Storing trained model...")

scripted_model =  torch.jit.script(mfield_mlp.model)
scripted_model.save('gluex_mfield_mlp.pt')

print("...done")
print(" ")



