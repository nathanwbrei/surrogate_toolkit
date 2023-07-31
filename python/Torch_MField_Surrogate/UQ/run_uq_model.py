import torch
import optuna
import numpy as np
import mlflow
from urllib.parse import urlparse

import pandas as pd
from torch_mlp_uq import TorchMLPUQ
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import joblib

plt.rcParams.update({'font.size':20})

# Basice settings:
#////////////////////////////////////////////////
# Set up a device:
dev = 'cpu'

# Name of the data:
csv_data = '../orig_bfield_data.csv'

# Define a name for the result folder:
n_model_calls = 20
result_folder = "mfield_uq_" + str(n_model_calls)

# Set the validation split:
val_split = 0.25

# Config for the surrogate
mlp_config = {
    'n_inputs': 3,
    'n_outputs': 3,
    'architecture': [50,50,50],
    'activations': ['leaky_relu']*3,
    'output_activation': 'linear',
    'init_dropout': 0.2,
    'dropout_temp':10.0,
    'dropout_regularizer':1e-3,
    'weight_regularizer':1e-3,
    'learning_rate': 1e-4,
    'n_epochs': 20000,
    'n_mon_epoch': 2000,
    'n_read_epoch': 200,
    'batch_size': 32,
    'n_calls':n_model_calls
}
#////////////////////////////////////////////////

print(" ")
print("***************************************")
print("*                                     *")
print("*   PHASM M-Field Surrogate wtih UQ   *")
print("*                                     *")
print("***************************************")
print(" ")


# Load the data:
#**********************************
print("Load csv-data...")

df = pd.read_csv(csv_data)

features = df[['x',' y',' z']].values
targets = df[[' Bx',' By',' Bz']].values

X_train, X_test, Y_train, Y_test = train_test_split(features,targets,test_size=val_split)

x_train_torch = torch.as_tensor(X_train,dtype=torch.float32,device=dev)
x_test_torch = torch.as_tensor(X_test,dtype=torch.float32,device=dev)
y_train_torch = torch.as_tensor(Y_train,dtype=torch.float32,device=dev)
y_test_torch = torch.as_tensor(Y_test,dtype=torch.float32,device=dev)

print("...done!")
print(" ")
#**********************************

# Create a result folder:
#**********************************
print("Create result folder...")

if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)

print("...done!")
print(" ")
#**********************************

# Set up UQ-model:
#**********************************
print("Set up UQ-model...")

uq_model = TorchMLPUQ(mlp_config,dev)

print("...done!")
print(" ")
#**********************************

# Run the training: 
#**********************************
print("Train surrogate...")

loss_dict = uq_model.fit(x=x_train_torch,y=y_train_torch,x_test=x_test_torch,y_test=y_test_torch)

print("...done!")
print(" ")
#**********************************

# Visualize results:
#**********************************
print("Visualize results...")

# Check losses:
figl,axl = plt.subplots(1,2,figsize=(15,8))
figl.subplots_adjust(wspace=0.5)

axl[0].plot(loss_dict['loss'],linewidth=3.0,label='Training')
axl[0].plot(loss_dict['val_loss'],linewidth=3.0,label='Validation')
axl[0].grid(True)
axl[0].legend(fontsize=15)
axl[0].set_xlabel('Epochs per ' + str(mlp_config['n_read_epoch']))
axl[0].set_ylabel('Loss')

axl[1].plot(loss_dict['mc_dropout'],linewidth=3.0,label='Training')
axl[1].plot(len(loss_dict['mc_dropout'])*[mlp_config['init_dropout']],'k--',linewidth=3.0,label='Init')
axl[1].set_xlabel('Epochs per ' + str(mlp_config['n_read_epoch']))
axl[1].set_ylabel('MC Dropout')
axl[1].grid(True)
axl[1].legend(fontsize=15)

# Get the residuals and uncertainty:
y_pred = uq_model.predict(x_train_torch).detach().cpu().numpy()
y_std = np.std(y_pred,axis=0)

np.save(result_folder+'/model_std.npy',y_std)
np.save(result_folder+'/model_pred.npy',y_pred)

residuals = y_train_torch.detach().cpu().numpy() - y_pred
np.save(result_folder+'/residual_data.npy',residuals)

figr,axr = plt.subplots(1,3,figsize=(18,8),sharey=True)
figr.subplots_adjust(hspace=0.5)

# Residuals:
axr[0].hist(residuals[:,0],200,histtype='step',color='k',linewidth=3.0)
axr[0].grid(True)
axr[0].set_xlabel('Residual ' + r'$B_{X}$')
axr[0].set_ylabel('Entries')

axr[1].hist(residuals[:,1],200,histtype='step',color='k',linewidth=3.0)
axr[1].grid(True)
axr[1].set_xlabel('Residual ' + r'$B_{Y}$')

axr[2].hist(residuals[:,2],200,histtype='step',color='k',linewidth=3.0)
axr[2].grid(True)
axr[2].set_xlabel('Residual ' + r'$B_{Z}$')

# Uncertainties:
figu,axu = plt.subplots(figsize=(12,8))
axu.plot(y_std,'ko',markersize=15)
axu.set_xticks([0,1,2])
axu.set_xticklabels(['Bx','By','Bz'])
axu.set_ylabel(r'$\sigma_{model}$')
axu.grid(True)

figl.savefig(result_folder+'/learning_curves.png')
plt.close(figl)

figu.savefig(result_folder+'/model_sigma.png')
plt.close(figu)

figr.savefig(result_folder+'/residuals.png')
plt.close(figr)


print("...done! Have a great day!")
print(" ")
#**********************************



