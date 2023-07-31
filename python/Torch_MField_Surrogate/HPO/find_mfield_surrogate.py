import torch
import optuna
import numpy as np
import mlflow
from urllib.parse import urlparse

import pandas as pd
from torch_mlp import TorchMLP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import joblib

plt.rcParams.update({'font.size':20})

# Basice settings:
#////////////////////////////////////////////////
# Set up a device:
dev = 'cpu'

# MLFlow settings:
mlflow_experiment_name = "mfield_hpo_search_v5"
mflow_tracking_uri = None

# Name of the data:
csv_data = '../orig_bfield_data.csv'

# Set the number of trials:
n_hpo_trials = 30

# Decide whether to use the mean chisquare as objective value or not:
use_mean_chisquare_objective = True

# Folder where the hp search results are stored:
hp_results = 'mfield_hp_search_results_v5'

# Set the validation split:
val_split = 0.1

# Set up a config for the HP search and the MLP training:
hpo_config = {
    'model_name': 'test_mfield_hpo_v5',
   'n_epochs': 5000,
   'n_mon_epoch': 5001,
   'n_read_epoch': 2,
   'batch_size': 256,
   'learning_rate': 1e-4,
   'n_max_layers': 4,
   'step_layers': 1,
   'n_min_neurons': 10,
   'n_max_neurons': 100,
   'step_neurons': 10,
   'min_dropout': 0.0,
   'max_dropout': 0.2,
   'step_dropout':0.02,
   'n_inputs': 3,
   'n_outputs': 3
}
#////////////////////////////////////////////////

print(" ")
print("***************************************")
print("*                                     *")
print("*   HPO for PHASM M-Field Surrogate   *")
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

# Create directionary to store the results:
#**********************************
print("Create result folder...")

if os.path.exists(hpo_config['model_name']) == False:
    os.mkdir(hpo_config['model_name'])

print("...done!")
print(" ")
#**********************************

# Prepare ML-flow:
#**********************************
print("Prepare mlflow and track important settings...")

mlflow.set_experiment(mlflow_experiment_name)
mlflow_experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)

print("...done!")
print(" ")
#**********************************


# Define the main componenets for the HPO:
#**********************************
print("Set up HP search components...")

# i) Probe a few network hyper parameters:
def probe_hp(trial,config):
    # Get the number of hidden layers first:
    n_hidden_layers = trial.suggest_int('n_hidden_layers',1,config['n_max_layers'],config['step_layers'])
    mlflow.log_param('n_hidden_layers',n_hidden_layers)


    n_min_neurons = config['n_min_neurons']
    n_max_neurons = config['n_max_neurons']
    step_neurons = config['step_neurons']
    
    # Hidden layers
    architecture = []
    #+++++++++++++++++++++
    for k in range(n_hidden_layers):
        par_name = 'n_units_layer' + str(k)
        number_neurons = trial.suggest_int(par_name,n_min_neurons,n_max_neurons,step_neurons)
        architecture.append(number_neurons)

        # Track it via mlflow:
        mlflow.log_param(par_name,number_neurons)
    #+++++++++++++++++++++
    
    # Activation functions:
    core_activation = trial.suggest_categorical('activation',['relu','elu','leaky_relu','tanh','sigmoid'])
    activations = [core_activation]*n_hidden_layers
    mlflow.log_param('core_activation',core_activation)

    # Dropouts:
    core_dropout = trial.suggest_float(name='dropout',low=config['min_dropout'],high=config['max_dropout'],step=config['step_dropout'])
    dropouts = [core_dropout]*n_hidden_layers
    mlflow.log_param('core_dropout',core_dropout)

    # Output activation, here we limit the search space a bit:
    output_activation = trial.suggest_categorical('output_activation',['relu','elu','leaky_relu'])
    mlflow.log_param('output_activation',output_activation)
    
    # We copy a bunch of parameters from the input config, so that we can simply pass them to the network class:
    return{
        'architecture':architecture,
        'activations':activations,
        'dropouts': dropouts,
        'output_activation': output_activation,
        'n_epochs': config['n_epochs'],
        'batch_size': config['batch_size'],
        'n_mon_epoch': config['n_mon_epoch'],
        'n_read_epoch': config['n_read_epoch'],
        'learning_rate': config['learning_rate'],
        'model_name': config['model_name'],
        'n_inputs': config['n_inputs'],
        'n_outputs': config['n_outputs']
    }

#---------------------------------

# ii) Formulate the objective function --> This is what we are going to minimize:
def objective(trial):
    # Get current trial number and store the model:
    current_trial = trial.number
    current_run_name = "mfield_hpo_trial_" + str(current_trial)

    with mlflow.start_run(experiment_id=mlflow_experiment.experiment_id,run_name=current_run_name): 
       # Log a few important settings:
       mlflow.log_param('number_hpo_trials',n_hpo_trials)
       mlflow.log_param('validation_split',val_split)
       mlflow.log_param('n_epochs',hpo_config['n_epochs'])
       mlflow.log_param('batch_size',hpo_config['batch_size'])
       mlflow.log_param('learning_rate',hpo_config['learning_rate'])

       # Get the MLP parameters first:
       mlp_config = probe_hp(trial,hpo_config) 
    
       # Set up the MLP:
       current_mlp = TorchMLP(mlp_config,dev)
    
       # Run the training (just for a few epochs)
       loss_dict = current_mlp.fit(
          x=x_train_torch,
          y=y_train_torch,
          x_test=x_test_torch,
          y_test=y_test_torch
       )
    
       torch.save(current_mlp.state_dict(),mlp_config['model_name']+'/mfield_surrogate_trial'+str(current_trial)+'.pt')

       # We use the validation loss as an objective to be minimized:
       validation_loss = loss_dict['val_loss'][-1]
       mlflow.log_metric('objective_score',validation_loss)

       # Compute an additional metric that we might want to use instead of the validation MSE:
       # Get prediction on test data:
       y_test_pred = current_mlp.forward(x_test_torch).to(dev)
       # Formulate residual:
       test_residual = y_test_torch - y_test_pred
       # Get the prediction error:
       err_test_residual = torch.std(test_residual,dim=0)
       # Determine chi-square:
       arg = torch.square(test_residual) / err_test_residual
       mean_chisquare = torch.mean(torch.sum(arg,dim=1)).detach().cpu().item()
       mlflow.log_metric('mean_chisqaure',mean_chisquare)

       # And finally store the model:
       tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

       # Model registry does not work with file store
       if tracking_url_type_store != "file":

            mlflow.pytorch.log_model(current_mlp.model, "model", registered_model_name='mfield_surrogate_trial'+str(current_trial))
       else:
            mlflow.pytorch.log_model(current_mlp.model, "model")

       if use_mean_chisquare_objective == True:
           mlflow.log_param('use_mean_chisquare',1)
           return mean_chisquare
       
       mlflow.log_param('use_mean_chisquare',0)
       return validation_loss

print("...done!")
print(" ")
#**********************************

# Run the actual search:
#**********************************
print("Run HP search...")

study = optuna.create_study(direction='minimize',study_name=hp_results)
study.optimize(objective,n_trials=n_hpo_trials)
joblib.dump(study,hpo_config['model_name']+'/'+hp_results +'.pkl')

print("...done!")
print(" ")
#**********************************
