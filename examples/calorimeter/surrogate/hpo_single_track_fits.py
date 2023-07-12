import torch
import optuna
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
from torch_cvae import CVAE
import joblib

# BASIC SETTINGS:
#/////////////////////////////////////
plt.rcParams.update({'font.size':20})
dev = 'cpu'

particle = "pim"
cut = 1e-3
val_split = 0.25
model_name = 'sampl_cal_surrogate_' + particle + '_hpo'
hpo_results = 'hpo_cvae_' + particle

learning_rate = 1e-4

n_hpo_trials = 5
n_epochs_per_trial = 5000 #30000
batch_size = 1024
mon_epoch = 3000 #3000
read_out_epoch = 100 #300

hpo_cfg = {
    'n_min_neurons':[150,50,20],
    'n_max_neurons':[300,100,50],
    'step_neurons':[50,10,5],
   # 'rec_weight': [20.0,200.0,20.0],
    'latent_dim': [3,12,3]

}
#/////////////////////////////////////

if os.path.exists(model_name) == False:
    os.mkdir(model_name)

print(" ")
print("**********************************")
print("*                                *")
print("*   HPO Single Track Surrogate   *")
print("*                                *")
print("**********************************")
print(" ")

# Translate npy data to torch tensors:
def npy_to_torch(x_input,x_cond,acc_cut=None):
    if acc_cut is not None:
        acc = (x_input[:,0] > acc_cut) & (x_input[:,1] > acc_cut) & (x_cond[:,0] > acc_cut)

        x_input_acc = x_input[acc]
        x_cond_acc = x_cond[acc]

        torch_input = torch.as_tensor(x_input_acc,device=dev,dtype=torch.float)
        torch_cond = torch.as_tensor(x_cond_acc,device=dev,dtype=torch.float)

        return torch_input, torch_cond
    
    torch_input = torch.as_tensor(x_input,device=dev,dtype=torch.float)
    torch_cond = torch.as_tensor(x_cond,device=dev,dtype=torch.float)

    return torch_input, torch_cond


# Define a function that searches for the optimum module parameters
def search_model_parameters(trial,config):
    n_min_neurons = config['n_min_neurons']
    n_max_neurons = config['n_max_neurons']
    step_neurons = config['step_neurons']
    latent_dim_search = config['latent_dim']

    n_layers = len(step_neurons)
    architecture = []
    #+++++++++++++++++++++
    for k in range(n_layers):
        architecture.append(trial.suggest_int(f'n_units_layer{k}',n_min_neurons[k],n_max_neurons[k],step_neurons[k]))
    #+++++++++++++++++++++

    core_activation = trial.suggest_categorical('activation',['relu','elu','leaky_relu','tanh','sigmoid'])
    activations = [core_activation]*n_layers
    latent_dim = trial.suggest_int('latent_dim',latent_dim_search[0],latent_dim_search[1],step=latent_dim_search[2])

    return{
        'architecture':architecture,
        'activations':activations,
        'latent_dim':latent_dim
    }

# Load the data:
#**********************************
print("Load and prepare data...")

mom = np.load(particle + "_mom.npy")
theta = np.load(particle + "_theta.npy")
de_abs = np.load(particle + "_de_abs.npy")
de_gap = np.load(particle + "_de_gap.npy")

in_scaler = MinMaxScaler()
cond_scaler = MinMaxScaler()

cond_data = np.concatenate([
    np.expand_dims(mom,axis=1),
    np.expand_dims(theta,axis=1)
],axis=1)

input_data = np.concatenate([
    np.expand_dims(de_abs,axis=1),
    np.expand_dims(de_gap,axis=1)
],axis=1)

input_data, cond_data = shuffle(input_data,cond_data)

input_data_scaled = in_scaler.fit_transform(input_data)
cond_data_scaled = cond_scaler.fit_transform(cond_data)

x_in_train, x_in_test, x_cond_train, x_cond_test = train_test_split(input_data_scaled,cond_data_scaled,test_size=val_split)

torch_in_train, torch_cond_train = npy_to_torch(x_in_train,x_cond_train,cut)
torch_in_test, torch_cond_test = npy_to_torch(x_in_test,x_cond_test,cut)

print("...done!")
print(" ")
#**********************************

# Collect scalers:
#**********************************
print("Collect scaling factors...")

input_min = in_scaler.data_min_
input_max = in_scaler.data_max_

cond_min = cond_scaler.data_min_
cond_max = cond_scaler.data_max_

scaler_df = pd.DataFrame(
    data = {
       'de_abs_min': [input_min[0]],
       'de_gap_min': [input_min[1]],
       'de_abs_max': [input_max[0]],
       'de_gap_max': [input_max[1]],
       'mom_min': [cond_min[0]],
       'theta_min': [cond_min[1]],
       'mom_max': [cond_max[0]],
       'theta_max': [cond_max[1]]
    }
)

scaler_df.to_csv(model_name+"/feature_scalers.csv")

print("...done!")
print(" ")
#**********************************

#**********************************
print("Set up objective for HPO...")

# Now define the objective
def objective(trial):
    # Get the model parameters for the current trial:
    cvae_cfg = search_model_parameters(trial,hpo_cfg)
    cvae_cfg['learning_rate'] = learning_rate
    cvae_cfg['n_inputs'] = 4
    cvae_cfg['n_outputs'] = 2
    cvae_cfg['output_activation'] = 'sigmoid'
    cvae_cfg['rec_loss_weight'] = 200.0


    # Set up the model:
    cvae = CVAE(cvae_cfg,dev)

    # Run the fit:
    loss_dict = cvae.fit(
       x_input_train=torch_in_train,
       x_cond_train=torch_cond_train,
       n_epochs=n_epochs_per_trial,
       batch_size=batch_size,
       monitor_epoch=n_epochs_per_trial+10,
       read_out_epoch=read_out_epoch,
       x_input_test=torch_in_test,
       x_cond_test=torch_cond_test
    )

    # Get current trial number and store the model:
    current_trial = trial.number
    torch.save(cvae.state_dict(),model_name+'/cvae_trial'+str(current_trial)+'.pt')

    # We use the validation loss as an objective to be minimized:
    return loss_dict['val_loss'][-1]


print("...done!")
print(" ")
#**********************************

#**********************************
print("Run HP search...")

study = optuna.create_study(direction='minimize',study_name=hpo_results)
study.optimize(objective,n_trials=n_hpo_trials)
joblib.dump(study,model_name+'/'+hpo_results+'.pkl')

print("...done!")
print(" ")
#**********************************

    

