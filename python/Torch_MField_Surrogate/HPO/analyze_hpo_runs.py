import torch
import optuna
import numpy as np
import joblib
from torch_mlp import TorchMLP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# BASIC SETTINGS:
#/////////////////////////////////////
plt.rcParams.update({'font.size':20})

# Define the number of 'best' model that we wish to inspect / retrain:
n_best_models = 5

# Device
dev = 'cpu'

# Name of the data:
csv_data = 'orig_bfield_data.csv'

# Set the validation split:
val_split = 0.1

# Set names of the result folders
hp_results = 'test_mfield_hpo'
hp_study_name = 'mfield_hp_search_results'

# Set up a basic config which will be completed with the parameters found during the hp search:
mlp_config = {
    'n_inputs': 3,
    'n_outputs': 3,
    'n_epochs': 5000,
    'batch_size': 256,
    'n_mon_epoch': 500,
    'n_read_epoch': 5,
    'learning_rate': 1e-4
}
#/////////////////////////////////////

print(" ")
print("***********************************************")
print("*                                             *")
print("*   PHASM M-Field Surrogate: HPO Evaluation   *")
print("*                                             *")
print("***********************************************")
print(" ")

# Add a result folder where the final training results are stored:
result_folder = hp_results + '/final_training_results'
if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)

# Add a folder where the final models are stored:
fin_model_folder = hp_results + '/retrained_models'
if os.path.exists(fin_model_folder) == False:
    os.mkdir(fin_model_folder)

# Load the HPO results
#**********************************
print("Load HP search results...")

study = joblib.load(hp_results+'/'+hp_study_name+'.pkl')

print("...done!")
print(" ")
#**********************************

# Formulate function to load the best model parameters:
#**********************************
print("Set up functions for HP result retrieval...")

# Get model params from HP results:
def get_model_prams_from_hp_search(hp_params,init_dict):
    model_dict = init_dict.copy()

    n_layers = hp_params['n_hidden_layers']
    activations = [hp_params['activation']]*n_layers
    dropouts = [hp_params['dropout']]*n_layers
    
    architecture = []
    #++++++++++++++++++++++
    for h in range(n_layers):
        key = 'n_units_layer' + str(h)
        architecture.append(hp_params[key])
    #+++++++++++++++++++++
    
    model_dict['activations'] = activations
    model_dict['architecture'] = architecture
    model_dict['dropouts'] = dropouts
    model_dict['output_activation'] = hp_params['output_activation']

    return model_dict
    
#--------------------------------

# Get MLP model from HP results:
def retrieve_models_from_hp_search(hp_study):
    all_trials = hp_study.trials
    n_trials = len(all_trials)

    all_values = [t.value for t in all_trials]
    best_values = sorted(all_values)

    assert n_best_models < n_trials, f"ERROR: Number of best model {n_best_models} is not smaller than the number of all available trials {n_trials}"

    mlp_models = []
    mlp_idx = []
    opt_value = -1
    # We are checking for the n best models and the worst model:
    #++++++++++++++++++++++++
    for i in range(n_best_models+1):

        if i < n_best_models:
           opt_value = best_values[i]
        else:
            opt_value = best_values[-1]

        idx = all_values.index(opt_value)
        
        # Get the parameters, associated with the 'best' values:
        current_params = all_trials[idx].params
        # Get the model configuration:
        current_config = get_model_prams_from_hp_search(current_params,mlp_config)
        # And set up the mlp:
        current_mlp = TorchMLP(current_config,dev)
        
        # Now load the state dictionary:
        current_state_dict = torch.load(hp_results+'/mfield_surrogate_trial'+str(idx)+'.pt')
        # Set the model weights:
        current_mlp.load_state_dict(current_state_dict)

        # Register a new optimizer for each model, so that we ensure that the optimizer starts with the proper weights:
        current_mlp.optimizer = torch.optim.Adam(current_mlp.parameters(),lr=mlp_config['learning_rate'])
        
        # And finally register the model:
        mlp_models.append(current_mlp)
        mlp_idx.append(idx)
    #++++++++++++++++++++++++

    return mlp_models, mlp_idx


print("...done!")
print(" ")
#**********************************

# Load the data (again...)
#**********************************
print("Load csv-data...")

df = pd.read_csv(csv_data)

features = df[['x',' y',' z']].values
targets = df[[' Bx',' By',' Bz']].values

X_train, X_test, Y_train, Y_test = train_test_split(features,targets,test_size=val_split)

print("...done!")
print(" ")
#**********************************

# Load the best and worst model:
#**********************************
print("Load " + str(n_best_models) + " best and worst model from HP search...")

best_mlp_models, model_indices = retrieve_models_from_hp_search(study)

print("...done!")
print(" ")
#**********************************

# Do the continued training:
#**********************************
residuals_means = np.zeros((n_best_models+1,3))
residuals_sigmas = np.zeros((n_best_models+1,3))

#+++++++++++++++++++++++++++
for h in range(n_best_models+1):
    current_idx = model_indices[h]

    print("Continue training for model: " + str(current_idx))

    current_loss_dict = best_mlp_models[h].fit(
       x=torch.as_tensor(X_train,dtype=torch.float32,device=dev),
       y=torch.as_tensor(Y_train,dtype=torch.float32,device=dev),
       x_test=torch.as_tensor(X_test,dtype=torch.float32,device=dev),
       y_test=torch.as_tensor(Y_test,dtype=torch.float32,device=dev)
    )
    
    # Plot the loss distributions:
    figt,axt = plt.subplots(figsize=(10,8))

    axt.plot(current_loss_dict['loss'],linewidth=3.0,label='Training')
    axt.plot(current_loss_dict['val_loss'],linewidth=3.0,label='Validation')
    axt.grid(True)
    axt.legend(fontsize=15)
    axt.set_xlabel('Epochs')
    axt.set_ylabel('Loss')  

    figt.savefig(result_folder+'/loss_curves_model_' + str(current_idx) + ".png")
    plt.close(figt)

    # Plot the residuals:
    with torch.no_grad():
        residuals = targets - best_mlp_models[h].model(torch.as_tensor(features,dtype=torch.float32)).numpy()

        m_residuals = np.mean(residuals,axis=0)
        std_residuals = np.std(residuals,axis=0)

        residuals_means[h,0] = m_residuals[0]
        residuals_sigmas[h,0] = std_residuals[0]

        residuals_means[h,1] = m_residuals[1]
        residuals_sigmas[h,1] = std_residuals[1]

        residuals_means[h,2] = m_residuals[2]
        residuals_sigmas[h,2] = std_residuals[2]

        # Plot residuals:
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

        figr.savefig(result_folder+'/residuals_model_' + str(current_idx) + ".png")
        plt.close(figr)

        # Normalize the residuals wr.t. to the width:
        norm_residuals = residuals / std_residuals

        fignr, axnr = plt.subplots(1,3,figsize=(15,8),sharey=True)

        axnr[0].hist(norm_residuals[:,0],100,histtype='step',color='k',linewidth=3.0)
        axnr[0].set_xlabel('Normalized Residual Bx')
        axnr[0].set_ylabel('Entries')
        axnr[0].grid(True)
        axnr[0].set_xlim(-3.0,3.0)

        axnr[1].hist(norm_residuals[:,1],100,histtype='step',color='k',linewidth=3.0)
        axnr[1].set_xlabel('Normalized Residual By')
        axnr[1].grid(True)
        axnr[1].set_xlim(-3.0,3.0)

        axnr[2].hist(norm_residuals[:,2],100,histtype='step',color='k',linewidth=3.0)
        axnr[2].set_xlabel('Normalized Residual Bz')
        axnr[2].grid(True)
        axnr[2].set_xlim(-3.0,3.0)

        fignr.savefig(result_folder+'/normalized_residuals_model_' + str(current_idx) + ".png")
        plt.close(fignr)

    # Write the current model to file:
    scripted_model = torch.jit.script(best_mlp_models[h].model)
    scripted_model.save(fin_model_folder+'/cont_trained_mfield_surrogate_' + str(current_idx) + '.pt')
#+++++++++++++++++++++++++++

# Create a nice plot that compares the individual performance
figc,axc = plt.subplots(1,3,figsize=(18,8),sharey=True)

axc[0].errorbar(x=np.array(model_indices),y=residuals_means[:,0],yerr=residuals_sigmas[:,0],fmt='ko',capsize=10,markersize=10)
axc[0].set_xlabel('Model Idx')
axc[0].set_xticks(np.array(model_indices))
axc[0].set_ylabel('Residual: ' + r'$\mu\pm\sigma$')
axc[0].grid(True)
axc[0].set_title('Bx')

axc[1].errorbar(x=np.array(model_indices),y=residuals_means[:,1],yerr=residuals_sigmas[:,1],fmt='ko',capsize=10,markersize=10)
axc[1].set_xlabel('Model Idx')
axc[1].set_xticks(np.array(model_indices))
axc[1].grid(True)
axc[1].set_title('By')

axc[2].errorbar(x=np.array(model_indices),y=residuals_means[:,2],yerr=residuals_sigmas[:,2],fmt='ko',capsize=10,markersize=10)
axc[2].set_xlabel('Model Idx')
axc[2].set_xticks(np.array(model_indices))
axc[2].grid(True)
axc[2].set_title('Bz')

figc.savefig(result_folder+'/residual_comparison.png')
plt.close(figc)

# Decouple mean and sigma:
figm,axm = plt.subplots(1,2,figsize=(18,8))
figm.subplots_adjust(wspace=0.5)

axm[0].plot(np.array(model_indices),residuals_means[:,0],'ko',markersize=10,label='Bx')
axm[0].plot(np.array(model_indices),residuals_means[:,1],'rs',markersize=10,label='By')
axm[0].plot(np.array(model_indices),residuals_means[:,2],'bd',markersize=10,label='Bd')

axm[0].set_xlabel('Model Idx')
axm[0].set_xticks(np.array(model_indices))
axm[0].set_ylabel('Residual Mean ' + r'$\mu$')
axm[0].grid(True)
axm[0].legend()

axm[1].plot(np.array(model_indices),residuals_sigmas[:,0],'ko',markersize=10,label='Bx')
axm[1].plot(np.array(model_indices),residuals_sigmas[:,1],'rs',markersize=10,label='By')
axm[1].plot(np.array(model_indices),residuals_sigmas[:,2],'bd',markersize=10,label='Bd')

axm[1].set_xlabel('Model Idx')
axm[1].set_xticks(np.array(model_indices))
axm[1].set_ylabel('Residual Wdith ' + r'$\sigma$')
axm[1].grid(True)
axm[1].legend()

figm.savefig(result_folder+'/residual_properties.png')
plt.close(figm)

print(" ")
#**********************************

# Fianlly, retrieve performance plots from HP search
#**********************************
print("Get performance plots from HP search...")

# Optimization history:
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.gcf().set_size_inches(20,7)
plt.savefig(hp_results + '/hpo_history.png')
plt.close()

# Parameter importance:
optuna.visualization.matplotlib.plot_param_importances(study)
plt.gcf().set_size_inches(15,7)
plt.savefig(hp_results + '/hpo_param_importance.png')
plt.close()

print("...done!")
print(" ")
print("You are all done! Have a great day!")
print(" ")
#**********************************
















