import torch
import optuna
import numpy as np
from torch_cvae import CVAE
import joblib
import matplotlib.pyplot as plt

# BASIC SETTINGS:
#/////////////////////////////////////
plt.rcParams.update({'font.size':20})

dev = 'cpu'

particle = "pim"
model_name = 'sampl_cal_surrogate_' + particle + '_hpo'
hpo_results = 'hpo_cvae_' + particle

learning_rate = 1e-4
cvae_cfg = {
    'rec_loss_weight': 200.0,
    'n_inputs': 4,
    'n_outputs': 2,
    'learning_rate': learning_rate,
    'output_activation': 'sigmoid',
    
}
#/////////////////////////////////////


study = joblib.load(model_name+'/'+hpo_results+'.pkl')

# optuna.visualization.matplotlib.plot_optimization_history(study)

# optuna.visualization.matplotlib.plot_contour(study, params=["n_units_layer0", "n_units_layer1"])

# #optuna.visualization.matplotlib.plot_edf([study])

# optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["n_units_layer0", "n_units_layer1","n_units_layer2"])

# optuna.visualization.matplotlib.plot_param_importances(study)

# plt.tight_layout()
# plt.show()

# Get the 'best' trial:
trial = study.best_trial

# Get the trial number:
trial_number = trial.number




print("  Value:  ")
print(trial.value)

print("  Params: ")
for key, value in trial.params.items():
        print("    {}: {}".format(key, value))