import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
from torch_cvae import CVAE

# BASIC SETTINGS:
#/////////////////////////////////////
plt.rcParams.update({'font.size':20})
dev = 'cpu'

particle = "pim"
cut = 1e-3
val_split = 0.25
model_name = 'sampl_cal_surrogate_' + particle + '_v2'

latent_dim = 10
cvae_cfg = {
    'architecture':[100,50,30],
    'activations': ['relu','relu','relu'],
    'latent_dim': latent_dim,
    'n_inputs': 4,
    'n_outputs': 2,
    "output_activation": "sigmoid",
    "learning_rate": 1e-4,
    "rec_loss_weight": 200.0
}

n_epochs = 30000 #30000
batch_size = 1024
mon_epoch = 3000 #3000
read_out_epoch = 300 #300
#/////////////////////////////////////

if os.path.exists(model_name) == False:
    os.mkdir(model_name)

print(" ")
print("******************************")
print("*                            *")
print("*   Single Track Surrogate   *")
print("*                            *")
print("******************************")
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

# Set up the model: 
#**********************************
print("Set up CVAE...")

cvae = CVAE(cvae_cfg,dev)

print("...done!")
print(" ")
#**********************************

# Run the training:
#**********************************
print("Train model...")

loss_dict = cvae.fit(
    x_input_train=torch_in_train,
    x_cond_train=torch_cond_train,
    n_epochs=n_epochs,
    batch_size=batch_size,
    monitor_epoch=mon_epoch,
    read_out_epoch=read_out_epoch,
    x_input_test=torch_in_test,
    x_cond_test=torch_cond_test
)

print("...done!")
print(" ")
#**********************************

# Get the loss curve:
#**********************************
print("Get loss curve(s)...")

figt,axt = plt.subplots(figsize=(12,8))

axt.plot(loss_dict['loss'],linewidth=3.0,label='Training')
axt.plot(loss_dict['val_loss'],linewidth=3.0,label='Validation')
axt.set_xlabel("Epochs per " + str(read_out_epoch))
axt.set_ylabel("Loss")
axt.grid(True)
axt.legend(fontsize=15)

figt.savefig(model_name + "/loss_curve.png")
plt.close(figt)

print("...done!")
print(" ")
#**********************************

# Check the residuals on the validation data: 
#**********************************
print("Determine residuals from validation data...")

rec_data = cvae.reconstruct(torch_in_test,torch_cond_test).detach().numpy()
true_data = torch_in_test.detach().numpy()
rec_residuals = true_data - rec_data

z = torch.normal(mean=0.0,std=1.0,size=(torch_cond_test.size()[0],latent_dim))
gen_data = cvae.generate(z,torch_cond_test).detach().numpy()
gen_residuals = true_data - gen_data

de_abs_res_rec = rec_residuals[:,0]
de_gap_res_rec = rec_residuals[:,1]
de_abs_res_gen = gen_residuals[:,0]
de_gap_res_gen = gen_residuals[:,1]

figr,axr = plt.subplots(2,2,figsize=(17,8),sharey=True)
figr.suptitle("Sampling Calorimeter Surrogate Performance")
figr.subplots_adjust(hspace=0.5)

h_range = [0.0,0.7]
axr[0,0].hist(true_data[:,0],100,histtype='step',color='k',linewidth=3.0,range=h_range,log=True,label='Truth')
axr[0,0].hist(rec_data[:,0],100,histtype='step',color='r',linewidth=3.0,range=h_range,log=True,label='Reconstructed')
axr[0,0].hist(gen_data[:,0],100,histtype='step',color='b',linewidth=3.0,range=h_range,log=True,label='Generated')
axr[0,0].set_xlabel("E(absorber) [a.u.]")
axr[0,0].set_ylabel('Entries')
axr[0,0].grid(True)
axr[0,0].legend(fontsize=15)

axr[0,1].hist(true_data[:,1],100,histtype='step',color='k',linewidth=3.0,range=h_range,log=True,label='Truth')
axr[0,1].hist(rec_data[:,1],100,histtype='step',color='r',linewidth=3.0,range=h_range,log=True,label='Reconstructed')
axr[0,1].hist(gen_data[:,1],100,histtype='step',color='b',linewidth=3.0,range=h_range,log=True,label='Generated')
axr[0,1].set_xlabel("E(detection layer) [a.u.]")
axr[0,1].grid(True)
axr[0,1].legend(fontsize=15)

axr[1,0].hist(de_abs_res_rec,100,range=[-0.25,0.25],histtype='step',color='r',label='Reconstructed',linewidth=3.0)
axr[1,0].hist(de_abs_res_gen,100,range=[-0.25,0.25],histtype='step',color='b',label='Generated',linewidth=3.0)
axr[1,0].set_xlabel('Residual E(absorber)')
axr[1,0].set_ylabel('Entries')
axr[1,0].grid(True)
axr[1,0].legend(fontsize=15)

axr[1,1].hist(de_gap_res_rec,100,range=[-0.25,0.25],histtype='step',color='r',label='Reconstructed',linewidth=3.0)
axr[1,1].hist(de_gap_res_gen,100,range=[-0.25,0.25],histtype='step',color='b',label='Generated',linewidth=3.0)
axr[1,1].set_xlabel('Residual E(detection layer)')
axr[1,1].grid(True)
axr[1,1].legend(fontsize=15)

figr.savefig(model_name + "/predictions_and_residuals.png")
plt.close(figt)

print("...done!")
print(" ")
#**********************************

# Data set comparison:
#**********************************
print("Inpsect generated data...")

true_de = in_scaler.inverse_transform(true_data)
true_kin = cond_scaler.inverse_transform(torch_cond_test.numpy())

gen_de = in_scaler.inverse_transform(gen_data)

figcm,axcm = plt.subplots(2,2,figsize=(17,8),sharey=True)
figcm.subplots_adjust(hspace=0.5)
figcm.suptitle('Energy Deposit in absorbtion Layer')

axcm[0,0].hist2d(true_kin[:,0],true_de[:,0],100,norm=LogNorm())
axcm[0,1].hist2d(true_kin[:,1],true_de[:,0],100,norm=LogNorm())
axcm[0,0].set_xlabel('Momentum p [MeV]')
axcm[0,0].set_ylabel(r'$E_{abs}$' + '(True) [MeV]')
axcm[0,1].set_xlabel(r'$\theta$' + ' [deg]')
axcm[0,0].grid(True)
axcm[0,1].grid(True)

axcm[1,0].hist2d(true_kin[:,0],gen_de[:,0],100,norm=LogNorm())
axcm[1,1].hist2d(true_kin[:,1],gen_de[:,0],100,norm=LogNorm())
axcm[1,0].set_ylabel(r'$E_{abs}$' + '(Model) [MeV]')
axcm[1,0].set_xlabel('Momentum p [MeV]')
axcm[1,1].set_xlabel(r'$\theta$' + ' [deg]')
axcm[1,0].grid(True)
axcm[1,1].grid(True)

figcm.savefig(model_name + "/E_abs_vs_mom_theta.png")
plt.close(figt)

figct,axct = plt.subplots(2,2,figsize=(17,8),sharey=True)
figct.subplots_adjust(hspace=0.5)
figct.suptitle('Energy Deposit in gap detection Layer')

axct[0,0].hist2d(true_kin[:,0],true_de[:,1],100,norm=LogNorm())
axct[0,1].hist2d(true_kin[:,1],true_de[:,1],100,norm=LogNorm())
axct[0,0].set_xlabel('Momentum p [MeV]')
axct[0,0].set_ylabel(r'$E_{gap}$' + '(True) [MeV]')
axct[0,1].set_xlabel(r'$\theta$' + ' [deg]')
axct[0,0].grid(True)
axct[0,1].grid(True)


axct[1,0].hist2d(true_kin[:,0],gen_de[:,1],100,norm=LogNorm())
axct[1,1].hist2d(true_kin[:,1],gen_de[:,1],100,norm=LogNorm())
axct[1,0].set_xlabel('Momentum p [MeV]')
axct[1,0].set_ylabel(r'$E_{gap}$' + '(Model) [MeV]')
axct[1,1].set_xlabel(r'$\theta$' + ' [deg]')
axct[1,0].grid(True)
axct[1,1].grid(True)

figct.savefig(model_name + "/E_gap_vs_mom_theta.png")
plt.close(figt)

print("...done!")
print(" ")
#**********************************

# Store the model:
#**********************************
print("Store model...")

torch.save(cvae.state_dict(),model_name + "/surrogate_model.pt")

print("...done!")
print(" ")
#**********************************