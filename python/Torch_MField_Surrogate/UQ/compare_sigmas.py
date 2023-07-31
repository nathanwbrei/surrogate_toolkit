import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})

core_name = 'mfield_ensemble'
ensemble_sizes = [1,5,10,15,20]
uq_calls = [1,5,10,15,20]

fig,ax = plt.subplots(1,2,figsize=(15,8),sharey=True)
ax[0].set_title('Ensemble Analysis')
#+++++++++++++++++++++++
for size in ensemble_sizes:
    current_folder = core_name + '_' + str(size)
    current_std = np.load(current_folder+'/model_std.npy')
   
    ax[0].plot(current_std,markersize=15,marker="o",linewidth=0.0,label='Ne='+str(size))
#+++++++++++++++++++++++

core_name = 'mfield_uq'
ax[1].set_title('MC Dropout Analysis')
#+++++++++++++++++++++++
for size in uq_calls:
    current_folder = core_name + '_' + str(size)
    current_std = np.load(current_folder+'/model_std.npy')
    
    ax[1].plot(current_std,markersize=15,marker="o",linewidth=0.0,label='Nc='+str(size))
#+++++++++++++++++++++++

ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(['Bx','By','Bz'])
ax[0].set_ylabel(r'$\sigma_{model}$')
ax[0].grid(True)
ax[0].legend(fontsize=15)

ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(['Bx','By','Bz'])
ax[1].grid(True)
ax[1].legend(fontsize=15)

plt.show()
