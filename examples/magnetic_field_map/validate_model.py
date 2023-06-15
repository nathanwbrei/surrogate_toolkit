import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv > 1):
    training_data_filename = sys.argv[1]
else:
    training_data_filename = "training_captures.csv"
print("Loading training data from '" + training_data_filename + "'")

if len(sys.argv > 2):
    validation_data_filename = sys.argv[2]
else:
    validation_data_filename = "validation_captures.csv"
print("Loading validation data from '" + training_data_filename + "'")

if !os.path.exists(validation_data_filename):
    print("Could not find validation data. Re-generating validation data from model instead")
    generate_validation_data = True
else:
    generate_validation_data = False


# Load the training data from the CSV file
training_data = pd.read_csv(training_data_filename)
features = training_data[['x',' y',' z']].values
targets = training_data[[' Bx',' By',' Bz']].values


if generate_validation_data:
    # Load the model
    device = torch.device("cpu")
    model = torch.load("gluex_mfield_mlp.pt", map_location=device)
    model.eval()

    # Print the dimensions of each layer
    for name, param in model.named_parameters():
        print(f"Layer: {name}\tShape: {param.shape}")

    with torch.no_grad():
        predictions = model(torch.as_tensor(features, dtype=torch.float32)).numpy()
        residuals = targets - predictions

else:
    validation_data = pd.read_csv(training_data_filename)
    predictions = validation_data[[' Bx',' By',' Bz']].values
    residuals = targets - predictions

mse_per_row = np.mean(residuals ** 2, axis=1)

# Calculate the summation of MSE values
mse_sum = np.sum(mse_per_row)

# Print the summation of MSE values
print("MSE loss:", mse_sum/len(residuals))

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
plt.save("residuals.pdf")


