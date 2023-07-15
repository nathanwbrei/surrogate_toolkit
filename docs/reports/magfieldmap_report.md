# Magnetic Field Mapping
Governor's School JLab Summer Internship

Hari Gopal

Last Updated: 7/23/22

## Objective

The objective of this project was to create an accurate and efficient neural network trained on the data from the Magnetic Field Map. It should take in 3 inputs x, y, and z, and should be able to predict Bx, By, and Bz.


## Methodology
I learned about what neural networks are and wrote my first neural networks are. After practicing writing basic neural networks, I started understanding how to use .csv files with neural networks. I had a few challenges with getting neural networks to work properly with .csv files.  I then looked into the magfieldmap3.csv file and removed any errors. I learned how to load this file into the model using pandas. I then started writing and adjusting the model until it worked and trained successfully. I wrote a predict function that takes 3 inputs and prints the 3 outputs from the model. I learned about matplot lib and used the predict function to graph the actual vs predicted values for z and Bz. I also graphed the loss over 250 epochs. I adjusted the layers, learning rate, and number of epochs until I got the network to be much more accurate. 

## Code Summary
1. Understanding and Loading the data:
The data consists of 3 inputs (x, y, z) and 3 outputs (Bx, By, Bz).
The field faces in the z direction.
Separate the .csv file into training and testing data using pandas.
Separate the .csv file into inputs and outputs(labels, images).
2. Creating the Neural Network:
The neural network has 2 hidden layers.
3 input features, 3 output features.
Has a forward function.
Uses the ReLu activation function.
3. Training the Model:
Model uses MSELoss (Mean Squared Error Loss).
The model is optimized. 
Training loop that trains the model through 250 epochs.
Graphing the loss over 250 epochs using Matplotlib.
4. Prediction
Predict function that runs 3 inputs through the model and prints the 3 predicted outputs.
Graphing the actual vs. predicted values for z with respect to Bz.






 


## Results
I was able to create an accurate neural network that is able to predict Bx, By, and Bz values given x, y , and z outputs. I graphed the loss over 250 epochs and graphed the actual vs. predicted values for z with respect to Bz. During this internship I learned about neural networks, and how to write them, as well as learning how to use pandas and matplotlib. 


## Future Steps
Tuning the accuracy of the model to be much more accurate.
Making models trained on different magnetic field maps. 
Using PHASM to implement the model in C++
