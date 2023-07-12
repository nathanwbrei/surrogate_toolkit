import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import MeanMetric
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd


class CVAE(nn.Module):

    # Initialize:
    #**********************************************
    def __init__(self,config,devices):
        super(CVAE, self).__init__()

        self.devices = devices
        self.n_inputs = config['n_inputs']
        self.n_output = config['n_outputs']
        self.architecture = config['architecture']
        self.activations = config['activations']
        self.output_activation = config['output_activation']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.rec_loss_weight = config['rec_loss_weight']

        self.n_layers = len(self.architecture)
        self.encoder = nn.Sequential(self.set_encoder()).to(self.devices)
        self.decoder = nn.Sequential(self.set_decoder()).to(self.devices)

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()),lr=self.learning_rate)
        self.rec_loss_fn = nn.CrossEntropyLoss(reduction='none')
    #**********************************************
    
    # Get the activation function:
    #**********************************************
    def get_activation_function(self,act_str):
        if act_str.lower() == "relu":
            return nn.ReLU()
        
        if act_str.lower() == "elu":
            return nn.ELU()
        
        if act_str.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2)
        
        if act_str.lower() == "tanh":
            return nn.Tanh()
        
        if act_str.lower() == "sigmoid":
            return nn.Sigmoid()
        
        if act_str.lower() == "softmax":
            return nn.Softmax()
    #**********************************************

    # Encoder:
    #**********************************************
    def set_encoder(self):
        encoder_layers = OrderedDict()

        n_prev_nodes = self.n_inputs
        #++++++++++++++++++++++++++++++
        for j in range(self.n_layers):
            layer_name = 'enc_layer_' + str(j)
            encoder_layers[layer_name] = nn.Linear(n_prev_nodes,self.architecture[j])
            encoder_layers[layer_name+'_act'] = self.get_activation_function(self.activations[j])

            n_prev_nodes = self.architecture[j]
        #++++++++++++++++++++++++++++++

        encoder_layers['latent_layer'] = nn.Linear(n_prev_nodes,self.latent_dim*2)
        return encoder_layers
    #**********************************************

    # Decoder:
    #**********************************************
    def set_decoder(self):
        decoder_layers = OrderedDict()

        n_prev_nodes = self.latent_dim + (self.n_inputs - self.n_output)
        #++++++++++++++++++++++++++++++
        for j in range(self.n_layers):
            layer_name = 'dec_layer_' + str(j)
            decoder_layers[layer_name] = nn.Linear(n_prev_nodes,self.architecture[::-1][j])
            decoder_layers[layer_name+'_act'] = self.get_activation_function(self.activations[::-1][j])

            n_prev_nodes = self.architecture[::-1][j]
        #++++++++++++++++++++++++++++++

        decoder_layers['output'] = nn.Linear(n_prev_nodes,self.n_output)
        if self.output_activation != "" and self.output_activation != "linear":
            decoder_layers['output_activation'] = self.get_activation_function(self.output_activation)

        
        return decoder_layers
    #**********************************************

    # Model responses:
    #**********************************************
    # Encode:
    def encode(self,x,y):
        enc_in = torch.cat([x,y],dim=1)
        enc_out = self.encoder(enc_in)

        vars = torch.split(enc_out,split_size_or_sections=self.latent_dim,dim=1)
        return vars[0], vars[1]
    
    #-----------------

    # Reparameterize:
    def reparameterize(self,mean,logvar):
        return mean + torch.exp(0.5*logvar) * torch.normal(mean=0.0,std=1.0,size=mean.size())
    
    #-----------------

    # Generate
    def generate(self,z,y):
        dec_in = torch.cat([z,y],dim=1)
        dec_out = self.decoder(dec_in)

        return dec_out
    
    #-----------------

    # Reconstruct:
    def reconstruct(self,x,y):
        mean, logvar = self.encode(x,y)
        z = self.reparameterize(mean,logvar)
        x_rec = self.generate(z,y)

        return x_rec
    #**********************************************

    # Compute loss functions:
    #**********************************************
    # Log normal:
    def log_normal_pdf(self,z,mean,logvar):
        log2pi = torch.log(2. * torch.tensor([np.pi]))
        arg = -0.5 * ( torch.square(z - mean) * torch.exp(-logvar) + logvar + log2pi)
        return torch.sum(arg,dim=1)
    
    #-----------------
    
    # Reconstruction loss:
    def compute_rec_loss(self,x,x_rec):
        return torch.sum(torch.square(x-x_rec),dim=1)
        
    #-----------------

    # Get the overall loss:
    def compute_loss(self,x,y):
        mean, logvar = self.encode(x,y)
        z = self.reparameterize(mean,logvar)

        x_rec = self.generate(z,y)

        logpx_z = -(self.compute_rec_loss(x,x_rec))*self.rec_loss_weight
        logpz = self.log_normal_pdf(z,torch.zeros(z.size()),torch.zeros(z.size()))
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return torch.mean(-logpx_z - logpz + logqz_x)
    #**********************************************

    # Update the weights:
    #**********************************************
    # Training:
    def train_step(self,x,y):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(x,y)
        loss.backward()
        self.optimizer.step()

        return loss
    
    #-----------------

    # Testing:
    def test_step(self,x,y):
        loss = self.compute_loss(x,y)
        return loss
    #**********************************************
      
    # Fit the model:
    #**********************************************
    # Sample random batch:
    def sample_rnd_batch(self,x_input,x_cond,batch_dim):
        idx = torch.randint(low=0,high=x_input.size()[0],size=(batch_dim,))
        batch_input = x_input[idx].to(self.devices)
        batch_cond = x_cond[idx].to(self.devices)

        return batch_input, batch_cond
   
    #-----------------

    # Fit the model:
    def fit(self,x_input_train,x_cond_train,n_epochs,batch_size,monitor_epoch,read_out_epoch,x_input_test=None,x_cond_test=None):
        mon_loss = []
        loss_tracker = MeanMetric()
        mon_val_loss = []
        val_loss_tracker = MeanMetric()

        run_testing = False
        if x_input_test is not None and x_cond_test is not None:
            run_testing = True

        #+++++++++++++++++++++++++++++
        for epoch in range(1,1+n_epochs):
            # Run the training:
            batch_input_train, batch_cond_train = self.sample_rnd_batch(x_input_train,x_cond_train,batch_size)
            current_train_loss = self.train_step(batch_input_train,batch_cond_train)
            loss_tracker.update(current_train_loss)

            # Run testing, if test data is provided:
            if run_testing:
                batch_input_test, batch_cond_test = self.sample_rnd_batch(x_input_test,x_cond_test,batch_size)
                current_test_loss = self.test_step(batch_input_test,batch_cond_test)
                val_loss_tracker.update(current_test_loss)

            # Print info:
            if epoch % monitor_epoch == 0:
                print(" ")
                print("Epoch: " + str(epoch) + "/" + str(n_epochs))
                print("Training Loss: " + str(loss_tracker.compute().cpu().item()))

                if run_testing == True:
                     print("Validation Loss: " + str(val_loss_tracker.compute().cpu().item()))
            
            # Collect results:
            if epoch % read_out_epoch == 0:
                mon_loss.append(loss_tracker.compute().cpu().item())
                loss_tracker.reset()

                if run_testing == True:
                    mon_val_loss.append(val_loss_tracker.compute().cpu().item())
                    val_loss_tracker.reset()
        #+++++++++++++++++++++++++++++
        print(" ")

        return {
            "loss": mon_loss,
            "val_loss": mon_val_loss
        }
    #**********************************************
