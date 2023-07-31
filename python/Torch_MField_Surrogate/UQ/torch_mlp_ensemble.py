import torch
from torch import nn
from torchmetrics import MeanMetric
from collections import OrderedDict

class TorchMLPEnsemble(nn.Module):

    # Initialize:
    #**********************************************
    def __init__(self,config,devices):
        super(TorchMLPEnsemble, self).__init__()

        self.devices = devices

        # Model hyper parameters:
        self.n_inputs = config['n_inputs']
        self.n_outputs = config['n_outputs']
        self.architecture = config['architecture']
        self.dropouts = config['dropouts']
        self.activations = config['activations']
        self.output_activation = config['output_activation']
        self.learning_rate = config['learning_rate']

        # Training hyper parameters:
        self.n_epochs = config['n_epochs']
        self.n_mon_epoch = config['n_mon_epoch']
        self.n_read_epoch = config['n_read_epoch']
        self.batch_size = config['batch_size']
        
        self.n_layers = len(self.architecture)

        # Create the ensemble:
        self.ensemble_size = config['ensemble_size']
        self.ensemble = []
        ensemble_params = list()
        #+++++++++++++++++++++++
        for _ in range(self.ensemble_size):
            current_model = nn.Sequential(self.set_mlp()).to(devices)
            self.ensemble.append(current_model)
            ensemble_params += list(current_model.parameters())
        #+++++++++++++++++++++++
        

        self.optimizer = torch.optim.Adam(ensemble_params,lr=self.learning_rate)
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

    # Set the mlp:
    #**********************************************
    def set_mlp(self):
        layers = OrderedDict()
        n_prev_nodes = self.n_inputs
        #++++++++++++++++++++++++++++++
        for j in range(self.n_layers):
            layer_name = 'layer_' + str(j)
            layers[layer_name] = nn.Linear(n_prev_nodes,self.architecture[j])
            layers[layer_name+'_act'] = self.get_activation_function(self.activations[j])

            if self.dropouts[j] > 0.0:
              layers[layer_name+'_mc_dropout'] = nn.Dropout(self.dropouts[j])

            n_prev_nodes = self.architecture[j]
        #++++++++++++++++++++++++++++++

        layers['output_layer'] = nn.Linear(n_prev_nodes,self.n_outputs)
        if self.output_activation != "linear":
          layers['output_activation'] = self.get_activation_function(self.output_activation)

        return layers
    #**********************************************

    # Parameter updates:
    #**********************************************
    def forward(self,x):
        prediction_list = []
        #+++++++++++++++++++
        for k in range(self.ensemble_size):
            current_prediction = self.ensemble[k](x).to(self.devices)
            prediction_list.append(current_prediction)
        #+++++++++++++++++++
        predictions = torch.cat(prediction_list,dim=0).to(self.devices)

        pred_mean = torch.mean(predictions,0).to(self.devices)
        pred_sigma = torch.std(predictions,0).to(self.devices)

        return pred_mean + torch.normal(mean=0.0,std=1.0,size=(x.size()[0],pred_mean.size()[0]),device=self.devices) * pred_sigma
    
    #----------------------

    # Define the loss function:  
    def compute_loss(self,x,y):
        y_pred = self.forward(x).to(self.devices)

        dy = y-y_pred
        sigma2 = torch.square(torch.std(y_pred,dim=0)).to(self.devices) + torch.tensor([1e-7],device=self.devices)
        arg = torch.square(dy) / sigma2 

        return torch.mean(torch.sum(arg + torch.log(sigma2),dim=1))

    #----------------------

    # Training step:
    def train_step(self,x,y):
        self.optimizer.zero_grad(set_to_none=True)
        
        loss = self.compute_loss(x,y)
        loss.backward()
        
        self.optimizer.step()

        return loss
    
    #----------------------

    # Testing step:
    def test_step(self,x,y):
        loss = loss = self.compute_loss(x,y)
        return loss

    #----------------------

    # Entire training loop:
    def fit(self,x,y,x_test=None,y_test=None):
        train_loss_tracker = MeanMetric().to(self.devices)
        test_loss_tracker = MeanMetric().to(self.devices)
        
        train_loss = []
        test_loss = [] 

        run_validation = False
        if x_test is not None and y_test is not None:
            run_validation = True

        #+++++++++++++++++++++++++
        for epoch in range(1,1+self.n_epochs):
            idx = torch.randint(x.size()[0],(self.batch_size,))
            
            x_train = x[idx].to(self.devices)
            y_train = y[idx].to(self.devices)
            
            loss = self.train_step(x_train,y_train)
            train_loss_tracker.update(loss)
    
            if run_validation:
                idx = torch.randint(x_test.size()[0],(self.batch_size,))
                x_val = x_test[idx]
                y_val = y_test[idx]

                val_loss = self.test_step(x_val,y_val)
                test_loss_tracker.update(val_loss)

            if epoch % self.n_read_epoch == 0:
                train_loss.append(train_loss_tracker.compute().cpu().item())
                train_loss_tracker.reset()
            
                if run_validation:
                    test_loss.append(test_loss_tracker.compute().cpu().item())
                    test_loss_tracker.reset()

            if epoch % self.n_mon_epoch == 0:
                print(" ")
                print("Epoch: " + str(epoch) + "/" + str(self.n_epochs))
                print("Loss: " + str(train_loss[-1]))

                if run_validation:
                    print("Test Loss: " + str(test_loss[-1]))
        #+++++++++++++++++++++++++
        print(" ")

        return {
            'loss': train_loss,
            'val_loss': test_loss
        }
    #**********************************************


