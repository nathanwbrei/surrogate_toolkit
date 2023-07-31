import torch
from torch import nn
from torchmetrics import MeanMetric
from collections import OrderedDict

class TorchMLPUQ(nn.Module):

    # Initialize:
    #**********************************************
    def __init__(self,config,devices):
        super(TorchMLPUQ, self).__init__()

        self.devices = devices

        # Model hyper parameters:
        self.n_inputs = config['n_inputs']
        self.n_outputs = config['n_outputs']
        self.architecture = config['architecture']
        self.activations = config['activations']
        self.output_activation = config['output_activation']
        self.learning_rate = config['learning_rate']

        # Handle the MC dropout layer:
        self.init_dropout = config['init_dropout'] if 'init_dropout' in config else 0.2
        self.dropout_temp = config['dropout_temp'] if 'dropout_temp' in config else 10.0
        self.n_calls = config['n_calls'] if 'n_calls' in config else 15
        self.weight_regularizer = config['weight_regularizer'] if 'weight_regularizer' in config else 0.0
        self.dropout_regularizer = config['dropout_regularizer'] if 'dropout_regularizer' in config else 1e-5
        self.dropout_layer = MCDropout(self.init_dropout,self.dropout_temp)

        # Training hyper parameters:
        self.n_epochs = config['n_epochs']
        self.n_mon_epoch = config['n_mon_epoch']
        self.n_read_epoch = config['n_read_epoch']
        self.batch_size = config['batch_size']
        
        self.n_layers = len(self.architecture)
        self.model = nn.Sequential(self.set_mlp()).to(self.devices)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
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

            #if self.dropouts[j] > 0.0:
            layers[layer_name+'_mc_dropout'] = self.dropout_layer

            n_prev_nodes = self.architecture[j]
        #++++++++++++++++++++++++++++++

        layers['output_layer'] = nn.Linear(n_prev_nodes,self.n_outputs)
        if self.output_activation != "linear":
          layers['output_activation'] = self.get_activation_function(self.output_activation)

        return layers
    #**********************************************

    # Define a KL regularization term which includes the dropout
    # See here: https://doi.org/10.48550/arXiv.1705.07832 
    # for more details
    #**********************************************
    def calc_KL_term(self):
        # Get the dropout parameter first:
        p = torch.sigmoid(self.dropout_layer.p_train)
        # Calcualte the entropy:
        H = -p * torch.log(p) - (1.0-p)*torch.log(1.0-p)

        # Now go through the model layers and collect the weights:
        kl_term = 0.0
        #+++++++++++++++++++++++++++
        for n,param in self.model.named_parameters():
            if "weight" in n:
                current_weights = param.data
                kl_term += self.weight_regularizer * (1.0-p) * torch.sum(torch.square(current_weights))
                kl_term -= self.dropout_regularizer * H * current_weights.size()[1]
        #+++++++++++++++++++++++++++

        return kl_term
    #**********************************************

    # Model response and parameter updates:
    #**********************************************
    # Forward pass:
    def forward(self,x):
        return self.model(x).to(self.devices)

    #----------------------

    # Prediction:
    def predict(self,x):
        # This seems a bit unnecessary, but we want to ensure that the dropout layer
        # is utilized everytime we call the model:
        prediction_list = []
        #+++++++++++++++++++
        for _ in range(self.n_calls):
            current_prediction = self.model(x).to(self.devices)
            prediction_list.append(current_prediction)
        #+++++++++++++++++++
        predictions = torch.cat(prediction_list,dim=0).to(self.devices)

        pred_mean = torch.mean(predictions,0).to(self.devices)
        pred_sigma = torch.std(predictions,0).to(self.devices)

        return pred_mean + torch.normal(mean=0.0,std=1.0,size=(x.size()[0],pred_mean.size()[0]),device=self.devices) * pred_sigma
        
    #----------------------

    # Define the loss function:
    def compute_loss(self,x,y):
        y_pred = self.predict(x).to(self.devices)

        dy = y-y_pred
        sigma2 = torch.square(torch.std(y_pred,dim=0)).to(self.devices) + torch.tensor([1e-7],device=self.devices)
        arg = torch.square(dy) / sigma2 

        kl_loss = self.calc_KL_term()
       # print(kl_loss)

        return torch.mean(torch.sum(arg + torch.log(sigma2),dim=1)) + kl_loss

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
        mc_dropout_rate_tracker = MeanMetric().to(self.devices)

        train_loss = []
        test_loss = [] 
        mc_dropout = []

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

            # Monitori the dropout rate:
            current_dropout_rate = torch.sigmoid(self.dropout_layer.p_train)
            mc_dropout_rate_tracker.update(current_dropout_rate)

            if run_validation:
                idx = torch.randint(x_test.size()[0],(self.batch_size,))
                x_val = x_test[idx]
                y_val = y_test[idx]

                val_loss = self.test_step(x_val,y_val)
                test_loss_tracker.update(val_loss)

            if epoch % self.n_read_epoch == 0:
                train_loss.append(train_loss_tracker.compute().cpu().item())
                train_loss_tracker.reset()
                
                mc_dropout.append(mc_dropout_rate_tracker.compute().cpu().item())
                mc_dropout_rate_tracker.reset()


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
            'val_loss': test_loss,
            'mc_dropout': mc_dropout
        }
    #**********************************************


# Define a custom dropout class:
class MCDropout(torch.nn.Module):
    
    def __init__(self,p0,temp):
        super(MCDropout, self).__init__()
        self.p_train = torch.nn.Parameter(self.calc_logit(p0),requires_grad=True)
        self.temp = temp

    def calc_logit(self,p0):
        p0_tensor = torch.tensor(p0)
        return torch.log(p0_tensor / (1. - p0_tensor))
    
    # The implementation used here has been taken from:
    # 'Concrete Dropout' Gal et al.,  arXiv:1705.07832v1 [stat.ML], 2017
    # Link to paper is here: https://doi.org/10.48550/arXiv.1705.07832
    def forward(self,x):
        eps = 1e-7
        u = torch.rand_like(x)
        pt = torch.ones_like(x)*torch.sigmoid(self.p_train)

        arg = torch.log(pt+eps) - torch.log(1.-pt+eps) + torch.log(u+eps) - torch.log(1.-u+eps)
        drop_prob = torch.sigmoid(self.temp*arg)
        
        x= torch.mul(x,(1.-drop_prob))
        x /= (1.-torch.sigmoid(self.p_train))
        return x