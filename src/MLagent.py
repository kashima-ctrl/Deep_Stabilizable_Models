from typing import Dict, List, Tuple, Type, Union, Any, Optional
import torch as th
import torch.nn as nn
from src.utils import get_device, SmoothReLU
from src.dynamics import Dynamics,CartPole,Pendulum,VanDerPol,BenchmarkExample,HJI_Example
from src.models import Stable_Dynamics,Safty_Dynamics,L2_Dynamics,Passification_Dynamics,HJI_Dynamics
from torch.utils.data import DataLoader,random_split
import numpy as np
import os

class MLAgent(nn.Module):
    def __init__(self,
                 name:str,
                 dynamics:Type[Dynamics],
                 model_class:Type[Stable_Dynamics],
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 batch_size:int = 256,
                 data_size:int =31,
                 lr:float = 0.01,
                 activation_fn: Type[nn.Module] = SmoothReLU,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: Union[th.device, str] = "auto") -> None:
        """
        Initializes the model with specified dynamics, model type, training parameters,
        and device configuration, preparing it for training or evaluation.

        Parameters:
            name (str): A unique identifier or name for this instance.
            dynamics (Type[Dynamics]): Class type for the dynamics system, responsible for
                providing the dataset and initializing environment parameters.
            model (Type[Stable_Dynamics]): Class type of the model to be used. Should be a subclass
                of Stable_Dynamics, which defines the neural network structure and dynamics.
            batch_size (int, optional): Number of samples per batch during training. Default is 256.
            data_size (int, optional): Number of data samples to generate or use from the dataset.
                Default is 31.
            lr (float, optional): Learning rate for the optimizer. Default is 0.01.
            activation_fn (Type[nn.Module], optional): Activation function for the model, defaulting to
                SmoothReLU. Must be a subclass of nn.Module.
            optimizer_class (Type[th.optim.Optimizer], optional): Optimizer class to use for training
                the model. Default is AdamW. Must be a subclass of torch.optim.Optimizer.
            optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the optimizer,
                passed as a dictionary. Default is None.
            device (Union[th.device, str], optional): Specifies the device to use (e.g., 'cpu', 'cuda',
                or 'auto' to select automatically based on availability). Default is "auto".
        """
        super().__init__()
        self.name = name
        self.device = get_device(device)
        os.makedirs("./figures/{}".format(name), exist_ok=True)

        dynamics = dynamics()
        self.dataset = dynamics.make_dataset(num=data_size,isRandom=False)
        self.size_dataset = len(self.dataset)
        
        if model_kwargs is None:
            model_kwargs = {}
        self.model = model_class(state_dim=dynamics.state_dim,
                                 action_dim=dynamics.action_dim,
                                 activation_fn=activation_fn,
                                 device=self.device,
                                 **model_kwargs)
                
        self.batch_size = batch_size
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(self.model.parameters(), lr = lr, **optimizer_kwargs)

    def train(self, epoches = 10, isSave = True):
        # [step 1] Prepare dataset
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # [step 2] Start training!!
        self.model.train()
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader:
                xu = input.to(self.device)
                dx = output.to(self.device).unsqueeze(dim=2)   
                x = xu[:,0:self.model.state_dim]
                u = xu[:,self.model.state_dim:].unsqueeze(dim=2)   
                x.requires_grad = True
                f, g, alpha, V = self.model(x)
                  
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(f + g @ u, dx)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            mean_loss =  np.mean(loss_list)
            print(epoch, "ave_loss=", mean_loss)
            if mean_loss <1e-3:
                break 
        if isSave:
            return self.save_model()
        else:
            return None

    def save_model(self):
        scripted_model = th.jit.script(self.model)
        scripted_model.save("./saved_model/{}.zip".format(self.name))
        return scripted_model
    
    def load_model(self):
        return th.jit.load("./saved_model/{}.zip".format(self.name))

class MLAgent_Passification(MLAgent):
    def __init__(self, name, dynamics, model_class,lambda_h=0.1, model_kwargs = None, batch_size = 256, data_size = 31, lr = 0.01, activation_fn = SmoothReLU, optimizer_class = th.optim.AdamW, optimizer_kwargs = None, device = "auto"):
        super().__init__(name, dynamics, model_class, model_kwargs, batch_size, data_size, lr, activation_fn, optimizer_class, optimizer_kwargs, device)
        self.lambda_h = lambda_h

    def train(self, epoches = 10, isSave = True):
        # [step 1] Prepare dataset
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # [step 2] Start training!!
        self.model.train()
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader:
                xu = input.to(self.device)
                dx_y = output.to(self.device)
                x = xu[:,0:self.model.state_dim]
                u = xu[:,self.model.state_dim:].unsqueeze(dim=2)
                dx = dx_y[:,0:self.model.state_dim].unsqueeze(dim=2) 
                y =  dx_y[:,self.model.state_dim:].unsqueeze(dim=2) 

                x.requires_grad = True
                f, g, alpha, V, h, beta = self.model(x)
                  
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(f + g @ u, dx) + \
                       self.lambda_h * nn.functional.mse_loss(y, h)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            mean_loss =  np.mean(loss_list)
            print(epoch, "ave_loss=", mean_loss)
            if mean_loss <1e-3:
                break 
        if isSave:
            return self.save_model()
        else:
            return None
        
class MLAgent_HJI(MLAgent):
    def __init__(self, name, dynamics, model_class,lambda_h=0.1, model_kwargs = None, batch_size = 256, data_size = 31, lr = 0.01, activation_fn = SmoothReLU, optimizer_class = th.optim.AdamW, optimizer_kwargs = None, device = "auto"):
        super().__init__(name, dynamics, model_class, model_kwargs, batch_size, data_size, lr, activation_fn, optimizer_class, optimizer_kwargs, device)
        self.lambda_h = lambda_h
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.8)  

           
    def train(self, epoches = 10, isSave = True):
        # [step 1] Prepare dataset
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # [step 2] Start training!!
        self.model.train()
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader:
                xu = input.to(self.device)
                dx = output.to(self.device)
                x = xu[:,0:self.model.state_dim]
                u = xu[:,self.model.state_dim:].unsqueeze(dim=2)
                dx = dx[:,0:self.model.state_dim].unsqueeze(dim=2) 
                
                x.requires_grad = True
                f, g, alpha, V, H = self.model(x)
                  
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(f + g @ u, dx) + \
                       self.lambda_h * H.pow(2).mean()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            mean_loss =  np.mean(loss_list)
            self.scheduler.step()
            print(epoch, "ave_loss=", mean_loss)
            if mean_loss <1e-4:
                break 
        if isSave:
            return self.save_model()
        else:
            return None
        
class MLAgent_SSG(MLAgent):
    def __init__(self, name, dynamics, model_class,lambda_h=0.1, model_kwargs = None, batch_size = 256, data_size = 31, lr = 0.01, activation_fn = SmoothReLU, optimizer_class = th.optim.AdamW, optimizer_kwargs = None, device = "auto"):
        super().__init__(name, dynamics, model_class, model_kwargs, batch_size, data_size, lr, activation_fn, optimizer_class, optimizer_kwargs, device)
        self.dataset_h = dynamics().make_ssdataset(num=16,isRandom=False)
        

    def train(self, epoches = 10, isSave = True):
        # [step 1] Prepare dataset (x_stable = h(u))
        train_dataloader_h = DataLoader(self.dataset_h, batch_size=self.batch_size, shuffle=True)
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # [step 2] Start training!!
        self.model.train()
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader_h:
                u = input.to(self.device).unsqueeze(1)
                stable_x = output.to(self.device)
                 
                x = self.model.forward_h(u)
                
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(x, stable_x)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            mean_loss =  np.mean(loss_list)
            print(epoch, "ave_loss=", mean_loss)
            if mean_loss <1e-4:
                break 

        # [step 3] Start training!!
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader:
                xu = input.to(self.device)
                dx = output.to(self.device).unsqueeze(dim=2)   
                x = xu[:,0:self.model.state_dim]
                u = xu[:,self.model.state_dim:]
                
                x.requires_grad = True
                f, h, V = self.model(x, u)
                
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(f, dx)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            mean_loss =  np.mean(loss_list)
            print(epoch, "ave_loss=", mean_loss)
            if mean_loss <1e-4:
                break 
        if isSave:
            return self.save_model()
        else:
            return None