from typing import Dict, List, Tuple, Type, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class LyapunovFunction(nn.Module):

    def __init__(self,
                 input_dim:int,
                 net_arch:List[int]=[32,32],
                 activation_fn:Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto",
                 eps=1e-3):
        """
        Create a LyapunovFunction 

        :param input_dim: Dimension of the input vector
        :param net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        :param activation_fn: The activation function
            to use after each layer: You can change it to a softReLu.
        :param eps: coefficent of |x|^2
        :return:    
        """
        super(LyapunovFunction, self).__init__()
        self.input_dim = input_dim
        self.activate_fn = activation_fn()
        self.icnn = ICNN(input_dim=input_dim,output_dim=1,net_arch=net_arch,activation_fn=activation_fn).to(device)
        self.x_stable = th.zeros([1,input_dim], dtype=th.float32,device=device)

        self.eps = eps
        for _ in range(100):
            optimizer = th.optim.Adam(self.parameters(),lr=1e-1)
            input = th.randn(256,input_dim,dtype=th.float32,device=device)
            loss = F.mse_loss(input.pow(2).sum(1,keepdim=True),self.forward(input))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def set_equilibium_point(self, x_stable:th.Tensor):
        self.x_stable = x_stable  
      
    def forward(self, x:th.Tensor)->th.Tensor:
        eta_x = self.icnn(x)
        eta_0 = self.icnn(self.x_stable)
        # V(x) := sigma(eta(x)-eta(0)) + eps*|x|^2
        V = self.activate_fn(eta_x - eta_0) + self.eps * (x - self.x_stable).pow(2).sum(1,keepdim=True)
        return V
    
class posLinear(nn.Linear):
    """Linear function with positive weights and without bias"""
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 device=None, 
                 dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: th.Tensor) -> th.Tensor:
        positive_weight = F.softplus(self.weight/self.out_features)
        return F.linear(input, positive_weight, self.bias)

class ICNN(nn.Module):
    def __init__(self,      
                 input_dim: int,
                 output_dim: int,
                 net_arch: List[int],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 with_bias:bool = True
                 ) -> None:
        """
        Create an ICNN (Input Convex Neural Network).

        :param input_dim: Dimension of the input vector
        :param output_dim:
        :param net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        :param activation_fn: The activation function
            to use after each layer.
        :param with_bias: If set to False, the layers will not learn an additive bias
        :return:
        """
       
        super(ICNN, self).__init__()
        self.activation_fn = activation_fn()
        self.net_arch = net_arch

        # layers for Wx+b len(net_arch) + 1 layers
        self.Wx_layers = nn.ModuleList()
        self.Wx_layers.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))
        for i in range(len(net_arch)-1): 
            self.Wx_layers.append(nn.Linear(input_dim, net_arch[i+1], bias=with_bias))
        self.Wx_layers.append(nn.Linear(input_dim, output_dim, bias=with_bias))
        
        # layers for Uz len(net_arch)  layers
        self.Uz_layers = nn.ModuleList()
        for i in range(len(net_arch) - 1):
            self.Uz_layers.append(posLinear(net_arch[i], net_arch[i+1]))
        self.Uz_layers.append(posLinear(net_arch[i+1], output_dim))
        
    def forward(self, x:th.Tensor)->th.Tensor:
        tmp_Wx = [Wx_layer(x) for Wx_layer in self.Wx_layers]
        zi = self.activation_fn(tmp_Wx[0])
        for i, Uz_layer in enumerate(self.Uz_layers):
            zi = tmp_Wx[i+1] + Uz_layer(zi)
            zi = self.activation_fn(zi)
        return zi

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    with_bias: bool = True,
) -> nn.Module:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:List[nn.Module]
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    return nn.Sequential(*modules)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utils
    import numpy as np
    V = LyapunovFunction(input_dim=1,device="cpu",activation_fn=utils.SmoothReLU)
    optimizer = th.optim.AdamW(V.parameters(),lr=0.01)

    V.train()
    for i in range(10000):
        x = th.tensor(np.random.uniform(-3,3,20).reshape(20,1),dtype=th.float32)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(V.forward(x),x**2)
        loss.backward()
        optimizer.step()
        print(loss)
    x = np.linspace(-3,3,20).reshape(20,1)
    plt.plot(x,V.forward(th.tensor(x,dtype=th.float32)).detach().numpy().flatten())
    plt.show()


