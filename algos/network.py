import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.base import Network

class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation_mu = None, last_activation_std = None, is_actor=True):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation_mu, is_actor, last_activation_std)

    def forward(self, x):
        mu, std = self._forward(x)
        return mu, std

    
class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None, is_actor=False):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation, is_actor)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)
    