import torch
import torch.nn as nn

import numpy as np


class VPGBuffer:

    def __init__(self, size, obs_dim, act_dim, gamma):
        self.obs_buf = np.repeat(np.zeros(obs_dim, dtype=np.float32)[None, :], size, axis=0)
        self.act_buf = np.repeat(np.zeros(act_dim, dtype=np.float32)[None, :], size, axis=0)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma

    ###TODO
    # 1. Check that obs_buf and act_buf are initialized correctly
    # 2. Implement store function for storing steps in the buffer
    # 3. Implement get function for getting the data stored in the buffer

    def store():
        pass

    
    def get():
        pass


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.

    Args:
        sizes: Tuple, list, or other iterable giving the number of units
            for each layer of the MLP. 

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A PyTorch module that can be called to give the output of the MLP.
        (Use an nn.Sequential module.)

    """
    layers = []

    for i in range(1, len(sizes)):
        act = activation if i < len(sizes)-1 else output_activation
        layers += [nn.Linear(sizes[i-1], sizes[i]), act()]
    
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim, activation, device='cpu'):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_sizes = hidden_sizes
        self.act_dim = act_dim
        self.device = device




class MLPCritic(nn.Module):
    pass