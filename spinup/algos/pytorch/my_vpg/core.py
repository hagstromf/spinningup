#from importlib.resources import path
import torch
import torch.nn as nn

import numpy as np


class VPGBuffer:

    ### TODO:
    # Document the VPGBuffer class and its functions

    def __init__(self, size, obs_dim, act_dim, gamma=0.99, device='cpu'):
        self.obs_buf = np.repeat(np.zeros(obs_dim, dtype=np.float32)[None, :], size, axis=0)
        self.act_buf = np.repeat(np.zeros(act_dim, dtype=np.float32)[None, :], size, axis=0)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)

        self.rtg_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma

        self.device = device

        self.curr_step, self.path_start, self.max_size = 0, 0, size


    def store(self, obs, act, rew, val):

        assert self.curr_step < self.max_size

        self.obs_buf[self.curr_step] = obs
        self.act_buf[self.curr_step] = act
        self.rew_buf[self.curr_step] = rew
        self.val_buf[self.curr_step]= val

        self.curr_step += 1


    def finish_path(self, last_val):
        
        path_idx = slice(self.path_start, self.curr_step)
        path_vals = np.append(self.val_buf[path_idx], last_val)
        path_rews = self.rew_buf[path_idx]

        self.adv_buf[path_idx] = path_rews + self.gamma * path_vals[1:] + path_vals[:-1]
        self.rtg_buf[path_idx] = np.cumsum(path_rews[::-1])[::-1]

        # Optional TODO: Implement infinite-horizon discounted rewards-to-go. Not necessary, the finite-horizon undiscounted
        # version, i.e., only counting observed rewards on the trajectory, is fine as well. 
        # path_rews = np.append(self.rew_buf[path_idx], last_val)
        
        self.path_start = self.curr_step
    
    def get(self):

        assert self.curr_step == self.max_size

        # Normalize advantages 
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std(ddof=1)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, 
                    rew=self.rew_buf, val=self.val_buf, 
                    rtg=self.rtg_buf, adv=self.adv_buf)

        self.curr_step, self.path_start = 0, 0

        return {k : torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}


def test_buffer(device='cpu'):
    size = 32
    obs_dim = (10, 3)
    act_dim = 5

    buf = VPGBuffer(size, obs_dim, act_dim, device=device)

    for i in range(size):
        o = np.random.random_sample(obs_dim)
        a = np.random.random_sample(act_dim)
        r = np.random.rand()
        v = np.random.rand()

        buf.store(o, a, r, v)
    
    last_val = 0
    buf.finish_path(last_val)

    data = buf.get()
    #obs, act, rew, val, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['rtg'], data['adv']

    # Test that we can fill the buffer again during next epoch
    for i in range(size):
        o = np.random.random_sample(obs_dim)
        a = np.random.random_sample(act_dim)
        r = np.random.rand()
        v = np.random.rand()

        buf.store(o, a, r, v)

    last_val = 0
    buf.finish_path(last_val)

    data = buf.get()
    obs, act, rew, val, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['rtg'], data['adv']

    print()
    print("Check shapes of different buffers:")
    assert obs.shape == (32, 10, 3)
    print(obs.shape)
    assert act.shape == (32, 5)
    print(act.shape)
    assert rew.shape == (32, )
    print(rew.shape)
    print(val.shape)
    print(rtg.shape)
    print(adv.shape)
    print()

    # Check advantage normalization
    print("Check that advantages are properly normalized:")
    #print(torch.mean(adv), torch.std(adv), "\n")
    print(adv.mean(), adv.std(), "\n")


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
    pass
    #def __init__(self, obs_dim, hidden_sizes, act_dim, activation, device='cpu'):
    #    super().__init__()

    #    self.obs_dim = obs_dim
    #    self.hidden_sizes = hidden_sizes
    #    self.act_dim = act_dim
    #    self.device = device




class MLPCritic(nn.Module):
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    test_buffer(device)