#from importlib.resources import path
import torch
import torch.nn as nn

import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import gym
from gym.spaces import Discrete, Box

class VPGBuffer:

    ### TODO:
    # Document the VPGBuffer class and its functions

    def __init__(self, size, obs_dim, act_dim, gamma=0.99, device='cpu'):
        self.obs_buf = np.repeat(np.zeros(obs_dim, dtype=np.float32)[None, :], size, axis=0)
        self.act_buf = np.repeat(np.zeros(act_dim, dtype=np.float32)[None, :], size, axis=0)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        #self.logp_buf = np.zeros(size, dtype=np.float32)

        self.rtg_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.device = device

        self.curr_step, self.path_start, self.max_size = 0, 0, size


    def store(self, obs, act, rew, val): #, logp_a):

        assert self.curr_step < self.max_size

        self.obs_buf[self.curr_step] = obs
        self.act_buf[self.curr_step] = act
        self.rew_buf[self.curr_step] = rew
        self.val_buf[self.curr_step] = val
        #self.logp_buf[self.curr_step] = logp_a

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
                     rtg=self.rtg_buf, adv=self.adv_buf)#, logp_a=self.logp_buf)

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
        #logp_a = np.random.rand()

        buf.store(o, a, r, v) #, logp_a)
    
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
        #logp_a = np.random.rand()

        buf.store(o, a, r, v) #, logp_a)

    last_val = 0
    buf.finish_path(last_val)

    data = buf.get()
    
    obs, act, rew, val, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['rtg'], data['adv']
    #obs, act, rew, val, logp_a, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['logp_a'], data['rtg'], data['adv']

    print()
    print("Check shapes of different buffers:")
    assert obs.shape == (size, *obs_dim)
    print(obs.shape)
    assert act.shape == (size, act_dim)
    print(act.shape)
    assert rew.shape == (size, )
    print(rew.shape)
    print(val.shape)
    #print(logp_a.shape)
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
    #print("Layer sizes: ", sizes)
    for i in range(1, len(sizes)):
        act = activation if i < len(sizes)-1 else output_activation
        layers += [nn.Linear(sizes[i-1], sizes[i]), act()]
    
    return nn.Sequential(*layers)


class Actor(nn.Module):

    # TODO: Document Actor class and its subclasses

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        #if obs.ndimension() > 1:
        #    obs = torch.flatten(obs, start_dim=1)
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob(pi, act)

        return pi, logp_a


class MLPDiscreteActor(Actor):

    def __init__(self, obs_dim, hidden_sizes, act_dim, activation, device='cpu'):
        super().__init__()

        #print("Obs before: ", obs_dim)
        if not np.isscalar(obs_dim):
            obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)


    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def _log_prob(self, pi, act):
        return pi.log_prob(act)


class MLPContinuousActor(Actor):
    
    def __init__(self, obs_dim, hidden_sizes, act_dim, activation, device='cpu'):
        super().__init__()

        #print("Obs before: ", obs_dim)
        if not np.isscalar(obs_dim):
            obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        self.mu = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.log_std = nn.parameter.Parameter(-0.5 * torch.ones(act_dim)) #.to(device)

    
    def _distribution(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


    def _log_prob(self, pi, act):
        return torch.sum(pi.log_prob(act), dim=1)


class MLPCritic(nn.Module):

    # TODO: Document Critic class

    def __init__(self, obs_dim, hidden_sizes, activation, device='cpu'):
        super().__init__()

        #print("Obs before: ", obs_dim)
        if not np.isscalar(obs_dim):
            obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        #print(obs.ndimension())
        #if obs.ndimension() > 1:
        #    obs = torch.flatten(obs, start_dim=1)
        #    print("Flattened obs", obs.shape)
        #print(obs.shape)
        return self.v(obs).squeeze() # Ensure v has shape (batch_size, ) instead of (batch_size, 1)


class MLPActorCritic(nn.Module):

    # TODO: Document ActorCritic class
    
    def __init__(self, obs_space, act_space, hidden_sizes=[128, 128], activation=nn.ReLU, device='cpu'):
        super().__init__()

        self.obs_dim = obs_space.shape

        #print()
        #print("Building critic:")

        self.critic = MLPCritic(self.obs_dim, hidden_sizes, activation, device)

        #print()
        #print("Building actor:")
        if isinstance(act_space, Box):
            #act_dim = act_space.shape[0]
            self.act_dim = act_space.shape[0]
            #print("Act dimension: ", act_dim)
            self.actor = MLPContinuousActor(self.obs_dim, hidden_sizes, self.act_dim, activation, device)
        elif isinstance(act_space, Discrete):
            self.act_dim = act_space.n
            #print("Act dimension: ", act_dim)
            self.actor = MLPDiscreteActor(self.obs_dim, hidden_sizes, self.act_dim, activation, device)
        else:
            raise Exception("Action space type should be either Box or Discrete, please use another environment!")

        #print()

    def act(self, obs):
        with torch.no_grad():
            pi, _ = self.actor(obs)
        return pi.sample().cpu().numpy()
        #return pi.sample()


    def step(self, obs):
        with torch.no_grad():
            pi, _ = self.actor(obs)
            act = pi.sample().squeeze()
            #logp_a = self.actor._log_prob(pi, act)
            v = self.critic(obs)
        return act.cpu().numpy(), v.cpu().numpy()#, logp_a.cpu().numpy()
        #return a, v, logp_a


def test_MLPmodules(env_fn, device='cpu'):
    from gym.wrappers import FlattenObservation

    #env = env_fn()
    env = FlattenObservation(env_fn())
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    print()
    print("Obs dim: ", obs_dim)
    print("Act dim: ", act_dim)
    print()

    ac = MLPActorCritic(env.observation_space, env.action_space, device=device).to(device)
        
    obs = np.random.random_sample((2, *obs_dim))
    #obs = np.random.random_sample(obs_dim)
    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
    print("Obs shape: ", obs.shape)

    
    v = ac.critic(obs)
    print()
    print("Check shapes of critic's forward function:")
    print(v.shape)
    print(v)

    a_rand = np.random.random_sample((2, *act_dim))
    a_rand = torch.as_tensor(a_rand, dtype=torch.float32).to(device)
    pi, logp_a = ac.actor(obs, a_rand)

    print()
    print("Check shapes of actor's forward function:")
    print("pi: ", pi)
    print("logp_a: ", logp_a)
    print()

    a, v = ac.step(obs)
    #a, v, logp_a = ac.step(obs)

    print("Check output of step method:")
    print("Act: ", a)
    print("Val: ", v)
    #print("logp_a: ", logp_a)
    print()

    a = ac.act(obs)

    print("Check output of act method:")
    print("Act: ", a)
    print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("On device: ", device)
    test_buffer(device)

    #test_MLPmodules(lambda: gym.make(args.env), device)