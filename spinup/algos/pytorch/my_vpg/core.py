#from importlib.resources import path
import torch
import torch.nn as nn

import numpy as np
import scipy

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import gym
from gym.spaces import Discrete, Box

from spinup.utils.mpi_tools import mpi_statistics_scalar


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class VPGBuffer:

    ### TODO:
    # Document the VPGBuffer class and its functions

    def __init__(self, size, obs_dim, act_dim, gamma=0.99, lam=0.95, device='cpu'):

        #obs_dim = obs_space.shape
        self.obs_buf = np.repeat(np.zeros(obs_dim, dtype=np.float32)[None, :], size, axis=0)

        #if isinstance(act_space, Discrete):
        #    act_dim = 1
        #else:
        #    act_dim = act_space.shape

        # Ensure that act_buf has correct shape. When action space is discrete, act_buf should have shape (size,), same as adv_buf,
        # otherwise actor loss will be calculated incorrectly!
        if not act_dim:
            self.act_buf = np.zeros(size, dtype=np.float32)
        else:
            self.act_buf = np.repeat(np.zeros(act_dim, dtype=np.float32)[None, :], size, axis=0)
        
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        #self.logp_buf = np.zeros(size, dtype=np.float32)

        self.rtg_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.device = device

        self.curr_step, self.path_start, self.path_count, self.max_size = 0, 0, 0, size


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
        
        ### TD advantage estimation
        #path_rews = self.rew_buf[path_idx]
        #self.adv_buf[path_idx] = path_rews + self.gamma * path_vals[1:] - path_vals[:-1]
        #self.rtg_buf[path_idx] = np.cumsum(path_rews[::-1])[::-1]

        ### TD advantage estimation and last val added to rewards-to-go
        #path_rews = np.append(self.rew_buf[path_idx], last_val)
        #self.adv_buf[path_idx] = path_rews[:-1] + self.gamma * path_vals[1:] - path_vals[:-1]
        #self.rtg_buf[path_idx] = np.cumsum(path_rews[::-1])[::-1][:-1]

        ### Monte Carlo advantage estimation
        #path_rews = self.rew_buf[path_idx]
        #self.rtg_buf[path_idx] = discount_cumsum(path_rews, self.gamma)
        #self.adv_buf[path_idx] = self.rtg_buf[path_idx] - path_vals[:-1]

        ### Generalized Advantage Estimation
        path_rews = np.append(self.rew_buf[path_idx], last_val)
        delta = path_rews[:-1] + self.gamma * path_vals[1:] - path_vals[:-1]
        self.adv_buf[path_idx] = discount_cumsum(delta, self.gamma * self.lam)
        self.rtg_buf[path_idx] = discount_cumsum(path_rews, self.gamma)[:-1]
        
        self.path_start = self.curr_step
        self.path_count += 1
    
    def get(self):

        assert self.curr_step == self.max_size

        # Normalize advantages 
        #adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() 
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_norm = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, 
                    #rew=self.rew_buf, val=self.val_buf, 
                    rtg=self.rtg_buf, adv=adv_norm)#, logp_a=self.logp_buf)

        self.curr_step, self.path_start = 0, 0

        return {k : torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}


def test_buffer(device='cpu'):
    from spinup.algos.pytorch.vpg.vpg import VPGBuffer as SpinupBuf

    size = 32
    obs_dim = (10, 3)
    #act_dim = 5
    act_dim = ()

    lam = 0.95

    buf = VPGBuffer(size, obs_dim, act_dim, device=device, lam=lam)
    buf_ref = SpinupBuf(obs_dim, act_dim, size, lam=lam)

    for i in range(size):
        o = np.random.random_sample(obs_dim)
        a = np.random.random_sample(act_dim)
        r = np.random.rand()
        v = np.random.rand()
        #logp_a = np.random.rand()

        buf.store(o, a, r, v) #, logp_a)
        #buf_ref.store(o, a, r, v)
    
    last_val = 0
    buf.finish_path(last_val)
    #buf_ref.finish_path(last_val)

    data = buf.get()
    #obs, act, rew, val, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['rtg'], data['adv']

    # Test that we can fill the buffer again during next epoch 
    for i in range(size):
        o = np.random.random_sample(obs_dim)
        a = np.random.random_sample(act_dim)
        r = np.random.rand()
        v = np.random.rand()
        logp_a = np.random.rand()

        buf.store(o, a, r, v) #, logp_a)
        buf_ref.store(o, a, r, v, logp_a)

    last_val = 0
    buf.finish_path(last_val)
    buf_ref.finish_path(last_val)

    data = buf.get()
    data_ref = buf_ref.get()
    
    #obs, act, rew, val, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['rtg'], data['adv']
    obs, act, rtg, adv = data['obs'], data['act'], data['rtg'], data['adv']
    obs_ref, act_ref, rtg_ref, adv_ref = data_ref['obs'], data_ref['act'], data_ref['ret'], data_ref['adv']

    #obs, act, rew, val, logp_a, rtg, adv = data['obs'], data['act'], data['rew'], data['val'], data['logp_a'], data['rtg'], data['adv']

    print()
    print("Check shapes of different buffers:")
    #assert obs.shape == (size, *obs_dim)
    print("Obs shape: ", obs.shape)
    print("Ref Obs shape: ", obs_ref.shape)
    #assert act.shape == (size, act_dim)
    print("Act shape: ", act.shape)
    print("Ref Act shape: ", act_ref.shape)
    #assert rew.shape == (size, )
    #print(rew.shape)
    #print(val.shape)
    #print(logp_a.shape)
    print("RTG shape: ", rtg.shape)
    print("Ref RTG shape: ", rtg_ref.shape)
    print("Adv shape: ", adv.shape)
    print("Ref Adv shape: ", adv_ref.shape)
    print()

    # Check advantage normalization
    print("Check that advantages are properly normalized:")
    #print(torch.mean(adv), torch.std(adv), "\n")
    print(adv.mean(), adv.std(), "\n")

    print("Reference Advantage buffer:")
    print(adv_ref.mean(), adv_ref.std(), "\n")

    print("Check action buffer:")
    print(act, "\n")

    print("Reference action buffer:")
    print(act_ref, "\n")


    print("Check RTG buffer:")
    print(rtg, "\n")

    print("Reference RTG buffer:")
    print(rtg_ref, "\n")



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
            #print(pi)
            #print(act)
            logp_a = self._log_prob(pi, act)

        return pi, logp_a


class MLPDiscreteActor(Actor):

    def __init__(self, obs_dim, hidden_sizes, act_dim, activation):
        super().__init__()

        #print("Obs before: ", obs_dim)
        #if not np.isscalar(obs_dim):
        #    obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)


    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def _log_prob(self, pi, act):
        return pi.log_prob(act)


class MLPContinuousActor(Actor):
    
    def __init__(self, obs_dim, hidden_sizes, act_dim, activation):
        super().__init__()

        #print("Obs before: ", obs_dim)
        #if not np.isscalar(obs_dim):
        #    obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        self.mu = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32)) #.to(device)

    
    def _distribution(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


    def _log_prob(self, pi, act):
        return torch.sum(pi.log_prob(act), dim=-1)


class MLPCritic(nn.Module):

    # TODO: Document Critic class

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()

        #print("Obs before: ", obs_dim)
        #if not np.isscalar(obs_dim):
        #    obs_dim = np.prod(obs_dim)
        #print("Obs after: ", obs_dim)

        #print("Critic architecture:", [obs_dim] + list(hidden_sizes) + [1])

        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        #print(obs.ndimension())
        #if obs.ndimension() > 1:
        #    obs = torch.flatten(obs, start_dim=1)
        #    print("Flattened obs", obs.shape)
        #print(obs.shape)

        # Ensure v has shape (batch_size, ) instead of (batch_size, 1). Important when calculating critic loss that v and rtg_buf have same shape!
        return self.v(obs).squeeze(dim=-1) 


class MLPActorCritic(nn.Module):

    # TODO: Document ActorCritic class
    
    def __init__(self, obs_space, act_space, hidden_sizes=[128]*3, activation=nn.Tanh):
        super().__init__()

        #obs_dim = obs_space.shape
        obs_dim = obs_space.shape[0]

        #print("Obs dim:", obs_dim)

        #print()
        #print("Building critic:")

        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)
        #print(self.critic, "\n")

        #print()
        #print("Building actor:")
        if isinstance(act_space, Box):
            #act_dim = act_space.shape[0]
            act_dim = act_space.shape[0]
            #print("Act dimension: ", act_dim)
            self.actor = MLPContinuousActor(obs_dim, hidden_sizes, act_dim, activation)
        elif isinstance(act_space, Discrete):
            act_dim = act_space.n
            #print("Act dimension: ", act_dim)
            self.actor = MLPDiscreteActor(obs_dim, hidden_sizes, act_dim, activation)
        else:
            raise Exception("Action space type should be either Box or Discrete, please use another environment!")

        #print(self.actor, ("\n"))

    def act(self, obs):
        with torch.no_grad():
            pi, _ = self.actor(obs)
        return pi.sample().cpu().numpy()
        #return pi.sample()


    def step(self, obs):
        with torch.no_grad():
            pi, _ = self.actor(obs)
            act = pi.sample() #.squeeze()
            #print(act.shape)
            #logp_a = self.actor._log_prob(pi, act)
            v = self.critic(obs)
        return act.cpu().numpy(), v.cpu().numpy()#, logp_a.cpu().numpy()
        #return a, v, logp_a


def test_MLPmodules(env_fn, device='cpu'):
    from gym.wrappers import FlattenObservation

    from spinup.algos.pytorch.vpg.core import MLPActorCritic as ReferenceAC

    env = env_fn()
    #env = FlattenObservation(env_fn())
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    print()
    print("Obs dim: ", obs_dim)
    print("Act dim: ", act_dim)
    print()

    hid = [128,128,128]

    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hid).to(device)
    print(ac.critic)
    print(ac.actor)
        
    obs = np.random.random_sample((2, *obs_dim))
    #obs = np.random.random_sample(obs_dim)
    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
    print("Obs shape: ", obs.shape)

    
    v = ac.critic(obs)
    print()
    print("Check shapes of critic's forward function:")
    print(v.shape)
    print(v)

    a_rand = np.random.random_sample((1, *act_dim))
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


    print("Checking reference actor-critic!")
    print()

    ac = ReferenceAC(env.observation_space, env.action_space, hidden_sizes=hid).to(device)
    print(ac.v)
    print(ac.pi)
        
    obs = np.random.random_sample((2, *obs_dim))
    #obs = np.random.random_sample(obs_dim)
    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
    print("Obs shape: ", obs.shape)

    
    v = ac.v(obs)
    print()
    print("Check shapes of critic's forward function:")
    print(v.shape)
    print(v)

    a_rand = np.random.random_sample((1, *act_dim))
    a_rand = torch.as_tensor(a_rand, dtype=torch.float32).to(device)
    pi, logp_a = ac.pi(obs, a_rand)

    print()
    print("Check shapes of actor's forward function:")
    print("pi: ", pi)
    print("logp_a: ", logp_a)
    print()

    a, v, _ = ac.step(obs)
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