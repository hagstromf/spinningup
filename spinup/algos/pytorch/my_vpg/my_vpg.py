from cmath import log
import torch
import numpy as np
from spinup.utils.logx import EpochLogger

import gym
from gym.wrappers import FlattenObservation

import spinup.algos.pytorch.my_vpg.core as core
import time

def my_vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
           steps_per_epoch=4000, epochs=50, gamma=0.99, 
           pi_lr=0.0003, vf_lr=0.001, train_v_iters=80,
           max_ep_len=1000, logger_kwargs=dict(), save_freq=10, device='cpu'):

    # TODO: Document entire my_vpg function

    # Optional TODO: Implement observation normalization to see if it makes a difference
    
    # Initialize Logger object used to log and print training info
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_fn()

    if isinstance(actor_critic, core.MLPActorCritic):
        env = FlattenObservation(env)

    # Initialize Actor-Critic network
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    #print("AC is on CUDA:", ac.actor.is_cuda)

    # Initialize buffer for collecting trajectories
    buf = core.VPGBuffer(steps_per_epoch, env.observation_space, env.action_space, gamma, device)


    # Estimate the policy gradient
    def compute_loss_actor(data):
        o, a, adv = data['obs'], data['act'], data['adv']

        #print(o.shape)
        #print(a.shape)
        pi, logp_a = ac.actor(o, a)

        #logp_a, adv = data['logp_a'], adv['adv']

        # Store some useful info for tracking progress
        ent = pi.entropy().mean().item()

        loss_info = dict(Entropy=ent)
        
        # If algo is bork, maybe the gradient should be positive instead!!!!
        return -torch.mean(logp_a * adv), loss_info

    # Compute the MSE loss for estimating the Value function
    def compute_loss_critic(data):
        o, rtg = data['obs'], data['rtg']
        v = ac.critic(o)
        #print(v.is_cuda)
        return torch.mean((v - rtg)**2)

    # Optimizers for actor and critic networks
    actor_optim = torch.optim.Adam(ac.actor.parameters(), lr=pi_lr)
    critic_optim = torch.optim.Adam(ac.critic.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Perform one policy gradient step and train_v_iter value function gradient steps
    def update():
        data = buf.get()

        loss_actor, actor_info = compute_loss_actor(data)
        actor_optim.zero_grad()
        loss_actor.backward()
        actor_optim.step()

        logger.store(LossPolicy=loss_actor.item(), **actor_info)

        for i in range(train_v_iters):
            loss_critic = compute_loss_critic(data)
            #print("Loss Critic on CUDA:", loss_critic.is_cuda)
            critic_optim.zero_grad()
            loss_critic.backward()
            critic_optim.step()

            logger.store(LossVfunc=loss_critic.item())

        # TODO: Logging of relevant info
    
    # Set up for environment interaction
    o, ep_ret, ep_len = env.reset(), 0, 0
    start_time = time.time()

    # Main training loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            #a, v, logp_a = ac.step(o)
            #o_prime, r, d, _ = env.step(a)
            #buf.store(o, a, r, v, logp_a)

            # Take one step in the environment
            a, v = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
            #print(a)
            o_prime, r, done, _ = env.step(a)
            
            # Store info of current step
            buf.store(o, a, r, v)

            logger.store(VfuncVals=v)

            # Update observation and episode counters
            o = o_prime
            ep_ret += r
            ep_len += 1

            epoch_ended = t == steps_per_epoch-1
            episode_maxed_out = ep_len == max_ep_len
            terminated = done or episode_maxed_out 

            if terminated or epoch_ended:
                if done:
                    last_v = 0
                else:
                    # If agent is still alive at end of episode we bootstrap the value of the next observation 
                    _, last_v = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
                    #last_v = ac.critic(o)
                
                buf.finish_path(last_v)

                if terminated:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                # Reset the environment for new episode
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Update policy and value function network at end of epoch
        update()


        # Save model 
        if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, itr=None)


        # Print out info summary of epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('LossPolicy', average_only=True)
        logger.log_tabular('LossVfunc', average_only=True)
        logger.log_tabular('VfuncVals', with_min_and_max=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
            
    return





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default='my_vpg')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("\n On device: ", device)    

    ac_kwargs = dict(hidden_sizes=[args.hid] * args.depth)
    my_vpg(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs,
           seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, gamma=args.gamma,
           logger_kwargs=logger_kwargs, device=device)