import torch
from utils.logx import EpochLogger

import core

def my_vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
           steps_per_epoch=4000, epochs=50, gamma=0.99, 
           pi_lr=0.0003, vf_lr=0.001, train_v_iters=80,
           max_ep_len=1000, logger_kwargs=dict(), save_freq=10, device='cpu'):

    # TODO: Document entire my_vpg function

    # Optional TODO: Implement observation normalization to see if it makes a difference
    
    # Initialize Logger object used to log and print training info
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    env = env_fn()

    # Initialize Actor-Critic network
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Initialize buffer for collecting trajectories
    buf = core.VPGBuffer(steps_per_epoch, ac.obs_dim, ac.act_dim, gamma, device)


    # Estimate the policy gradient
    def compute_loss_actor(data):
        o, a, adv = data['obs'], data['act'], data['adv']

        pi, logp_a = ac.actor(o, a)

        # Store some useful info for tracking progress
        ent = pi.entropy().mean().item()

        loss_info = dict(Entropy=ent)
        
        # If algo is bork, maybe the gradient should be positive instead!!!!
        return - torch.mean(logp_a * adv), loss_info

    # Compute the MSE loss for estimating the Value function
    def compute_loss_critic(data):
        o, rtg = data['obs'], data['rtg']
        v = ac.critic(o)
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

        logger.store(LossActor=loss_actor.item(), **actor_info)

        for i in range(train_v_iters):
            loss_critic = compute_loss_critic(data)
            critic_optim.zero_grad()
            loss_critic.backward()
            critic_optim.step()

            logger.store(LossCritic=loss_critic.item())

        # TODO: Logging of relevant info
    

    # Main training loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            pass

    return





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()


    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'