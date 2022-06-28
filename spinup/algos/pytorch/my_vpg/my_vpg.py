import torch
from utils.logx import EpochLogger

def my_vpg(env_fn, actor_critic, ac_kwargs=dict(), seed=0, 
           steps_per_epoch=4000, epochs=50, gamma=0.99, 
           pi_lr=0.0003, vf_lr=0.001, train_v_iters=80,
           max_ep_len=1000, logger_kwargs=dict(), save_freq=10):

    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    
    
    
    return


# TODO implement observation normalization to see if it makes a difference




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