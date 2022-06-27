import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()


    if args.cuda and torch.cuda.is_available():
        device = 'gpu:0'
    else:
        device = 'cpu'