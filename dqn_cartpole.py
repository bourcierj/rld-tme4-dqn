import argparse

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import torch
import random

import gym
from gym import wrappers, logger

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import timedelta

from dqn import DeepQlearningAgent
from dqn_hparams import hparams_cartpole as hparams

from utils import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deep Q-Network on Cartpole-v1"
                                                 "Gym environment")
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--no-rendering', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()  # get command line arguments
    # Load optimal hyperparameters
    for param, value in hparams.items():
        setattr(args, param, value)

    env = gym.make('CartPole-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = get_experiment_name('__DQN__Cartpole-v1__', args)
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter('runs/'+exp_name)

    # register the Agent
    state_dim = 4
    n_actions = 2
    args.lr_decay = 1.
    agent = DeepQlearningAgent(state_dim, n_actions,
                               args.replay_memory_capacity, args.ctarget, args.layers,
                               args.batch_size, args.lr, args.gamma, args.epsilon,
                               args.epsilon_decay, args.lr_decay, device=device)

    outdir = 'cartpole-v1/dqn-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    env.seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    episode_count = 2400
    show_every = 100
    reward = 0
    done = False
    env.verbose = True

    rsum = 0
    loss_sum = 0
    best_rsum = -1e10
    i_best_rsum = 0
    env._max_episode_steps = 10000
    print('Train for {} episodes: show every {} episodes'.format(episode_count, show_every))
    print('Using device ', device.type.upper())
    print('Name of experiment: {}'.format(exp_name))

    episode = 0
    since = time.time()
    for i in range(episode_count):
        obs = envm.reset()
        if args.no_rendering:
            env.verbose = False
        else:
            env.verbose = (i % show_every == 0 and i > 0)
        if env.verbose:
            # show episode
            env.render()
        j = 0  # length of episode
        rsum = 0  # cumulated reward
        loss_sum = 0  # cumulated loss
        Qsum = 0  # cumulated Q-value
        while True:

            action = agent.act(obs, reward, done)
            loss = agent.optimize(done)
            loss_sum += loss

            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if agent.lQ_value is not None:
                Qsum += agent.lQ_value
            if env.verbose:
                env.render()
            if done:
                print("Episode: {}: cumulated reward: {}, avg loss: {}, {} actions"
                      .format(str(i), rsum, loss_sum / j, str(j)))
                if writer is not None:
                    writer.add_scalar('Cumulated_Reward', rsum, i)
                    # writer.add_scalar('Avg_Reward', rsum / j, i)  # prints 1.0
                    writer.add_scalar('Avg_Loss', loss_sum / j, i)
                    writer.add_scalar('Avg_Q_Value', Qsum / j, i)

                # if rsum > best_rsum:
                #     best_rsum = rsum
                #     i_best_rsum = agent.checkpoint.epoch
                #     agent.checkpoint.save(suffix='_best')

                # agent.checkpoint.save()
                episode += 1
                break

    print("Finished. Trained on {} episodes, time: {}"
          .format(episode, timedelta(seconds=time.time() - since)))
    env.close()


#@TODO:
# - add checkpointing to save tthe model with highest cumulated reward (maybe bad, how to do better?)
# - run the experiment several times (eg 10), and observe variability in the curves / results
# - use stats tracker to do that (see RDFIA)
# - DONT USE EARLY STOPPING

