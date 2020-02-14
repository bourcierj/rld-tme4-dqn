import argparse

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import torch
import random

import gym
import gridworld
from gym import wrappers, logger

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import timedelta

from dqn import DeepQlearningAgent
from dqn_hparams import hparams_gridworld

from utils import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deep Q-Network on Gridworld-v0"
                                                 "Gym environment")
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--no-rendering', action='store_true')
    parser.add_argument('--plan', type=int, default=0)
    return parser.parse_args()


def get_features(obs):
    """Extract a feature vector from a gridworld map.
    Returns:
        (torch.Tensor): vector of features containing:
            - the position of the agent
            - the position of the yellow elements
            - the position of the pink elements
    """
    state = np.zeros((3, obs.shape[0], obs.shape[1]))
    state[0] = np.where(obs == 2, 1, state[0])
    state[1] = np.where(obs == 4, 1, state[1])
    state[2] = np.where(obs == 6, 1, state[2])
    # flattened vector shape: 6x6x3 = 108
    return state.reshape(-1)


if __name__ == '__main__':

    args = parse_args()  # get command line arguments
    # Load optimal hyperparameters
    for param, value in hparams_gridworld.items():
        setattr(args, param, value)

    hparams = get_hyperparams_dict(args)
    exp_name = get_experiment_name('__DQN__Gridworld-v0-plan{}__'.format(args.plan),
                                   hparams)

    env = gym.make('gridworld-v0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter('runs/'+exp_name)

    # register the Agent
    state_dim = get_features(env.reset()).size
    n_actions = 4
    agent = DeepQlearningAgent(state_dim, n_actions,
                               args.replay_memory_capacity, args.ctarget, args.layers,
                               args.batch_size, args.lr, args.gamma, args.epsilon,
                               args.epsilon_decay, args.lr_decay, device=device)

    outdir = 'gridworld-v0/dqn-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan{}.txt".format(args.plan),
                {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    episode_count = args.episode_count
    show_every = 100
    reward = 0
    done = False
    FPS = 0.0001
    env.verbose = True

    rsum = 0
    loss_sum = 0
    best_rsum = -1e10
    i_best_rsum = 0

    print('Train for {} episodes; show every {} episodes if rendering is enabled.'.format(episode_count, show_every))
    print('Using device ', device.type.upper())
    print('Name of experiment: {}'.format(exp_name))

    episode = 0
    since = time.time()
    i = 0
    for i in range(episode_count):
        obs = envm.reset()
        if args.no_rendering:
            env.verbose = False
        else:
            env.verbose = (i % show_every == 0 and i > 0)
        if env.verbose:
            # show episode
            env.render(FPS)
        j = 0  # length of episode
        rsum = 0  # cumulated reward
        loss_sum = 0  # cumulated loss
        Qsum = 0  # cumulated Q-value
        while True:

            action = agent.act(get_features(obs), reward, done)
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
                avg_loss = loss_sum / j
                avg_Q = Qsum / j
                if writer is not None:
                    writer.add_scalar('Cumulated_Reward', rsum, i)
                    # writer.add_scalar('Avg_Reward', rsum / j, i)  # prints 1.0
                    writer.add_scalar('Avg_Loss', avg_loss, i)
                    writer.add_scalar('Avg_Q_Value', avg_Q, i)

                if rsum > best_rsum:
                    best_rsum = rsum
                    i_best_rsum = i
                    best_rsum_loss = avg_loss
                break

    print("Finished. Trained on {} episodes, time: {}.\n Max cumulated reward: {} (episode {}, with loss: {}) "
          .format(i, timedelta(seconds=time.time() - since), best_rsum, i_best_rsum, best_rsum_loss))

    if writer is not None:
        writer.add_hparams(hparams, {'hparam/Cumulated_Reward': best_rsum,
                                     'hparam/Episode': i_best_rsum,
                                     'hparam/Avg_Loss': best_rsum_loss})
        writer.close()

    env.close()


#@TODO:
# - add checkpointing to save tthe model with highest cumulated reward (maybe bad, how to do better?)
# - run the experiment several times (eg 10), and observe variability in the curves / results
# - use stats tracker to do that (see RDFIA)
# - DONT USE EARLY STOPPING

