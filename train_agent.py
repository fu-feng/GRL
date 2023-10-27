import os 
from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
from gym.wrappers import Monitor
import envs_gene
import numpy as np
import multiprocessing as mp
import random

from algos.ppo import PPO
from utils.utils import make_transition, Dict_cfg, RunningMeanStd
from learngene.inherit import inherit_from_ancestor
from utils.global_rewards import Global_rewards

def wrap_env(env, agent_id):
  env = Monitor(env, './video/{}'.format(agent_id), video_callable=lambda epoch_id: epoch_id == 10, force=True)
  return env

def train(generation_id, agent_id, task_id, load_gene, args, global_reward):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = (args.out_dir + args.generation_dir + args.agent_dir).format(generation_id, generation_id, agent_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    env = gym.make(args.env_name, task_id=task_id, args=args)
    # env = wrap_env(env, agent_id)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    state_rms = RunningMeanStd(shape=state_dim)

    agent = PPO(device, state_dim, action_dim, args)
    agent = agent.to(device)

    if load_gene != 'no':
        ancestry = PPO(device, state_dim, action_dim, args)
        ancestry_model = torch.load(load_gene[1])
        ancestry.load_state_dict(ancestry_model[0])
        agent = inherit_from_ancestor(ancestry, agent, load_gene[0], args)

    score_lst = []
    score_epoch_lst = []
    state_lst = []
    reward_lst = []
    eposide = 0

    score = 0.0
    state_ = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    average_score_save = []
    
    for n_epi in range(1, args.epochs+1):
        for t in range(args.traj_length):
            if n_epi > args.render_after_epoch and args.render: 
                env.render(mode='human')
            state_lst.append(state_)
            mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))

            dist = torch.distributions.Normal(mu,sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1,keepdim = True)
            
            next_state_, reward, done, info = env.step(action.cpu().numpy())
        
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,\
                                        action.cpu().numpy(),\
                                        np.array([reward*args.reward_scaling]),\
                                        next_state,\
                                        np.array([done]),\
                                        log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += reward
            reward_lst.append(reward)
            if done:
                eposide += 1
                state_ = env.reset()
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                score_epoch_lst.append(score)
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net()
        state_rms.update(np.vstack(state_lst))
        if n_epi%args.print_interval==0:
            print("agent{}_{}_{} of episode :{}, avg score : {:.1f}".format(generation_id, agent_id, task_id, n_epi, sum(score_epoch_lst)/len(score_epoch_lst)))
            average_score_save.append(sum(score_epoch_lst)/len(score_epoch_lst))
            score_epoch_lst = []
        if n_epi%args.save_interval==0:
            torch.save([agent.state_dict(), agent.data, state_rms], save_path + '/{}_{}_{}.pth'.format(generation_id, agent_id, task_id))
            np.save(save_path + '/score_epoch_{}_{}.npy'.format(generation_id, agent_id), average_score_save)
            np.save(save_path + '/score_episode_{}_{}.npy'.format(generation_id, agent_id), score_lst)
            np.save(save_path + '/rewards_{}_{}.npy'.format(generation_id, agent_id), reward_lst)

    agent_score = 0
    cal_num = 0
    for s in range(args.epoch_calculate_score):
        if average_score_save[-(s+1)] > args.epoch_score_threshold:
            agent_score += average_score_save[-(s+1)]
            cal_num += 1
    if cal_num == 0:
        agent_score = args.epoch_score_threshold
    else:
        agent_score /= cal_num
    agent_score += args.init_reward
    global_reward.set_value(save_path+'/{}_{}_{}.pth'.format(generation_id, agent_id, task_id), round(agent_score, 1))


