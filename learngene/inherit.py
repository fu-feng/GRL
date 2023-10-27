import torch
import gym
import numpy as np
import os
import random
from configparser import ConfigParser
from argparse import ArgumentParser
from algos.ppo import PPO
from utils.utils import Dict_cfg


def inherit_from_ancestor(ancestry, descendant, layer_pos, args):
    for i in layer_pos:
        if 0 <= i < args.layer_num_actor-1:
            layer_weight = f'actor.layers.{i}.weight'
            layer_bias = f'actor.layers.{i}.bias'
            descendant.state_dict()[layer_weight].copy_(ancestry.state_dict()[layer_weight])
            descendant.state_dict()[layer_bias].copy_(ancestry.state_dict()[layer_bias])
        elif i == args.layer_num_actor-1:
            layer_weight_mu = f'actor.last_layer.weight'
            layer_bias_mu = f'actor.last_layer.bias'
            layer_weight_std = f'actor.last_layer_std.weight'
            layer_bias_std = f'actor.last_layer_std.bias'
            descendant.state_dict()[layer_weight_mu].copy_(ancestry.state_dict()[layer_weight_mu])
            descendant.state_dict()[layer_bias_mu].copy_(ancestry.state_dict()[layer_bias_mu])
            descendant.state_dict()[layer_weight_std].copy_(ancestry.state_dict()[layer_weight_std])
            descendant.state_dict()[layer_bias_std].copy_(ancestry.state_dict()[layer_bias_std])
        else:
            print(f'ancestry model do not have enough layer num, the maximum number is {args.layer_num_actor-1}')
            exit()

    return descendant