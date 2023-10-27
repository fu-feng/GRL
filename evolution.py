import numpy as np
import random
import torch
from itertools import combinations
import math
import os 
import multiprocessing as mp
from multiprocessing import Process
from train_agent import train
from configparser import ConfigParser
from argparse import ArgumentParser
import pickle
import torch
import gym
import envs_gene
import numpy as np
from utils.save import *
from utils.global_rewards import Global_rewards
from utils.utils import make_transition, Dict_cfg, RunningMeanStd
import copy

parser = ConfigParser()
parser.read('config.ini')
args = Dict_cfg(parser)

class Agent:
    def __init__(self, generation_id, agent_id, task_id, model_path, reward, 
                 score=0, father=None, father_gene=None) -> None:
        self.generation_id = generation_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.model_path = model_path
        self.reward = reward
        self.score = score
        self.father = father
        self.father_gene = father_gene
        if father_gene is not None:
            self.father_gene_feature_num = sum([math.sqrt(args.model_feature_num[i]) for i in self.father_gene])
        else:
            self.father_gene_feature_num = 0

class GenePool:
    def __init__(self, agent_list) -> None:
        self.genepool = {}
        for c in combinations([i for i in range(args.model_layer_num)], args.learngene_layer_num):
            self.genepool[c] = copy.deepcopy(random.sample(agent_list,args.genepool_maxnum))
            feature_num = sum([math.sqrt(args.model_feature_num[j]) for j in c])  #  参数量
            for j in self.genepool[c]:
                j.score = j.score / feature_num
        
        self.logits = {}
        self.update_logits()

    def update_logits(self):
        for c in combinations([i for i in range(args.model_layer_num)], args.learngene_layer_num):
            self.logits[c] = sum([i.score for i in self.genepool[c]])
        total_score = 0
        for s in self.logits.values():
            total_score += s
        for c in self.logits:
            self.logits[c] = self.logits[c] / total_score


    def generation_decay(self,alpha):
        for c in self.genepool:
            for i in range(args.genepool_maxnum):
                self.genepool[c][i].score *= alpha

class Generation(Process):
    def __init__(self, generation_id, agent_id, task_id, gene, parser, global_reward):
        super().__init__()
        self.generation_id = generation_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.gene = gene
        self.parser = parser
        self.global_reward = global_reward

    def run(self):
        print(f'Agent{self.generation_id}_{self.agent_id}_{self.task_id} starting...PID {os.getpid()}')
        # torch.cuda.set_device(self.gpu_id)
        args = Dict_cfg(self.parser)
        train(self.generation_id, self.agent_id, self.task_id, self.gene, args, self.global_reward)

def train_generation(generation_id, agent_ids, task_ids, gene, parser):
    parser.read('config.ini')
    info_manager = mp.Manager()
    gr = Global_rewards(info_manager)
    # gpu_num = torch.cuda.device_count()
    if gene:
        agent_procs = [Generation(generation_id, agent_id, task_ids[agent_id], gene[agent_id], parser, gr) for agent_id in agent_ids]
    else:
        agent_procs = [Generation(generation_id, agent_id, task_ids[agent_id], 'no', parser, gr) for agent_id in agent_ids]
    for p in agent_procs:
        p.start()

    for p in agent_procs:
        p.join()
    
    score_dict = gr.get_dict()
    score_list = save_agent_rewards(score_dict, generation_id, agent_ids, gene, args)
    return score_list

    
def train_random_init(agent_num):
    agent_list = []
    task_ids = []
    agent_ids = []
    for i in range(agent_num):
        task_ids.append(random.randint(0,args.task_num-1))  # 随机选择一个任务
        agent_ids.append(i)
    res = train_generation(0,agent_ids,task_ids, None, parser)
    for i in range(agent_num):
        agent_list.append(Agent(0, agent_ids[i], task_ids[i], res[i][0], res[i][1]))
    average_reward = sum([i.reward for i in agent_list]) / len(agent_list)
    agent_list = normalization(agent_list,True)
    return agent_list, average_reward


def normalization(agent_list, init=False):
    generation_average_reward = 0
    for i in agent_list:
        generation_average_reward += i.reward
    generation_average_reward /= len(agent_list)
    for i in range(args.task_num):
        task_agent_list = []
        for j in agent_list:
            if j.task_id == i:
                task_agent_list.append(j)
        if not task_agent_list:
            continue
        max_reward = max(task_agent_list,key=lambda x:x.reward).reward
        
        for j in task_agent_list:
            if init:
                j.score = j.reward / max_reward * generation_average_reward
            else:
                j.score = j.reward / max_reward * generation_average_reward / j.father_gene_feature_num
    return agent_list

def competition(agent_list):
    keep_agent_list = []
    temp_group = []
    for j in range(len(agent_list)):
        temp_group.append(agent_list[j])
        if len(temp_group) == args.competition_num:
            keep_agent_list.append(max(temp_group,key=lambda x:x.score))
            temp_group = []
    if len(temp_group) > 0:
        keep_agent_list.append(max(temp_group,key=lambda x:x.score))
    return keep_agent_list

def train_random_generation(agent_num, genepool, generation_id):
    agent_list = []
    task_ids = []
    agent_ids = []
    learngenes = []
    father_list = []
    father_gene_list = []
    for i in range(agent_num):
        task_ids.append(random.randint(0,args.task_num-1))  # 随机选择一个任务
        agent_ids.append(i)
        gene_layer = random.choices([j for j in genepool.genepool.keys()],weights=genepool.logits.values())[0]  # 随机选择一个基因层
        gene_agent = random.choices(
            genepool.genepool[gene_layer],
            weights=[j.score for j in genepool.genepool[gene_layer]])[0]  # 随机选择一个父亲
        learngenes.append([gene_layer, gene_agent.model_path])
        father_list.append(gene_agent)
        father_gene_list.append(gene_layer)
    res = train_generation(generation_id,agent_ids,task_ids,learngenes, parser)
    for i in range(agent_num):
        agent_list.append(Agent(generation_id, agent_ids[i], task_ids[i], res[i][0], res[i][1], 
                                father=father_list[i], father_gene=father_gene_list[i]))
    average_reward = sum([i.reward for i in agent_list]) / len(agent_list)
    agent_list = normalization(agent_list)
    return agent_list, average_reward

def update_father_score(agent_list, beta):
    for i in agent_list:
        temp_beta = beta
        score = i.score
        node = i
        similarity_beta = 1.0
        while node.father is not None:
            node.father.score += temp_beta * score * similarity_beta
            similarity_beta = agent_similarity(node,node.father)
            node = node.father
            temp_beta *= temp_beta

def agent_similarity(agent1,agent2):
    if agent2.father is None:
        return 0
    gene1 = agent1.father_gene
    gene2 = agent2.father_gene
    union = set(gene1) | set(gene2)
    intersection = set(gene1) & set(gene2)
    res = sum([math.sqrt(args.model_feature_num[i]) for i in intersection]) / \
        sum([math.sqrt(args.model_feature_num[i]) for i in union])
    return res

def extract_gene(agent_list, genepool):
    new_pool = {}
    for i in agent_list:
        weight = []
        for c in genepool.logits.keys():
            if c == i.father_gene:
                weight.append(genepool.logits[c]+1)
            else:
                weight.append(genepool.logits[c])
        gene_layer = random.choices([i for i in genepool.genepool.keys()],weights=weight)[0]  # 随机选择一个基因层
        feature_num = sum(math.sqrt(args.model_feature_num[j]) for j in gene_layer)
        i.score = i.score * i.father_gene_feature_num / feature_num
        if gene_layer in new_pool:
            new_pool[gene_layer].append(i)
        else:
            new_pool[gene_layer] = []
            new_pool[gene_layer].append(i)
    return new_pool

def merge(genepool,new_pool):
    for gene_layer in genepool.genepool.keys():
        genepool.genepool[gene_layer].sort(key=lambda x:x.score,reverse=True) # 从大到小
        if gene_layer in new_pool:
            for i in range(args.obsolete_num):
                new_pool[gene_layer].append(genepool.genepool[gene_layer].pop())
            new_pool[gene_layer].sort(key=lambda x:x.score) # 从小到大
            for i in range(args.obsolete_num):
                genepool.genepool[gene_layer].append(new_pool[gene_layer].pop())
    genepool.update_logits()
    return genepool

def evolution():
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    use_checkpoint = False
    use_checkpoint_id = 0
    if use_checkpoint:
        with open(args.checkpoint_dir.format(use_checkpoint_id, use_checkpoint_id),"rb") as f:
            checkpoint = pickle.load(f)
        print(f'------------Get the information of generation {checkpoint["i"]}------------')
        genepool = checkpoint["genepool"]
        start_generation_id = checkpoint["i"] + 1
        best_reward = checkpoint["best_reward"]
        patience = checkpoint["patience"]
        
    else:
        print(f'------------Generation {0}------------')
        agent_list, average_reward_all = train_random_init(args.init_agent_num)
        agent_list = competition(agent_list)
        genepool = GenePool(agent_list)
        best_reward = 0
        patience = 0
        start_generation_id = 1
        save_genepool(genepool, average_reward_all, 0, args)
        save_checkpoint(genepool, 0, best_reward, patience, args)

    for i in range(start_generation_id, args.iter_num+1):
        print(f'------------Generation {i}------------')
        agent_list, average_reward_all = train_random_generation(args.generation_agent_num, genepool, i) 
        agent_list = competition(agent_list)  
        genepool.generation_decay(args.alpha)  
        update_father_score(agent_list, args.beta)  
        new_pool = extract_gene(agent_list,genepool)  
        genepool = merge(genepool,new_pool) 

        save_genepool(genepool, average_reward_all, i, args)
        average_reward = sum([i.reward for i in agent_list]) / len(agent_list)
        if average_reward > best_reward:
            best_reward = average_reward
            patience = 0
        else:
            patience += 1
        if patience > args.patience_num:
            print("Early Stop!")
            break
        save_checkpoint(genepool, i, best_reward, patience, args)
        
    best_gene = max(genepool.logits,key=lambda x:genepool.logits[x])
    print("The best gene is" + str(best_gene))

if __name__ == "__main__":
    evolution()
    print("finish!")