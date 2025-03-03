import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import Config

class Actor(nn.Module): #PolicyNN
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], action_dim),
            nn.Tanh()
        )
        self.maxaction = maxaction

    def forward(self, state):
        actions = self.model(state) * self.maxaction
        return actions

class Q_Critic(nn.Module): #CriticNN
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim + action_dim, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], 1)
        )

    def forward(self, state, action):  
        input_state_action = torch.cat((state, action), 1)
        return self.value(input_state_action)
    

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)


#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')