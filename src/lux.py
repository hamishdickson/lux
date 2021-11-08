# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of lux environment from kaggle

import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# You need to install kaggle_environments, requests
# from kaggle_environments import make
from luxai2021.game.game import Game
from luxai2021.game.actions import *
from luxai2021.game.constants import LuxMatchConfigs_Default

from .environment import BaseEnvironment



class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(20, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p


class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        configs = LuxMatchConfigs_Default
        self.env = Game(configs)
        self.ACTION = [Constants.DIRECTIONS.CENTER, Constants.DIRECTIONS.NORTH, Constants.DIRECTIONS.EAST, Constants.DIRECTIONS.SOUTH, Constants.DIRECTIONS.WEST]
        self.reset()

    def reset(self, args={}):
        obs = self.env.to_state_object()
        obs['status'] = 'OVER' if self.env.match_over() else 'ACTIVE'
        self.env.reset()
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)


    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]['observation']
        return obs

    def step(self, actions):
        # state transition
        obs = self.env.step([self.action2str(actions.get(p, None) or 0) for p in self.players()])
        self.update((obs, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return [p for p in self.players()]

    def terminal(self):
        if self.obs_list[-1]['status'] == 'ACTIVE':
            return False
        return True

    def outcome(self):
        # return terminal outcomes
        rewards = {o['observation']['player']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        outcomes[0] = 1.0 if rewards[0] > rewards[1] else -1.0
        outcomes[1] = 1.0 if rewards[1] > rewards[0] else -1.0

        # tie break
        if rewards[0] == rewards[1]:
            outcomes[0] = 0.0
            outcomes[1] = 0.0

        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return [0, 1]

    def net(self):
        return LuxNet

    def observation(self, player=None):
        NUM_STATES = 5
        if player is None:
            player = 0
        
        obs = self.obs_list[-1][0]['observation']
        # biggest board is 32,32 so build that and mask out unused area when smaller
        b = np.zeros((32, 32, NUM_STATES), dtype=np.float32)

        # print(obs)

        for p, update in enumerate(obs['updates']):
            us = update.split(" ")

            # 1 is player number
            # 2 is board size
            if len(us) == 3:
                # research points
                b[int(us[1]), int(us[2]), 0] = 1.
            elif (len(us) == 5) and (us[0] == 'r'):
                if us[1] == 'coal':
                    b[int(us[2]), int(us[3]), 2] = float(us[4])
                elif us[1] == 'wood':
                    b[int(us[2]), int(us[3]), 3] = float(us[4])
                else:
                    b[int(us[2]), int(us[3]), 4] = float(us[4])

        return b


if __name__ == '__main__':
    e = Environment()
    for _ in range(1):
        e.reset()
        while not e.terminal():
            print(e)
            actions = {p: e.legal_actions(p) for p in e.turns()}
            print([[e.action2str(a, p) for a in alist] for p, alist in actions.items()])
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e)
        print(e.outcome())
