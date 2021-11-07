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
from kaggle_environments import make

from .environment import BaseEnvironment

class Constants:
    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"
    class DIRECTIONS:
        NORTH = "n"
        WEST = "w"
        SOUTH = "s"
        EAST = "e"
        CENTER = "c"
    class UNIT_TYPES:
        WORKER = 0
        CART = 1
    class RESOURCE_TYPES:
        WOOD = "wood"
        URANIUM = "uranium"
        COAL = "coal"


class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        # self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        # self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}


class Environment(BaseEnvironment):
    DIRECTIONS = Constants.DIRECTIONS
    ACTION = [DIRECTIONS.NORTH, DIRECTIONS.WEST, DIRECTIONS.EAST, DIRECTIONS.SOUTH]
    # DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    NUM_AGENTS = 2

    def __init__(self, args={}):
        super().__init__()
        self.env = make("lux_ai_2021")
        # print(self.env)
        self.reset()

    def reset(self, args={}):
        # print("*******", self.env.configuration)
        obs = self.env.reset()
        # print(obs)
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

    def direction(self, pos_from, pos_to):
        if pos_from is None or pos_to is None:
            return None
        x, y = pos_from // 11, pos_from % 11
        for i, d in enumerate(self.DIRECTION):
            nx, ny = (x + d[0]) % 7, (y + d[1]) % 11
            if nx * 11 + ny == pos_to:
                return i
        return None

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
        return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs['status'] == 'ACTIVE':
                return False
        return True

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        # print(self.obs_list)
        rewards = {o['observation']['player']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        action_map = {'N': Action.NORTH, 'S': Action.SOUTH, 'W': Action.WEST, 'E': Action.EAST}

        agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
        agent.last_action = action_map[self.ACTION[self.last_actions[player]][0]] if player in self.last_actions else None
        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def net(self):
        return LuxNet

    def observation(self, player=None):
        NUM_STATES = 5
        if player is None:
            player = 0
        
        obs = self.obs_list[-1][0]['observation']
        b = np.zeros((obs['width'], obs['height'], NUM_STATES), dtype=np.float32)

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
