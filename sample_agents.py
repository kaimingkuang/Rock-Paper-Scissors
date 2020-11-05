import random

import numpy as np
from kaggle_environments.envs.rps.utils import get_score


class CopyCat:

    def act(self, obs, cfg):
        if obs.step > 0:
            return obs.lastOpponentAction
        else:
            return random.randrange(0, cfg.signs)


class Reactionary:

    def __init__(self):
        self.last_react_action = None

    def act(self, obs, cfg):
        if obs.step == 0:
            self.last_react_action = random.randrange(0, cfg.signs)
        elif get_score(self.last_react_action, obs.lastOpponentAction) <= 1:
            self.last_react_action = (obs.lastOpponentAction + 1) % cfg.signs

        return self.last_react_action


class CounterReactionary:

    def __init__(self):
        self.last_react_action = None

    def act(self, obs, cfg):
        if obs.step == 0:
            self.last_counter_action = random.randrange(0, cfg.signs)
        elif get_score(self.last_counter_action, obs.lastOpponentAction) == 1:
            self.last_counter_action =\
                (self.last_counter_action + 2) % cfg.signs
        else:
            self.last_counter_action =\
                (obs.lastOpponentAction + 1) % cfg.signs

        return self.last_counter_action


class RandomAgent:

    def act(self, obs, cfg):
        return np.random.randint(0, 3)
