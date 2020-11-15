from itertools import chain

import numpy as np


def get_score(x, y):
    score = np.where((x - y) % 2 != 0, x - y, np.sign(y - x))

    return score


class BaseAgent:

    def __init__(self):
        self.history = []

    def update_history(self, oppo_move):
        raise NotImplementedError

    def decide(self, obs, cfg):
        raise NotImplementedError

    def act(self, obs, cfg):
        raise NotImplementedError

    def get_recent_score(self, momentum):
        if not hasattr(self, "recent_score"):
            self.recent_score = 0
        else:
            last_match_score = get_score(self.history[-1][0],
                self.history[-1][1])
            self.recent_score = self.recent_score * momentum\
                + (1 - momentum) * last_match_score

        return self.recent_score


class RandomAgent(BaseAgent):

    def __init__(self):
        self.history = []

    def update_history(self, oppo_move):
        self.history[-1] = (self.history[-1], oppo_move)

    def decide(self):
        decision = np.random.randint(0, 3)
        self.history.append(decision)

        return decision

    def act(self, obs, cfg):
        decision = self.decide(obs, cfg)

        return decision
    
    def get_recent_score(self, momentum):
        return 0


class MarkovAgent(BaseAgent):

    def __init__(self, momentum, shift):
        self.momentum = momentum
        self.shift = shift
        self.history = []
        self.markov_chain = np.zeros((3, 3, 3))

    def update_markov_chain(self):
        NotImplementedError

    def update_history(self, oppo_move):
        if oppo_move is not None:
            self.history[-1] = (self.history[-1], oppo_move)
        if len(self.history) >= 2:
            self.update_markov_chain()

    def search_markov_chain(self, memory_key):
        if np.all(self.markov_chain[memory_key] == 0):
            return np.random.randint(0, 3)

        max_cnt = self.markov_chain[memory_key].max()
        if (self.markov_chain[memory_key] == max_cnt).sum() > 1:
            max_oppo_acts = np.argwhere(
                self.markov_chain[memory_key] == max_cnt).squeeze()
            return np.random.choice(max_oppo_acts)
        else:
            return np.argmax(self.markov_chain[memory_key])

    def decide(self):
        raise NotImplementedError

    def act(self, obs, cfg):
        oppo_move = obs.get("lastOpponentAction")
        self.update_history(oppo_move)
        decision = self.decide()

        return decision


class OpponentMarkovAgent(MarkovAgent):

    def update_markov_chain(self):
        memory_key = (self.history[-2][0], self.history[-2][1],
            self.history[-1][1])
        self.markov_chain *= self.momentum
        self.markov_chain[memory_key] += 1 - self.momentum

    def decide(self):
        if len(self.history) < 2:
            decision = np.random.randint(0, 3)
        else:
            memory_key = (self.history[-1][0], self.history[-1][1])
            decision = self.search_markov_chain(memory_key)

        decision = int((decision + self.shift) % 3)
        self.history.append(decision)

        return decision


class SelfMarkovAgent(MarkovAgent):

    def update_markov_chain(self):
        memory_key = (self.history[-2][0], self.history[-2][1],
            self.history[-1][0])
        self.markov_chain *= self.momentum
        self.markov_chain[memory_key] += 1 - self.momentum

    def decide(self):
        if len(self.history) < 2:
            decision = np.random.randint(0, 3)
        else:
            memory_key = (self.history[-1][0], self.history[-1][1])
            decision = self.search_markov_chain(memory_key)

        decision = int((decision + self.shift) % 3)
        self.history.append(decision)

        return decision


class MetaAgent(BaseAgent):

    def __init__(self, agents, momentum):
        self.agents = agents
        self.momentum = momentum
        self.history = []

    def update_history(self, oppo_move):
        if oppo_move is not None:
            self.history[-1] = (self.history[-1], oppo_move)
            for i in range(len(self.agents)):
                self.agents[i].update_history(oppo_move)

    def decide(self):
        recent_scores = [self.agents[i].get_recent_score(self.momentum)
            for i in range(len(self.agents))]
        for i in range(len(self.agents)):
            self.agents[i].decide()

        decision = self.agents[np.argmax(recent_scores)].history[-1]
        self.history.append(decision)

        return decision

    def act(self, obs, cfg):
        oppo_move = obs.get("lastOpponentAction")
        self.update_history(oppo_move)
        decision = self.decide()

        return decision


my_agent = MetaAgent([
    RandomAgent(),
    OpponentMarkovAgent(0.5, 1),
    OpponentMarkovAgent(0.5, 0),
    OpponentMarkovAgent(0.5, 2),
    SelfMarkovAgent(0.5, 2),
    SelfMarkovAgent(0.5, 1),
    SelfMarkovAgent(0.5, 0),
], 0.7)


def play_rps(observation, configuration):
    return my_agent.act(observation, configuration)
