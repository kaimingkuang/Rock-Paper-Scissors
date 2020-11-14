from itertools import chain

import numpy as np


def get_score(x, y):
    score = np.where((x - y) % 2 != 0, x - y, np.sign(y - x))

    return score


class BaseAgent:

    def update_history(self, oppo_move):
        raise NotImplementedError

    def decide(self, obs, cfg):
        raise NotImplementedError

    def act(self, obs, cfg):
        raise NotImplementedError


class RandomAgent(BaseAgent):

    def update_history(self, oppo_move):
        pass

    def decide(self, obs, cfg):
        return np.random.randint(0, 3)

    def act(self, obs, cfg):
        self.last_move = self.decide(obs, cfg)

        return self.last_move

    def cal_latest_score(self, hist_len):
        return 0.5


class UniMarkovAgent(BaseAgent):

    def __init__(self, momentum, shift):
        self.momentum = momentum
        self.shift = shift
        self.history = []
        self.markov_chain = np.zeros((3, 3))

    def _update_markov_chain(self):
        memory_key = (self.history[-2][1], self.history[-1][1])
        self.markov_chain[memory_key[:-1]] *= self.momentum
        self.markov_chain[memory_key] += 1 - self.momentum

    def update_history(self, oppo_move):
        if oppo_move is not None:
            self.history[-1] = (self.history[-1], oppo_move)
        if len(self.history) >= 2:
            self._update_markov_chain()

    def _search_markov_chain(self, memory_key):
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
        if len(self.history) < 2:
            decision = np.random.randint(0, 3)
        else:
            memory_key = self.history[-1][1]
            decision = self._search_markov_chain(memory_key)

        decision = int((decision + self.shift) % 3)
        self.history.append(decision)

        return decision

    def act(self, obs, cfg):
        oppo_move = obs.get("lastOpponentAction")
        self.update_history(oppo_move)
        decision = self.decide()

        return decision


class BiMarkovAgent(UniMarkovAgent):

    def __init__(self, momentum, shift):
        self.momentum = momentum
        self.shift = shift
        self.history = []
        self.markov_chain = np.zeros((3, 3, 3))

    def _update_markov_chain(self):
        memory_key = (self.history[-2][0], self.history[-2][1],
            self.history[-1][1])
        self.markov_chain[memory_key[:-1]] *= self.momentum
        self.markov_chain[memory_key] += 1 - self.momentum

    def decide(self):
        if len(self.history) < 2:
            decision = np.random.randint(0, 3)
        else:
            memory_key = (self.history[-1][0], self.history[-1][1])
            decision = self._search_markov_chain(memory_key)

        decision = int((decision + self.shift) % 3)
        self.history.append(decision)

        return decision


class MetaAgent(BaseAgent):

    def __init__(self, agents, momentum):
        self.agents = agents
        self.momentum = momentum
        self.last_move = None

    def update_history(self, self_move, oppo_move):
        for i in range(len(self.agents)):
            self.agents[i].update_history(self.agents[i].last_move, oppo_move)

    def decide(self, obs, cfg):
        for i in range(len(self.agents)):
            self.agents[i].last_move = self.agents[i].decide(obs, cfg)

        recent_scores = [self.agents[i].cal_latest_score(self.recent_hist_len)
            for i in range(len(self.agents))]
        cur_best_agent = self.agents[np.argmax(recent_scores)]

        return cur_best_agent.last_move

    def act(self, obs, cfg):
        self.update_history(self.last_move, obs.get("lastOpponentAction"))
        self.last_move = self.decide(obs, cfg)

        return self.last_move


# def play_rps(observation, configuration):
#     return my_agent.act(observation, configuration)


if __name__ == "__main__":
    my_agent = BiMarkovAgent(0.9, 1)
    for i in range(100):
        obs = {"lastOpponentAction": None if i == 0 else np.random.randint(0, 3)}
        my_agent.act(obs, None)
