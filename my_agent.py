from itertools import chain

import numpy as np


def get_score(x, y):
    score = np.where((x - y) % 2 != 0, x - y, np.sign(y - x))

    return score


class BaseAgent:

    def update_history(self, self_move, oppo_move):
        raise NotImplementedError

    def decide(self, obs, cfg):
        raise NotImplementedError

    def act(self, obs, cfg):
        raise NotImplementedError

    def cal_latest_score(self, hist_len):
        if len(self.history) == 0:
            return 0

        score = get_score(np.array([x[0] for x in self.history[-hist_len:]]),
            np.array([x[1] for x in self.history[-hist_len:]])).sum()

        return score


class MarkovAgent(BaseAgent):

    def __init__(self, mem_len, shift):
        self.mem_len = mem_len
        self.shift = shift
        self.history = []
        markov_chain_shape = tuple([3] * (mem_len * 2 + 1))
        self.markov_chain = np.zeros(markov_chain_shape)
        self.last_move = None

    def update_history(self, self_move, oppo_move):
        if oppo_move is not None:
            self.history.append((self_move, oppo_move))

        if len(self.history) > self.mem_len:
            self._update_markov_chain()

    def _get_memory_idx(self):
        return tuple(chain(*self.history[-(self.mem_len + 1):-1],
            (self.history[-1][1],)))

    def _update_markov_chain(self):
        memory_idx = self._get_memory_idx()
        self.markov_chain[memory_idx] += 1

    def _search_markov_chain(self):
        # if no such memory found, act randomly
        memory_idx = self._get_memory_idx()[:-1]
        if np.all(self.markov_chain[memory_idx] == 0):
            return np.random.randint(0, 3)

        max_prob = np.max(self.markov_chain[memory_idx])
        if np.sum(self.markov_chain[memory_idx] == max_prob) > 1:
            rnd_move = np.random.choice(np.argwhere(
                self.markov_chain[memory_idx] == max_prob).squeeze().tolist())
            return int((rnd_move + self.shift) % 3)

        return int((np.argmax(self.markov_chain[memory_idx]) + self.shift) % 3)

    def decide(self, obs, cfg):
        oppo_last_move = obs.get("lastOpponentAction")
        # act randomly if it's the first round
        if oppo_last_move is None:
            return np.random.randint(0, 3)

        # if the current history is too short, act randomly
        if len(self.history) <= self.mem_len:
            return np.random.randint(0, 3)

        # search for appropriate act in the markov chain
        decision = self._search_markov_chain()

        return decision

    def act(self, obs, cfg):
        self.update_history(self.last_move, obs.get("lastOpponentAction"))
        self.last_move = self.decide(obs, cfg)

        return self.last_move


class RandomAgent(BaseAgent):

    def __init__(self, mc_times=100):
        self.history = []
        self.mc_times = mc_times
        self.last_move = None

    def update_history(self, self_move, oppo_move):
        if oppo_move is not None:
            self.history.append((self_move, oppo_move))

    def decide(self, obs, cfg):
        return np.random.randint(0, 3)

    def act(self, obs, cfg):
        self.update_history(self.last_move, obs.get("lastOpponentAction"))
        self.last_move = self.decide(obs, cfg)

        return self.last_move

    def cal_latest_score(self, hist_len):
        if len(self.history) == 0:
            return 0

        hist_len = min(hist_len, len(self.history))
        oppo_plays = np.tile(np.array(self.history[-hist_len:])[:, 1],
            (self.mc_times, 1))
        rnd_plays = np.random.randint(0, 3, (self.mc_times, hist_len))
        score = get_score(rnd_plays, oppo_plays).sum(axis=1).mean()

        return score


class MetaAgent(BaseAgent):

    def __init__(self, agents, recent_hist_len):
        self.agents = agents
        self.recent_hist_len = recent_hist_len
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


my_agent = MarkovAgent(2, 1)


def play_rps(observation, configuration):
    return my_agent.act(observation, configuration)
