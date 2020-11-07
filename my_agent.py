import numpy as np
from kaggle_environments.envs.rps.utils import get_score


class BaseAgent:

    def update_history(self, self_move, oppo_move):
        raise NotImplementedError

    def decide(self, obs, cfg):
        raise NotImplementedError

    def act(self, obs, cfg):
        raise NotImplementedError


class MarkovMomentum(BaseAgent):

    def __init__(self, momentum, mem_len):
        self.momentum = momentum
        self.mem_len = mem_len
        self.history = []
        self.markov_chain = {}
        self.last_move = None
    
    @staticmethod
    def _make_markov_key(cur_memory):
        return ",".join([repr(x) for x in cur_memory])
    
    def update_history(self, self_move, oppo_move):
        if oppo_move is not None:
            self.history.append((self_move, oppo_move))

        if len(self.history) <= self.mem_len:
            self._update_markov_chain(self._make_markov_key(self.history[:-1]),
                oppo_move)

    def _update_markov_chain(self, cur_memory_key, oppo_last_move):
        if self.markov_chain.get(cur_memory_key) is None:
            self.markov_chain[cur_memory_key] = {
                oppo_last_move: self.momentum * 1.0
            }
        else:
            for k, v in self.markov_chain[cur_memory_key].items():
                self.markov_chain[cur_memory_key][k] = v * self.momentum
            self.markov_chain[cur_memory_key][oppo_last_move] =\
                self.markov_chain[cur_memory_key].get(oppo_last_move, 0)\
                + (1 - self.momentum)

    def _search_markov_chain(self, cur_memory):
        # if no such memory found, act randomly
        if self.markov_chain.get(cur_memory) is None:
            return np.random.randint(0, 3)

        # if there is a tie between two acts, choose randomly between them
        max_cnt = 0
        max_act = []
        for k, v in self.markov_chain[cur_memory].items():
            if v > max_cnt:
                max_cnt = v
                max_act = [k]
            elif v == max_act:
                max_act.append(k)
            else:
                pass

        if len(max_act) > 1:
            return int((np.random.choice(max_act) + 1) % 3)
        else:
            return int((max_act[0] + 1) % 3)
    
    def decide(self, obs, cfg):
        oppo_last_move = obs.get("lastOpponentAction")
        # act randomly if it's the first round
        if oppo_last_move is None:
            self.last_move = np.random.randint(0, 3)
            return self.last_move

        # if the current history is too short, act randomly
        if len(self.history) <= self.mem_len:
            self.last_move = np.random.randint(0, 3)
            return self.last_move

        # search for appropriate act in the markov chain
        decision = self._search_markov_chain(self._make_markov_key(
            self.history[-2:]))

        return decision

    def act(self, obs, cfg):
        self.update_history(self.last_move, obs.get("lastOpponentAction"))
        self.last_move = self.decide(obs, cfg)

        return self.last_move


my_agent = MarkovMomentum(0.5, 2)


def play_rps(observation, configuration):
    return my_agent.act(observation, configuration)
