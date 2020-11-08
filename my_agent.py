import numpy as np


def get_score(x, y):
    score = np.where((x - y) % 2 != 0, x - y, np.sign(y - x))
    
    return score


def kl_divergence(p0, p1):
    return -p0 * np.log(p1 / (p0 + 1e-8))


def symmetric_cross_entropy(p0, p1):
    return (kl_divergence(p0, p1) + kl_divergence(p1, p0)).mean()


class BaseAgent:

    def __init__(self, *args, **kwargs):
        self.history = []

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

        if len(self.history) > self.mem_len:
            self._update_markov_chain(self._make_markov_key(self.history[-3:-1]),
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


class YouGoFirst(BaseAgent):

    def __init__(self, momentum, sce_thresh, t_wait):
        self.momentum = momentum 
        self.sce_thresh = sce_thresh
        self.t_wait = t_wait
        self.history = []
        self.last_move = None
        self.pred_move = 0
        self.recent_pred_v_oppo = []
        self.recent_score = 0
        self.markov_matrix = np.zeros((3, 3, 3))
        self.rand_markov_matrix = np.ones(3) / 3
    
    def _update_markov_matrix(self):
        self.markov_matrix[self.history[-2][0], self.history[-2][1]] =\
            self.markov_matrix[self.history[-2][0], self.history[-2][1]]\
            * self.momentum

        self.markov_matrix[
            self.history[-2][0],
            self.history[-2][1],
            self.history[-1][1]
        ] += (1 - self.momentum)

    def update_history(self, self_move, oppo_move):
        if oppo_move is not None:
            self.history.append((self_move, oppo_move))

        if len(self.history) >= 2:
            self._update_markov_matrix()
            if len(self.recent_pred_v_oppo) > 10:
                self.recent_pred_v_oppo.pop(0)
            self.recent_pred_v_oppo.append(
                get_score(self.pred_move, oppo_move))
            self.recent_score = sum(self.recent_pred_v_oppo)
            self.pred_move = (np.argmax(self.markov_matrix[
                self.history[-1][0], self.history[-1][1]
                ]) + 1) % 3

    def decide(self, obs, cfg):
        if len(self.history) >= self.t_wait:
            cond_prob = self.markov_matrix[
                self.history[-1][0],
                self.history[-1][1]
            ]
            cond_prob /= cond_prob.sum()
            sce = symmetric_cross_entropy(cond_prob, self.rand_markov_matrix)
            if sce >= self.sce_thresh and self.recent_score >= 0:
                return int((np.argmax(cond_prob) + 1) % 3)
            else:
                return np.random.randint(0, 3)
        else:
            return np.random.randint(0, 3)

    def act(self, obs, cfg):
        self.update_history(self.last_move, obs.get("lastOpponentAction"))
        self.last_move = self.decide(obs, cfg)

        return self.last_move


# my_agent = MetaAgent([RandomAgent(), MarkovMomentum(0.5, 2)], 20)
my_agent = YouGoFirst(0.5, 0.2, 100)


def play_rps(observation, configuration):
    return my_agent.act(observation, configuration)
