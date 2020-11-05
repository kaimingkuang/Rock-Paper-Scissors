import numpy as np


class RPSAgent:

    def __init__(self, momentum, mem_len):
        self.momentum = momentum        # memory momentum
        self.mem_len = mem_len          # memory length
        self.markov_chain = {}          # markov chain status graph
        self.last_move = None           # agent's last move
        self.cur_memory = []            # current memory

    @staticmethod
    def _make_markov_key(cur_memory):
        return ",".join([repr(x) for x in cur_memory])

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
            return int(np.random.choice(max_act))

        # else just pick the most probable act
        return max_act[0]

    def act(self, observation, configuration):
        oppo_last_move = observation.get("lastOpponentAction")
        # act randomly if it's the first round
        if oppo_last_move is None:
            self.last_move = np.random.randint(0, 3)
            return self.last_move

        # add the latest update to the memory
        self.cur_memory.append((self.last_move, oppo_last_move))

        # if the current memory is too short, act randomly
        if len(self.cur_memory) <= self.mem_len:
            self.last_move = np.random.randint(0, 3)
            return self.last_move

        # update markov chain
        self._update_markov_chain(self._make_markov_key(self.cur_memory[:-1]),
            oppo_last_move)

        # pop the earliest memory
        self.cur_memory.pop(0)

        # search for appropriate act in the markov chain
        self.last_move = self._search_markov_chain(self._make_markov_key(
            self.cur_memory))

        return self.last_move


rps_agent = RPSAgent(0.9, 2)


def rps_play(observation, configuration):
    return rps_agent.act(observation, configuration)


# if __name__ == "__main__":
#     for i in range(100):
#         if i == 0:
#             rps_play({}, None)
#         else:
#             rps_play({"lastOpponentAction": np.random.randint(0, 3)}, None)

#         print("=" * 16, f"Round {i + 1}", "=" * 16)
#         print(rps_agent.markov_chain)
