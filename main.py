import numpy as np


class RPSAgent:

    def __init__(self, momentum, mem_len):
        self.momentum = momentum
        self.mem_len = mem_len
        self.memory = {}
        self.last_move = None
        self.cur_state = []

    def act(self, observation, configuration):
        if observation.get("lastOpponentAction") is None\
                or len(self.cur_state) < self.mem_len:
            self.last_move = np.random.randint(0, 3)
            return self.last_move

        self.cur_state.append(
            (self.last_move, observation.get("lastOpponentAction")))
        self.cur_state.pop(0)
