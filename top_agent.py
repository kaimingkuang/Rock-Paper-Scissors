import random

# how many steps in a row are in the pattern (multiplied by two)
memory_length = 6
# current memory of the agent
current_memory = []
# list of memory patterns
memory_patterns = []

def find_pattern(memory):
    """ find appropriate pattern in memory """
    for pattern in memory_patterns:
        actions_matched = 0
        for i in range(memory_length):
            if pattern["actions"][i] == memory[i]:
                actions_matched += 1
            else:
                break
        # if memory fits this pattern
        if actions_matched == memory_length:
            return pattern
    # appropriate pattern not found
    return None

def my_agent(obs, conf):
    """ your ad here """
    # if it's not first step, add opponent's last action to agent's current memory
    if obs["step"] > 0:
        current_memory.append(obs["lastOpponentAction"])
    # if length of current memory is bigger than necessary for a new memory pattern
    if len(current_memory) > memory_length:
        # get momory of the previous step
        previous_step_memory = current_memory[:memory_length]
        previous_pattern = find_pattern(previous_step_memory)
        if previous_pattern == None:
            previous_pattern = {
                "actions": previous_step_memory.copy(),
                "opp_next_actions": [
                    {"action": 0, "amount": 0, "response": 1},
                    {"action": 1, "amount": 0, "response": 2},
                    {"action": 2, "amount": 0, "response": 0}
                ]
            }
            memory_patterns.append(previous_pattern)
        for action in previous_pattern["opp_next_actions"]:
            if action["action"] == obs["lastOpponentAction"]:
                action["amount"] += 1
        # delete first two elements in current memory (actions of the oldest step in current memory)
        del current_memory[:2]
    my_action = random.randint(0, 2)
    pattern = find_pattern(current_memory)
    if pattern != None:
        my_action_amount = 0
        for action in pattern["opp_next_actions"]:
            # if this opponent's action occurred more times than currently chosen action
            # or, if it occured the same amount of times, choose action randomly among them
            if (action["amount"] > my_action_amount or
                    (action["amount"] == my_action_amount and random.random() > 0.5)):
                my_action_amount = action["amount"]
                my_action = action["response"]
    current_memory.append(my_action)
    return my_action
