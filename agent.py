import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1 # If epsilon =1, that means that is it never greedy
        self.alpha = 0.01
        self.gamma = 1.0
        self.num_episodes =1


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # We want to define an Epsilon-Greedy policy that:
        #         1. with a probability of 1-epsilon selects the greedy action
        #         2. with probablity epsilon, the agent selects an action uniformly at random,
        #            from the available (non-greedy AND greedy actions)
        #  This is done in two steps:
        #         1. First all policies with epsilon/nA
        #         2. Fill the greedy action with 1-epsilon +epsilon/nA

        policy = np.ones(self.nA) * self.epsilon/ self.nA
        best_action = np.argmax(self.Q[state])
        policy[best_action] += 1 - self.epsilon
        return np.random.choice(self.nA, p = policy) 


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Here use the alpha to already learn from the current findings (the learned value)
        # With the code shown here I have obtained after 20000 runs, a best average award of 9.1699

        if not done:
            learned_value = reward + self.gamma*np.max(self.Q[next_state])
            self.Q[state][action] = (1-self.alpha)*self.Q[state][action] +self.alpha*learned_value
        else:
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward-self.Q[state][action])
            self.num_episodes +=1
            self.epsilon = 2 / self.num_episodes