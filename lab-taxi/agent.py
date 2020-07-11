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
        self.epsilon = 1
        self.gamma = 0.9
        self.alpha = 0.2
        self.episode = 0
        print("gamma: "+str(self.gamma), " || epsilon: "+str(self.epsilon), " || alpha: "+str(self.alpha))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probability = np.ones(self.nA) * self.epsilon/self.nA
        # if state in self.Q:
        # probability *= self.epsilon
        probability[np.argmax(self.Q[state])] += 1 - self.epsilon
        return np.random.choice(self.nA, p=probability)

    def greedy_action(self, state):
        return np.argmax(self.Q[state])

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

        target_reward = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target_reward - self.Q[state][action])
        if done:
            self.episode += 1
            self.epsilon = 1/self.episode