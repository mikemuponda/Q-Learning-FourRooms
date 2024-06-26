from FourRooms import FourRooms
import numpy as np
import matplotlib
import sys
import random

#discount factor= df
#learning rate = lr
#epsilon the probability by which the agent will choose a value at random

#Notes
#state S = tuple [position on grid S(x,y) + packages_left + visited (experimenting to reduce unnecessary exploration)] but the information for each state is uniquely encoded into
#a single quantity using the formula : position[0]*11 + position[1]*packages_left+ (11*11)
#action A = UP, DOWN, LEFT, RIGHT = 0,1,2,3,4
#Terminating condition = while not in the terminal state & for n episodes

class RLAgent:
    def __init__(self, lr, df, epsilon, env):
        self.q_table = {}
        self.lr = lr
        self.df = df
        self.epsilon = epsilon
        self.actions = [env.UP, env.DOWN, env.RIGHT, env.LEFT]
        self.env = env
        self.visited = set()

    def reset(self):
        self.visited.clear()

    def current_state(self):
        position = self.env.getPosition()
        packages_left = self.env.getPackagesRemaining()
        position_tuple = tuple(position)
        return (position_tuple, packages_left)

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        best_action = np.argmax(self.q_table[next_state])
        target = reward + self.df * self.q_table[next_state][best_action]
        delta = target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * delta

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            if state in self.q_table:
                return np.argmax(self.q_table[state])
            else:
                return random.choice(self.actions)

    def calculate_reward(self, isTerminal, packages_remaining_before, packages_remaining_after):
        reward = -3
        new_packages = packages_remaining_before - packages_remaining_after
        reward += 10 * new_packages
        if isTerminal:
            if packages_remaining_after == 0:
                reward += 40
        else:
            reward -= 30
        #reward = -5 if not isTerminal else 0 
        return reward

def main():
    lr = 0.1
    df = 0.99
    epsilon = 0.1
    n_episodes = 1000
    env = FourRooms('multi', False)
    agent = RLAgent(lr, df, epsilon, env)

    for e in range(n_episodes):
        env.newEpoch()
        agent.reset()
        while not env.isTerminal():
            current_state = agent.current_state()
            action = agent.choose_action(current_state)
            cellType, newPos, packagesRemaining, isTerminal = env.takeAction(action)
            reward = agent.calculate_reward(isTerminal, packagesRemaining, env.getPackagesRemaining())
            next_state = agent.current_state()
            agent.update(current_state, action, reward, next_state)

    env.showPath(-1)

if __name__ == "__main__":
    main()