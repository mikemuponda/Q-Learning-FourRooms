from FourRooms import FourRooms
import numpy as np
import matplotlib
import sys
import random

#discount factor= df
#learning rate = lr
#epsilon the probability by which the agent will choose a value at random

#Notes
#state S = tuple [position on grid S(x,y) + packages_left + packages_collected] but the information for each state is uniquely encoded into
#a single quantity using the formula : position[0]*11 + position[1]*packages_left+ (11*11)
#action A = UP, DOWN, LEFT, RIGHT = 0,1,2,3,4
#Terminating condition = while not in the terminal state & for n episodes

class RLAgent:
   def __init__(self,states, actions, lr, df, epsilon, env):
        self.Q_table=np.zeros((states, actions))
        self.df = df
        self.lr=lr
        self.epsilon=epsilon
        self.actions=actions

   def current_state(self):
        position = self.env.getPosition
        packages_left=self.env.getPackagesRemaining
        return (position[0],position[1],packages_left)
   
   def update(self, state, action, reward, next_state):
        best_action = np.argmax(self.Q_table[next_state])
        target = reward + self.df * self.Q_table[next_state][best_action]
        delta = target - self.Q_table[state][action]
        self.Q_table[state][action] += self.lr * delta

   def choose_action(self, state_index):
        if random.random() < self.epsilon:
            return random.choice([self.env.UP,self.env.DOWN,self.env.RIGHT,self.env.LEFT])
        else:
            return np.argmax(self.Q_table[state_index])
        
   def package_index(self,packages_collected):
        package_index = 0
        for i, is_collected in enumerate(packages_collected):
            if is_collected:
             package_index+=2**i
        return package_index

       
   def state_index(self, position, packages_left, packages_collected):
        position_index = position[0] * 11 + position[1]
        pkg_index = self.package_index(packages_collected)
        total_position_states = 11*11  # 121 positions
        num_package_states = 2*len(packages_collected)
        return position_index * num_package_states + pkg_index
   
   def env_package_collected(self, packages_remaining):
        packages_collected = [False, False, False]
        package_count = 3 - packages_remaining
        for p in range(package_count):
         packages_collected[p] = True
        return packages_collected
   
   


def main():
    lr=0.1
    df=0.9
    epsilon=0
    action_space={0,1,2,3}
    n_episodes =100
    states=11*11*8
    actions=len(action_space)
    env = FourRooms('multi',False)
    agent=RLAgent(states,actions,lr,df,epsilon, env)
    packages_collected={False,False,False}

    episodes = 100
    for e in range(n_episodes):
        env.newEpoch()
        while not env.isTerminal():
            current_position = env.getPosition()
            packages_left = env.getPackagesRemaining()
            packages_collected=agent.env_package_collected(env.getPackagesRemaining())
            state_index = agent.state_index(current_position, packages_left,packages_collected)
            action = agent.choose_action(state_index)
            e, newPos, packagesRemaining, isTerminal = env.takeAction(action)
            reward = -1 if not isTerminal else 0 
            next_state_index = agent.state_index(newPos, packagesRemaining, packages_collected)
            agent.update(state_index, action, reward, next_state_index)
    env.showPath(-1)

if __name__ == "__main__":
    main()

    