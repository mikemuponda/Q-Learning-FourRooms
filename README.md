# Q-Learning-FourRooms


#about the project
The project is an implementation of Reinforcement Learning to teach RL ageants how to pick up packages in a 13*13 gridworld using Q-learning Algorithm.
The agent will pick up the packages from the environment based on 3 main scenarios which are :

1) Simple Package Collection (Scenario1.py)
   There's only one package in the environment and the agent's goal is to locate and pick up this pakcage
2) Multi Package Collection (Scenario2.py)
   There are multiple packages located in the environment and the agent's goal is to locate and pick up all the apackages
3) Ordered Multiple Package Collection (Scenario3.py)
   Teh agent has to locate and pick up three packages each marked red(R),green(G) and blue(B) and they must be collected
   in the same order R -> G -> B

There is a 4th scenario however and which adds a stochastic element to the action space such that when the RL agents take action there will be a 20% chance they move to a different grid than the expected one thus adding some randomness


Scenario 1 : 
Agent uses Q-table with a fixed size according to the number of states defined by grid size and action space size 11 x 11 x 4
Agent uses a state indexing function based on the indexing formula position[0]*11 + position[1]*packages_left+ (11*11)
Epsilon value is fixed and algorithm is mostly greedy when choosing actions
State is based on just position and packagesRemaining variables
Very simple reward function just based on reaching the terminal state ie package picked up


Scenario 2 :
Agent uses dictionary as Qtable to take advantage of key mapping, states are looked up as keys in the table
Qtable size is now dynamic and agents can directly look up states without indexes
Epsilon is now a decaying function however restricted to a narrow range
State now adds cellType variable for whether an agent is in a cell of type 0 (empty) 1 (Red) 2 (Green) 3 (Blue)
More advanced reward function that takes into account the packages collected and optimal path

Scenario 3:
The main difference from Scenario 2 now is that agent adds order of package collection to the state under the variable packages_collected and the correct states here would be [1], [1,2],[1,2,3]
Reward function penalizes incorrect order of package collection but also gives partial rewards for intermediate correct states [1], [1,2] but also rewards the final state linearly 
Epsilon has decay function but in practice this is kept tight, going above 0.2 yielded not so great results
Agent is not very efficient due to the difficulty in balancing prioritizing order and efficient routes





Project Package Requirements :
Requirements are contained in the included requirements.txt file
   
