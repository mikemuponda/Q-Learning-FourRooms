# Q-Learning-FourRooms


#about the project
The project is an implementation of Reincofecement Learning to teach RL ageants how to pick up packages in a 13*13 gridworld using Q-learning Algorithm.
The agent will pick up the packages from the environment based on 3 main scenarios which are :

1) Simple Package Collection (Scenario1.py)
   There's only one package in the environment and the agent's goal is to locate and pick up this pakcage
2) Multi Package Collection (Scenario2.py)
   There are multiple packages located in the environment and the agent's goal is to locate and pick up all the apackages
3) Ordered Multiple Package Collection (Scenario3.py)
   Teh agent has to locate and pick up three packages each marked red(R),green(G) and blue(B) and they must be collected
   in the same order R -> G -> B

There is a 4th scenario however and which adds a stochastic element to the action space such that when the RL agents take action
there will be a 20% chance they move to a different grid than the expected one thus adding some randomness



Project Package Requirements :
Requirements are contained in the included requirements.txt file
   
