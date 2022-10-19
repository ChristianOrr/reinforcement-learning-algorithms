# Reinforcement Learning Algorithms
This repository contains a suite of popular classic reinforcement learning algorithms. The algorithms include tabular SARSA, tabular Q-Learning, tabular Double Q-Learning, SARSA with Function Approximation, Q-Learning with Function Approximation, tabular SARSA using Acme and tabular Q-Learning using Acme. All the theory required to understand the algorithms are contained in Richard S. Sutton and Andrew G. Barto's book: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf). Alternative implementations of the algorithms are provided in the references below. 


## Implementation Details
The algorithms are trained and tested on games provided by Open AI's gym. Training is performed by allowing the agent to interact with the environment over multiple episodes and learn the optimal actions by evaluating the accumulation of rewards (return) from the agents actions over the episode. The agent's follow an $\epsilon$-greedy policy to allow appropriate amounts of exploration and exploitation during training. Once the agent has been trained it is tested on a single episode, which is rendered on the screen. During testing the agent follows the greedy policy. 


## Installation Requirements
Open AI gym 0.26 or higher, numpy and matplotlib is required for q-learning and sarsa notebooks. In addition Scikit-learn is required for the sarsa and q-learning function approximation notebooks. Acme, Jax and Pandas are required for the sarsa and q-learning Acme notebooks. The play_games notebook has Atari games which requires the rom licence to be installed. You can install all the requirements by running the block below:

```
pip install gym[atari,accept-rom-license]==0.26
pip install matplotlib==3.5
pip install pandas==1.4
pip install jupyter==1.0
pip install scikit-learn==1.1
pip install dm-acme[jax, envs]==0.4
```
Alternativaly you could install jax for GPU, by running the following first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```



## References
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)
- [Q-Learning Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb)
- [Solving Taxi by Justin Francis](https://github.com/wagonhelm/Reinforcement-Learning-Introduction/blob/master/Solving%20Taxi.ipynb)
- [SARSA Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb)
- [Q-Learning Value Function Approximation Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb)
- [Intro to DeepMind’s Reinforcement Learning Framework “Acme”](https://towardsdatascience.com/deepminds-reinforcement-learning-framework-acme-87934fa223bf)
