# Reinforcement Learning Algorithms
This repository contains a suite of reinforcement learning algorithms implemented from scratch. The algorithms include a wide variety of algorithms ranging from tabular to deep reinforcement learning. The foundational knowledge required to understand the algorithms are provided in Richard S. Sutton and Andrew G. Barto's book [1] and OpenAI's Spinning Up blog [2].


## Implementation Details
All the RL algorithms will follow the Markov decision process (MDP) and all of them will use environments from OpenAI's Gym (now its called to Gymnasium and maintained by a different organisation). All agents are trained from scratch with either random or zeroed initialization of the model/s parameters. The problems get progressively harder as the RL methods get better, which results in longer training times.

## Algorithms
The RL algorithms get progressively more complex and simultaniously more powerful from value function approximation to policy gradient methods, to the most powerful, actor-critic methods.   
### Value Function Approximation
#### Q-Learning

#### SARSA

#### Double Q-Learning

#### Q-Learning with Function Approximation

#### SARSA with Function Approximation


### Policy Gradient Methods
#### REINFORCE

#### REINFORCE with Baseline

#### Synchronous Advantage Actor-Critic (A2C) Monte-Carlo

#### Generalized Advantage Estimation (GAE) Monte-Carlo


### Actor-Critic Methods
#### Actor-Critic

#### Synchronous Advantage Actor-Critic (A2C)

#### Generalized Advantage Estimation (GAE) 

#### Proximal Policy Optimization (PPO)



## Installation Requirements
If you have a GPU you can install Jax by running the following first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
All the requirements are provided below:
```
pip install gymnasium[atari, toy_text, accept-rom-license, box2d]
pip install moviepy
pip install matplotlib==3.5
pip install pandas==1.4
pip install jupyter==1.0
pip install scikit-learn==1.1
pip install dm-acme[jax, envs]==0.4
```



## References
- [1] [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)
- [2] [OpenAI, Spinning Up](https://spinningup.openai.com/en/latest/index.html)
- [3] [Q-Learning Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb)
- [4] [Solving Taxi by Justin Francis](https://github.com/wagonhelm/Reinforcement-Learning-Introduction/blob/master/Solving%20Taxi.ipynb)
- [5] [SARSA Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb)
- [6] [Q-Learning Value Function Approximation Example by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb)
- [7] [Intro to DeepMind’s Reinforcement Learning Framework “Acme”](https://towardsdatascience.com/deepminds-reinforcement-learning-framework-acme-87934fa223bf)
