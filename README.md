# Reinforcement Learning Algorithms
This repository contains a suite of reinforcement learning algorithms implemented from scratch. The algorithms include a wide variety of algorithms ranging from tabular to deep reinforcement learning. The foundational knowledge required to understand the algorithms are provided in Richard S. Sutton and Andrew G. Barto's book [1] and OpenAI's Spinning Up blog [2].


## Implementation Details
All the RL algorithms will follow the Markov decision process (MDP) and all of them will use environments from OpenAI's Gym (now its called to Gymnasium and maintained by a different organisation). All agents are trained from scratch with either random or zeroed initialization of the model/s parameters. The problems get progressively harder as the RL methods get better, which results in longer training times.

## Algorithms
The RL algorithms get progressively more complex and simultaniously more powerful from value function approximation to policy gradient methods, to the most powerful, actor-critic methods.   
### Value Function Approximation
Value function approximation methods mainly focus on finding the optimal value function (usually an action-value function). The policy function will then be defined around the value function. $\epsilon$-Greedy is the most common policy used with these methods. $\epsilon$-Greedy policies will mostly choose the action with highest value (from the value function estimate). Then on the odd occasion it will choose a random action with some small probability, $\epsilon$. This provides a good balance of exploration with exploitation. The observation and action spaces are usually discrete. The algorithms are the easiest to understand and implement.
#### SARSA
SARSA is the simplest on-policy temporal difference method. It uses bootstrapping to estimate the return before the episode has completed. The return estimate is then used to update the action-value functions prediction of the return given the state, action pair. 

##### Features:
- On-Policy:            Yes
- Observation Space:    Discrete
- Action Space:         Discrete
- Value Function:       Tabular
- Policy Function:      $\epsilon$-Greedy
- Update Period:        1-step (n-step and TD($\lambda$) versions are also available)

#### Q-Learning
Q-Learning is the simplest off-policy temporal difference method. There are many different variants of this algorithm from tabular to deep learning. Its very similar to SARSA, but instead bootstraps with the optimal return estimate rather than the return estimate from following the current policy. This results in it becoming an off-policy algorithm. 

##### Features:
- On-Policy:            No
- Observation Space:    Discrete
- Action Space:         Discrete
- Value Function:       Tabular
- Policy Function:      $\epsilon$-Greedy
- Update Period:        1-step (n-step and TD($\lambda$) versions are also available)

#### Double Q-Learning
Double Q-Learning is a variation of Q-Learning. The primary aim of this algorithm is to reduce the maximization bias problem of Q-Learning and SARSA. Both methods follow greedy policies, which result in them having a bias towards the maximum value. Due to the variation in the value estimates the maximum value can often be larger than the real maximum value. The value estimates will get better as more updates are performed, but the maximization bias can significantly delay learning. Double Q-Learning helps solve this issue by using two action-value functions. The one value function is used to choose the optimal action, while the other is used to estimate the action's value. The roles are then reversed in the next update cycle. This provides an unbiased estimate of the value of an action. Only a single value function is updated per step, so they both experience different updates and will therefore provide different value estimates.  

##### Features:
- On-Policy:            No
- Observation Space:    Discrete
- Action Space:         Discrete
- Value Function:       Tabular
- Policy Function:      $\epsilon$-Greedy
- Update Period:        1-step (n-step and TD($\lambda$) versions are also available)

#### SARSA with Function Approximation
SARSA with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Continuous observation spaces can contain exponentially more observations than discrete observation spaces (sometimes infinite observations). Making it infeasable or even impossible to store each action observation pair (state) in a table. Linear function approximation solves this problem by converting the state/s into a feature vector. The feature vector is trained to represent the entire state space. This results in a representation of the state space with far lower dimensions.

##### Features:
- On-Policy:            Yes
- Observation Space:    Continuous
- Action Space:         Discrete
- Value Function:       Linear Function Approximation
- Policy Function:      $\epsilon$-Greedy
- Update Period:        1-step (n-step and TD($\lambda$) versions are also available)

#### Q-Learning with Function Approximation
Q-Learning with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Uses the same function approximation technique as SARSA with Function Approximation, but applied to Q-Learning.

##### Features:
- On-Policy:            No
- Observation Space:    Continuous
- Action Space:         Discrete
- Value Function:       Linear Function Approximation
- Policy Function:      $\epsilon$-Greedy
- Update Period:        1-step (n-step and TD($\lambda$) versions are also available)


### Policy Gradient Methods
#### REINFORCE

#### REINFORCE with Baseline

#### Synchronous Advantage Actor-Critic (A2C) Monte Carlo

#### Generalized Advantage Estimation (GAE) Monte Carlo


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
