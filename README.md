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
- On-Policy:                Yes
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step
- Parallel Environments:    No

#### Q-Learning
Q-Learning is the simplest off-policy temporal difference method. There are many different variants of this algorithm from tabular to deep learning. Its very similar to SARSA, but instead bootstraps with the optimal return estimate rather than the return estimate from following the current policy. This results in it becoming an off-policy algorithm. 

##### Features:
- On-Policy:                No
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### Double Q-Learning
Double Q-Learning is a variation of Q-Learning. The primary aim of this algorithm is to reduce the maximization bias problem of Q-Learning and SARSA. Both methods follow greedy policies, which result in them having a bias towards the maximum value. Due to the variation in the value estimates the maximum value can often be larger than the real maximum value. The value estimates will get better as more updates are performed, but the maximization bias can significantly delay learning. Double Q-Learning helps solve this issue by using two action-value functions. The one value function is used to choose the optimal action, while the other is used to estimate the action's value. The roles are then reversed in the next update cycle. This provides an unbiased estimate of the value of an action. Only a single value function is updated per step, so they both experience different updates and will therefore provide different value estimates.  

##### Features:
- On-Policy:                No
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### SARSA with Function Approximation
SARSA with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Continuous observation spaces can contain exponentially more observations than discrete observation spaces (sometimes infinite observations). Making it infeasable or even impossible to store each action observation pair (state) in a table. Linear function approximation solves this problem by converting the state/s into a feature vector. The feature vector is trained to represent the entire state space. This results in a representation of the state space with far lower dimensions.

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Linear Function Approximation
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### Q-Learning with Function Approximation
Q-Learning with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Uses the same function approximation technique as SARSA with Function Approximation, but applied to Q-Learning.

##### Features:
- On-Policy:                No
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Linear Function Approximation
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

### Policy Gradient Methods
Policy gradient methods learn a parameterized policy function that can choose actions without the need of a value function. We will be using neural networks for the parametrized policy function and use gradient descent with the Adam optimizer to update the parameters. A value function can be used in pure policy gradient methods, but it can only be used in the first state of the state transition. Actor-critic methods on the other hand use the value function on a later state for bootstrapping. So actor-critic methods are essentially pure policy gradient methods but with bootstrapping. You will notice that all of the algorithms shown in this section perform Monte Carlo updates, which is due to the lack of bootstrapping. This is the main weakness of pure policy gradient methods and hence why actor-critic methods are generally prefered over them. 

#### REINFORCE
REINFORCE is the simplest pure policy gradient method. It uses a neural network as the policy function. The policy function is learned directly unlike the value function approximation methods, which don't learn a policy function, their policy is based on the value estimates from the value function. REINFORCE doesn't use a value function. The policy function uses a soft-max function to create a probability distribution for action selection. A major benifit of using a neural network policy with soft-max action selection is that the policy will initially be stochastic, promoting more exploration, and gradually get more deterministic as it is trained. In later methods you will notice an entropy function is used to prevent the policy from becoming too deterministic. 

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           None
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    No

#### REINFORCE with Baseline
REINFORCE with Baseline improves upon the REINFORCE algorithm by reducing variance. This allows the policy function to converge faster to the optimal solution. Instead of using the return to update the policy functions parameters, the advantage gained (or lost) over the expected return is used. This provides a clearer indication of the performance of the policy's actions, hence reducing the variance during training. Despite using a value function this algorithm can't be categorized as an actor-critic method. This is due to the value function only being used on the first state in each state transition. No bootstrapping is used on this method.

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    No

#### Synchronous Advantage Actor-Critic (A2C) Monte Carlo
This is a Monte Carlo variant of A2C. The original A2C algorithm is an actor-critic, because it uses bootstrapping to estimate the expected return. Estimating the returns is not needed with this method because the updates are only performed after the episode has completed. Other than this alteration the algorithm functions the same as the original. A2C's main feature is to synchronously run multiple environments in parallel. The parallel environments will go through different state transitions. The gradient update of the value and policy functions are calculated from the average experience from the parallel agents. If the policy parameters are poor, a single agent may achieve high return during an episode, but the average return from multiple agents in distinct environments with the same poor parameters will be much closer to the real performance of the policy. This reduces variance during training and allows the agent to improve faster. A2C also uses and advantage function instead of the return for the policy parameter update like REINFORCE with baseline. 

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    Yes


### Actor-Critic Methods
#### Actor-Critic

#### Synchronous Advantage Actor-Critic (A2C)

#### Generalized Advantage Estimation (GAE) Monte Carlo
This is the Monte Carlo version of GAE. 

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    Yes

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
