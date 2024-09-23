# Reinforcement Learning Algorithms
This repository contains a suite of reinforcement learning algorithms implemented from scratch. The algorithms include a wide variety of algorithms ranging from tabular to deep reinforcement learning. The algorithm implementations are designed to contain only the fundamental parts of the algorithms needed to make them functional. This ensures that the code is simple short and easy to understand. This is a valuable resource for anyone looking to understand how to apply RL algorithms without having to go through many layers of abstraction. The foundational knowledge required to understand the algorithms are provided in Richard S. Sutton and Andrew G. Barto's book [1] and OpenAI's Spinning Up blog [2].


## Implementation Details
All the RL algorithms will follow the Markov decision process (MDP) and all of them will use environments from OpenAI's Gym (now its called to Gymnasium and maintained by a different organisation). All agents are trained from scratch with either random or zeroed initialization of the model/s parameters. The problems get progressively harder as the RL methods get better, which results in longer training times.

## Algorithms
The RL algorithms get progressively more complex and simultaniously more powerful from value function methods to policy gradient methods, to the most powerful, actor-critic methods. 

### Value Function Methods
Value function methods mainly focus on finding the optimal value function (usually an action-value function). The policy function will then be defined around the value function. $\epsilon$-Greedy is the most common policy used with these methods. $\epsilon$-Greedy policies will mostly choose the action with highest value (from the value function estimate). Then on the odd occasion it will choose a random action with some small probability, $\epsilon$. This provides a good balance of exploration with exploitation. The observation and action spaces are usually discrete. The algorithms are the easiest to understand and implement.

#### SARSA
**Algorithm Notebook:** `sarsa.ipynb`

SARSA is the simplest on-policy temporal difference method. It uses bootstrapping to estimate the return before the episode has completed. The return estimate is then used to update the action-value functions prediction of the return given the state, action pair. 

**Before Training:**

<video controls>
<source src="./videos/sarsa/Taxi-v3-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/sarsa/Taxi-v3-episode-400.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step
- Parallel Environments:    No

#### Q-Learning
**Algorithm Notebook:** `q_learning.ipynb`

Q-Learning is the simplest off-policy temporal difference method. There are many different variants of this algorithm from tabular to deep learning. Its very similar to SARSA, but instead bootstraps with the optimal return estimate rather than the return estimate from following the current policy. This results in it becoming an off-policy algorithm. 

**Before Training:**

<video controls>
<source src="./videos/q_learning/Taxi-v3-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/q_learning/Taxi-v3-episode-400.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                No
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### Double Q-Learning
**Algorithm Notebook:** `double_q_learning.ipynb`

Double Q-Learning is a variation of Q-Learning. The primary aim of this algorithm is to reduce the maximization bias problem of Q-Learning and SARSA. Both methods follow greedy policies, which result in them having a bias towards the maximum value. Due to the variation in the value estimates the maximum value can often be larger than the real maximum value. The value estimates will get better as more updates are performed, but the maximization bias can significantly delay learning. Double Q-Learning helps solve this issue by using two action-value functions. The one value function is used to choose the optimal action, while the other is used to estimate the action's value. The roles are then reversed in the next update cycle. This provides an unbiased estimate of the value of an action. Only a single value function is updated per step, so they both experience different updates and will therefore provide different value estimates.  

**Before Training:**

<video controls>
<source src="./videos/double_q_learning/Taxi-v3-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/double_q_learning/Taxi-v3-episode-400.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                No
- Observation Space:        Discrete
- Action Space:             Discrete
- Value Function:           Tabular
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### SARSA with Function Approximation
**Algorithm Notebook:** `sarsa_fa.ipynb`

SARSA with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Continuous observation spaces can contain exponentially more observations than discrete observation spaces (sometimes infinite observations). Making it infeasable or even impossible to store each action observation pair (state) in a table. Linear function approximation solves this problem by converting the state/s into a feature vector. The feature vector is trained to represent the entire state space. This results in a representation of the state space with far lower dimensions.

**Before Training:**

<video controls>
<source src="./videos/sarsa_fa/MountainCar-v0-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/sarsa_fa/MountainCar-v0-episode-40.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Linear Function Approximation
- Policy Function:          $\epsilon$-Greedy
- Update Period:            1-step 
- Parallel Environments:    No

#### Q-Learning with Function Approximation
**Algorithm Notebook:** `q_learning_fa.ipynb`

Q-Learning with Function Approximation improves on the Q-Learning algorithm by providing the ability to work on continuous observation spaces. Uses the same function approximation technique as SARSA with Function Approximation, but applied to Q-Learning.

**Before Training:**

<video controls>
<source src="./videos/q_learning_fa/MountainCar-v0-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/q_learning_fa/MountainCar-v0-episode-40.mp4" type="video/mp4">
</video>

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
**Algorithm Notebook:** `reinforce.ipynb`

REINFORCE is the simplest pure policy gradient method. It uses a neural network as the policy function. The policy function is learned directly unlike the value function methods, which don't learn a policy function, their policy is based on the value estimates from the value function. REINFORCE doesn't use a value function. The policy function uses a soft-max function to create a probability distribution for action selection. A major benifit of using a neural network policy with soft-max action selection is that the policy will initially be stochastic, promoting more exploration, and gradually get more deterministic as it is trained. In later methods you will notice an entropy function is used to prevent the policy from becoming too deterministic. 

**Before Training:**

<video controls>
<source src="./videos/reinforce/CartPole-v1-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/reinforce/CartPole-v1-episode-50.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           None
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    No

#### REINFORCE with Baseline
**Algorithm Notebook:** `reinforce_baseline.ipynb`

REINFORCE with Baseline improves upon the REINFORCE algorithm by reducing variance. This allows the policy function to converge faster to the optimal solution. Instead of using the return to update the policy functions parameters, the advantage gained (or lost) over the expected return is used. This provides a clearer indication of the performance of the policy's actions, hence reducing the variance during training. Despite using a value function this algorithm can't be categorized as an actor-critic method. This is due to the value function only being used on the first state in each state transition. No bootstrapping is used on this method.

**Before Training:**

<video controls>
<source src="./videos/reinforce_baseline/CartPole-v1-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/reinforce_baseline/CartPole-v1-episode-50.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    No

#### Synchronous Advantage Actor-Critic (A2C) Monte Carlo
**Algorithm Notebook:** `a2c_discrete_monte-carlo.ipynb`

This is a Monte Carlo variant of A2C. The original A2C algorithm is an actor-critic, because it uses bootstrapping to estimate the expected return. Estimating the returns is not needed with the Monte Carlo implementation because the updates are performed after the episode has completed. Other than this alteration the algorithm functions the same as the original. A2C's main feature is to synchronously run multiple environments in parallel. The parallel environments will go through different state transitions. The gradient update of the value and policy functions are calculated from the average experience from the parallel agents. If the policy parameters are poor, a single agent may achieve high return during an episode, but the average return from multiple agents in distinct environments with the same poor parameters will be much closer to the real performance of the policy. This reduces variance during training and allows the agent to improve faster. A2C also uses and advantage function instead of the return for the policy parameter update like REINFORCE with baseline. 

**Before Training:**

<video controls>
<source src="./videos/a2c_discrete_monte-carlo/CartPole-v1-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/a2c_discrete_monte-carlo/CartPole-v1-episode-5.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    Yes


### Actor-Critic Methods
Actor-critic methods use both a policy function (the actor) and a value function (the critic). The difference between actor-critic methods and pure policy gradient methods with both a value and policy function (like REINFORCE with baseline) is that the value function is used to criticise the policy functions actions. in technical terms the value function is used on later states in the state transition. This is mainly performed to bootstrap the expected return. The bootstrapped return will provide a more accurate value estimate then the value estimate in the current state, therefore it can be used to criticise the action/s taken by the policy function. The criticising is performed in the form of the advantage estimate, which tells us how much advantage was gained or lost by performing those actions. 

#### Actor-Critic
**Algorithm Notebook:** `actor_critic.ipynb`

This is the simplest actor-critic algorithm. Actor-critic slightly adjusts the REINFORCE with baseline by using the value function to calculate the bootstrapped the return value, which is then used to calculate the temporal difference (TD) error, $\delta$. This enables parameter updates every step. The TD error is a simple advantage estimate, used to criticise the action taken by the policy in the first step.

**Before Training:**

<video controls>
<source src="./videos/actor-critic/CartPole-v1-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/actor-critic/CartPole-v1-episode-50.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            1-step
- Parallel Environments:    No


#### Synchronous Advantage Actor-Critic (A2C)
**Algorithm Notebook:** `a2c_discrete.ipynb`

A2C is an actor-critic algorithm that synchronously runs multiple environments in parallel. All the agents run the same policy, but the environment's will be slightly different, resulting in different experiences for all the agents. The returns from the experiences are then aggregated in the advantage function, which is then used to update the policy and value function parameters. The policy and value functions are then updated after n-steps, where n is preferably greater than 1 but less than the average episode length. Some of the agents may complete before n-steps. In this case they will automatically start a new episode. The return calculation will start from scratch, so the previous episodes rewards won't influence the new episodes return. The algorithm will even work for values of n far greater than the episode length, although this won't be optimal. 

**Before Training:**

<video controls>
<source src="./videos/a2c_discrete/LunarLander-v2-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/a2c_discrete/LunarLander-v2-episode-5.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            n-steps
- Parallel Environments:    Yes

#### Generalized Advantage Estimation (GAE) Monte Carlo
**Algorithm Notebook:** `gae_discrete_monte-carlo.ipynb`

This is the Monte Carlo version of GAE. GAE proposes a better advantage estimation function, the generalized advantage estimation function. The GAE function improves on A2C advantage estimate by lowering the bias at the cost of a minor variance increase. Although this algorithm performs Monte Carlo updates, it still uses bootstrapping in the GAE function, so it is an actor-critic method. This implementation will use A2C's technique of running multiple environments in parallel, but replace the advantage estimation function with the GAE function. 

**Before Training:**

<video controls>
<source src="./videos/gae_discrete_monte-carlo/CartPole-v1-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/gae_discrete_monte-carlo/CartPole-v1-episode-5.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            Monte Carlo (updates after episode termination)
- Parallel Environments:    Yes

#### Generalized Advantage Estimation (GAE)
**Algorithm Notebook:** `gae_discrete.ipynb`

This implemenation of GAE only differs from the Monte Carlo implementation in when the parameter updates are performed. The parameters are now updated every n-steps like the vanilla A2C. See the A2C and GAE Monte Carlo algorithm descriptions for a better understanding of this implementation.   

**Before Training:**

<video controls>
<source src="./videos/gae_discrete/LunarLander-v2-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/gae_discrete/LunarLander-v2-episode-5.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            n-steps
- Parallel Environments:    Yes

#### Proximal Policy Optimization (PPO)
**Algorithm Notebook:** `ppo_discrete.ipynb`

This is the clipped surrogate objective version of PPO. The aim of PPO is to mimic the behaviour of TRPO, but only use first-order derivatives in the policy optimization, resulting in a simpler algorithm. PPO and TRPO differ from other policy gradient algorithms with their approach to policy parameter updates. They both restrict the objective function (loss function) when the objective improves too much in a single step. This has the desired effect of preventing the policy updates from getting so large that the policy function can become worse. This implementation utilises the synchronous parallel environments technique from A2C and the generalized advantage estimator function as the advantage estimator from GAE. 

**Before Training:**

<video controls>
<source src="./videos/ppo_discrete/LunarLander-v2-episode-0.mp4" type="video/mp4">
</video>

**After Training:**

<video controls>
<source src="./videos/ppo_discrete/LunarLander-v2-episode-5.mp4" type="video/mp4">
</video>

##### Features:
- On-Policy:                Yes
- Observation Space:        Continuous
- Action Space:             Discrete
- Value Function:           Deep Neural Network
- Policy Function:          Deep Neural Network
- Update Period:            n-steps
- Parallel Environments:    Yes

## Installation Requirements
If you have a GPU you can install Jax by running the following first:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
All the requirements are provided below:
```
pip install "gymnasium[atari, toy_text, accept-rom-license, box2d]"==0.29.1
pip install moviepy==1.0.3
pip install matplotlib==3.8.4
pip install pandas==2.2.2
pip install jupyter==1.0
pip install scikit-learn==1.4.2
pip install dm-acme[jax, envs]==0.4
pip install flax==0.8.3
```



## References
- [1] [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)
- [2] [Spinning Up, OpenAI](https://spinningup.openai.com/en/latest/index.html)
- [3] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [4] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [5] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [6] [Intro to DeepMind’s Reinforcement Learning Framework “Acme”](https://towardsdatascience.com/deepminds-reinforcement-learning-framework-acme-87934fa223bf)
- [7] [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
