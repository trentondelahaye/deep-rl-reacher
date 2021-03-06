# Deep Reinforcement Learning - Reacher Continuous Control

## Overview

This project is part of the fulfillment of the Udacity Deep Reinforcement Learning nanodegree. Deep Deterministic Policy Gradient (DDPG) reinforcement learning agents were programmed to solve a unity based double jointed robotic arm environment. 

A gif of a trained DDPG agent interacting with the environment is shown below.
![](report/images/trained_agent.gif)

## Environment description

In this environment, a double-jointed arm can apply torque to its joints in order to move towards target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. The action space is a vector with four numbers, corresponding to torque applicable to two joints, with continuous values in the interval [-1, 1].

The environment also has capability for 20 concurrent agents for distributed training. This allows and is useful for algorithms such as PPO, DDPG, D4PG and A2C. Since there is natural synchronisation of the agents, A3C cannot be implemented.

The environment is considered solved when the average score across the 20 agents exceeds +30 over 100 consecutive episodes.

## Dependencies

Set up of the Python environment in order to run this project.

1. (Suggested) Create and activate a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drl-reacher python=3.6
	source activate drl-reacher
	```
	- __Windows__: 
	```bash
	conda create --name drl-reacher python=3.6 
	activate drl-reacher
	```
	
2. Clone this repository provided by Udacity, then install dependencies.
  ```bash
  git clone https://github.com/udacity/deep-reinforcement-learning.git
  cd deep-reinforcement-learning/python
  pip install .
  ```
  
3. Install the additional dependency for this project.
  ```bash
  pip install click
  ```

4. Download the environment as per your operating system and unzip to the project directory:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - macOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    
    
## Training and watching agents

In order to start the program, run `python main.py` in the project directory (using the virtual env). Running `python main.py --help` will show the following options:

```
Options:
  --unity-env TEXT            Path to UnityEnvironment
  --agent-cfg TEXT            Section of config used to load agent
  --no-graphics / --graphics  Load environment without graphics
  --help                      Show this message and exit
```

The `unity-env` needs to point to the directory that you placed the environment from [Dependencies](#Dependencies) Step 4 and the `agent-cfg` can point to any config section specified in `agents/configs.cfg`. You can change hyperparameters and configurations of the DeepQ agents in `agents/configs.cfg`. For example 

```
python main.py --unity-env ./Reacher.app --agent-cfg DDPG
```

Once the program has started, it will display environment information and await a command. The following commands are currently available:

```
{'save-agent', 'exit', 'plot', 'load-agent', 'watch', 'train'}
```

Using the flag `--help` after a command will show the keyword arguments available for the command, if any. For example, the user can train an agent with

```
train number_episodes=20 exit_when_solved=False
```

which will train the agent for 20 more episodes, not quitting when the average over the past 100 is above 13. The command `watch` is unavailable if the unity environment has been started in `no-graphics` mode.
