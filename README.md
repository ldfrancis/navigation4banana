# navigation4banana [udacity-DRLND]
## Project Details
This is the first project in the deep reinforcement learning nanodegree on udacity. It involves training an agent to navigate a world littered with bananas and collect as many good bananas as possible while avoiding the bad bananas. The good bananas are yellow while the bad ones are blue.
A reward of +1 is given to the agent for each yellow banana collected and -1 for each blue banana.
The state space is a 37 dimension vector whose values describe agent's velocity and objects in front of the agent, and It is continous with values in the range [0,1]. The action space is descrete
## Getting Started
This project uses python 3 and some of its packages. To get started, first, install anaconda/miniconda  and then create the conda environment using;

```conda create --name=drlndbanana python=3.6```

```source activate drlndbanana```

Clone the repository

```git clone https://github.com/ldfrancis/navigation4banana.git```

Change the current working directory to the projects base folder

```cd navigation4banana```

Then proceed to installing the required packages by running

```pip install -r requirement.txt```

Having installed all the required packages, the unity environment files can then be downloaded and placed in the banana_env folder. Below are links to download the unity environments for the popular operating systems;

[linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) <br/>
[mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) <br/>
[windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) <br/>
[windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) <br/>

Now, all is set for the experiments

## Instructions
To run the experiment use

```python main.py```

This would use the default configs specified in ```config.py```. The file config.py contains variables whose values are necessary to configure the environment, the dqn agent, and the experiment. Below is a sample setting for the variables in config.py
```
ENV_PATH = f"./banana_env/{ENV_FILE}"
NUM_OBS = 37
NUM_ACT = 4
TARGET_SCORE = 13

# dqn agent
BUFFER_SIZE = 100
BATCH_SIZE = 32
LR = 5e-4
GAMMA = 0.99
TAU = 1e-2
EPS_DECAY_STEP = 1e-1
HIDDEN_DIM = [64, 128]
```

The experiment would continue running for several episodes till the agent achieves a score of +13 averaged over the last 100 episodes.