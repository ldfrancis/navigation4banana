from unityagents import UnityEnvironment
import numpy as np
from config import ENV_PATH
from utils import BananaEnv
from dqn.agent import DQN

env_file_path = ENV_PATH

env = BananaEnv(env_file_path)
agent = DQN(env.action_space.n, env.observation_space.shape[0])


def train_agent(agent, env, target_return=None):
    average_return = 0
    while average_return < target_return:
        pass


def train_for_an_episode(agent, env, epsilon=0):
    done = False
    episode_score = 0
    while not done:
        obs = env.reset(train_mode=True)
        action = agent.take_action(obs, epsilon=epsilon, train=True)
        next_obs, reward, done, info = env.step(action)
        episode_score += reward
        agent.save_transition((obs, action, reward, next_obs, done))
        obs = next_obs

import pdb; pdb.set_trace()
