from collections import defaultdict

from unityagents import UnityEnvironment
import numpy as np
from config import ENV_PATH, TARGET_SCORE
from utils import BananaEnv
from dqn.agent import DQN


def train_agent(agent, env, target_score=None):
    average_return = 0
    episodes_so_far = 0
    epsilon = 1
    train_info = defaultdict(lambda: [])
    print(f"Training: last 100 episodes score mean = {0}", end="")
    last_100_episodes_mean_score = 0
    while last_100_episodes_mean_score < target_score:
        episodes_so_far += 1
        episode_info = train_for_an_episode(agent, env, epsilon)
        train_info["episode_scores"] += [episode_info["score"]]
        last_100_episodes_window = max(episodes_so_far-100, 0)
        last_100_episodes_mean_score = np.mean(train_info["episode_scores"][last_100_episodes_window:])
        train_info["last_100_episodes_mean_score"] += [last_100_episodes_mean_score]
        print(f"\rTraining: last 100 episodes score mean = {last_100_episodes_mean_score:2f}", end="")

    print(f"Training: last 100 episodes score mean = {last_100_episodes_mean_score:2f}")

    return train_info


def train_for_an_episode(agent, env, epsilon=0):
    done = False
    episode_score = 0
    episode_info = {}
    obs = env.reset(train_mode=True)
    while not done:
        action = agent.take_action(obs, epsilon=epsilon, train=True)
        next_obs, reward, done, info = env.step(action)
        episode_score += reward
        agent.save_transition((obs, action, reward, next_obs, done))
        obs = next_obs
        agent.update_q_networks()
    episode_info["score"] = episode_score
    return episode_info


if __name__=="__main__":
    env_file_path = ENV_PATH
    env = BananaEnv(env_file_path)
    agent = DQN()
    info = train_agent(agent, env, target_score=TARGET_SCORE)