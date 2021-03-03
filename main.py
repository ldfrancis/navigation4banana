from collections import defaultdict

from unityagents import UnityEnvironment
import numpy as np
from config import ENV_PATH, TARGET_SCORE, EPS_DECAY
from utils import BananaEnv
from dqn.agent import DQN
import sys
import matplotlib.pyplot as plt
from pathlib import Path

import os

os.makedirs("./plots", exist_ok=True)


def train_agent(agent, env, target_score=None):
    average_return = 0
    episodes_so_far = 0
    best_score = -np.inf
    epsilon = 1
    train_info = defaultdict(lambda: [])
    print(f"Episode {episodes_so_far}: last 100 episodes score mean = {0:2f} epsilon = {epsilon:2f}", end="")
    last_100_episodes_mean_score = 0
    while last_100_episodes_mean_score < target_score:
        episodes_so_far += 1
        episode_info = train_for_an_episode(agent, env, epsilon)
        epsilon = episode_info["epsilon"]
        train_info["episode_scores"] += [episode_info["score"]]
        last_100_episodes_window = max(episodes_so_far - 100, 0)
        last_100_episodes_mean_score = np.mean(train_info["episode_scores"][last_100_episodes_window:])
        train_info["last_100_episodes_mean_score"] += [last_100_episodes_mean_score]
        print(f"\rEpisode {episodes_so_far}: last 100 episodes score mean = "
              f"{last_100_episodes_mean_score:2f} epsilon = {epsilon:2f}", end="")
        if episode_info["score"] > best_score:
            agent.save()
            best_score = episode_info["score"]
        plot_score(train_info["last_100_episodes_mean_score"], train_info["episode_scores"])

    print(f"\rEpisode {episodes_so_far}: last 100 episodes score mean = "
          f"{last_100_episodes_mean_score:2f} epsilon = {epsilon:2f}")

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
    new_epsilon = max(epsilon * EPS_DECAY, 0.05)
    episode_info["score"] = episode_score
    episode_info["epsilon"] = new_epsilon
    return episode_info


def parse_args():
    valid_commands = ["train", "test", "eval"]
    args = sys.argv
    assert len(args) == 2 and args[0] == "main.py"
    if args[1] not in valid_commands:
        raise ValueError(f"command not identified! use either of valid_commands")

    return args[1]


def evaluate(agent, env):
    done = False
    episode_score = 0
    obs = env.reset(train_mode=False)
    while not done:
        action = agent.take_action(obs, epsilon=0, train=False)
        next_obs, reward, done, info = env.step(action)
        episode_score += reward
        obs = next_obs

    return episode_score


def plot_score(last_100_episode_score_mean, episode_scores):
    filename = "score_plot"
    plt.figure(figsize=(30, 10))
    for plot, score, title in zip((121, 122), (last_100_episode_score_mean, episode_scores),
                                  ("last_100_episode_score_mean", "episode_scores")):
        plt.subplot(plot)
        plt.plot(score)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel("episodes")
        plt.savefig(f"./plots/{filename}.png")
    plt.close()


if __name__ == "__main__":
    command = parse_args()
    env_file_path = ENV_PATH
    env = BananaEnv(env_file_path)
    agent = DQN()
    if command == "train":
        info = train_agent(agent, env, target_score=TARGET_SCORE)
    else:
        assert Path("./checkpoint/checkpoint.pt").exists()
        agent.restore("./checkpoint/checkpoint.pt")
        score = evaluate(agent, env)
        print(score)
