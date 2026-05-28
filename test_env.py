import torch
import gymnasium as gym
from stable_baselines3 import PPO
from flask import Flask

print("torch:", torch.__version__)

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

print("环境正常，可以训练")