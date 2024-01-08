import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import highway_env
from RL.DQN import DQN
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

highway_env.register_highway_envs()

TRAIN = True
# TRAIN = False
if __name__ == "__main__":
    n_cpu = 6
    batch_size = 64
    env = make_vec_env("merge-v1", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="merge_ppo/",
    )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e5))
        model.save("merge_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("merge_ppo/model", env=env)

    env = gym.make("merge-v1", render_mode="rgb_array")
    env = RecordVideo(
        env, video_folder="merge_ppo/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 5})
    ep_reward = 0
    reward_list = []
    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            print(info)
            # Render
            ep_reward += reward
            env.render()
        reward_list.append(ep_reward)
        ep_reward = 0
    # Save reward_list
    import numpy as np

    np.save("merge_ppo/return_list1.npy", reward_list)
    env.close()
