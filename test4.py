import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import highway_env
from RL.DQN import DQN
from datetime import datetime
from pathlib import Path

highway_env.register_highway_envs()

TRAIN = True
# TRAIN = False
if __name__ == "__main__":
    # Create the environment
    env = gym.make("merge-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model
    model = DQN(
        env=env,
        lr=5e-4,
        gamma=0.99,
        epsilon=0.9,
        MEMORY_CAPACITY=15000,
        state_dim=25,
        action_dim=5,
        targe_update_f=50,
        batch_size=32,
    )
    path = "D:\python\HighwayEnv\save_models\\DQN.pkl"
    if TRAIN:
        model.learn(total_timesteps=int(5e3))
        model.save(path)
        # del model

    # Run the trained model and record video
    model.load(load_path=path)
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    # simulation_frequency越大越慢
    env.configure({"simulation_frequency": 120})  # Higher FPS for rendering
    episode_rewards = 0
    return_list = []
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        _last_obs = obs.flatten()
        while not (done or truncated):
            # Predict
            action = model.choose_action(_last_obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            _last_obs = obs.flatten()
            # Render
            env.render()
            episode_rewards += reward
        return_list.append(episode_rewards)
        episode_rewards = 0

    import matplotlib.pyplot as plt

    plt.plot(return_list)
    plt.show()
    env.close()
