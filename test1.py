import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO

import highway_env

highway_env.register_highway_envs()

TRAIN = True
# TRAIN = False
if __name__ == "__main__":
    # Create the environment
    env = gym.make("merge-v0", render_mode="rgb_array")
    obs, info = env.reset()
    print(obs)
    # print(env.config)
    # env.render()

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        # batch_size=128,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",
    )
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    #     n_steps=batch_size * 12 // n_cpu,
    #     batch_size=batch_size,
    #     n_epochs=10,
    #     learning_rate=5e-4,
    #     gamma=0.8,
    #     verbose=2,
    #     tensorboard_log="highway_ppo/",
    # )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e3))
        model.save("highway_dqn/model")
        del model

    # Run the trained model and record video
    model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    # simulation_frequency越大越慢
    env.configure({"simulation_frequency": 120})  # Higher FPS for rendering
    episode_rewards = 0
    return_list = []
    for videos in range(100):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
            episode_rewards += reward
        return_list.append(episode_rewards)
        episode_rewards = 0

    import matplotlib.pyplot as plt

    plt.plot(return_list)
    plt.show()
    env.close()
