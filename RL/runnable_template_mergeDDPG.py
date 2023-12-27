import os
import carla
import random
import argparse
import sys
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/home/lifei/carla-real-traffic-scenarios")
print(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario
from carla_real_traffic_scenarios.ngsim.scenario_merge_copyDDPG import (
    NGSimLaneChangeScenario_merge,
)
from carla_real_traffic_scenarios.utils.carla import (
    RealTrafficVehiclesInCarla,
)
from carla_real_traffic_scenarios.ngsim.ngsim_recording import (
    NGSimRecording,
)
from DDPG import DDPG
from noise import OUNoise


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ngsim", "opendd"], default="opendd")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--num-episodes", default=10, type=int)
    args = parser.parse_args()
    return args


def prepare_ngsim_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("NGSIM_DIR")
    assert data_dir, "Path to the directory with NGSIM dataset is required"
    # # 返回i80、us101
    # ngsim_map = NGSimDatasets.list()
    # # 在i80和us101中随即选择一个
    # ngsim_dataset = random.choice(ngsim_map)
    ngsim_dataset = NGSimDatasets.list()
    # 加载地图
    client.load_world(ngsim_dataset.carla_map.level_path)
    return NGSimLaneChangeScenario_merge(
        ngsim_dataset,
        dataset_mode=DatasetMode.TRAIN,
        data_dir=data_dir,
        reward_type=RewardType.DENSE,
        client=client,
    )


def prepare_opendd_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("OPENDD_DIR")
    assert data_dir, "Path to the directory with openDD dataset is required"
    maps = ["rdb1", "rdb2", "rdb3", "rdb4", "rdb5", "rdb6", "rdb7"]
    map_name = random.choice(maps)
    carla_map = getattr(CarlaMaps, map_name.upper())
    client.load_world(carla_map.level_path)
    return OpenDDScenario(
        client,
        dataset_dir=data_dir,
        dataset_mode=DatasetMode.TRAIN,
        reward_type=RewardType.DENSE,
        place_name=map_name,
    )


def prepare_ego_vehicle(world: carla.World) -> carla.Actor:
    car_blueprint = world.get_blueprint_library().find("vehicle.audi.a2")

    # This will allow external scripts like manual_control.py or no_rendering_mode.py
    # from the official CARLA examples to take control over the ego agent
    car_blueprint.set_attribute("role_name", "hero")

    # spawn points doesnt matter - scenario sets up position in reset
    ego_vehicle = world.spawn_actor(
        car_blueprint, carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    )

    assert ego_vehicle is not None, "Ego vehicle could not be spawned"

    # Setup any car sensors you like, collect observations and then use them as input to your model
    return ego_vehicle


torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parser_args()

    print(f"Trying to connect to CARLA server via {args.host}:{args.port}", end="...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    # 场景准备
    scenario = prepare_ngsim_scenario(client)

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # 同步模式
    settings.fixed_delta_seconds = 0.1  # 固定时间步长
    world.apply_settings(settings)

    spectator = world.get_spectator()
    # 车辆准备
    # ego_vehicle = prepare_ego_vehicle(world)
    # dqn = DQN(
    #     lr=0.001, gamma=0.9, epsilon=0, MEMORY_CAPACITY=50, state_dim=2, action_dim=3
    # )
    state_dim = 18
    action_dim = 1
    # hidden_dim = 400
    hidden_dim = 1
    # buffer_size = 10000
    buffer_size = 10
    # batch_size = 256
    batch_size = 1
    # actor_lr = 5e-4
    actor_lr = 1e-4
    critic_lr = 1e-3
    tau = 0.001
    gamma = 0.995
    ddpg = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_size=buffer_size,
        batch_size=batch_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamme=gamma,
    )
    # s = scenario.reset()
    # OpenAI Gym interface:
    reward_list = []
    for ep_idx in range(args.num_episodes):
        print(f"Running episode {ep_idx+1}/{args.num_episodes}")
        # while not done:
        done = False
        s = scenario.reset()
        step = 0
        while not done:
            a = ddpg.choose_action(state=s, step=step)
            s_, r, done = scenario.step(a)
            ddpg.store_tansition(state=s, action=a, reward=r, next_state=s_, done=done)
            ddpg.learn()
            s = s_
            step += 1
            world.tick()
        reward_list.append(r)
        # 每100个episode计算一次平均reward
        if ep_idx % 100 == 0:
            print("average reward", np.mean(reward_list))
            reward_list = []
        print("done", done)
        print("reward", r)
        # scenario.close()
