import os
import carla
import random
import argparse
import torch

from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario
from carla_real_traffic_scenarios.ngsim.scenario_merge_copy import (
    NGSimLaneChangeScenario_merge,
)
from carla_real_traffic_scenarios.utils.carla import (
    RealTrafficVehiclesInCarla,
)
from carla_real_traffic_scenarios.ngsim.ngsim_recording_copy import (
    NGSimRecording,
)


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


if __name__ == "__main__":
    args = parser_args()

    print(f"Trying to connect to CARLA server via {args.host}:{args.port}", end="...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)

    # scenario = prepare_ngsim_scenario(client)
    data_dir = os.environ.get("NGSIM_DIR")
    ngsim_dataset = NGSimDatasets.list()
    # 加载地图
    client.load_world(ngsim_dataset.carla_map.level_path)

    world = client.get_world()
    spectator = world.get_spectator()

    settings = world.get_settings()
    settings.synchronous_mode = True  # 同步模式
    settings.fixed_delta_seconds = 0.1  # 固定时间步长
    world.apply_settings(settings)
    # 场景准备
    scenario = prepare_ngsim_scenario(client)
    # ngsim replay
    # _ngsim_vehicles_in_carla = RealTrafficVehiclesInCarla(client, world)
    # _ngsim_recording = NGSimRecording(
    #     data_dir=data_dir,
    #     ngsim_dataset=ngsim_dataset,
    # )
    # _ngsim_recording.reset(ngsim_dataset.timeslots[2], 50)
    # vehicles = _ngsim_recording.step()
    # _ngsim_vehicles_in_carla.step(vehicles)

    # scenario.reset(ego_vehicle)vehicles

    # OpenAI Gym interface:
    for ep_idx in range(args.num_episodes):
        print(f"Running episode {ep_idx+1}/{args.num_episodes}")
        scenario.reset()
        index = 0
        while index < 3000:
            scenario.step()
            index += 1
            world.tick()
        # ngsim replay
        # _ngsim_recording.reset(ngsim_dataset.timeslots[2], 50)
        # vehicles = _ngsim_recording.step()
        # _ngsim_vehicles_in_carla.step(vehicles)
        # # while not done:
        # index = 0
        # while index < 3000:
        #     vehicles = _ngsim_recording.step()
        #     if vehicles is None:
        #         break
        #     _ngsim_vehicles_in_carla.step(vehicles)
        #     index = index + 1
        #     world.tick()
