from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv_v1(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "speed_range": (10, 30),
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "collision_reward": -1,
                # "lane_change_reward": 1,
                "lane_change_reward": 2.0,
                "reward_speed_range": [20, 30],
                # "staright_speed_reward": 0.2,
                "staright_speed_reward": 0.0,
                "distance_reward": 0.0,
                "lane_centering_cost": 4.0,  #
                # "lane_centering_reward": 0.8,
                "lane_centering_reward": 1.2,
                "action_reward": -0.2,
                "action_flag": 0,
                "high_speed_reward": 0.4,
            }
        )
        return cfg

    def _reward(self, action: np.ndarray) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        # if (
        #     self.vehicle.lane_index == (("b", "c", 2))
        #     and self.vehicle.position[0] > 230 + 5
        # ):
        #     if rewards["action_flag"] > 0:
        #         reward = reward + 2 * (
        #             self.config["action_reward"] * -1 * rewards["action_reward"]
        #         )
        #     reward = reward - (
        #         self.config["lane_centering_reward"] * rewards["lane_centering_reward"]
        #     )
        #     reward = utils.lmap(
        #         reward,
        #         [
        #             self.config["collision_reward"],
        #             +self.config["staright_speed_reward"]
        #             + self.config["action_reward"] * -1
        #             + self.config["high_speed_reward"],
        #         ],
        #         [0, 1],
        #     )
        # elif (
        #     self.vehicle.lane_index == (("b", "c", 1))
        #     and self.vehicle.position[0] > 230 + 5
        # ):
        #     reward = reward - 2 * (
        #         self.config["staright_speed_reward"] * rewards["staright_speed_reward"]
        #     )
        #     reward = utils.lmap(
        #         reward,
        #         [
        #             self.config["collision_reward"]
        #             + self.config["action_reward"]
        #             + self.config["staright_speed_reward"] * -1,
        #             self.config["lane_centering_reward"]
        #             + self.config["high_speed_reward"],
        #         ],
        #         [0, 1],
        #     )
        # elif (
        #     self.vehicle.lane_index == (("b", "c", 1))
        #     and self.vehicle.position[0] < 230 + 5
        # ):
        #     reward = 0
        # else:
        #     reward = reward - (
        #         self.config["staright_speed_reward"] * rewards["staright_speed_reward"]
        #     )
        #     reward = utils.lmap(
        #         reward,
        #         [
        #             self.config["collision_reward"] + self.config["action_reward"],
        #             self.config["lane_centering_reward"]
        #             + self.config["high_speed_reward"],
        #         ],
        #         [0, 1],
        #     )
        reward = utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["action_reward"],
                self.config["lane_centering_reward"]
                + self.config["high_speed_reward"]
                + self.config["lane_change_reward"],
            ],
            [0, 1],
        )
        reward *= rewards["on_road_reward"]
        # print(reward)
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        # 换道奖励
        scaled_x_position = utils.lmap(
            80 - (self.vehicle.position[0] - 230), [5, 80], [0, 1]
        )
        scaled_speed = utils.lmap(self.vehicle.speed, [10, 30], [0, 1])
        if (
            self.vehicle.lane_index == (("b", "c", 1))
            and self.vehicle.position[0] > 230 + 5
        ):
            lane_change_flag = 1
        else:
            lane_change_flag = 0
        if self.vehicle.lane_index != self.delay_lane:
            lane_change_reward = 1
        else:
            lane_change_reward = 0
        self.delay_lane = self.vehicle.lane_index
        if action[1] > 0:
            action_flag = 1
        else:
            action_flag = -1
        crashed_reward = 0
        if self.vehicle.crashed:
            if self.vehicle.position[0] > 310:
                crashed_reward = 1
            else:
                crashed_reward = 0.5
        else:
            crashed_reward = 0
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action[1]),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
            "staright_speed_reward": sum(  # 影响其他车辆速度
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 1)
                and isinstance(vehicle, ControlledVehicle)
            ),
            "lane_change_reward": lane_change_reward
            * lane_change_flag
            * scaled_x_position,
            "action_flag": action_flag,
            "high_speed_reward": scaled_speed,
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        # print("crash" + str(self.vehicle.crashed))
        # print("over" + str(self.vehicle.position[0] > 370))
        # print("over" + str(bool(self.vehicle.position[0] > 460)))
        # return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
        if self.vehicle.on_road:
            return self.vehicle.crashed or bool(self.vehicle.position[0] > 310)  # 310
        else:
            return True

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.delay_lane = self.vehicle.lane_index

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        rng = self.np_random
        road = self.road
        # ego_vehicle = self.action_type.vehicle_class(
        #     road, road.network.get_lane(("j", "k", 0)).position(130, 0), speed=30
        # )
        ego_vehicle = self.action_type.vehicle_class.make_on_lane(
            road, ("j", "k", 0), speed=20, longitudinal=130
        )
        # ego_vehicle = self.action_type.vehicle_class(
        #     road, road.network.get_lane(("j", "k", 0)).position(100, 0), speed=30
        # )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
        #     lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
        #     position = lane.position(position + self.np_random.uniform(-5, 5), 0)
        #     speed += self.np_random.uniform(-1, 1)
        #     road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        # for position, speed in [
        #     (105, 20),
        #     (90, 21),
        #     (75, 20),
        #     (60, 18),
        #     (45, 16),
        #     (130, 20),
        #     (145, 20),
        #     (160, 20),
        #     # (175, 20),
        #     # (190, 20),
        #     (205, 23),
        #     # (220, 25),
        # ]:
        #     lane = road.network.get_lane(("a", "b", 1))
        #     position = lane.position(position + self.np_random.uniform(-5, 5), 0)
        #     speed += self.np_random.uniform(-1, 1)
        #     road.vehicles.append(
        #         other_vehicles_type(
        #             road, position, speed=speed, enable_lane_change=False
        #         )
        #     )
        # 随机获取一个8到12的整数
        vehicle_nums = np.random.randint(8, 10)
        # 0到200根据车辆数目等分
        vehicle_position = np.linspace(20, 200, vehicle_nums)
        for position in vehicle_position:
            lane = road.network.get_lane(("a", "b", 1))
            position = lane.position(position + np.random.uniform(-5, 5), 0)
            speed = 20 + np.random.uniform(-5, 5)
            road.vehicles.append(
                other_vehicles_type(
                    road, position, speed=speed, enable_lane_change=False
                )
            )

        # for position, speed in [(95, 10), (75, 10), (10, 10)]:
        #     lane = road.network.get_lane(("b", "c", 0))
        #     position = lane.position(position + self.np_random.uniform(-5, 5), 0)
        #     speed += self.np_random.uniform(-1, 1)
        #     road.vehicles.append(
        #         other_vehicles_type(
        #             road, position, speed=speed, enable_lane_change=False
        #         )
        #     )
        # merging_v = other_vehicles_type(
        #     road, road.network.get_lane(("j", "k", 0)).position(130, 0), speed=20
        # )
        # merging_v.target_speed = 30
        # road.vehicles.append(merging_v)

        # self.vehicle为环境返回observation的对象
        self.vehicle = ego_vehicle
