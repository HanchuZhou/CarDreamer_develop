from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

from car_dreamer.toolkit import Command, Command2Index, FourLaneCommandPlanner, Index2Command

from .toolkit import (
    BasePlanner,
    EnvMonitorOpenCV,
    Observer,
    TTCCalculator,
    WorldManager,
    get_location_distance,
    get_vehicle_pos,
    get_vehicle_velocity,
)


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self.num_agents = config.num_agents
        self._config = config

        self._monitor = EnvMonitorOpenCV(self._config)
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observers = {i: Observer(self._world, self._config.observation) for i in range(self.num_agents)}

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    @abstractmethod
    def on_reset(self) -> None:
        pass

    @abstractmethod
    def apply_control(self, actions) -> None:
        pass

    @abstractmethod
    def on_step(self) -> None:
        pass

    @abstractmethod
    def reward(self) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, Dict[str, bool]]:
        pass

    def get_ego_vehicles(self) -> Dict[int, carla.Actor]:
        return {i: self.egos[i] for i in range(self._config.num_agents)}

    def get_state(self) -> Dict:
        return self._state

    def _get_action_space(self):
        pass

    def _get_observation_space(self):
        return self._observers[0].get_observation_space()

    def reset(self):
        print("[CARLA] Reset environment")

        for observer in self._observers.values():
            observer.destroy()
        self._world.reset()
        for i, ego in self.egos.items():
            self._observers[i].reset(ego)

        self._time_step = 0

        print("[CARLA] Environment reset")
        self.obs, _ = self._get_observation()
        return self.obs

    def _get_observation(self):
        obs = {}
        info = {}
        for i in range(self._config.num_agents):
            agent_obs, agent_info = self._observers[i].get_observation(self.get_state()[i])
            obs[str(i)] = agent_obs
            info[str(i)] = agent_info
        return obs, info

    def get_vehicle_control(self, action):
        action_config = self._config.action
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for agent_idx, agent_conds in terminal_conds.items():
            for k, v in agent_conds.items():
                if v:
                    print(f"[CARLA] Terminal condition triggered for {agent_idx}: {k}")
                    terminal = True
                agent_conds[k] = np.array([v], dtype=np.bool_)
            if terminal:
                agent_conds["episode_timesteps"] = self._time_step
        return terminal, terminal_conds

    def step(self, actions):
        self.apply_control(actions)
        self._world.step()
        self._time_step += 1

        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._get_observation()
        reward, _ = self.reward()

        info = {
            **terminal_conds,
            **obs_info,
        }
        if self._config.display.enable:
            self._render(self.obs, info)

        return (self.obs, reward, is_terminal, info)

    def is_collision(self, agent_idx):
        return self.obs[str(agent_idx)]["collision"][0] > 0

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = str(prefix) + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _render(self, obs, info):
        obs = self._flatten(obs)
        info = self._flatten(info)
        self._monitor.render(obs, info)


class CarlaWptEnv(CarlaBaseEnv):
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def get_ego_planner(self, agent_idx) -> BasePlanner:
        pass

    def get_state(self):
        state = {}
        for i in range(self._config.num_agents):
            state[i] = {"ego_waypoints": self.waypoints[i]}
        return state

    def apply_control(self, actions) -> None:
        for agent_idx, action in enumerate(actions):
            control = self.get_vehicle_control(action)
            self.egos[agent_idx].apply_control(control)

    def on_step(self) -> None:
        for i in range(self._config.num_agents):
            self.waypoints[i], self.planner_stats[i] = self.get_ego_planner(i).run_step()
            self.num_completed[i] = self.planner_stats[i]["num_completed"]

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        terminal_conds = {}
        for i in range(self._config.num_agents):
            ego_location = get_vehicle_pos(self.egos[i])
            conds = {
                "is_collision": self.is_collision(i),
                "time_exceeded": self._time_step > terminal_config.time_limit,
                "out_of_lane": self.get_wpt_dist(ego_location, i) > terminal_config.out_lane_thres,
                "destination_reached": len(self.waypoints[i]) == 0,
            }
            terminal_conds[i] = conds
        return terminal_conds

    def get_wpt_dist(self, ego_location, agent_idx):
        if len(self.waypoints[agent_idx]) == 0:
            return 0
        else:
            return get_location_distance(ego_location, self.waypoints[agent_idx][0])


class CarlaMessageMaEnv(CarlaWptEnv):
    """
    An environment that requires the agent to follow a text command.
    """

    def __init__(self, config):
        super().__init__(config)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self._e_command_repeat_counts = 64
        self._e_enable_replan = True

    def get_ego_planner(self, agent_idx) -> BasePlanner:
        return self.ego_planners[agent_idx]

    def get_state(self):
        state = super().get_state()
        for i in range(self.num_agents):
            state[i] = {
                **state[i],
                "dest_lane_idx": self.dest_lanes[i],
                "command": self.commands[i],
                "dest_x": self.dest_points[i][0],
                "actor_actions": {ego.id: self.commands[i] for (i, ego) in self.egos.items()},
            }
        return state

    def on_reset(self) -> None:
        self.dest_lanes = [3, 0]
        self.ego_srcs = [self._config.lane_start_points[i] for i in [0, 3]]
        self.dest_points = [self._config.lane_end_points[i] for i in [3, 0]]
        self.egos = {}
        self.ego_planners = {}
        self.waypoints = {}
        self.planner_stats = {}
        self.commands = [Command.LaneFollow] * self.num_agents
        self.num_completed = [0] * self.num_agents
        self.command_not_completed = [False] * self.num_agents
        self.command_completed_count = [0] * self.num_agents

        for i in range(self.num_agents):
            ego_x = self.ego_srcs[i][0]
            ego_y = self.ego_srcs[i][1] + np.random.uniform(-6, 6)
            ego_z = self.ego_srcs[i][2]
            ego_transform = carla.Transform(carla.Location(x=ego_x, y=ego_y, z=ego_z), carla.Rotation(yaw=-90))
            ego = self._world.spawn_actor(transform=ego_transform)
            self.egos[i] = ego
            self.ego_planners[i] = FourLaneCommandPlanner(
                ego, [p[0] for p in self._config.lane_start_points], force_reset_command=self._config.force_reset_command
            )
            self.ego_planners[i].set_command(Command.LaneFollow)
            self.waypoints[i], self.planner_stats[i] = self.ego_planners[i].run_step()

        self._world.spawn_auto_actors(self._config.num_vehicles)

    def apply_control(self, actions) -> None:
        indices = []
        action_num = self.n_acc * self.n_steer
        for i in range(self.num_agents):
            command, control = actions[i] // action_num, actions[i] % action_num
            command = Index2Command[command]
            if command != self.commands[i]:
                if self.ego_planners[i].get_ego_lane_id() != self.ego_planners[i].get_goal_lane_id() and not self._e_enable_replan:
                    self.command_not_completed[i] = True
                    continue
                self.commands[i] = command
                self.command_completed_count[i] = 0
                self.ego_planners[i].set_command(command)
            if self.ego_planners[i].get_ego_lane_id() == self.ego_planners[i].get_goal_lane_id():
                self.command_completed_count[i] += 1
                if self.command_completed_count[i] >= self._e_command_repeat_counts:
                    self.command_completed_count[i] = 0
                    self.ego_planners[i].set_command(self.commands[i])
            indices.append(control)
        super().apply_control(indices)

    def reward(self):
        return 0, {}

    def get_terminal_conditions(self):
        terminal_conditions = super().get_terminal_conditions()
        for i in range(self.num_agents):
            info = {}
            info["invalid_command"] = self.is_command_invalid(i)
            info["destination_reached"] = self.is_road_end_reached(i) and self.is_dest_lane_reached(i)
            info["road_end_reached"] = self.is_road_end_reached(i)
            terminal_conditions[i] = {**terminal_conditions[i], **info}
        return terminal_conditions

    def is_road_end_reached(self, agent_idx):
        return self.egos[agent_idx].get_location().y < self.dest_points[agent_idx][1]

    def is_dest_lane_reached(self, agent_idx):
        return self.ego_planners[agent_idx].get_ego_lane_id() == self.dest_lanes[agent_idx]

    def is_command_invalid(self, agent_idx):
        return not self.ego_planners[agent_idx].is_command_valid() or self.command_not_completed[agent_idx]

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc + action_config.n_commands)
        else:
            raise NotImplementedError("Continuous action space is not supported yet.")
