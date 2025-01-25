import carla
import numpy as np
from abc import abstractmethod
from typing import Dict, Tuple
import gym
from gym import spaces

# from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner

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
            state[i] = {
                "ego_waypoints": self.waypoints[i],
                "ma_agent_waypoints": self.waypoints,
            }
        return state

    def apply_control(self, actions) -> None:
        for agent_idx, action in enumerate(actions):
            control = self.get_vehicle_control(action)
            self.egos[agent_idx].apply_control(control)

    def on_step(self) -> None:
        for i in range(self._config.num_agents):
            self.waypoints[i], self.planner_stats[i] = self.get_ego_planner(i).run_step()
        self.num_completed = self.planner_stats[0]["num_completed"]
    
    def reward(self):
        reward_scales = self._config.reward.scales
        ego = self.egos[0]
        ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = np.linalg.norm(ego_velocity)

        # Reward for reaching waypoints
        r_waypoints = 0.0
        if self.num_completed > 0:
            r_waypoints = reward_scales["waypoint"]

        # Reward for speed
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints[0]) > 0:
            # compute the wpt line direction
            next_waypoint = self.waypoints[0][0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])

            # compute the perpendicular direction
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            if perp_direction_norm > 0.05:
                perp_direction = perp_direction / perp_direction_norm
            else:
                perp_direction = np.array([0.0, 0.0])

            # compute the speed reward
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.dot(ego_velocity, perp_direction)
            r_speed = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * max(speed_perpendicular, -0.5)) * reward_scales["speed"]

        # Reward for collision
        r_collision = 0.0
        if reward_scales["collision"] > 0 and self.is_collision(0):
            r_collision = -reward_scales["collision"] * np.abs(speed_norm)

        # Reward for going out of lane
        r_out_of_lane = 0.0
        if len(self.waypoints[0]) > 0:
            dist = perp_direction_norm
            if dist > 0.5:
                r_out_of_lane = -reward_scales["out_of_lane"] * (dist - 0.5)

        # Reward for reaching the destination
        r_destination = 0.0
        if self.is_destination_reached():
            r_destination = reward_scales["destination_reached"]

        # Time penalty
        time_penalty = -reward_scales["time"]

        # Total reward
        total_reward = r_waypoints + r_speed + r_collision + r_out_of_lane + r_destination + time_penalty

        ttc = TTCCalculator.get_ttc(ego, self._world.carla_world, self._world.carla_map)

        info = {
            **self.planner_stats,
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(ego_location, 0),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_out_of_lane": r_out_of_lane,
            "ttc": ttc,
        }

        return total_reward, info
    
    def is_destination_reached(self):
        return len(self.waypoints[0]) <= 3

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        terminal_conds = {}
        for i in range(self._config.num_agents):
            ego_location = get_vehicle_pos(self.egos[i])
            conds = {
                "is_collision": self.is_collision(i),
                "time_exceeded": self._time_step > terminal_config.time_limit,
                "out_of_lane": self.get_wpt_dist(ego_location, i) > terminal_config.out_lane_thres,
                "destination_reached": self.is_destination_reached(),
            }
            terminal_conds[i] = conds
        return terminal_conds

    def get_wpt_dist(self, ego_location, agent_idx):
        if len(self.waypoints[agent_idx]) == 0:
            return 0
        else:
            return get_location_distance(ego_location, self.waypoints[agent_idx][0])


class CarlaFourLaneMaEnv(CarlaWptEnv):
    """
    Four lane tasks for multi-agent training
    """

    def __init__(self, config):
        super().__init__(config)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def get_state(self):
        state = super().get_state()
        
        # Add WM agent IDs to each state
        ma_agent_ids = [ego.id for ego in self.egos.values()]
        for i in range(self._config.num_agents):
            state[i]["ma_agent_ids"] = ma_agent_ids
            
        return state

    def get_ego_planner(self, agent_idx) -> BasePlanner:
        return self.ego_planners[agent_idx]

    def on_reset(self) -> None:
        self.ego_srcs = [path[0] for path in self._config.ego_path]
        self.egos = {}
        self.ego_planners = {}
        self.waypoints = {}
        self.planner_stats = {}
        self.ego_paths = self._config.ego_path

        for i in range(self.num_agents):
            ego_transform = carla.Transform(
                carla.Location(x=self.ego_srcs[i][0], y=self.ego_srcs[i][1], z=self.ego_srcs[i][2]), 
                carla.Rotation(yaw=-90)
            )
            ego = self._world.spawn_actor(transform=ego_transform)
            self.egos[i] = ego
            self.ego_path = self._config.ego_path[i]
            self.use_road_waypoints = self._config.use_road_waypoints[i]
            self.ego_planners[i] = FixedPathPlanner(
                vehicle=ego,
                vehicle_path=self.ego_path,
                use_road_waypoints=self.use_road_waypoints,
            )
            self.waypoints[i], self.planner_stats[i] = self.ego_planners[i].run_step()

        self._world.spawn_auto_actors(self._config.num_vehicles)

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            raise NotImplementedError("Continuous action space is not supported yet.")