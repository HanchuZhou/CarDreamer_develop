import collections

import numpy as np

from .basics import convert


class MADriver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, num_agents, **kwargs):
        self._env = env
        self._num_agents = num_agents
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        print("[MA] act_space", self._env.act_space)
        self._acts = {}
        for i in range(self._num_agents):
            self._acts[i] = {k: convert(np.zeros((len(self._env),) + v.shape, v.dtype)) for k, v in self._env.act_space.items()}
            self._acts["reset"] = np.ones(1, bool)
        self._eps = {}
        self._eps_info = {}
        for i in range(self._num_agents):
            self._eps[i] = [collections.defaultdict(list) for _ in range(len(self._env))]
            self._eps_info[i] = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = {i: None for i in range(self._num_agents)}

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        """
        Args:
            policy: Can be either a single policy function or a list of policy functions
                   If single policy: Applied to all agents
                   If list: policies[0] applied to agent 0, policies[1] applied to rest
        """
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        acts = self._acts
        obs, info = self._env.step(acts)

        tmp_obs = {}
        for i in range(self._num_agents):
            tmp_obs[str(i)] = {}
            for k, v in obs.items():
                if k.startswith(f"{i}/"):
                    tmp_obs[str(i)][k[len(f"{i}/"):]] = v
                else:
                    tmp_obs[str(i)][k] = v
        for i in range(self._num_agents):
            obs[str(i)] = tmp_obs[str(i)]

        tmp_info = {}
        for i in range(self._num_agents):
            tmp_info[str(i)] = {}
            for k, v in info.items():
                if k.startswith(f"{i}/"):
                    tmp_info[str(i)][k[len(f"{i}/"):]] = v
                else:
                    tmp_info[str(i)][k] = v
        for i in range(self._num_agents):
            info[str(i)] = tmp_info[str(i)]

        # Handle different policy formats
        if isinstance(policy, (list, tuple)):
            # Multi-agent mode: policy[0] for training agent, policy[1] for others
            for i in range(self._num_agents):
                if i == 0:
                    agent_acts, self._state[i] = policy[0](obs[str(i)], self._state[i], **self._kwargs)
                else:
                    agent_acts, self._state[i] = policy[1](obs[str(i)], self._state[i], **self._kwargs)
                for k, v in agent_acts.items():
                    acts[i][k] = convert(v)
        else:
            # Single agent mode
            for i in range(self._num_agents):
                agent_acts, self._state[i] = policy(obs[str(i)], self._state[i], **self._kwargs)
                for k, v in agent_acts.items():
                    acts[i][k] = convert(v)

        for i in range(self._num_agents):
            if obs["is_last"].any():
                mask = 1 - obs["is_last"]
                acts[i] = {k: v * self._expand(mask, len(v.shape)) for k, v in acts[i].items()}
            acts["reset"] = obs["is_last"].copy()
        self._acts = acts

        trns = {}
        for i in range(self._num_agents):
            trns[i] = {**obs[str(i)], **acts[i]}

        if obs["is_first"].any():
            for i in range(self._num_agents):
                for j, first in enumerate(obs["is_first"]):
                    if first:
                        self._eps[i][j].clear()
                        self._eps_info[i][j].clear()

        for i in range(self._num_agents):
            for j in range(len(self._env)):
                trn = {k: v[j] for k, v in trns[i].items()}
                inf = {k: v[j] for k, v in info[str(i)].items()}
                [self._eps[i][j][k].append(v) for k, v in trn.items()]
                [self._eps_info[i][j][k].append(v) for k, v in inf.items()]
                [fn(trn, inf, i, **self._kwargs) for fn in self._on_steps]

        step += 1

        if obs["is_last"].any():
            for i in range(self._num_agents):
                for j, done in enumerate(obs["is_last"]):
                    if done:
                        ep = {k: convert(v) for k, v in self._eps[i][j].items()}
                        ep_info = {k: convert(v) for k, v in self._eps_info[i][j].items()}
                        [fn(ep.copy(), ep_info.copy(), i, **self._kwargs) for fn in self._on_episodes]
                        # self._state[i][1]["step"] = np.zeros_like(self._state[i][1]["step"], np.int8)
                        # self._state[i][2]["step"] = np.zeros_like(self._state[i][1]["step"], np.int8)
                        # self._state[i] = None
            episode += 1

        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
