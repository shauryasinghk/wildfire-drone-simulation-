import os
import sys
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box

if "WEBOTS_CONTROLLER_URL" in os.environ:
    del os.environ["WEBOTS_CONTROLLER_URL"]
os.environ["WEBOTS_ROBOT_NAME"] = "swarm_supervisor"

# Ensure Python can find the Webots API on Windows
WEBOTS_PATH = "C:/Program Files/Webots"
os.environ['WEBOTS_HOME'] = WEBOTS_PATH
sys.path.append(os.path.join(WEBOTS_PATH, 'lib', 'controller', 'python'))
from controller import Supervisor

INITIAL_HEIGHT = 5

# Global reference to cache the single permitted Webots Supervisor instance
_SHARED_SUPERVISOR = None

class WebotsDroneCoverageEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "wildfire_drone_coverage"}

    def __init__(self, num_drones=3, map_size=50, spacing=5):
        super().__init__()
        global _SHARED_SUPERVISOR
        self.num_drones = num_drones
        self.map_size = map_size
        self.agents = [f"drone_{i}" for i in range(num_drones)]
        self.possible_agents = self.agents[:]
        
        # Enforce Singleton pattern across BenchMARL validation re-runs
        if _SHARED_SUPERVISOR is None:
            print("Initializing Webots Supervisor Instance...")
            _SHARED_SUPERVISOR = Supervisor()
            print("Supervisor instance created successfully.")
        else:
            print("Re-using active Webots Supervisor Instance...")
            
        self.supervisor = _SHARED_SUPERVISOR
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        print("Dynamically generating swarm agents...")
        self.drones = []
        for i in range(num_drones):
            # Safe check: Avoid re-injecting nodes if they already exist in the active world tree
            existing_drone = self.supervisor.getFromDef(f'drone_{i}')
            if existing_drone is not None:
                self.drones.append(existing_drone)
                continue
                
            if i == 0:
                drone = self.supervisor.getFromDef('drone_0')
                self.drones.append(drone)
                continue
            
            # Linear spacing along the X axis on the ground
            x = i * spacing
            y = 0.0
            z = INITIAL_HEIGHT
            
            drone_string = f'DEF drone_{i} Mavic2Pro {{ name "drone_{i}" translation {x} {y} {z} controller "<none>" }}'
            
            root = self.supervisor.getRoot()
            children_field = root.getField('children')
            children_field.importMFNodeFromString(-1, drone_string)
            
            drone = self.supervisor.getFromDef(f'drone_{i}')
            self.drones.append(drone)
        print("Swarm network initialization complete.")

        self.initial_states = {}
        for i in range(num_drones):
            node = self.drones[i]
            if node is not None:
                self.initial_states[i] = {
                    "translation": node.getField("translation").getSFVec3f(),
                    "rotation": node.getField("rotation").getSFRotation()
                }

        # Coverage Grid: 50x50 matrix (0 = Unvisited, 1 = Covered)
        self.coverage_grid = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # MARL Spaces: Local observations per drone
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Action Spaces: Target flight velocities [vx, vy, vz]
        self.action_spaces = {
            agent: Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
            for agent in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        for i in range(self.num_drones):
            node = self.drones[i]
            if node is not None:
                node.getField("translation").setSFVec3f(self.initial_states[i]["translation"])
                node.getField("rotation").setSFRotation(self.initial_states[i]["rotation"])
                node.resetPhysics()

        self.supervisor.simulationResetPhysics()
        self.coverage_grid.fill(0)
        self.supervisor.step(self.timestep)
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_observations(self):
        obs = {}
        for i, drone in enumerate(self.drones):
            if drone is not None:
                pos = drone.getPosition()   # [X, Y, Z]
                vel = drone.getVelocity()   # [Vx, Vy, Vz, Wx, Wy, Wz]
            
                obs[self.agents[i]] = np.array([pos[2], vel[0], vel[1], 1.0], dtype=np.float32)
        return obs

    def step(self, actions):
        for i, action in enumerate(actions):
            node = self.drones[i]
            if node is not None:
                if np.isnan(action).any() or np.isinf(action).any():
                    print(f"\n[WARNING] NaN/Inf action detected for drone_{i}! Forcing hover.")
                    action = np.zeros(3, dtype=np.float32)
                node.setVelocity([float(action[0]), float(action[1]), float(action[2]), 0, 0, 0])

        for _ in range(12): 
            self.supervisor.step(self.timestep)

        newly_covered_cells = 0
        for i, drone in enumerate(self.drones):
            if drone is not None:
                pos = drone.getPosition()
                grid_x = int(np.clip((pos[0] + 25), 0, self.map_size - 1))
                grid_y = int(np.clip((pos[1] + 25), 0, self.map_size - 1))
                
                if self.coverage_grid[grid_x, grid_y] == 0:
                    self.coverage_grid[grid_x, grid_y] = 1
                    newly_covered_cells += 1

        team_reward = float(newly_covered_cells * 1)
        rewards = {agent: team_reward for agent in self.agents}

        crashed = any(d.getPosition()[2] < 0.2 for d in self.drones)
        terminations = {agent: crashed for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        observations = self._get_observations()
        return observations, rewards, terminations, truncations, {agent: {} for agent in self.agents}
    

from typing import Optional
import torch
from torchrl.data import Composite
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, UnboundedContinuous, Bounded, UnboundedDiscrete

class BenchMarlWebotsWrapper(EnvBase):
    def __init__(self, pz_env, device="cpu"):
        # BenchMARL uses an empty root batch size for the environment level
        super().__init__(device=device, batch_size=torch.Size([]))
        self.pz_env = pz_env
        self.num_drones = pz_env.num_drones
        self.agents = pz_env.agents
        
        # Sample an observation and action to infer shape sizes automatically
        sample_obs = pz_env.observation_space(self.agents[0]).sample()
        sample_act = pz_env.action_space(self.agents[0]).sample()
        self.obs_dim = sample_obs.shape[0]
        self.action_dim = sample_act.shape[0]
        
        # 1. Observation Spec: Structured under the "agents" group name
        self.observation_spec = Composite({
            "agents": Composite({
                "observation": UnboundedContinuous(
                    shape=torch.Size([self.num_drones, self.obs_dim]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 2. Action Spec: Structured under the "agents" group name with bounds
        act_space = pz_env.action_space(self.agents[0])
        low = torch.tensor(act_space.low, device=device).unsqueeze(0).expand(self.num_drones, -1)
        high = torch.tensor(act_space.high, device=device).unsqueeze(0).expand(self.num_drones, -1)
        
        self.action_spec = Composite({
            "agents": Composite({
                "action": Bounded(
                    low=low, high=high,
                    shape=torch.Size([self.num_drones, self.action_dim]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 3. Reward Spec: Stacked reward per agent group
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuous(
                    shape=torch.Size([self.num_drones, 1]),
                    device=device
                )
            }, batch_size=torch.Size([self.num_drones]))
        })
        
        # 4. Done/Terminal Specs (Global tracking at the root level)
        self.done_spec = Composite({
            "done": UnboundedDiscrete(shape=torch.Size([1]), dtype=torch.bool, device=device),
            "terminated": UnboundedDiscrete(shape=torch.Size([1]), dtype=torch.bool, device=device)
        })

    def _reset(self, tensordict=None):
        # Reset the underlying custom Webots PettingZoo environment
        obs_dict = self.pz_env.reset()[0]
        
        # Stack individual drone observations into a single multi-agent tensor [num_drones, obs_dim]
        obs_list = [torch.tensor(obs_dict[agent], dtype=torch.float32, device=self.device) for agent in self.agents]
        stacked_obs = torch.stack(obs_list, dim=0)
        
        return TensorDict({
            "agents": TensorDict({"observation": stacked_obs}, batch_size=torch.Size([self.num_drones]))
        }, batch_size=torch.Size([]))

    def _step(self, tensordict):
        # 1. Pull the stacked action tensor from MAPPO [num_drones, action_dim]
        action_tensor = tensordict["agents", "action"]
        
        # 2. Convert to an ordered list of numpy actions for your custom env step method
        actions_list = [action_tensor[i].cpu().numpy() for i in range(self.num_drones)]
        
        # 3. Step the environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.pz_env.step(actions_list)
        
        # 4. Collect and restack observations and rewards into BenchMARL format
        obs_list = [torch.tensor(obs_dict[agent], dtype=torch.float32, device=self.device) for agent in self.agents]
        reward_list = [torch.tensor([reward_dict[agent]], dtype=torch.float32, device=self.device) for agent in self.agents]
        
        stacked_obs = torch.stack(obs_list, dim=0)
        stacked_reward = torch.stack(reward_list, dim=0)
        
        # Determine episode completion conditions
        done = any(terminated_dict.values()) or any(truncated_dict.values())
        terminated = any(terminated_dict.values())
        
        return TensorDict({
            "agents": TensorDict({
                "observation": stacked_obs,
                "reward": stacked_reward
            }, batch_size=torch.Size([self.num_drones])),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device)
        }, batch_size=torch.Size([]))

    def _set_seed(self, seed: Optional[int]):
        pass