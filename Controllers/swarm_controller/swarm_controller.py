import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import the environment we built in the previous step
# Assuming it is saved as 'swarm_env.py'
from swarm_env import WildfireSwarmEnv, NUM_DRONES 

class WildfireGymWrapper(gym.Env):
    """
    Wraps our custom Webots Swarm Environment into a standard Gymnasium interface
    so Stable-Baselines3 can train it.
    """
    def __init__(self, supervisor):
        super().__init__()
        self.webots_env = WildfireSwarmEnv(supervisor)
        
        # Action Space: 7 discrete actions (0-6) for each of the drones.
        # We use MultiDiscrete to send an array of actions, one for each drone.
        self.action_space = spaces.MultiDiscrete([7] * NUM_DRONES)
        
        # Observation Space: Flattened array of all drones' local observations
        # Each drone has 6 values: [X, Y, Z, Rel_Fire_X, Rel_Fire_Y, Neighbor_Dist]
        obs_length_per_drone = 6
        total_obs_length = obs_length_per_drone * NUM_DRONES
        
        # Define limits for the observation variables (e.g., -100 to 100 meters)
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, 
            shape=(total_obs_length,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the Swarm environment without invalidating Webots node references
        raw_obs = self.webots_env.reset()
        flat_obs = np.concatenate([raw_obs[i] for i in range(NUM_DRONES)])
        flat_obs = np.clip(flat_obs, -100.0, 100.0).astype(np.float32)
        return flat_obs, {}

    def step(self, action_array):
        # Convert Gym's action array back into the dictionary our environment expects
        actions_dict = {i: action for i, action in enumerate(action_array)}
        
        # Step the Webots simulation
        next_raw_obs, reward, done = self.webots_env.step(actions_dict)
        
        # Flatten the new observations
        flat_obs = np.concatenate([next_raw_obs[i] for i in range(NUM_DRONES)])
        flat_obs = np.clip(flat_obs, -100.0, 100.0).astype(np.float32)
        
        # Stable-Baselines expects (obs, reward, terminated, truncated, info)
        return flat_obs, reward, done, False, {}

# --- EXECUTION & TRAINING ---
if __name__ == "__main__":
    from controller import Supervisor
    supervisor = Supervisor()
    
    # 1. Initialize the wrapper
    gym_env = WildfireGymWrapper(supervisor)
    
    # 2. Verify the environment complies with gym standards
    check_env(gym_env, warn=True)
    
    print("Starting PPO Neural Network Training...")
    
    # 3. Build the Neural Network Model
    # MlpPolicy creates a standard feed-forward neural network
    model = PPO("MlpPolicy", gym_env, verbose=1, tensorboard_log="./ppo_wildfire_tensorboard/")
    
    # 4. Train the drones for 500,000 timesteps
    model.learn(total_timesteps=500000)
    
    # 5. Save the artificial brain!
    model.save("ppo_wildfire_swarm")
    print("Training Complete. Model saved as 'ppo_wildfire_swarm.zip'")