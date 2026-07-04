import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
from datetime import datetime
from pathlib import Path

# Import the environment we built in the previous step
# Assuming it is saved as 'swarm_env.py'
from swarm_env import WildfireSwarmEnv, NUM_DRONES, FOREST_SIZE, GRID_RESOLUTION, COLLISION_DIST, DETECTION_RADIUS, MAX_EPISODE_STEPS

class CheckpointCallback(BaseCallback):
    """Custom callback to save model checkpoints at regular intervals during training."""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.checkpoint_count = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}"
            self.model.save(str(save_path))
            self.checkpoint_count += 1
            if self.verbose > 0:
                print(f"Checkpoint saved: {save_path}.zip (Step {self.num_timesteps})")
        return True

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
    
    # Setup training directories
    training_info_dir = Path("training_info")
    training_info_dir.mkdir(exist_ok=True)
    
    # Create timestamped subdirectory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = training_info_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Initialize supervisor and environment
    supervisor = Supervisor()
    gym_env = WildfireGymWrapper(supervisor)
    
    # Verify the environment complies with gym standards
    check_env(gym_env, warn=True)
    
    # Log environment configuration
    env_config = {
        "num_drones": NUM_DRONES,
        "forest_size": FOREST_SIZE,
        "grid_resolution": GRID_RESOLUTION,
        "collision_distance": COLLISION_DIST,
        "detection_radius": DETECTION_RADIUS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "observation_space": f"Box({6 * NUM_DRONES},)",
        "action_space": f"MultiDiscrete([7] * {NUM_DRONES})"
    }
    with open(run_dir / "environment_config.json", "w") as f:
        json.dump(env_config, f, indent=2)
    
    # Log training hyperparameters
    training_config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 500000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 10,
        "n_steps": 2048,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "batch_size": 64,
        "checkpoint_frequency": 50000,
        "training_start_time": timestamp
    }
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    print(f"Starting PPO Neural Network Training...")
    print(f"Training info will be saved to: {run_dir}")
    
    # Create model with explicit hyperparameters
    model = PPO(
        "MlpPolicy", 
        gym_env, 
        verbose=1,
        learning_rate=training_config["learning_rate"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        n_epochs=training_config["n_epochs"],
        n_steps=training_config["n_steps"],
        clip_range=training_config["clip_range"],
        tensorboard_log="./ppo_wildfire_tensorboard/"
    )
    
    # Create checkpoint callback (saves every 50,000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(checkpoints_dir),
        name_prefix="checkpoint",
        verbose=1
    )
    
    # Train the model with checkpoints
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback
    )
    
    # Save final model
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    
    # Log training completion metadata
    metadata = {
        "training_end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_timesteps_completed": 500000,
        "checkpoints_saved": checkpoint_callback.checkpoint_count,
        "final_model_saved": True,
        "status": "completed"
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training Complete!")
    print(f"Final model saved: {final_model_path}.zip")
    print(f"All training info saved to: {run_dir}")