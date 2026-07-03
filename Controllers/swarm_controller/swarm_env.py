from controller import Supervisor
import numpy as np
import math
from typing import List, Dict, Tuple

# --- RL ENVIRONMENT HYPERPARAMETERS ---
NUM_DRONES = 4
SPACING = 3.0
FOREST_SIZE = 40.0       # Size of the forest simulation area (meters)
GRID_RESOLUTION = 2.0    # Each grid cell is 2x2 meters
COLLISION_DIST = 4.0     # Penalty threshold if drones get too close
DETECTION_RADIUS = 6.0   # Distance inside which a drone "detects" the fire

class WildfireSwarmEnv:
    def __init__(self, supervisor_instance: Supervisor):
        self.sv = supervisor_instance
        self.timestep = int(self.sv.getBasicTimeStep())
        
        # Initialize Drones
        self.drones = []
        self._spawn_swarm()
        
        # Track down the Fire/Wildfire node in the Webots world file
        self.fire_node = self.sv.getFromDef("FIRE_0")
        if self.fire_node is None:
            print("WARNING: No node named 'FIRE_0' found in Scene Tree. Using origin fallback.")
            
        # Initialize Forest Coverage Grid Tracking Matrix
        self.grid_cells = int(FOREST_SIZE / GRID_RESOLUTION)
        self.coverage_grid = np.zeros((self.grid_cells, self.grid_cells), dtype=np.int32)

    def _spawn_swarm(self):
        root = self.sv.getRoot()
        children_field = root.getField('children')
        
        for i in range(NUM_DRONES):
            if i == 0:
                self.drones.append(self.sv.getFromDef('drone_0'))
                continue
                
            # 100ms staggering buffer during creation loop
            wait_steps = max(1, int(10 / self.timestep))
            for _ in range(wait_steps):
                self.sv.step(self.timestep)
                
            x = i * SPACING
            drone_string = (
                f'DEF drone_{i} Mavic2Pro {{ '
                f'name "drone_{i}" '
                f'translation {x} 0.0 0.12 '
                f'controller "drone_controller" '
                f'}}'
            )
            children_field.importMFNodeFromString(-1, drone_string)
            self.drones.append(self.sv.getFromDef(f'drone_{i}'))

    def get_global_fire_pos(self) -> np.ndarray:
        """Retrieves global fire target matrix via Supervisor spatial node lookups."""
        if self.fire_node:
            return np.array(self.fire_node.getPosition())
        return np.array([10.0, 10.0, 0.5])  # Default fallback coordinates

    def get_observations(self) -> Dict[int, np.ndarray]:
        """
        Builds the localized observation space vector for each individual agent actor.
        Vector includes: [Self X, Self Y, Self Z, Rel_Fire_X, Rel_Fire_Y, Closest_Neighbor_Dist]
        """
        obs = {}
        fire_pos = self.get_global_fire_pos()
        
        # Retrieve all positions first
        positions = [np.array(d.getPosition()) if d else np.array([0.,0.,0.]) for d in self.drones]
        
        for i, d in enumerate(self.drones):
            if d is None: continue
            my_pos = positions[i]
            
            # Vector vector pointing to the fire source
            rel_fire = fire_pos - my_pos
            
            # Find distance to the closest teammate drone
            min_neighbor_dist = 999.0
            for j, other_pos in enumerate(positions):
                if i != j:
                    dist = np.linalg.norm(my_pos - other_pos)
                    if dist < min_neighbor_dist:
                        min_neighbor_dist = dist
                        
            # Flatten into a raw continuous sensory observation array
            obs[i] = np.array([
                my_pos[0], my_pos[1], my_pos[2],  # Self coordinates
                rel_fire[0], rel_fire[1],         # Location of fire relative to drone
                min_neighbor_dist                 # Radar distance to teammates
            ], dtype=np.float32)
            
        return obs

    def compute_step_rewards(self) -> Tuple[float, Dict[int, bool]]:
        """
        Calculates the global collective cooperative reward pool.
        Encourages dynamic exploration, penalizes overlaps, rewards fire sweeps.
        """
        shared_reward = 0.0
        detections = {i: False for i in range(NUM_DRONES)}
        fire_pos = self.get_global_fire_pos()
        positions = [np.array(d.getPosition()) if d else np.array([0.,0.,0.]) for d in self.drones]

        # 1. Coverage Reward Allocation Loop
        for i, my_pos in enumerate(positions):
            # Map physical continuous position coordinates down into our virtual index array grid
            grid_x = int((my_pos[0] + FOREST_SIZE/2) / GRID_RESOLUTION)
            grid_y = int((my_pos[1] + FOREST_SIZE/2) / GRID_RESOLUTION)
            
            if 0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells:
                if self.coverage_grid[grid_x, grid_y] == 0:
                    self.coverage_grid[grid_x, grid_y] = 1
                    shared_reward += 5.0  # Shared bonus points for exploring new territory

        # 2. Team Proximity Proximity Penalties
        for i in range(NUM_DRONES):
            for j in range(i + 1, NUM_DRONES):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < COLLISION_DIST:
                    shared_reward -= 10.0 * (COLLISION_DIST - dist)  # Progressive safety constraint

        # 3. Target Search / Fire Spotting Bonus
        for i, my_pos in enumerate(positions):
            dist_to_fire = np.linalg.norm(my_pos[:2] - fire_pos[:2])  # 2D Ground distance
            if dist_to_fire <= DETECTION_RADIUS:
                detections[i] = True
                shared_reward += 20.0  # Massive team incentive for locking coordinates onto the fire

        return shared_reward, detections

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], float, bool]:
        """Executes selected model steps across all agents simultaneously in Webots."""
        # Inject the action keys down to the customData interfaces
        for i, action in actions.items():
            if self.drones[i]:
                pos_field = self.drones[i].getField('customData')
                if pos_field:
                    pos_field.setSFString(str(action))
                    
        # Advance the physical simulation engine frame
        self.sv.step(self.timestep)
        
        # Extract new states and feedback metrics
        next_obs = self.get_observations()
        reward, detections = self.compute_step_rewards()
        
        # Check termination state: Game over if all drones successfully register fire contact
        done = all(detections.values())
        
        return next_obs, reward, done

# --- CHOREOGRAPHED REINFORCEMENT LEARNING RUNTIME LOOP ---
if __name__ == "__main__":
    # 1. Initialize the Webots Supervisor and Timestep
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    
    # 2. Now you can safely pass the supervisor into your environment
    env = WildfireSwarmEnv(supervisor)
    print("RL Environment Framework Initialized. Running Policy Networks...")
    
    # Simple rollout loop showing where a framework like MAPPO, RLlib, or StableBaselines injects
    while supervisor.step(timestep) != -1:
        # Fetch current system perceptions
        current_observations = env.get_observations()
        
        # MOCK POLICY INTERFACE PLACEHOLDER: 
        # In actual training, you pass 'current_observations' into your MAPPO Actor Neural Networks:
        # actions = policy.predict(current_observations)
        actions = {}
        for i in range(NUM_DRONES): # Ensure NUM_DRONES is also defined above!
            # For demonstration purposes: Pick a random exploration action (0 to 6)
            actions[i] = np.random.randint(0, 7)
            
        # Execute the action configurations across the entire swarm ensemble
        next_observations, step_reward, is_terminated = env.step(actions)
        
        if step_reward != 0.0:
            print(f"Global Swarm Step Reward Pool: {step_reward:.2f}")
            
        if is_terminated:
            print("SUCCESS: Fire Encircled completely by all agents! Resetting forest matrix...")
            env.coverage_grid.fill(0)