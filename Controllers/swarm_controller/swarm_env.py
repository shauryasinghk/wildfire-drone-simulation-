from controller import Supervisor
import numpy as np
import math
from typing import List, Dict, Tuple

# --- RL ENVIRONMENT HYPERPARAMETERS ---
NUM_DRONES = 4
SPACING = 3.0
FOREST_SIZE = 85      # Side length of the square forest area in meters
FOREST_ORIGIN = np.array([-FOREST_SIZE / 2.0, -FOREST_SIZE / 2.0])
FOREST_MAX = FOREST_ORIGIN + np.array([FOREST_SIZE, FOREST_SIZE])
FOREST_CENTER = FOREST_ORIGIN + np.array([FOREST_SIZE / 2.0, FOREST_SIZE / 2.0])
GRID_RESOLUTION = 2.0    # Each grid cell is 2x2 meters
COLLISION_DIST = 4.0     # Penalty threshold if drones get too close
DETECTION_RADIUS = 6.0   # Distance inside which a drone "detects" the fire
MAX_EPISODE_STEPS = 5000 # Maximum steps per episode to prevent infinite loops

# --- NEW SWARM COORDINATION REWARDS ---
NEIGHBOR_TARGET_DIST = 8.0      # Desired minimum separation between drones
NEIGHBOR_PENALTY_WEIGHT = 1.5   # Penalty strength for clustering
NOVELTY_REWARD_WEIGHT = 0.35    # Reward for visiting under-observed cells
NEW_CELL_REWARD = 1.0           # Bonus for visiting a completely new cell
VELOCITY_REWARD_WEIGHT = 0.03   # Reward for useful motion

class WildfireSwarmEnv:
    def __init__(self, supervisor_instance: Supervisor):
        self.sv = supervisor_instance
        self.timestep = int(self.sv.getBasicTimeStep())
        
        # Initialize Drones
        self.drones = []
        self._spawn_swarm()
        
        # Cache initial pose data for reset without invalidating node references
        self.initial_states = {}
        for i, node in enumerate(self.drones):
            if node is not None:
                self.initial_states[i] = {
                    "translation": node.getField("translation").getSFVec3f(),
                    "rotation": node.getField("rotation").getSFRotation()
                }
        # Track which drones have already received a detection reward this episode
        self.has_detected_fire = {i: False for i in range(NUM_DRONES)}
        self.last_fire_distances = {}
        self.prev_positions = {}
        self.step_count = 0
        self.flipped = False
        
        # Track down the Fire/Wildfire node in the Webots world file
        self.fire_node = self.sv.getFromDef("FIRE_0")
        if self.fire_node is None:
            print("WARNING: No node named 'FIRE_0' found in Scene Tree. Using origin fallback.")
        
        # Track the camera node with fixed viewport position and orientation
        self.camera_node = self.sv.getFromDef("camera") or self.sv.getFromDef("Camera")
        self.initial_camera_state = {
            "translation": [-25.9, -4.64, 1.93],
            "rotation": [0.132, 0.514, -0.848, 0.588]
        }
            
        # Initialize Forest Coverage Grid Tracking Matrix
        self.grid_cells = int(np.ceil(FOREST_SIZE / GRID_RESOLUTION))
        self.coverage_grid = np.zeros((self.grid_cells, self.grid_cells), dtype=np.int32)
        self.visitation_grid = np.zeros((self.grid_cells, self.grid_cells), dtype=np.int32)

    def _spawn_swarm(self):
        root = self.sv.getRoot()
        children_field = root.getField('children')

        base_drone = self.sv.getFromDef('drone_0')
        if base_drone is not None:
            base_pos = np.array(base_drone.getPosition())
        else:
            base_pos = np.array([0.0, 0.0, 0.12])

        for i in range(NUM_DRONES):
            if i == 0:
                self.drones.append(base_drone)
                continue

            # 100ms staggering buffer during creation loop
            wait_steps = max(1, int(10 / self.timestep))
            for _ in range(wait_steps):
                self.sv.step(self.timestep)

            offset_x = (i % 2) * SPACING
            offset_y = (i // 2) * SPACING
            x = base_pos[0] + offset_x
            y = base_pos[1] + offset_y
            z = base_pos[2]
            drone_string = (
                f'DEF drone_{i} Mavic2Pro {{ '
                f'name "drone_{i}" '
                f'translation {x} {y} {z} '
                f'controller "drone_controller" '
                f'bodySlot DistanceSensor {{ name "ground_distance_sensor" }} '
                f'}}'
            )
            children_field.importMFNodeFromString(-1, drone_string)
            self.drones.append(self.sv.getFromDef(f'drone_{i}'))

    def reset(self):
        """Reset the swarm state without invalidating dynamic Webots node references."""
        for i, node in enumerate(self.drones):
            if node is not None:
                node.getField("translation").setSFVec3f(self.initial_states[i]["translation"])
                node.getField("rotation").setSFRotation(self.initial_states[i]["rotation"])
                node.resetPhysics()

        self.sv.simulationResetPhysics()
        
        # Reset camera to fixed viewport position and orientation
        if self.camera_node is not None:
            self.camera_node.getField("translation").setSFVec3f(self.initial_camera_state["translation"])
            self.camera_node.getField("rotation").setSFRotation(self.initial_camera_state["rotation"])
        
        self.coverage_grid.fill(0)
        self.visitation_grid.fill(0)
        # Clear per-drone detection flags and reward shaping state at the start of each episode
        self.has_detected_fire = {i: False for i in range(NUM_DRONES)}
        self.last_fire_distances = {}
        self.prev_positions = {
            i: np.array(self.initial_states[i]["translation"][:2], dtype=np.float32)
            if i in self.initial_states else np.array([0.0, 0.0], dtype=np.float32)
            for i in range(NUM_DRONES)
        }
        self.step_count = 0
        self.flipped = False
        self.sv.step(self.timestep)
        return self.get_observations()

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
            if d is None:
                obs[i] = np.zeros(6, dtype=np.float32)
                continue
            my_pos = positions[i]
            
            # Vector vector pointing to the fire source
            rel_fire = fire_pos - my_pos
            
            # Find distance to the closest teammate drone
            min_neighbor_dist = float('inf')
            for j, other_pos in enumerate(positions):
                if i != j:
                    dist = np.linalg.norm(my_pos - other_pos)
                    if dist < min_neighbor_dist:
                        min_neighbor_dist = dist
                        
            if min_neighbor_dist == float('inf'):
                min_neighbor_dist = 100.0
            else:
                min_neighbor_dist = min(min_neighbor_dist, 100.0)

            # Flatten into a raw continuous sensory observation array
            obs[i] = np.array([
                my_pos[0], my_pos[1], my_pos[2],  # Self coordinates
                rel_fire[0], rel_fire[1],         # Location of fire relative to drone
                min_neighbor_dist                 # Radar distance to teammates
            ], dtype=np.float32)
            
        return obs

    def is_drone_flipped(self, node) -> bool:
        """Check if drone has flipped (inverted orientation).
        Uses rotation vector to compute if local Z-axis (up) points downward.
        """
        if node is None:
            return False
        
        # Get rotation as axis-angle: [axis_x, axis_y, axis_z, angle_radians]
        rotation = node.getField('rotation').getSFRotation()
        axis_x, axis_y, axis_z, angle = rotation
        
        # For local up vector (0, 0, 1) rotated by axis-angle (n, theta):
        # Z_component_rotated = cos(theta) + (1 - cos(theta)) * axis_z^2
        # If this is < 0.3, drone is severely tilted or upside-down
        cos_angle = math.cos(angle)
        z_component = cos_angle + (1 - cos_angle) * (axis_z ** 2)
        
        return z_component < 0.3

    def compute_step_rewards(self) -> Tuple[float, Dict[int, bool]]:
        """
        Calculates the global collective cooperative reward pool.
        Encourages broad coverage, penalizes clustering, and rewards useful motion.
        """
        shared_reward = 0.0
        detections = {i: False for i in range(NUM_DRONES)}
        fire_pos = self.get_global_fire_pos()
        positions = [np.array(d.getPosition()) if d else np.array([0.,0.,0.]) for d in self.drones]

        # 1. Coverage / novelty reward allocation loop
        for i, my_pos in enumerate(positions):
            grid_x = int((my_pos[0] - FOREST_ORIGIN[0]) / GRID_RESOLUTION)
            grid_y = int((my_pos[1] - FOREST_ORIGIN[1]) / GRID_RESOLUTION)

            if 0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells:
                if self.coverage_grid[grid_x, grid_y] == 0:
                    self.coverage_grid[grid_x, grid_y] = 1
                    shared_reward += NEW_CELL_REWARD

                visit_count = int(self.visitation_grid[grid_x, grid_y])
                novelty_bonus = NOVELTY_REWARD_WEIGHT / (1.0 + visit_count)
                shared_reward += novelty_bonus
                self.visitation_grid[grid_x, grid_y] += 1

        # 1.5 Boundary constraint: penalize wandering too far from the forest area
        for i, my_pos in enumerate(positions):
            boundary = (FOREST_SIZE / 2.0) * 0.75
            distance_from_center = np.linalg.norm(my_pos[:2] - FOREST_CENTER)
            if distance_from_center > boundary:
                excess = distance_from_center - boundary
                shared_reward -= 0.5 * excess

        # 2. Team proximity penalties to discourage clustering
        for i in range(NUM_DRONES):
            for j in range(i + 1, NUM_DRONES):
                dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                if dist < NEIGHBOR_TARGET_DIST:
                    shared_reward -= NEIGHBOR_PENALTY_WEIGHT * (NEIGHBOR_TARGET_DIST - dist)

        # 3. Dense reward shaping for approaching the fire
        for i, my_pos in enumerate(positions):
            dist_to_fire = np.linalg.norm(my_pos[:2] - fire_pos[:2])
            prev_dist = self.last_fire_distances.get(i)
            prev_pos = self.prev_positions.get(i)
            grid_x = int((my_pos[0] - FOREST_ORIGIN[0]) / GRID_RESOLUTION)
            grid_y = int((my_pos[1] - FOREST_ORIGIN[1]) / GRID_RESOLUTION)

            if prev_dist is not None:
                dist_change = prev_dist - dist_to_fire
                if dist_change > 0:
                    shared_reward += 0.5 * dist_change
                else:
                    shared_reward -= 0.20 * abs(dist_change)

            if prev_pos is not None:
                travel = np.linalg.norm(my_pos[:2] - prev_pos[:2])
                if travel > 0.05:
                    useful_motion = False
                    if prev_dist is not None and (prev_dist - dist_to_fire) > 0.05:
                        useful_motion = True
                    if 0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells:
                        if self.coverage_grid[grid_x, grid_y] == 1 and self.visitation_grid[grid_x, grid_y] == 1:
                            useful_motion = True
                    if useful_motion:
                        shared_reward += VELOCITY_REWARD_WEIGHT * travel

            if dist_to_fire <= DETECTION_RADIUS:
                detections[i] = True
                if not self.has_detected_fire.get(i, False):
                    self.has_detected_fire[i] = True
                    shared_reward += 50.0

            self.last_fire_distances[i] = dist_to_fire

        # Update previous positions for the next step after reward evaluation
        self.prev_positions = {i: np.array(pos[:2], dtype=np.float32) for i, pos in enumerate(positions)}

        # Per-step cost: encourage efficient search without wasting time
        shared_reward -= 0.10

        return shared_reward, detections

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], float, bool]:
        """Executes selected model steps across all agents simultaneously in Webots."""
        self.step_count += 1
        
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
        
        # Check if any drone has flipped (inverted orientation)
        drone_flipped = False
        for i, drone in enumerate(self.drones):
            if self.is_drone_flipped(drone):
                if not self.flipped:
                    print(f"Drone {i} flipped! Episode terminated.")
                    reward -= 100.0  # Harsh penalty for flipping
                    self.flipped = True
                drone_flipped = True
                reward -= 100.0  # Harsh penalty for flipping
                break
        
        # Check termination state: Game over if all drones detect fire, max steps reached, OR any drone flipped
        done = all(detections.values()) or self.step_count >= MAX_EPISODE_STEPS or drone_flipped
        
        # Penalty if episode timeout occurs without detection
        if self.step_count >= MAX_EPISODE_STEPS and not all(detections.values()):
            reward -= 50.0
        
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