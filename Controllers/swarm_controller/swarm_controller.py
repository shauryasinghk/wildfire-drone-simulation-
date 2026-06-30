from controller import Supervisor
import math

NUM_DRONES = 4  # Change this to 2, 4, 9, etc.
SPACING = 3.0
TARGET_HEIGHT = 3.0

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# FIXED: Fetch the dedicated Keyboard device instance and enable it
keyboard = supervisor.getKeyboard()
keyboard.enable(timestep)

drones = []

root = supervisor.getRoot()
children_field = root.getField('children')

# --- DYNAMIC SWARM SPAWNING BLOCKS ---
for i in range(NUM_DRONES):
    if i == 0:
        drone = supervisor.getFromDef('drone_0')
        drones.append(drone)
        continue

    wait_steps = max(1, int(10 / timestep))  # 100 ms spacing buffer
    for _ in range(wait_steps):
        supervisor.step(timestep)

    x = i * SPACING
    y = 0.0
    z = 0.12

    drone_string = (
        f'Mavic2Pro {{ '
        f'name "drone_{i}" '
        f'translation {x} {y} {z} '
        f'controller "drone_controller" '
        f'}}'
    )

    children_field.importMFNodeFromString(-1, drone_string)
    drone = supervisor.getFromDef(f'drone_{i}')
    drones.append(drone)

# Default configuration state at startup
current_mode = 1
print("Swarm Init Completed. Click 3D Window and press '1', '2', or '3' to change formations!")

# --- MAIN SUPERVISOR RUNTIME LOOP ---
while supervisor.step(timestep) != -1:
    # FIXED: Capture user keyboard strokes from the keyboard device instance
    key = keyboard.getKey()
    
    if key == ord('1'):
        current_mode = 1
        print("Swarm Command: Mode 1 -> Broad Grid Line Search Formation")
    elif key == ord('2'):
        current_mode = 2
        print("Swarm Command: Mode 2 -> Fire Encirclement Perimeter Ring")
    elif key == ord('3'):
        current_mode = 3
        print("Swarm Command: Mode 3 -> Low Altitude Safe Cluster/Rally")

    # 2. Process positions and send discrete commands to individual drones
    for i, drone in enumerate(drones):
        if drone is None:
            continue
        
        # Determine the unique target $(x, y, z)$ position for this drone based on selected mode
        if current_mode == 1:
            # Formation 1: Evenly spaced row layout along the Y-axis
            target_x = 0.0
            target_y = (i - (NUM_DRONES - 1) / 2) * SPACING
            target_z = TARGET_HEIGHT
            
        elif current_mode == 2:
            # Formation 2: Radial circle pattern around origin center
            angle = (2 * math.pi * i) / NUM_DRONES
            radius = 6.0
            target_x = radius * math.cos(angle)
            target_y = radius * math.sin(angle)
            target_z = TARGET_HEIGHT
            
        else:
            # Formation 3: Tight low-altitude hover collection point
            angle = (2 * math.pi * i) / NUM_DRONES
            radius = 1.5
            target_x = radius * math.cos(angle)
            target_y = radius * math.sin(angle)
            target_z = 1.5

        # 3. Get actual drone position via the supervisor spatial tracking nodes
        current_pos = drone.getPosition() # Returns [X, Y, Z] global coordinates
        
        err_x = target_x - current_pos[0]
        err_y = target_y - current_pos[1]
        err_z = target_z - current_pos[2]
        
        # 4. Convert structural errors into discrete actions (0-6) matching the drone's input map
        # Tolerance margin prevents continuous jittering once inside the destination boundary
        TOLERANCE = 0.3 
        action = 0  # Default: Hover / Keep Position
        
        if abs(err_x) > TOLERANCE:
            action = 1 if err_x > 0 else 2  # 1: Move Forward (+X), 2: Move Backward (-X)
        elif abs(err_y) > TOLERANCE:
            action = 3 if err_y > 0 else 4  # 3: Move Left (+Y), 4: Move Right (-Y)
        elif abs(err_z) > TOLERANCE:
            action = 5 if err_z > 0 else 6  # 5: Ascend (+Z), 6: Descend (-Z)
        
        # 5. Push the single integer action instruction down to the customData string pipeline
        pos_field = drone.getField('customData')
        if pos_field:
            pos_field.setSFString(str(action))