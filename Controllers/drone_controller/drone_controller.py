from controller import Robot, Motor, GPS, Gyro, Camera, InertialUnit
import numpy as np
import math
from typing import Dict, cast

# 1. Initialize Robot Context
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# 2. Set Up Motors
motors: Dict[str, Motor] = {
    'front_left': cast(Motor, robot.getDevice('front left propeller')),
    'front_right': cast(Motor, robot.getDevice('front right propeller')),
    'rear_left': cast(Motor, robot.getDevice('rear left propeller')),
    'rear_right': cast(Motor, robot.getDevice('rear right propeller'))
}

for motor in motors.values():
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# 3. Set Up Sensors
gps = cast(GPS, robot.getDevice('gps'))
gps.enable(timestep)

gyro = cast(Gyro, robot.getDevice('gyro'))
gyro.enable(timestep)

imu = cast(InertialUnit, robot.getDevice('inertial unit'))
imu.enable(timestep)

camera = cast(Camera, robot.getDevice('camera'))
camera.enable(4 * timestep)

# --- PID FLIGHT TUNING CONSTANTS ---
HOVER_SPEED = 68.5      # Baseline motor RPM required to counteract gravity

K_P_ALT = 8.0           # Altitude proportional gain
K_D_ALT = 4.0           # Vertical velocity braking (stops vertical bouncing)

K_P_POS = 0.2           # Horizontal positioning tracking speed
K_D_POS = 0.15          # NEW: Horizontal braking gain (stops the drone from overshooting)

K_P_ATT = 16.0          # Leveling sharpness (Attitude correction force)
K_D_ANG = 4.0           # Rotational angular velocity dampening (Stops the wobbling)
K_YAW_D = 3.0           # Anti-spin angular brake (Stops the spinning)

MAX_ANGLE = 0.4         # Max tilt pitch/roll constraint
STEP_SIZE = 0.04        # NEW: Distance (meters) the target shifts per timestep when moving

# Historical positions for velocity calculation
past_x = 0.0
past_y = 0.0
past_z = 0.12
target_pos = np.array([0.0, 0.0, 3.0]) # Internal tracking target initialization
first_frame = True

def get_rl_action() -> int:
    """
    Reads the discrete action integer injected into CustomData by the RL wrapper/Supervisor.
    Expected values: 0 to 6. Default to 0 (Hover) if empty or invalid.
    """
    custom_data = robot.getCustomData()
    if not custom_data:
        return 0
    try:
        return int(custom_data.strip())
    except ValueError:
        return 0

# --- MAIN CONTROL LOOP ---
while robot.step(timestep) != -1:
    # Read Sensors
    pos = np.array(gps.getValues())
    ang_vel = np.array(gyro.getValues())    # [roll_rate, pitch_rate, yaw_rate]
    rpy = np.array(imu.getRollPitchYaw())   # [Roll, Pitch, Yaw]

    # Guard Clause against Webots NaN Initialization Trap
    if np.isnan(pos).any() or np.isnan(ang_vel).any() or np.isnan(rpy).any():
        continue

    roll = rpy[0] 
    pitch = rpy[1]
    yaw = rpy[2]

    dt = timestep / 1000.0

    if first_frame:
        past_x = pos[0]
        past_y = pos[1]
        past_z = pos[2]
        # Snap target coordinates to current position, but at safe cruising altitude
        target_pos = np.array([pos[0], pos[1], 3.0]) 
        first_frame = False
        
    # Calculate velocities via sensor derivation
    vel_x = (pos[0] - past_x) / dt
    vel_y = (pos[1] - past_y) / dt
    vertical_vel = (pos[2] - past_z) / dt
    
    past_x, past_y, past_z = pos[0], pos[1], pos[2]
    
    # --- ACTION SPACE EXECUTION ---
    action = get_rl_action()
    
    if action == 1:    # Move Forward (North / +X)
        target_pos[0] += STEP_SIZE
    elif action == 2:  # Move Backward (South / -X)
        target_pos[0] -= STEP_SIZE
    elif action == 3:  # Move Left (West / +Y)
        target_pos[1] += STEP_SIZE
    elif action == 4:  # Move Right (East / -Y)
        target_pos[1] -= STEP_SIZE
    elif action == 5:  # Ascend (Up / +Z)
        target_pos[2] += STEP_SIZE
    elif action == 6:  # Descend (Down / -Z)
        target_pos[2] -= STEP_SIZE
    # Action 0 (Hover) falls through and leaves target_pos unchanged, forcing stabilization.
        
    # Calculate spatial deviations
    error = target_pos - pos  # [Error_X, Error_Y, Error_Z]
    
    # --- CASCADING CONTROL LOGIC ---
    
    # 1. Altitude Control (Thrust)
    thrust = HOVER_SPEED + (K_P_ALT * error[2]) - (K_D_ALT * vertical_vel)
    
    # 2. Position Control -> Map global errors and velocities into local body frame
    c = math.cos(yaw)
    s = math.sin(yaw)

    body_x = c * error[0] + s * error[1]
    body_y = -s * error[0] + c * error[1]
    
    body_vel_x = c * vel_x + s * vel_y
    body_vel_y = -s * vel_x + c * vel_y

    # FIXED: Added the D-term damping subtraction to force braking before arrival
    target_pitch = np.clip((K_P_POS * body_x) - (K_D_POS * body_vel_x), -MAX_ANGLE, MAX_ANGLE) 
    target_roll  = np.clip((-K_P_POS * body_y) + (K_D_POS * body_vel_y), -MAX_ANGLE, MAX_ANGLE) 
    
    # 3. Attitude & Gyro Dampening Control (Stops the violent wobbling)
    roll_input = K_P_ATT * (roll - target_roll) + K_D_ANG * ang_vel[0]
    pitch_input = K_P_ATT * (target_pitch - pitch) - K_D_ANG * ang_vel[1] 
    
    # 4. Anti-Spin Correction (Stops the spinning)
    yaw_input = K_YAW_D * ang_vel[2] 
    
    # 5. Fixed Mixer Matrix for Webots Mavic 2 Pro Structure
    fl = thrust - roll_input - pitch_input + yaw_input
    fr = thrust + roll_input - pitch_input - yaw_input
    rl = thrust - roll_input + pitch_input - yaw_input
    rr = thrust + roll_input + pitch_input + yaw_input
    
    # 6. Apply Motor Limits & Handle Reverse Thrust Properties for Counter-Rotators
    motors['front_left'].setVelocity(np.clip(fl, 0.0, 95.0))
    motors['front_right'].setVelocity(np.clip(-fr, -95.0, 0.0))
    motors['rear_left'].setVelocity(np.clip(-rl, -95.0, 0.0))
    motors['rear_right'].setVelocity(np.clip(rr, 0.0, 95.0))