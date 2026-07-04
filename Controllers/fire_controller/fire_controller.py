from controller import Supervisor
import random

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
print("Supervisor started")

fire = robot.getFromDef("FIRE_0")
print("Fire node:", fire)
if fire is None:
    raise RuntimeError("DEF FIRE_0 not found.")

translation = fire.getField("translation")

# Preassigned fire locations
LOCATIONS = (
    (-4.05, 0.1, 4.0),
    (-4.05, 18.68, 5.93),
    (20.05, 0.05, 3.11995),
)

# Set the fire position once
robot.step(timestep)
location = list(random.choice(LOCATIONS))
print(location)

translation.setSFVec3f(location)
print("Current:", translation.getSFVec3f())
translation.setSFVec3f(location)
robot.step(timestep)

# Keep the Supervisor alive
while robot.step(timestep) != -1:
    pass