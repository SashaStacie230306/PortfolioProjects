from sim_class import Simulation
import numpy as np
import time

# Initialize the simulation
sim = Simulation(num_agents=1)
print("Simulation initialized. Determining working envelope...")

# Move the pipette and record its position
def move_pipette_and_get_position(velocity_x, velocity_y, velocity_z, num_steps=200):
    actions = [[velocity_x, velocity_y, velocity_z, 0]]  # [velocity_x, velocity_y, velocity_z, drop_command]
    state = sim.run(actions, num_steps=num_steps)
    if 'robotId_1' in state and 'pipette_position' in state['robotId_1']:
        position = state['robotId_1']['pipette_position']
        print(f"Velocity: ({velocity_x}, {velocity_y}, {velocity_z}) -> Position: {position}")
        return position
    else:
        raise ValueError("Pipette position data is missing in the state.")

# Define velocity ranges for each axis
velocity_limits = [-0.5, 0.5] 
working_envelope = []

# Move the pipette to each corner of the cube
print("\nMoving pipette to each corner of the cube...\n")
for vx in velocity_limits:
    for vy in velocity_limits:
        for vz in velocity_limits:
            print(f"➡ Moving pipette to (vx={vx}, vy={vy}, vz={vz})...")
            position = move_pipette_and_get_position(vx, vy, vz)
            if position:
                working_envelope.append(position)
                print(f"Corner recorded: {position}")
            else:
                print(f" Error: No position recorded for (vx={vx}, vy={vy}, vz={vz})")

# Verify the working envelope contains exactly 8 unique points
unique_points = set(tuple(point) for point in working_envelope)
print("\n Unique Corner Points Logged:")
for i, point in enumerate(unique_points, start=1):
    print(f" Corner {i}: {point}")

if len(unique_points) == 8:
    print("\n The working envelope contains exactly **8 distinct points.")
else:
    print(f"\n Error: The working envelope has {len(unique_points)} unique points. Expected 8.")

# Define expected bounds for each axis
expected_bounds = {
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (-0.5, 0.5)
}

# Check if each point is within bounds
print("\nVerifying if all points are within bounds...")
for i, point in enumerate(working_envelope, start=1):
    x, y, z = point
    if (expected_bounds['x'][0] <= x <= expected_bounds['x'][1] and
        expected_bounds['y'][0] <= y <= expected_bounds['y'][1] and
        expected_bounds['z'][0] <= z <= expected_bounds['z'][1]):
        print(f"Point {i} is within bounds: {point}")
    else:
        print(f"Error: Point {i} is out of bounds: {point}")
 
# Print the working envelope (8 points)
print("\nWorking Envelope (Pipette Coordinates):")
for i, point in enumerate(working_envelope, start=1):
    print(f"Corner {i}: {point}")

# End the simulation
print("Done! Simulation reset.")