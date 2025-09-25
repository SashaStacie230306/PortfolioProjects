import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sim_class import Simulation

# CONFIGURATION 
MODEL_PATH = "models/model_v2"
# Path to trained model
CSV_PATH = "coordinates.csv"          # CSV file with target coordinates
MAX_STEPS = 200
THRESHOLD = 0.001                     # 1 mm accuracy requirement
RENDER = True                         # Set to False to disable PyBullet GUI

# PLOTTING FUNCTION 
def plot_trajectory(x_vals, y_vals, z_vals, target):
    steps = list(range(len(x_vals)))
    fig = plt.figure(figsize=(12, 4))

    axes = [x_vals, y_vals, z_vals]
    labels = ['X', 'Y', 'Z']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(steps, axes[i], label=f'{labels[i]} Position', color='blue')
        plt.hlines(target[i], 0, len(x_vals), colors='green', linestyles='dashed', label='Target')
        plt.xlabel("Step")
        plt.ylabel("Position")
        plt.title(f"{labels[i]} Axis")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# LOAD COORDINATES
def load_coordinates(csv_path):
    coords = []
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return coords

# RUN EPISODE
def run_single_test(model, sim, target):
    observation = sim.reset()
    robot_index = 0 
    pipette_pos = np.array(sim.get_pipette_position(robot_index), dtype=np.float32)
    obs = np.append(pipette_pos, target).astype(np.float32)

    xs, ys, zs = [], [], []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action = np.append(action, 0)  # Add dummy drop_command
        observation = sim.run([action])
        pipette_pos = np.array(sim.get_pipette_position(robot_index), dtype=np.float32)
        obs = np.append(pipette_pos, target).astype(np.float32)

        xs.append(pipette_pos[0])
        ys.append(pipette_pos[1])
        zs.append(pipette_pos[2])

        dist = np.linalg.norm(pipette_pos - target)
        print(f"Step {step}: Position={pipette_pos}, Distance={dist:.5f}")

        if dist <= THRESHOLD:
            return True, step + 1, xs, ys, zs

    return False, MAX_STEPS, xs, ys, zs

# MAIN 
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP error on Windows

    print("Loading model...")
    model = PPO.load(MODEL_PATH)

    print("Loading coordinates...")
    targets = load_coordinates(CSV_PATH)

    successes = 0
    steps_list = []

    for i, target in enumerate(targets):
        print(f"\n Test {i + 1}: Target = {target}")
        sim = Simulation(num_agents=1, render=RENDER)

        success, steps, x_vals, y_vals, z_vals = run_single_test(model, sim, target)
        sim.close()

        plot_trajectory(x_vals, y_vals, z_vals, target)

        if success:
            print(f"Target reached in {steps} steps.")
            successes += 1
        else:
            print(f"Failed to reach target within {MAX_STEPS} steps.")

        steps_list.append(steps)

    print("\n EVALUATION SUMMARY")
    print(f"Total Successes: {successes}/{len(targets)}")
    print(f"Average Steps Taken: {np.mean(steps_list):.2f}")
    print("=======")

if __name__ == "__main__":
    main()
