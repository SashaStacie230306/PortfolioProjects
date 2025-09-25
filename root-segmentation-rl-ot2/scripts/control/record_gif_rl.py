import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import imageio
import csv
import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
from sim_class import Simulation

# CONFIG 
MODEL_PATH = "models/model_v1.1"
GIF_PATH = "week_8/simulation_run.gif"
GIF_DURATION = 4.0  # seconds
STEPS = 100  # number of steps to record
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

def record_gif(model_path, gif_path, steps=100):
    print("Initializing environment and loading model...")
    sim = Simulation(num_agents=1, render=False, rgb_array=True)
    model = PPO.load(model_path)

    # Reset environment
    sim.reset()
    sim.set_start_position(0, 0, 0.17)
    robot_id = 0

    with open("coordinates.csv", mode="r") as file:
        reader = csv.DictReader(file)
        first_row = next(reader)
        target_pos = np.array([
            float(first_row["x"]),
            float(first_row["y"]),
            float(first_row["z"])
        ], dtype=np.float32)
    
    current_pos = np.array(sim.get_pipette_position(0))
    observation = np.append(current_pos, target_pos).astype(np.float32)

    print("Recording frames...")
    frames = []

    for _ in range(steps):
        action, _ = model.predict(observation, deterministic=True)
        action = np.append(action, 0)
        sim.run([action])

        robot_id = 0
        current_pos = np.array(sim.get_pipette_position(robot_id))
        observation = np.append(current_pos, target_pos).astype(np.float32)

        frame = sim.get_rgb_array()
        if frame is not None:
            frames.append(frame)

    sim.close()

    print(f"Saving {len(frames)} frames to GIF...")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=GIF_DURATION / len(frames))
    print(f"GIF saved at: {gif_path}")

if __name__ == "__main__":
    record_gif(MODEL_PATH, GIF_PATH, STEPS)
