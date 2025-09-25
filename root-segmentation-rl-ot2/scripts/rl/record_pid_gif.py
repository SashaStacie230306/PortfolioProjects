# Robotics_tasks/pid_record_gif.py

import os
import time
import imageio
import numpy as np
from sim_class import Simulation
from cv_target_detection import detect_dark_target_coordinates  # Import vision pipeline

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt=1/240):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def run_pid_gif(target, gains, max_steps=300, gif_path="week_8/pid_controller_run.gif"):
    sim = Simulation(num_agents=1, render=False, rgb_array=True)
    sim.reset()
    time.sleep(1)

    pid_x = PID(*gains['x'])
    pid_y = PID(*gains['y'])
    pid_z = PID(*gains['z'])

    robot_id = 0
    frames = []

    for step in range(max_steps):
        pos = np.array(sim.get_pipette_position(robot_id))
        error = target - pos
        if np.linalg.norm(error) <= 0.001:
            print(f"Target reached in {step} steps!")
            break

        vx = pid_x.compute(error[0])
        vy = pid_y.compute(error[1])
        vz = pid_z.compute(error[2])

        sim.run([[vx, vy, vz, 0]])
        frame = sim.get_rgb_array()
        if frame is not None:
            frames.append(frame)

    sim.close()

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=4 / len(frames))
    print(f"GIF saved at: {gif_path}")

if __name__ == "__main__":
    # Automatically detect target from image
    target_pos = detect_dark_target_coordinates("Robotics_tasks/textures/03.png")

    # Tuned PID gains
    gains = {
        'x': (3.5, 0.0, 0.4),
        'y': (3.5, 0.0, 0.4),
        'z': (6.0, 0.0, 0.6)
    }

    run_pid_gif(target_pos, gains)
