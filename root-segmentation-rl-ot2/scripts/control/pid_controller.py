# Pid_controller

import numpy as np
import time
from sim_class import Simulation
from cv_target_detection import detect_dark_target_coordinates

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

def run_pid(target, gains, max_steps=300):
    sim = Simulation(num_agents=1, render=True)
    sim.reset()
    time.sleep(1)

    pid_x = PID(*gains['x'])
    pid_y = PID(*gains['y'])
    pid_z = PID(*gains['z'])

    robot_id = 0
    positions = []

    for step in range(max_steps):
        pos = np.array(sim.get_pipette_position(robot_id))
        positions.append(pos)

        error = target - pos
        if np.linalg.norm(error) <= 0.001:
            print(f"Target reached in {step} steps!")
            break

        vx = pid_x.compute(error[0])
        vy = pid_y.compute(error[1])
        vz = pid_z.compute(error[2])

        action = [vx, vy, vz, 0]
        sim.run([action])
        time.sleep(1 / 240)

    sim.close()
    return positions

if __name__ == "__main__":
    # Final target to validate accuracy
    target_pos = detect_dark_target_coordinates("Robotics_tasks/textures/03.png")

    # Final tuned gains for best accuracy
    gains = {
        'x': (3.5, 0.0, 0.4),
        'y': (3.5, 0.0, 0.4),
        'z': (6.0, 0.0, 0.6)
    }

    run_pid(target_pos, gains)
