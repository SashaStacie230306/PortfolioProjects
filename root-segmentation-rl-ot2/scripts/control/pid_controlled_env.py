import numpy as np
import time
from ot2_gym_wrapper import OT2Env
from pid_controller import PIDController

class PIDControlledEnv:
    def __init__(self):
        self.env = OT2Env(render=True, max_steps=1000, success_threshold=0.001)
        self.goal = None
        self.dt = 0.05

        self.pid_x = PIDController(kp=20.0, ki=0.1, kd=0.005)
        self.pid_y = PIDController(kp=20.0, ki=0.1, kd=0.005)
        self.pid_z = PIDController(kp=20.0, ki=0.1, kd=0.005)

    def run(self):
        obs, info = self.env.reset()
        self.goal = obs[3:]

        for step in range(self.env.max_steps):
            pos = obs[:3]
            error = self.goal - pos

            vx = self.pid_x.compute(error[0], self.dt)
            vy = self.pid_y.compute(error[1], self.dt)
            vz = self.pid_z.compute(error[2], self.dt)

            action = np.clip([vx, vy, vz], self.env.action_space.low, self.env.action_space.high)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            time.sleep(self.dt)

            if np.linalg.norm(error) < 0.001:
                print(f"Goal reached at step {step}")
                break

        self.env.close()
