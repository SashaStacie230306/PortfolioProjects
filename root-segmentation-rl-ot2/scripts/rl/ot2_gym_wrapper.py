import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Define workspace boundaries
top_corner = [0.258, 0.2195, 0.344]
bottom_corner = [-0.217, -0.233, 0.297]

NO_DROP = 0  # Drop command fixed at 0 for now

def generate_random_goal(corner_a, corner_b):
    return [
        random.uniform(min(corner_a[0], corner_b[0]), max(corner_a[0], corner_b[0])),
        random.uniform(min(corner_a[1], corner_b[1]), max(corner_a[1], corner_b[1])),
        random.uniform(min(corner_a[2], corner_b[2]), max(corner_a[2], corner_b[2]))
    ]

class PipetteBotGym(gym.Env):
    def __init__(self, show=False, max_iters=1500):
        super().__init__()
        self.visuals = show
        self.max_iters = max_iters
        self.sim_world = Simulation(num_agents=1, render=self.visuals)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.threshold = 0.001
        self.agent_index = 0
    
    def reset(self, seed=None, options=None):
     super().reset(seed=seed)
     if seed is not None:
        np.random.seed(seed)

    # Reset simulation state
     start_state = self.sim_world.reset()

     robot_keys = list(start_state.keys())
     if not robot_keys:
        raise RuntimeError("No robot data found in simulation state. Is the simulation initialized correctly?")

     self.agent_index = self.sim_world.robotIds.index(int(robot_keys[0].split("_")[1]))

     self.goal = generate_random_goal(top_corner, bottom_corner)
     pipette_coords = np.array(self.sim_world.get_pipette_position(self.agent_index), dtype=np.float32)
     state = np.concatenate([pipette_coords, self.goal]).astype(np.float32)

     self.previous_distance = np.linalg.norm(pipette_coords - self.goal)
     self.iter_count = 0

     self.flag_reached = False
     self.hold_start = None
     self.reward_cache = 0

     return state, {}


    def step(self, action):
        done = False
        out_of_time = False

        action = np.clip(action + np.random.normal(0, 0.2, size=action.shape), -1, 1)
        full_action = np.append(action, NO_DROP)

        sim_state = self.sim_world.run([full_action])
        current_pos = np.array(self.sim_world.get_pipette_position(self.agent_index), dtype=np.float32)
        state = np.concatenate([current_pos, self.goal]).astype(np.float32)

        distance_now = np.linalg.norm(current_pos - self.goal)
        base_reward = (self.previous_distance - distance_now) * 10
        self.previous_distance = distance_now

        reward = base_reward + (-0.1 + 0.05 * distance_now)

        if distance_now < self.threshold:
            if not self.flag_reached:
                self.hold_start = self.iter_count
                self.flag_reached = True

            if self.iter_count - self.hold_start >= self.stopping_delay:
                if self.consecutive_hits < 100:
                    self.consecutive_hits += 1
                self.stopping_delay = max((self.consecutive_hits // 10) - 1, 0)
                self.reward_cache = self.stopping_delay * 100 + (0.1 ** 3) * ((self.max_iters - self.iter_count) ** 2)
                reward += self.reward_cache
                done = True
            else:
                reward += 40 * ((self.stopping_delay - 1) ** 2)

        if self.iter_count + 1 >= self.max_iters:
            reward -= 50
            out_of_time = True

        self.iter_count += 1

        info = {
            'distance_to_goal': distance_now,
            'debug': {
                'raw_reward': base_reward,
                'stop_bonus': self.reward_cache,
                'step_count': self.iter_count
            }
        }

        return state, float(reward), done, out_of_time, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.sim_world.close()

# Alias for compatibility with other scripts
OT2Env = PipetteBotGym
