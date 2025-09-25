import pybullet as p
import time
import pybullet_data
import math
import os
import random
import numpy as np

class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array
        self.num_agents = num_agents
        mode = p.GUI if render else p.DIRECT

        print("Initializing PyBullet simulation")
        self.physicsClient = p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.urdf_dir = base_path
        self.textures_dir = os.path.join(base_path, "textures")
        self.plates_dir = os.path.join(self.textures_dir, "_plates")

# Prioritize textures from textures/ (not _plates), excluding "-Fish Eye Corrected.png"
        texture_list = [
            f for f in os.listdir(self.textures_dir)
            if f.endswith(".png") and not f.endswith("-Fish Eye Corrected.png")
        ]   

        if not texture_list:
         raise RuntimeError("No valid textures found in 'textures' folder!")

        random_texture = random.choice(texture_list)
        self.textureId = p.loadTexture(os.path.join(self.textures_dir, random_texture))
        print(f"Loaded texture: {random_texture}")


        self.baseplaneId = p.loadURDF("plane.urdf")

        self.robotIds = []
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1
        self.robotIds = []

        for i in range(num_agents):
            position = [i * spacing, 0, 0.03]
            robot_urdf = os.path.join(self.urdf_dir, "ot_2_simulation_v6.urdf")
            specimen_urdf = os.path.join(self.urdf_dir, "custom.urdf")

            if not os.path.exists(robot_urdf):
                raise FileNotFoundError(f"ERROR: '{robot_urdf}' not found!")
            if not os.path.exists(specimen_urdf):
                raise FileNotFoundError(f"ERROR: '{specimen_urdf}' not found!")

            robotId = p.loadURDF(robot_urdf, position, [0, 0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)
            p.createConstraint(self.baseplaneId, -1, robotId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position)

            specimenId = p.loadURDF(specimen_urdf, [position[0], position[1], position[2] + 0.05])
            p.setCollisionFilterPair(robotId, specimenId, -1, -1, enableCollision=0)
            p.changeVisualShape(specimenId, -1, textureUniqueId=self.textureId)

            self.robotIds.append(robotId)

        print(f" {num_agents} Robot(s) Loaded!")
    
    def set_start_position(self, x, y, z):
        if not self.robotIds:
            raise RuntimeError("No robot initialized to set position.")
        # Teleport the robot to the given position
        p.resetBasePositionAndOrientation(self.robotIds[0], [x, y, z], [0, 0, 0, 1])

    def reset(self):
        dummy_action = [[0, 0, 0, 0] for _ in range(len(self.robotIds))]
        return self.run(dummy_action)

    def run(self, actions, num_steps=1):
        if not self.robotIds:
            print(" ERROR: No robots initialized!")
            return {}

        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()
            if self.render:
                time.sleep(1 / 240)

        return self.get_states()

    def apply_actions(self, actions):
        for i, robotId in enumerate(self.robotIds):
            if len(actions) <= i:
                continue

            p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, targetVelocity=actions[i][0], force=50)
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=actions[i][1], force=50)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=100)

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            pipette_position = p.getLinkState(robotId, 2)[0]
            states[f'robotId_{robotId}'] = {"pipette_position": pipette_position}

        return states

    def get_pipette_position(self, robot_index):
        if robot_index >= len(self.robotIds):
            raise IndexError("Robot index out of range. Check if robots are initialized correctly.")
        return p.getLinkState(self.robotIds[robot_index], 2)[0]

    def get_rgb_array(self):
        if not self.rgb_array:
            return None

        width = 640
        height = 480

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.3, 0, 0.5],
            cameraTargetPosition=[0, 0, 0.2],
            cameraUpVector=[0, 0, 1]
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=2.0
        )

        _, _, px, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.reshape(px, (height, width, 4))[:, :, :3]  # Drop alpha
        return rgb_array

    def close(self):
        p.disconnect()
