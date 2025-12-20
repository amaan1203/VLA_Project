import gymnasium as gym
import panda_mujoco_gym  
import numpy as np
import time


env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="human")

#seed=12
obs, info = env.reset()

def get_ee_pos(obs):
    return obs["observation"][:3]

def get_cube_pos(obs):
    return obs["achieved_goal"]

def get_goal_pos(obs):
    return obs["desired_goal"]

def clip_action(action, max_delta=0.05):
    return np.clip(action, -max_delta, max_delta)


def perform_task(target_pos, destination_pos, is_stacking=False , is_ball=False):
    global obs
    
    # STAGE 1 & 2: Alignment & Hover
    print(f"Targeting position: {target_pos}")
    for step in range(1000):
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        action = np.append(delta * 10.0, 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        if np.linalg.norm(delta[:2]) < 0.01: break

    # STAGE 3: Precise Deep Descent
    print("Descending...")
    for step in range(500):
        ee_pos = get_ee_pos(obs)
        target_z = target_pos[2] - 0.005 
        action = np.array([(target_pos[0]-ee_pos[0])*15, (target_pos[1]-ee_pos[1])*15, (target_z-ee_pos[2])*5, 1.0])
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.2))
        if abs(target_z - ee_pos[2]) < 0.005: break

    # STAGE 4: Grasping
    print("Grasping...")
    for _ in range(60):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        
    if is_ball: 
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))

    # STAGE 5 & 6: Lift & Move to Destination
    print("Moving to destination...")
    for step in range(1200):
        ee_pos = get_ee_pos(obs)
        # If stacking, we target a bit higher than the destination
        offset_z = 0.08 if is_stacking else 0.05
        target_dest = destination_pos + np.array([0, 0, offset_z])
        delta = target_dest - ee_pos
        action = np.append(delta * 10.0, -1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        if np.linalg.norm(delta) < 0.02: break

    # STAGE 7: Release & Retract
    print("Releasing...")
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))



# executing the logic #
cube_starting_pos = get_cube_pos(obs)


target_goal_pos = get_goal_pos(obs)
final_destination = np.array([target_goal_pos[0], target_goal_pos[1], 0.02]) 


print("\n trying to move the cube to the location of ball ")

perform_task(cube_starting_pos, final_destination, is_stacking=False)
env.close()
