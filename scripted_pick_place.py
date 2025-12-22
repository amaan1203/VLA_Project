import gymnasium as gym
import panda_mujoco_gym
import numpy as np
import mujoco
import cv2
import time
from arm_state_logger import ArmStateLogger 


env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="human")

mj_env = env.unwrapped
model = mj_env.model
data = mj_env.data

main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)

arm_logger = ArmStateLogger()

obs, info = env.reset()


def get_ee_pos(obs):
    return obs["observation"][:3]

def get_cube_pos(obs):
    return obs["achieved_goal"]

def get_goal_pos(obs):
    return obs["desired_goal"]

def clip_action(action, max_delta=0.05):
    return np.clip(action, -max_delta, max_delta)


main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)


def render_dashboard():
    # 1. Main View (Free Camera / 'Human' Mode style)
    # Using the variable 'main_renderer' defined above
    main_renderer.update_scene(data, camera=-1) 
    main_rgb = main_renderer.render() 
    main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)

    # 2. Front Camera
    # Using the variable 'cam_renderer' defined above
    cam_renderer.update_scene(data, camera="front_cam")
    front_rgb = cam_renderer.render()
    front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)

    # 3. Gripper Camera
    cam_renderer.update_scene(data, camera="gripper_cam")
    gripper_rgb = cam_renderer.render()
    gripper_bgr = cv2.cvtColor(gripper_rgb, cv2.COLOR_RGB2BGR)
    
    # --- Layout Logic ---
    bottom_row = np.hstack((front_bgr, gripper_bgr))
    dashboard = np.vstack((main_bgr, bottom_row))

    cv2.imshow("Robot Control Dashboard", dashboard)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True


from arm_state_logger import ArmStateLogger

# Initialize globally
arm_logger = ArmStateLogger()

def perform_task(target_pos, destination_pos, is_stacking=False, is_ball=False):
    global obs
    
    # STAGE 1 & 2: Alignment & Hover
    print(f"Targeting position: {target_pos}")
    for step in range(1000):
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        action = np.append(delta * 10.0, 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01) 
        if np.linalg.norm(delta[:2]) < 0.01: break

    # STAGE 3: Precise Deep Descent
    print("Descending...")
    for step in range(500):
        ee_pos = get_ee_pos(obs)
        target_z = target_pos[2] - 0.005 
        action = np.array([(target_pos[0]-ee_pos[0])*15, (target_pos[1]-ee_pos[1])*15, (target_z-ee_pos[2])*5, 1.0])
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.2))
        
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01) 
        if abs(target_z - ee_pos[2]) < 0.005: break

    # STAGE 4: Grasping
    print("Grasping...")
    for _ in range(60):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01) 
        
    if is_ball: 
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
            arm_logger.log_state(obs, data) # Added Logger
            render_dashboard()
            time.sleep(0.01)

    print("Moving to destination...")
    for step in range(1200):
        ee_pos = get_ee_pos(obs)
        offset_z = 0.08 if is_stacking else 0.05
        target_dest = destination_pos + np.array([0, 0, offset_z])
        delta = target_dest - ee_pos
        action = np.append(delta * 10.0, -1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01) 
        if np.linalg.norm(delta) < 0.02: break

    # STAGE 7: Release & Retract
    print("Releasing...")
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01) 
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))
        arm_logger.log_state(obs, data) # Added Logger
        render_dashboard()
        time.sleep(0.01)# This forces the CPU to wait, stabilizing the physics

try:
    
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0]))

    cube_starting_pos = get_cube_pos(obs)
    target_goal_pos = get_goal_pos(obs)
    
   
    final_destination = np.array([target_goal_pos[0], target_goal_pos[1], 0.02]) 

    print("\nStarting unified task...")
    perform_task(cube_starting_pos, final_destination, is_stacking=False)
    print("Done.")

finally:
    env.close()
    cv2.destroyAllWindows()