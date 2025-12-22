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

DRIFT_HISTORY = []
START_REAL_TIME = time.time()

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

BASE_REAL_TIME = time.time()

def render_dashboard(obs, data):
    
    global BASE_REAL_TIME, START_REAL_TIME, DRIFT_HISTORY
    
    sim_time = data.time
    current_real_time = time.time() - START_REAL_TIME
    current_drift = (current_real_time - sim_time) * 1000 
    DRIFT_HISTORY.append(current_drift)
    
    relative_real = time.time() - BASE_REAL_TIME
    sync_offset = (relative_real - sim_time) * 1000

    
    main_renderer.update_scene(data, camera=-1) 
    t0=time.time()
    main_rgb = main_renderer.render() 
    latency = (time.time() - t0) * 1000
    print(f"Frame Latency: {latency:.2f}ms")
    main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
    
    
    ee_pos = get_ee_pos(obs)
    fingers = mj_env.get_fingers_width()
    f_val = fingers if np.isscalar(fingers) else fingers[0]

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thickness = 2
    
    
    drift_color = (0, 0, 255) if (len(DRIFT_HISTORY) > 1 and current_drift > DRIFT_HISTORY[-2]) else (0, 255, 0)
    sync_color = (0, 255, 0) if sync_offset < 100 else (0, 0, 255)
    text_color = (255, 255, 255) 


    cv2.putText(main_bgr, f"TIME: {sim_time:.2f}s", (15, 30), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_X: {ee_pos[0]:.3f}", (15, 60), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_Y: {ee_pos[1]:.3f}", (15, 90), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_Z: {ee_pos[2]:.3f}", (15, 120), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"GRIP: {f_val:.4f}", (15, 150), font, scale, text_color, thickness)


    cv2.putText(main_bgr, f"DRIFT: {current_drift:.1f}ms", (350, 30), font, scale, drift_color, thickness)
    cv2.putText(main_bgr, f"MAX LAG: {max(DRIFT_HISTORY):.1f}ms", (350, 60), font, scale, (0, 255, 255), thickness)
    cv2.putText(main_bgr, f"SYNC OFFSET: {sync_offset:.1f}ms", (350, 90), font, scale, sync_color, thickness)
    cv2.putText(main_bgr, f"REAL CLOCK: {current_real_time:.2f}s", (350, 120), font, scale, (200, 200, 200), thickness)

   
    cam_renderer.update_scene(data, camera="front_cam")
    front_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)

    cam_renderer.update_scene(data, camera="gripper_cam")
    gripper_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)
    

    bottom_row = np.hstack((front_bgr, gripper_bgr))
    dashboard = np.vstack((main_bgr, bottom_row))

    cv2.imshow("Robot Control Dashboard", dashboard)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True


from arm_state_logger import ArmStateLogger


arm_logger = ArmStateLogger()

def perform_task(target_pos, destination_pos, is_stacking=False, is_ball=False):
    global obs
    
    
    print(f"targeting position: {target_pos}")
    for step in range(1000):
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        action = np.append(delta * 10.0, 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) 
        render_dashboard(obs, data)
        time.sleep(0.01) 
        if np.linalg.norm(delta[:2]) < 0.01: break

    
    print("Descending...")
    for step in range(500):
        ee_pos = get_ee_pos(obs)
        target_z = target_pos[2] - 0.005 
        action = np.array([(target_pos[0]-ee_pos[0])*15, (target_pos[1]-ee_pos[1])*15, (target_z-ee_pos[2])*5, 1.0])
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.2))
        
        arm_logger.log_state(obs, data) 
        render_dashboard(obs, data)
        time.sleep(0.01) 
        if abs(target_z - ee_pos[2]) < 0.005: break

   
    print("Grasping...")
    for _ in range(60):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        
        arm_logger.log_state(obs, data)
        render_dashboard(obs, data)
        time.sleep(0.01) 
        
    if is_ball: 
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
            arm_logger.log_state(obs, data) 
            render_dashboard(obs, data)
            time.sleep(0.01)

    print("Moving to destination...")
    for step in range(1200):
        ee_pos = get_ee_pos(obs)
        offset_z = 0.08 if is_stacking else 0.05
        target_dest = destination_pos + np.array([0, 0, offset_z])
        delta = target_dest - ee_pos
        action = np.append(delta * 10.0, -1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) 
        render_dashboard(obs, data)
        time.sleep(0.01) 
        if np.linalg.norm(delta) < 0.02: break

    
    print("Releasing...")
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
        arm_logger.log_state(obs, data) 
        render_dashboard(obs, data)
        time.sleep(0.01) 
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))
        arm_logger.log_state(obs, data) 
        render_dashboard(obs, data)
        time.sleep(0.01)

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