import gymnasium as gym
import panda_mujoco_gym
import numpy as np
import mujoco
import cv2
import time
from arm_state_logger import ArmStateLogger 
import os
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

os.environ["MUJOCO_GL"] = "egl"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="rgb_array")

mj_env: MujocoEnv = env.unwrapped # type: ignore
model = mj_env.model
data = mj_env.data

video_out = cv2.VideoWriter('robot_dashboard.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 720)) # type: ignore

main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)

print(f"Using MuJoCo GL Backend: {os.environ['MUJOCO_GL']}")

DRIFT_HISTORY = []
START_REAL_TIME = time.time()
BASE_REAL_TIME = time.time()

# Fixed initialization
total_step_t0 = time.perf_counter()

obs, info = env.reset()

def get_ee_pos(obs):
    return obs["observation"][:3]

def get_cube_pos(obs):
    return obs["achieved_goal"]

def get_goal_pos(obs):
    return obs["desired_goal"]

def clip_action(action, max_delta=0.05):
    return np.clip(action, -max_delta, max_delta)

def render_dashboard(obs, data):
    global BASE_REAL_TIME, START_REAL_TIME, DRIFT_HISTORY, total_step_t0
    
    sim_time = data.time
    current_real_time = time.time() - START_REAL_TIME
    current_drift = (current_real_time - sim_time) * 1000 
    DRIFT_HISTORY.append(current_drift)
    
    relative_real = time.time() - BASE_REAL_TIME
    sync_offset = (relative_real - sim_time) * 1000

    main_renderer.update_scene(data, camera=-1) 
    t_render_start = time.perf_counter()
    main_rgb = main_renderer.render() 
    render_latency = (time.perf_counter() - t_render_start) * 1000
    
    main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
    
    ee_pos = get_ee_pos(obs)
    fingers = mj_env.get_fingers_width() # type: ignore
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

    cam_renderer.update_scene(data, camera="gripper_front_chase")
    gripper_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)

    border_color = (0, 255, 255) # Yellow in BGR
    cv2.rectangle(gripper_bgr, (0, 0), (320, 240), border_color, 10)
    cv2.putText(gripper_bgr, "GRIPPER VIEW", (10, 30), font, 0.7, border_color, 2)
    bottom_row = np.hstack((front_bgr, gripper_bgr))
    dashboard = np.vstack((main_bgr, bottom_row))
    
    video_out.write(dashboard)

    # Correct latency calculation
    total_latency = (time.perf_counter() - total_step_t0) * 1000
    print(f"Render: {render_latency:5.2f}ms | Step: {total_latency:6.2f}ms | Sync Offset: {sync_offset:7.2f}ms")
    
    return sync_offset

from arm_state_logger import ArmStateLogger
arm_logger = ArmStateLogger()

def perform_task(target_pos, destination_pos, is_stacking=False, is_ball=False):
    global obs, total_step_t0
    
    TARGET_STEP_TIME = 0.01 # 10ms target per step

    def apply_throttle():
        """Helper to apply dynamic sleep based on accumulated drift."""
        current_sync = render_dashboard(obs, data)
        
        # 1. Calculate how much time this loop iteration has actually taken so far
        loop_work_time = time.perf_counter() - total_step_t0
        
        # 2. Base sleep is what's left of our 10ms target
        sleep_time = TARGET_STEP_TIME - loop_work_time
        
        # 3. If we have a negative offset (we are ahead), add a correction
        # We use a 'proportional' correction (e.g., 0.5) so we don't over-correct
        if current_sync < 0:
            correction = (abs(current_sync) / 1000.0) * 0.5
            sleep_time += correction 

        # 4. Final check: Only sleep if there is time to kill
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 1. Targeting Position
    print(f"targeting position: {target_pos}")
    for step in range(1000):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        action = np.append(delta * 10.0, 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) 
        apply_throttle()
        if np.linalg.norm(delta[:2]) < 0.01: break

    # 2. Descending
    print("Descending...")
    for step in range(500):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_z = target_pos[2] - 0.005 
        action = np.array([(target_pos[0]-ee_pos[0])*15, (target_pos[1]-ee_pos[1])*15, (target_z-ee_pos[2])*5, 1.0])
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.2))
        
        arm_logger.log_state(obs, data) 
        apply_throttle()
        if abs(target_z - ee_pos[2]) < 0.005: break

    # 3. Grasping
    print("Grasping...")
    for _ in range(60):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        arm_logger.log_state(obs, data)
        apply_throttle()
        
    if is_ball: 
        for _ in range(20):
            total_step_t0 = time.perf_counter()
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
            arm_logger.log_state(obs, data) 
            apply_throttle()

# --- PHASE 4: TRAVEL TO DESTINATION (STAY HIGH) ---
    print("Moving to destination hover...")
    for step in range(1200):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        
        # Keep the Z high (0.15) while moving horizontally to the goal
        target_dest_hover = np.array([destination_pos[0], destination_pos[1], 0.15])
        delta = target_dest_hover - ee_pos
        
        action = np.append(delta * 10.0, -1.0) # Keep gripper closed
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.3))
        
        arm_logger.log_state(obs, data) 
        apply_throttle()
        # Only break once we are centered over the goal horizontally
        if np.linalg.norm(delta[:2]) < 0.01: break

    # --- NEW PHASE 5: CONTROLLED DESCENT (THE FIX) ---
    print("Lowering to floor plane...")
    for step in range(300):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        
        # Target the floor surface + half the cube height (approx 0.02)
        target_floor = np.array([destination_pos[0], destination_pos[1], 0.02]) 
        
        delta = target_floor - ee_pos
        action = np.append(delta * 5.0, -1.0) # Still closed!
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.05))
        
        arm_logger.log_state(obs, data)
        apply_throttle()
        # Break once we are within 2mm of the floor target
        if abs(ee_pos[2] - 0.02) < 0.002: break

    # --- PHASE 6: RELEASE AND LIFT ---
    print("Releasing on floor...")
    for _ in range(50):
        total_step_t0 = time.perf_counter()
        # Send 1.0 to open gripper
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle()

    print("Clearing area...")
    for _ in range(30):
        total_step_t0 = time.perf_counter()
        # Lift up 20cm to finish cleanly
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle()

try:
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0]))

    cube_starting_pos = get_cube_pos(obs)
    site_id = model.site('target').id
    target_site_pos = data.site_xpos[site_id].copy()
    target_goal_pos = get_goal_pos(obs)
    final_destination = np.array([target_site_pos[0], target_site_pos[1], 0.0]) 

    print(f"\nDetecting Object at: {cube_starting_pos}")
    print(f"Target Destination: {final_destination}")

    # 4. Execute the task
    perform_task(cube_starting_pos, final_destination, is_stacking=False)
    print("Task Completed.")

finally:
    env.close()
    video_out.release()
    print("Video saved as robot_dashboard.mp4")