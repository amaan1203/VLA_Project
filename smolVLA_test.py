import gymnasium as gym
import panda_mujoco_gym
import numpy as np
import mujoco
import cv2
import time
import os
import torch
import threading
from PIL import Image
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from arm_state_logger import ArmStateLogger


os.environ["MUJOCO_GL"] = "egl"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="rgb_array")

arm_logger = ArmStateLogger()

mj_env: MujocoEnv = env.unwrapped 
model = mj_env.model
data = mj_env.data

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID = "lerobot/smolvla_base"

mj_env = env.unwrapped 
model, data = mj_env.model, mj_env.data

video_out = cv2.VideoWriter('vla_robot_dashboard.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 720))

main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)

print(f"Using MuJoCo GL Backend: {os.environ['MUJOCO_GL']}")

DRIFT_HISTORY = []

total_step_t0 = time.perf_counter()
obs, info = env.reset()

START_REAL_TIME = time.time()
BASE_REAL_TIME = START_REAL_TIME

def get_ee_pos(obs):
    return obs["observation"][:3]

def get_cube_pos(obs):
    return obs["achieved_goal"]

def get_goal_pos(obs):
    return obs["desired_goal"]

def clip_action(action, max_delta):
    action = action.copy()
    action[:3] = np.clip(action[:3], -max_delta, max_delta)
    return action



def render_dashboard(data, current_vla_action):
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
    
    cv2.putText(main_bgr, f"TIME: {sim_time:.2f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(main_bgr, f"DEVICE: {DEVICE}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if current_vla_action is not None:
         cv2.putText(main_bgr, f"VLA ACT: {np.round(current_vla_action[:3], 2)}", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    
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
    total_latency = (time.perf_counter() - total_step_t0) * 1000
    print(f"Render: {render_latency:5.2f}ms | Step: {total_latency:6.2f}ms | Sync Offset: {sync_offset:7.2f}ms")
    
    return main_rgb, sync_offset

class VLAPID:
    def __init__(self, model_id, device):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
        self.policy.eval()

        instruction = "Pick up the green block and move it to the target."
        tokens = self.policy.model.vlm_with_expert.processor.tokenizer(
            instruction, return_tensors="pt"
        ).to(device)

        self.lang_tokens = tokens["input_ids"]
        self.lang_mask = tokens["attention_mask"].bool()

    @torch.inference_mode()
    def residual(self, image, state):
        """
        image: HWC uint8
        state: (32,) float32   ← real env state
        returns: (3,) residual xyz delta
        """

        img_t = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device) / 255.0
        )

        state_t = (
        torch.from_numpy(state)
        .float()              # 👈 FORCE float32
        .unsqueeze(0)
        .to(self.device)
    )

        obs = {
            "observation.images.camera1": img_t,
            "observation.language.tokens": self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
            "observation.state": state_t,
        }

        action = self.policy.select_action(obs)
        return action[0, :3].cpu().numpy()



def perform_task_vla(target_pos, destination_pos):
    global obs, total_step_t0

    TARGET_STEP_TIME = 0.01
    vla = VLAPID("lerobot/smolvla_base", DEVICE)

    def apply_throttle(current_vla_action=None):
        render_dashboard(data, current_vla_action)
        loop_time = time.perf_counter() - total_step_t0
        sleep_time = TARGET_STEP_TIME - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)


    print("Targeting hover position...")
    for step in range(1000):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([*target_pos[:2], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        
        main_renderer.update_scene(data, camera=-1)
        img = main_renderer.render()
        state = obs["observation"].astype(np.float32)
        if state.shape[0] < 32:
            state = np.pad(state, (0, 32 - state.shape[0]))
        else:
            state = state[:32]
        residual = vla.residual(img, obs["observation"])

        xyz = delta * 10.0 + residual
        action = np.append(xyz, 1.0)

        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        arm_logger.log_state(obs, data) 
        apply_throttle(action)

        if np.linalg.norm(delta[:2]) < 0.01: break

    print("Descending...")
    for step in range(500):
        total_step_t0 = time.perf_counter()

        ee_pos = get_ee_pos(obs)
        target_z = target_pos[2] - 0.005

        delta = np.array([
            target_pos[0] - ee_pos[0],
            target_pos[1] - ee_pos[1],
            target_z - ee_pos[2]
        ])
        
        pid_xyz = delta * np.array([15.0, 15.0, 5.0])
        main_renderer.update_scene(data, camera=-1)
        img = main_renderer.render()
        
        state = obs["observation"].astype(np.float32)
        if state.shape[0] < 32:
            state = np.pad(state, (0, 32 - state.shape[0]))
        else:
            state = state[:32]

        residual_xyz = vla.residual(img, state)
                
        xyz = pid_xyz + residual_xyz
        action = np.append(xyz, 1.0)
        
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.2))
        
        arm_logger.log_state(obs, data) 
        apply_throttle(action)

        if abs(delta[2]) < 0.005: break
        
    print("Grasping...")
    for _ in range(60):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        arm_logger.log_state(obs, data)
        apply_throttle(action)
        
    is_ball=False
    if is_ball: 
        for _ in range(20):
            total_step_t0 = time.perf_counter()
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
            arm_logger.log_state(obs, data) 
            apply_throttle(action)
    
    print("Moving to destination hover...")
    for step in range(1200):
        total_step_t0 = time.perf_counter()

        ee_pos = get_ee_pos(obs)
        
        target_dest_hover = np.array([destination_pos[0], destination_pos[1], 0.15])
        delta = target_dest_hover - ee_pos
        
        main_renderer.update_scene(data, camera=-1)
        img = main_renderer.render()
        state = obs["observation"].astype(np.float32)
        if state.shape[0] < 32:
            state = np.pad(state, (0, 32 - state.shape[0]))
        else:
            state = state[:32]
        residual = vla.residual(img, state)
        
        xyz = delta * 10.0 + residual
        action = np.append(xyz, -1.0)

        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        arm_logger.log_state(obs, data) 
        apply_throttle(action)

        if np.linalg.norm(delta[:2]) < 0.01: break
    
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
        apply_throttle(action)
        # Break once we are within 2mm of the floor target
        if abs(ee_pos[2] - 0.02) < 0.002: break

    # --- PHASE 6: RELEASE AND LIFT ---
    print("Releasing on floor...")
    for _ in range(50):
        total_step_t0 = time.perf_counter()
        # Send 1.0 to open gripper
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle(action)

    print("Clearing area...")
    for _ in range(30):
        total_step_t0 = time.perf_counter()
        # Lift up 20cm to finish cleanly
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle(action)
    
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
    perform_task_vla(cube_starting_pos, final_destination)
    print("Task Completed.")

finally:
    env.close()
    video_out.release()
    print("Video saved as robot_dashboard.mp4")
    

    

    




    