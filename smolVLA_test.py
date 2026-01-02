import gymnasium as gym
import panda_mujoco_gym
import numpy as np
import mujoco
import cv2
import time
import os
import torch
import threading
import queue
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

video_out = cv2.VideoWriter('vla_robot_dashboard.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 720))

main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)

print(f"Using MuJoCo GL Backend: {os.environ['MUJOCO_GL']}")

DRIFT_HISTORY = []

total_step_t0 = time.perf_counter()
obs, info = env.reset()

START_REAL_TIME = time.time()
START_SIM_TIME = data.time
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
    """Render dashboard - called AFTER control step"""
    global BASE_REAL_TIME, START_REAL_TIME, DRIFT_HISTORY
    
    sim_time = data.time
    current_real_time = time.time() - START_REAL_TIME
    current_drift = (current_real_time - sim_time) * 1000 
    DRIFT_HISTORY.append(current_drift)
    
    main_renderer.update_scene(data, camera=-1) 
    t_render_start = time.perf_counter()
    main_rgb = main_renderer.render()
    render_latency = (time.perf_counter() - t_render_start) * 1000
    
    main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
    
    ee_pos = get_ee_pos(obs)
    fingers = mj_env.get_fingers_width()
    f_val = fingers if np.isscalar(fingers) else fingers[0]
    
    if current_vla_action is not None:
        cv2.putText(main_bgr, f"VLA ACT: {np.round(current_vla_action[:3], 2)}", 
                   (350, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thickness = 2
    
    drift_color = (0, 0, 255) if (len(DRIFT_HISTORY) > 1 and current_drift > DRIFT_HISTORY[-2]) else (0, 255, 0)
    text_color = (255, 255, 255) 
    
    cv2.putText(main_bgr, f"TIME: {sim_time:.2f}s", (15, 30), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_X: {ee_pos[0]:.3f}", (15, 60), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_Y: {ee_pos[1]:.3f}", (15, 90), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"EE_Z: {ee_pos[2]:.3f}", (15, 120), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"GRIP: {f_val:.4f}", (15, 150), font, scale, text_color, thickness)
    cv2.putText(main_bgr, f"DEVICE: {DEVICE}", (15, 180), font, scale, (0, 255, 0), thickness)

    cv2.putText(main_bgr, f"DRIFT: {current_drift:.1f}ms", (350, 30), font, scale, drift_color, thickness)
    cv2.putText(main_bgr, f"MAX LAG: {max(DRIFT_HISTORY) if DRIFT_HISTORY else 0:.1f}ms", 
               (350, 60), font, scale, (0, 255, 255), thickness)
    cv2.putText(main_bgr, f"RENDER: {render_latency:.1f}ms", (350, 90), font, scale, (0, 255, 0), thickness)
    cv2.putText(main_bgr, f"REAL CLOCK: {current_real_time:.2f}s", (350, 120), font, scale, (200, 200, 200), thickness)
    
    # Render two auxiliary cameras
    cam_renderer.update_scene(data, camera="front_cam")
    front_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)
    
    cam_renderer.update_scene(data, camera="gripper_front_chase")
    gripper_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)
    
    border_color = (0, 255, 255) 
    cv2.rectangle(gripper_bgr, (0, 0), (320, 240), border_color, 10)
    cv2.putText(gripper_bgr, "GRIPPER VIEW", (50, 30), font, 0.5, border_color, 2)

    bottom_row = np.hstack((front_bgr, gripper_bgr))
    dashboard = np.vstack((main_bgr, bottom_row))
    
    # ⚠️ THIS IS THE KILLER - write video frame asynchronously
    try:
        video_out.write(dashboard)
    except Exception as e:
        print(f"Video write error: {e}")
    
    return main_rgb


def apply_throttle_fixed(current_vla_action=None):
    """Control loop timing - NO rendering here"""
    global total_step_t0
    
    TARGET_STEP_TIME = 0.01  # 100Hz = 10ms per step
    
    # Measure time since last step
    actual_elapsed = time.perf_counter() - total_step_t0
    
    # Calculate sleep needed
    sleep_needed = TARGET_STEP_TIME - actual_elapsed
    
    # Sleep if ahead of schedule
    if sleep_needed > 0.001:
        time.sleep(sleep_needed)
    elif sleep_needed < -0.005:
        # We're >5ms late
        print(f"the control is lagging rendering by : {-sleep_needed*1000:.1f}ms")
    
    total_step_t0 = time.perf_counter()


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
        state: (32,) float32
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
            .float()             
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


class AsyncVLA:
    """Asynchronous VLA inference to prevent blocking the control loop"""
    def __init__(self, vla):
        self.vla = vla
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.latest_result = np.zeros(3)
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("AsyncVLA started")
    
    def _inference_loop(self):
        """Runs continuously in background thread"""
        while self.running:
            try:
                img, state = self.input_queue.get(timeout=0.1)
                t_start = time.perf_counter()
                result = self.vla.residual(img, state)
                inference_time = (time.perf_counter() - t_start) * 1000
                
                with self.lock:
                    self.latest_result = result
                
                # Clear old results
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                self.output_queue.put(result)
                
                
                print(f"  VLA inference: {inference_time:.1f}ms")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VLA inference error: {e}")
    
    def submit(self, img, state):
        try:
            self.input_queue.put_nowait((img, state))
        except queue.Full:
            # If queue full, remove old and add new
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            self.input_queue.put_nowait((img, state))
    
    def get_latest(self):
        """Get latest result"""
        with self.lock:
            return self.latest_result.copy()
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        print("AsyncVLA stopped")


def perform_task_vla(target_pos, destination_pos):
    global obs, total_step_t0, START_REAL_TIME, START_SIM_TIME

    vla = VLAPID("lerobot/smolvla_base", DEVICE)
    async_vla = AsyncVLA(vla)
    
    VLA_TICK_RATE = 30  # Update VLA every 30 steps (300ms at 100Hz)
    current_residual = np.zeros(3)
    
   
    START_REAL_TIME = time.time()
    START_SIM_TIME = data.time

    # --- PHASE 1: MOVE TO HOVER ---
    print("\n=== PHASE 1: MOVING TO HOVER ===")
    for step in range(1000):
        total_step_t0 = time.perf_counter() 
        
        ee_pos = get_ee_pos(obs)
         # Define target: cube location + 0.15m above it
        target_hover = np.array([*target_pos[:2], target_pos[2] + 0.15])
                                #  ^^^^^^ XY from cube  ^^^^^^ Z = cube_height + 15cm
        
        delta = target_hover - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            img = main_renderer.render()
            state = obs["observation"].astype(np.float32)
            state = np.pad(state, (0, max(0, 32 - state.shape[0])))[:32]
            # Send to background thread
            async_vla.submit(img, state)  
        
        current_residual = async_vla.get_latest()
        
        # HYBRID CONTROL: PID + VLA
        # PID: P gain of 10.0 on position error
        # VLA: Small correction (0.3 weight) from learned model       
        xyz = (delta * 10.0) + (current_residual * 0.3)
        action = np.append(xyz, 1.0)

        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        arm_logger.log_state(obs, data) 
        apply_throttle_fixed()  # Enforce 100Hz control rate
        render_dashboard(data, action)  # ← Render AFTER timing

        if np.linalg.norm(delta[:2]) < 0.01: 
            print("✓ Hover position reached")
            break


    # --- PHASE 2: DESCENDING TO CUBE ---
    print("\n=== PHASE 2: DESCENDING TO CUBE ===")
    START_REAL_TIME = time.time()
    START_SIM_TIME = data.time

    cube_body_id = mj_env.model.body('obj').id
    cube_pos = data.xpos[cube_body_id].copy()
    ee_pos = get_ee_pos(obs)
    print(f"Cube Position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"EE Position:   [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"Initial Distance: {np.linalg.norm(ee_pos[:2] - cube_pos[:2]):.4f}m\n")

    descent_complete = False
    for step in range(500):
        total_step_t0 = time.perf_counter()
        
        cube_pos = data.xpos[cube_body_id].copy()
        ee_pos = get_ee_pos(obs)
        dist_to_cube = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        delta = cube_pos - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            img = main_renderer.render()
            state = np.pad(obs["observation"].astype(np.float32), (0, max(0, 32 - obs["observation"].shape[0])))[:32]
            async_vla.submit(img, state)
        
        current_residual = async_vla.get_latest()
        
        if dist_to_cube > 0.08:
            pid_xyz = delta * np.array([25.0, 25.0, 10.0])
            xyz = (pid_xyz * 0.9) + (current_residual * 0.1)
        elif dist_to_cube > 0.05:
            pid_xyz = delta * np.array([20.0, 20.0, 8.0])
            xyz = (pid_xyz * 0.8) + (current_residual * 0.2)
        else:
            pid_xyz = delta * np.array([15.0, 15.0, 6.0])
            xyz = (pid_xyz * 0.7) + (current_residual * 0.3)
        
        action = np.append(xyz, 1.0)
        
        if step % 20 == 0:
            print(f"Step {step:3d} | Dist: {dist_to_cube:.4f}m | Z: {ee_pos[2]:.3f}m")

        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.15))
        arm_logger.log_state(obs, data)
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, action)  # ← Render AFTER
        
        if dist_to_cube < 0.035 and ee_pos[2] < (cube_pos[2] + 0.01):
            print(f"✓ Descent complete at step {step}")
            descent_complete = True
            break

    print(f"Descent phase: {'COMPLETE' if descent_complete else 'TIMEOUT'}")


    # --- PHASE 3: GRASP ---
    print("\n=== PHASE 3: GRASPING ===")
    for i in range(150):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        arm_logger.log_state(obs, data)
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, None)  # ← Render AFTER
        
        if i % 30 == 0:
            fingers = mj_env.get_fingers_width()
            f_val = fingers if np.isscalar(fingers) else fingers[0]
            print(f"  Gripper closing... width: {f_val:.4f}")

    print("✓ Gripper closed")


    # --- PHASE 4: MOVE TO DESTINATION ---
    print("\n=== PHASE 4: MOVING TO DESTINATION ===")
    for step in range(1200):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        current_target_site = data.site_xpos[mj_env.model.site('target').id].copy()
        target_dest_hover = np.array([current_target_site[0], current_target_site[1], 0.15])
        delta = target_dest_hover - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            img = main_renderer.render()
            state = np.pad(obs["observation"].astype(np.float32), (0, max(0, 32 - obs["observation"].shape[0])))[:32]
            async_vla.submit(img, state)

        current_residual = async_vla.get_latest()
        xyz = (delta * 10.0) + (current_residual * 0.5)
        action = np.append(xyz, -1.0)
        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        arm_logger.log_state(obs, data) 
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, action)  # ← Render AFTER
        
        if np.linalg.norm(delta[:2]) < 0.01: 
            print("✓ Destination hover reached")
            break


    # --- PHASE 5: LOWER TO FLOOR ---
    print("\n=== PHASE 5: LOWERING TO FLOOR ===")
    for step in range(300):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_floor = np.array([destination_pos[0], destination_pos[1], 0.02])
        delta = target_floor - ee_pos
        action = np.append(delta * 5.0, -1.0)
        obs, _, _, _, _ = env.step(clip_action(action, max_delta=0.05))
        arm_logger.log_state(obs, data)
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, action)  # ← Render AFTER
        
        if abs(ee_pos[2] - 0.02) < 0.002:
            print("✓ Floor level reached")
            break


    # --- PHASE 6: RELEASE ---
    print("\n=== PHASE 6: RELEASING ===")
    for _ in range(50):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, None)  # ← Render AFTER
    print("✓ Gripper opened")


    # --- PHASE 7: CLEAR AREA ---
    print("\n=== PHASE 7: CLEARING AREA ===")
    for _ in range(30):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0.2, 1.0]))
        arm_logger.log_state(obs, data) 
        apply_throttle_fixed()  # ← Just timing
        render_dashboard(data, None)  # ← Render AFTER
    print("✓ Area cleared")

    async_vla.stop()


try:
    START_REAL_TIME = time.time()
    START_SIM_TIME = data.time
    BASE_REAL_TIME = START_REAL_TIME
    
    # Stabilize
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0]))

    cube_starting_pos = get_cube_pos(obs)
    site_id = model.site('target').id
    target_site_pos = data.site_xpos[site_id].copy()
    target_goal_pos = get_goal_pos(obs)
    final_destination = np.array([target_site_pos[0], target_site_pos[1], 0.0]) 

    print(f"\n{'='*60}")
    print(f"TASK INITIALIZATION")
    print(f"{'='*60}")
    print(f"Object Position:  {cube_starting_pos}")
    print(f"Target Position:  {final_destination}")
    print(f"Device:           {DEVICE}")
    print(f"{'='*60}\n")

    # Execute the task
    perform_task_vla(cube_starting_pos, final_destination)
    
    print(f"\n{'='*60}")
    print(f"✓ TASK COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")

finally:
    env.close()
    video_out.release()
    print("Video saved as vla_robot_dashboard.mp4")