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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True 
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not found, using CPU")

env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="rgb_array")
arm_logger = ArmStateLogger()
mj_env: MujocoEnv = env.unwrapped 
model = mj_env.model
data = mj_env.data


main_renderer = mujoco.Renderer(model, height=480, width=640)
cam_renderer = mujoco.Renderer(model, height=240, width=320)



def project_robot_state(mj_env, mj_data):
    """
    Extracts Franka Panda proprioception and pads to 32-dim.
    Structure: [Joint_Pos (7), Gripper_Width (1), Padding (24)]
    """
    # 1. Get 7-DOF Joint Positions
    # Franka joints in MuJoCo are usually indexed 0-6
    qpos = mj_data.qpos[:7].copy() 
    
    # 2. Get Gripper Width
    # mj_env usually provides a helper, otherwise extract from qpos
    fingers = mj_env.get_fingers_width()
    f_val = fingers if np.isscalar(fingers) else fingers[0]
    
    # 3. Combine and Pad
    state_vector = np.concatenate([qpos, [f_val]]) # length 8
    padded_state = np.zeros(32, dtype=np.float32)
    padded_state[:len(state_vector)] = state_vector
    
    return padded_state

def project_robot_state(mj_env, mj_data):
    qpos = mj_data.qpos[:7].copy() 
    fingers = mj_env.get_fingers_width()
    f_val = fingers if np.isscalar(fingers) else fingers[0]
    
    state_vector = np.concatenate([qpos, [f_val]]) 
    padded_state = np.zeros(32, dtype=np.float32)
    padded_state[:len(state_vector)] = state_vector
    
    # --- ADD VERIFICATION LOG ---
    # Log the first time this is called to verify shape and non-zero content
    if not hasattr(project_robot_state, "verified"):
        print(f"\n[VERIFICATION] State Projector Initialized")
        print(f" -> Raw State (7 DOF + Gripper): {state_vector.shape}")
        print(f" -> Padded Vector Shape: {padded_state.shape}")
        print(f" -> Non-zero count: {np.count_nonzero(padded_state)}")
        project_robot_state.verified = True
    
    return padded_state


class VideoWriterThread:
    def __init__(self, filename, fps, resolution):
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        self.queue = queue.Queue(maxsize=128)
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                if frame is not None:
                    self.writer.write(frame)
            except queue.Empty:
                continue

    def write(self, frame):
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass 

    def stop(self):
        self.running = False
        self.thread.join()
        self.writer.release()

video_thread = VideoWriterThread('vla_robot_dashboard.mp4', 30, (640, 720))


DRIFT_HISTORY = []
total_step_t0 = time.perf_counter()
obs, info = env.reset()
START_REAL_TIME = time.time()

def get_ee_pos(obs): return obs["observation"][:3]
def get_cube_pos(obs): return obs["achieved_goal"]
def get_goal_pos(obs): return obs["desired_goal"]
def clip_action(action, max_delta):
    action = action.copy()
    action[:3] = np.clip(action[:3], -max_delta, max_delta)
    return action

def render_dashboard(data, current_vla_action):
    global START_REAL_TIME, DRIFT_HISTORY
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
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.50, 2
    
    if current_vla_action is not None:
        cv2.putText(main_bgr, f"VLA ACT: {np.round(current_vla_action[:3], 2)}", (350, 180), font, 0.5, (0, 255, 255), 2)
    
    drift_color = (0, 0, 255) if (len(DRIFT_HISTORY) > 1 and current_drift > DRIFT_HISTORY[-2]) else (0, 255, 0)
    
    cv2.putText(main_bgr, f"TIME: {sim_time:.2f}s", (15, 30), font, scale, (255,255,255), thickness)
    cv2.putText(main_bgr, f"EE_X: {ee_pos[0]:.3f}", (15, 60), font, scale, (255,255,255), thickness)
    cv2.putText(main_bgr, f"EE_Y: {ee_pos[1]:.3f}", (15, 90), font, scale, (255,255,255), thickness)
    cv2.putText(main_bgr, f"EE_Z: {ee_pos[2]:.3f}", (15, 120), font, scale, (255,255,255), thickness)
    cv2.putText(main_bgr, f"GRIP: {f_val:.4f}", (15, 150), font, scale, (255,255,255), thickness)
    cv2.putText(main_bgr, f"DEVICE: {DEVICE}", (15, 180), font, scale, (0, 255, 0), thickness)

    cv2.putText(main_bgr, f"DRIFT: {current_drift:.1f}ms", (350, 30), font, scale, drift_color, thickness)
    cv2.putText(main_bgr, f"MAX LAG: {max(DRIFT_HISTORY) if DRIFT_HISTORY else 0:.1f}ms", (350, 60), font, scale, (0, 255, 255), thickness)
    cv2.putText(main_bgr, f"RENDER: {render_latency:.1f}ms", (350, 90), font, scale, (0, 255, 0), thickness)
    cv2.putText(main_bgr, f"REAL CLOCK: {current_real_time:.2f}s", (350, 120), font, scale, (200, 200, 200), thickness)

    cam_renderer.update_scene(data, camera="front_cam")
    front_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)
    cam_renderer.update_scene(data, camera="gripper_front_chase")
    gripper_bgr = cv2.cvtColor(cam_renderer.render(), cv2.COLOR_RGB2BGR)
    
    dashboard = np.vstack((main_bgr, np.hstack((front_bgr, gripper_bgr))))
    video_thread.write(dashboard)
    return main_rgb

def apply_throttle_fixed():
    global total_step_t0
    TARGET_STEP_TIME = 0.01 
    actual_elapsed = time.perf_counter() - total_step_t0
    sleep_needed = TARGET_STEP_TIME - actual_elapsed
    
    if sleep_needed > 0:
        print(f"the control is leading rendering by : {sleep_needed*1000:.1f}ms")
        time.sleep(sleep_needed)
    else:
        print(f"the control is lagging rendering by : {-sleep_needed*1000:.1f}ms")
    total_step_t0 = time.perf_counter()

class VLAPID:
    def __init__(self, model_id, device):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
        self.policy.eval()
        instruction = "Pick up the green block, then move to the target location and then place the cube at the target location on the surface."
        tokens = self.policy.model.vlm_with_expert.processor.tokenizer(instruction, return_tensors="pt").to(device)
        self.lang_tokens = tokens["input_ids"]
        self.lang_mask = tokens["attention_mask"].bool()

    @torch.inference_mode()
    def residual(self, image, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device, non_blocking=True)

        if torch.all(state_t == 0):
         print("[WARNING] VLA received an ALL-ZERO state vector!")

        assert state_t.shape[1] == 32, f"Expected 32 dims, got {state_t.shape[1]}"

        img_t = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().to(self.device, non_blocking=True) / 255.0
        obs_vla = {"observation.images.camera1": img_t, 
                   "observation.language.tokens": self.lang_tokens, 
                   "observation.language.attention_mask": self.lang_mask, 
                   "observation.state": state_t}
        
        action = self.policy.select_action(obs_vla)
        return action[0, :3].cpu().numpy()

class AsyncVLA:
    def __init__(self, vla):
        self.vla = vla
        self.input_queue = queue.Queue(maxsize=1)
        self.latest_result = np.zeros(3)
        self.new_data_available = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("AsyncVLA started")

    def _inference_loop(self):
        while self.running:
            try:
                img, state = self.input_queue.get(timeout=0.1)
                t_start = time.perf_counter()
                result = self.vla.residual(img, state)
                inference_time = (time.perf_counter() - t_start) * 1000
                with self.lock:
                    self.latest_result = result
                    self.new_data_available = True
                print(f"  VLA inference took: {inference_time:.1f}ms")
            except queue.Empty: continue

    def submit(self, img, state):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put_nowait((img, state))

    def get_latest(self):
        with self.lock:
            is_new = self.new_data_available
            self.new_data_available = False
            return self.latest_result.copy(), is_new

    def stop(self): self.running = False; self.thread.join(); print("AsyncVLA stopped")

def perform_task_vla(target_pos, destination_pos):
    global obs, total_step_t0
    vla = VLAPID("lerobot/smolvla_base", DEVICE)
    async_vla = AsyncVLA(vla)
    
    VLA_TICK_RATE = 30
    vla_target_xyz = np.zeros(3)
    current_vla_residual = np.zeros(3)
    smoothing_alpha = 0.15 

    print("\n=== PHASE 1: MOVING TO HOVER ===")
    for step in range(1000):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            current_state_32 = project_robot_state(mj_env, data)
            img = main_renderer.render()
            async_vla.submit(img, current_state_32)
            # async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))
        
        raw_vla, is_new = async_vla.get_latest()
        if is_new: vla_target_xyz = raw_vla
        current_vla_residual = (smoothing_alpha * vla_target_xyz) + ((1 - smoothing_alpha) * current_vla_residual)
        
        action = np.append((delta * 10.0) + (current_vla_residual * 0.2), 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        apply_throttle_fixed()
        render_dashboard(data, action)
        if np.linalg.norm(delta[:2]) < 0.01: break

    
    print("\n=== PHASE 2: DESCENDING TO CUBE ===")
    for step in range(800):
        total_step_t0 = time.perf_counter()
        cube_pos = data.xpos[mj_env.model.body('obj').id].copy()
        ee_pos = get_ee_pos(obs)
        dist_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            
            main_renderer.update_scene(data, camera=-1)
            current_state_32 = project_robot_state(mj_env, data)
            img = main_renderer.render()
            async_vla.submit(img, current_state_32)
            # async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))
        
        raw_vla, is_new = async_vla.get_latest()
        if is_new: vla_target_xyz = raw_vla
        current_vla_residual = (smoothing_alpha * vla_target_xyz) + ((1 - smoothing_alpha) * current_vla_residual)
        
        target_z = cube_pos[2] + 0.005
        delta = np.array([cube_pos[0]-ee_pos[0], cube_pos[1]-ee_pos[1], target_z-ee_pos[2]])
        xyz = (delta * 12.0) + (current_vla_residual * 0.05)
        
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, 1.0), 0.15))
        apply_throttle_fixed(); render_dashboard(data, xyz)
        if dist_xy < 0.015 and abs(ee_pos[2] - target_z) < 0.01: break

    print("\n=== PHASE 3: GRASPING ===")
    for _ in range(100):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        apply_throttle_fixed(); render_dashboard(data, None)

    
    print("\n=== PHASE 4: ALIGNING WITH RED SPOT ===")
    target_site_id = mj_env.model.site('target').id
    for step in range(800):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        actual_red_spot = data.site_xpos[target_site_id].copy()
        delta = np.array([actual_red_spot[0], actual_red_spot[1], 0.15]) - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)

            main_renderer.update_scene(data, camera=-1)
            current_state_32 = project_robot_state(mj_env, data)
            img = main_renderer.render()
            async_vla.submit(img, current_state_32)
            # async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))

        raw_vla, is_new = async_vla.get_latest()
        if is_new: vla_target_xyz = raw_vla
        current_vla_residual = (smoothing_alpha * vla_target_xyz) + ((1 - smoothing_alpha) * current_vla_residual)
        
        xyz = (delta * np.array([15.0, 15.0, 5.0])) + (current_vla_residual * 0.02)
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, -1.0), 0.12))
        apply_throttle_fixed(); render_dashboard(data, xyz)
        if np.linalg.norm(delta[:2]) < 0.005: break

    print("\n=== PHASE 5: VERTICAL DROP ===")
    for step in range(400):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        actual_red_spot = data.site_xpos[target_site_id].copy()
        target_floor = np.array([actual_red_spot[0], actual_red_spot[1], 0.02])
        delta = target_floor - ee_pos
        xyz = delta * np.array([30.0, 30.0, 5.0])
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, -1.0), 0.03))
        apply_throttle_fixed(); render_dashboard(data, xyz)
        if abs(ee_pos[2] - 0.02) < 0.002: break

    
    print("\n=== PHASE 6 & 7: FINISHING ===")
    for _ in range(100): env.step(np.array([0, 0, 0, 1.0])); apply_throttle_fixed()
    for _ in range(100): env.step(np.array([0, 0, 0.2, 1.0])); apply_throttle_fixed()
    async_vla.stop()


try:
    print(f"Using MuJoCo GL Backend: {os.environ['MUJOCO_GL']}")
    for _ in range(10): obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0]))
    cube_starting_pos = get_cube_pos(obs)
    target_site_pos = data.site_xpos[model.site('target').id].copy()
    final_destination = np.array([target_site_pos[0], target_site_pos[1], 0.0]) 

    print(f"\n{'='*60}\nTASK INITIALIZATION\nDetecting Object at: {cube_starting_pos}\nTarget Destination: {final_destination}\nDevice: {DEVICE}\n{'='*60}\n")
    perform_task_vla(cube_starting_pos, final_destination)
    print(f"✓ TASK COMPLETED SUCCESSFULLY")

finally:
    del main_renderer
    del cam_renderer
    env.close()
    video_thread.stop()
    print("Video saved as vla_robot_dashboard.mp4")