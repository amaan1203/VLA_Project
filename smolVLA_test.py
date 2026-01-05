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
        
        # print("vla-act values to debug : ", np.round(current_vla_action[:3], 2))
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
        self.new_data_available = False
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
                    self.new_data_available = True
                
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                self.output_queue.put(result)
                print(f"  VLA inference took: {inference_time:.1f}ms")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VLA inference error: {e}")
    
    def submit(self, img, state):
        try:
            self.input_queue.put_nowait((img, state))
        except queue.Full:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            self.input_queue.put_nowait((img, state))
    
    def get_latest(self):
        """Get latest result"""
        with self.lock:
            is_new = self.new_data_available
            self.new_data_available = False
            return self.latest_result.copy() , is_new
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        print("AsyncVLA stopped")


def perform_task_vla(target_pos, destination_pos):
    global obs, total_step_t0, START_REAL_TIME, START_SIM_TIME

    vla = VLAPID("lerobot/smolvla_base", DEVICE)
    async_vla = AsyncVLA(vla)
    
    VLA_TICK_RATE = 30
    current_residual = np.zeros(3)
    
    # --- PHASE 1: HOVER ---
    print("\n=== PHASE 1: MOVING TO HOVER ===")
    for step in range(1000):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        target_hover = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.15])
        delta = target_hover - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))
        
        res, _ = async_vla.get_latest()
        action = np.append((delta * 10.0) + (res * 0.2), 1.0)
        obs, _, _, _, _ = env.step(clip_action(action, 0.3))
        apply_throttle_fixed()
        render_dashboard(data, action)
        if np.linalg.norm(delta[:2]) < 0.01: break

    # --- PHASE 2: DESCEND TO CUBE ---
    print("\n=== PHASE 2: DESCENDING TO CUBE ===")
    vla_alpha = 0.15
    smoothed_vla = np.zeros(3)
    for step in range(800):
        total_step_t0 = time.perf_counter()
        cube_body_id = mj_env.model.body('obj').id
        cube_pos = data.xpos[cube_body_id].copy()
        ee_pos = get_ee_pos(obs)
        dist_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))
        
        raw_vla, is_new = async_vla.get_latest()
        if is_new: 
            smoothed_vla = (vla_alpha * raw_vla) + ((1 - vla_alpha) * smoothed_vla)
        
        target_z = cube_pos[2] + 0.005
        delta = np.array([cube_pos[0]-ee_pos[0], cube_pos[1]-ee_pos[1], target_z-ee_pos[2]])
        xyz = (delta * 12.0) + (smoothed_vla * 0.05)
        
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, 1.0), 0.15))
        apply_throttle_fixed()
        render_dashboard(data, xyz)
        if dist_xy < 0.015 and abs(ee_pos[2] - target_z) < 0.01: break

    # --- PHASE 3: GRASP ---
    print("\n=== PHASE 3: GRASPING ===")
    for _ in range(100):
        total_step_t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
        apply_throttle_fixed(); render_dashboard(data, None)

    # --- PHASE 4: TRANSPORT ---
    print("\n=== PHASE 4: ALIGNING WITH RED SPOT ===")
    target_site_id = mj_env.model.site('target').id
    for step in range(800):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        actual_red_spot = data.site_xpos[target_site_id].copy()
        delta = np.array([actual_red_spot[0], actual_red_spot[1], 0.15]) - ee_pos
        
        if step % VLA_TICK_RATE == 0:
            main_renderer.update_scene(data, camera=-1)
            async_vla.submit(main_renderer.render(), obs["observation"].astype(np.float32))

        raw_vla, is_new = async_vla.get_latest()
        # High gain XY to fight VLA bias
        xyz = (delta * np.array([15.0, 15.0, 5.0])) + (raw_vla * 0.02)
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, -1.0), 0.12))
        apply_throttle_fixed()
        render_dashboard(data, xyz)
        if np.linalg.norm(delta[:2]) < 0.005: break

    # --- PHASE 5: PRECISION LOWERING (VLA OFF) ---
    print("\n=== PHASE 5: VERTICAL DROP ===")
    for step in range(400):
        total_step_t0 = time.perf_counter()
        ee_pos = get_ee_pos(obs)
        actual_red_spot = data.site_xpos[target_site_id].copy()
        
        # Target exact center, 2cm height
        target_floor = np.array([actual_red_spot[0], actual_red_spot[1], 0.02])
        delta = target_floor - ee_pos
        
        # No VLA here - Pure PID for 100% accuracy
        xyz = delta * np.array([30.0, 30.0, 5.0])
        obs, _, _, _, _ = env.step(clip_action(np.append(xyz, -1.0), 0.03))
        apply_throttle_fixed()
        render_dashboard(data, xyz)
        
        if abs(ee_pos[2] - 0.02) < 0.002: 
            # Stabilize momentum
            for _ in range(30):
                env.step(np.array([0,0,0,-1.0]))
                apply_throttle_fixed()
            break

    # --- PHASE 6 & 7: RELEASE & RETRACT ---
    print("\n=== PHASE 6 & 7: FINISHING ===")
    for _ in range(100): # Open
        env.step(np.array([0, 0, 0, 1.0]))
        apply_throttle_fixed()
    for _ in range(100): # Lift
        env.step(np.array([0, 0, 0.2, 1.0]))
        apply_throttle_fixed()
        
    async_vla.stop()


try:
    START_REAL_TIME = time.time()
    START_SIM_TIME = data.time
    BASE_REAL_TIME = START_REAL_TIME
    
   
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
    print(f"\nDetecting Object at: {cube_starting_pos}")
    print(f"Target Destination: {final_destination}")
    print(f"Device:           {DEVICE}")
    print(f"{'='*60}\n")
    
    perform_task_vla(cube_starting_pos, final_destination)

    print(f"✓ TASK COMPLETED SUCCESSFULLY")


finally:
    env.close()
    video_out.release()
    print("Video saved as vla_robot_dashboard.mp4")