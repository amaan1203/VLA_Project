# logger_utils.py
import time
import numpy as np

class ArmStateLogger:
    def __init__(self):
        # Header for the console output
        print(f"\n{'[LOG START]':=^80}")
        print(f"{'Sim_Time':>10} | {'Real_Time':>15} | {'EE_X':>8} | {'EE_Y':>8} | {'EE_Z':>8} | {'Grip_Width':>10}")
        print("-" * 80)

    def log_state(self, obs, data):
        """
        Extracts and prints the current state of the arm.
        """
        sim_time = data.time           # MuJoCo Simulation Clock
        real_time = time.time()        # System Clock

        # EE position is usually the first 3 elements of the observation
        ee_pos = obs["observation"][:3] 
        
        # Gripper width (assuming index 6 based on FrankaEnv)
        # Note: If FrankaEnv is block_gripper=True, this index might shift
        try:
            grip_width = obs["observation"][6]
        except IndexError:
            grip_width = 0.0

        log_str = (f"{sim_time:10.3f} | {real_time:15.4f} | "
                   f"{ee_pos[0]:8.3f} | {ee_pos[1]:8.3f} | {ee_pos[2]:8.3f} | "
                   f"{grip_width:10.4f}")
        
        print(log_str)