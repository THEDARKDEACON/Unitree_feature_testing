import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import csv

# Path to model (use h1_scene.xml)
model_path = os.path.join(os.path.dirname(__file__), "assets/h1_scene.xml")

# Path to a motion file
motion_file = os.path.join(os.path.dirname(__file__), "data/motions/h1/walk1_subject1.csv")

def load_motion(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert to float
            data.append([float(x) for x in row])
    return np.array(data)

def main():
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Loading motion: {motion_file}")
    motion_data = load_motion(motion_file)
    print(f"Loaded {len(motion_data)} frames. Columns: {motion_data.shape[1]}")
    
    # Check joint count
    # Model qpos = 7 (root) + joints.
    # Note: h1_scene.xml is likely h1_2 (with hands, 50+ joints). 
    # The motion data (h1) is 26 cols (7 root + 19 joints).
    # We need to map the 19 joints to the 50+ joints of model.
    # OR we need to load a simplified h1_19dof.urdf for playback.
    
    # Strategy: Map the 19 joints to the corresponding joints in H1_2 and zero the rest (hands).
    # We need to guess the mapping.
    
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        frame_idx = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            if frame_idx < len(motion_data):
                frame = motion_data[frame_idx]
                
                # ROOT POSE
                # CSV: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w (Assumed)
                # MuJoCo qpos: pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
                
                # Check quaternion magnitude to guess order
                # frame[3:7] -> 0.003, 0.007, 0.018, 0.999
                # This suggests [x, y, z, w]
                
                data.qpos[0:3] = frame[0:3]
                # Reorder to w, x, y, z
                data.qpos[3] = frame[6] # w
                data.qpos[4] = frame[3] # x
                data.qpos[5] = frame[4] # y
                data.qpos[6] = frame[5] # z
                
                # JOINTS (19 DoF Source -> 50+ DoF Model)
                # Source Order Assumption (Standard H1 19-DoF):
                # Legs: Yaw, Pitch, Roll, Knee, AnklePitch
                # Torso: Yaw
                # Arms: Pitch, Roll, Yaw, Elbow
                
                # ROBUST MAPPING FIX
                # Try swapping Pitch/Roll. Large walking motion (Pitch) was likely going to Roll joint.
                # New Assumption: Yaw, Roll, Pitch
                source_joints = [
                    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
                    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
                    "torso_joint",
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
                ]
                
                # Apply mapping
                for src_idx, name in enumerate(source_joints):
                    try:
                        joint_addr = model.joint(name).qposadr[0]
                        angle = frame[7 + src_idx]
                        data.qpos[joint_addr] = angle
                    except:
                        pass
                
                # ROTATION & HEIGHT FIX
                # 1. Rotate root by 90 deg if needed (typical mismatch).
                # 2. Add Z offset because model feet are different.
                # data.qpos[2] += 0.05 # Removed to prevent hovering

                
                mujoco.mj_forward(model, data)
                frame_idx = (frame_idx + 1) % len(motion_data)
                
            viewer.sync()
            
            # 30 FPS playback approx
            time.sleep(1/30.0)

if __name__ == "__main__":
    main()
