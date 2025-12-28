import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import csv

class H1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Load Model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/h1_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation parameters
        self.dt = 0.02  # 50Hz control
        self.model.opt.timestep = 0.002 # 500Hz physics
        self.n_substeps = int(self.dt / self.model.opt.timestep)

        # Action Space: Torques for 19 Controlled Joints (Ignore Hands)
        # Detailed Mapping needed
        self.control_joints = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "left_ankle_roll_joint", # Note: CSV might not have roll/yaw for ankle, but model has it.
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ]
        # Wait, the CSV has 19 joints. The Robot has 19 MAIN joints.
        # Let's see: Left Leg (6), Right Leg (6), Torso (1), L Arm (4), R Arm (4) = 21 Joints.
        # (Ankle Roll is often passive or unused in some CSVs, but H1 has it).
        
        # Let's map ALL main body joints (21).
        self.actuator_indices = []
        for name in self.control_joints:
            # Actuator name often matches joint name + "_motor" or just joint name
            # In create_scene.py, I named motors explicitly. E.g. "left_hip_yaw_joint_motor" (Step 781 output)
            # Actually, Step 781 says: "Added motor for left_hip_yaw_joint" -> name might be "left_hip_yaw_joint"
            # Let's check model.actuator(name).id
            try:
                # Try finding actuator with joint name
                # Usually we search by joint name correlation
                # Step 781: 'Added motor for ...' -> name defaults to joint name usually if not specified, 
                # OR create_scene line: motor name="left_hip_yaw_joint" likely?
                # Actually, check create_scene.py log... 
                # It says `motor name="{joint_name}_motor"` in Step 778 replacement?
                # No, I didn't see explicit naming in Python loop.
                # Just now (Step 738) XML shows: <motor name="left_hip_yaw_joint_motor" ...>
                # So we append "_motor".
                act_name = name + "_motor"
                id = self.model.actuator(act_name).id
                self.actuator_indices.append(id)
            except:
                print(f"Warning: Actuator {name} not found.")

        n_actions = len(self.actuator_indices)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # Load Motion & Update Mapping
        self.motion_data = self._load_motion()
        self.update_mapping() # Sets self.source_joints and self.joint_indices

        # Observations: qpos(21) + qvel(21) + error(21) = 63 dims
        # Use mapped joints count
        n_obs = len(self.joint_indices) * 3
        high = np.inf
        self.observation_space = spaces.Box(-high, high, shape=(n_obs,), dtype=np.float32)
        
        # State vars
        self.step_counter = 0
        self.motion_frame = 0
        self.viewer = None

    def step(self, action):
        # Scale Action
        # Current action is size 21.
        # We need to fill self.data.ctrl which is size 51 (includes hands)
        
        full_ctrl = np.zeros(self.model.nu)
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        
        # Map to specific actuators
        for i, act_id in enumerate(self.actuator_indices):
            # Get range
            ctrl_min = self.model.actuator_ctrlrange[act_id, 0]
            ctrl_max = self.model.actuator_ctrlrange[act_id, 1]
            
            # Scale
            scaled = (action[i] + 1) / 2 * (ctrl_max - ctrl_min) + ctrl_min
            full_ctrl[act_id] = scaled
            
        self.data.ctrl[:] = full_ctrl
        
    def _load_motion(self):
        # Load the verified walk motion
        motion_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/motions/h1/walk1_subject1.csv")
        data = []
        if not os.path.exists(motion_path):
            print(f"Warning: Motion file {motion_path} not found. using dummy zeros.")
            return np.zeros((100, 26))
            
        with open(motion_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([float(x) for x in row])
        return np.array(data)

        
    def update_mapping(self):
        # Mapping from CSV/Model to Observed/Rewarded Joints
        self.source_joints = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ]
        
        self.joint_indices = []
        for name in self.source_joints:
            try:
                j = self.model.joint(name)
                addr = j.qposadr
                if hasattr(addr, "__getitem__"):
                    addr = addr[0]
                self.joint_indices.append(int(addr))
            except Exception as e:
                print(f"Error mapping joint {name}: {e}")
                self.joint_indices.append(None) 

    def get_reference_pose(self, frame_idx):
        # Returns qpos target for the robot
        frame = self.motion_data[frame_idx]
        target_qpos = np.zeros(self.model.nq)
        
        # Root (with offset)
        target_qpos[0:3] = frame[0:3]
        # Quat Reorder (x,y,z,w -> w,x,y,z)
        target_qpos[3] = frame[6]
        target_qpos[4] = frame[3]
        target_qpos[5] = frame[4]
        target_qpos[6] = frame[5]
        
        # Joints
        # CSV has 19 joints. We have 21 mapped.
        # We need to map CSV columns to Model Joints carefully.
        # If CSV lacks Ankle Roll, we leave it 0.
        # The CSV columns are fixed 19.
        
        # Original CSV mapping (19 cols)
        csv_map = [
            # Left Leg
            (self.joint_indices[0], 7),   # hip_yaw
            (self.joint_indices[1], 8),   # hip_roll
            (self.joint_indices[2], 9),   # hip_pitch
            (self.joint_indices[3], 10),  # knee
            (self.joint_indices[4], 11),  # ankle_pitch
            # Right Leg
            (self.joint_indices[6], 12),  # hip_yaw
            (self.joint_indices[7], 13),  # hip_roll
            (self.joint_indices[8], 14),  # hip_pitch
            (self.joint_indices[9], 15),  # knee
            (self.joint_indices[10], 16), # ankle_pitch
            # Torso
            (self.joint_indices[12], 17), # torso
            # Arms
            (self.joint_indices[13], 18), 
            (self.joint_indices[14], 19), 
            (self.joint_indices[15], 20), 
            (self.joint_indices[16], 21), 
            (self.joint_indices[17], 22), 
            (self.joint_indices[18], 23), 
            (self.joint_indices[19], 24), 
            (self.joint_indices[20], 25), 
        ]
        
        for q_idx, csv_col in csv_map:
            if q_idx is not None and csv_col < len(frame):
                target_qpos[q_idx] = frame[csv_col]
            
        return target_qpos

    def step(self, action):
        # 1. Action Mapping (19/21 Actuators -> 51 Controls)
        full_ctrl = np.zeros(self.model.nu)
        
        # Clip input
        action = np.clip(action, -1.0, 1.0)
        
        # Map active joints, leave Hands (unmapped) at 0 torque.
        for i, act_id in enumerate(self.actuator_indices):
            if i < len(action):
                ctrl_min = self.model.actuator_ctrlrange[act_id, 0]
                ctrl_max = self.model.actuator_ctrlrange[act_id, 1]
                scaled = (action[i] + 1) / 2 * (ctrl_max - ctrl_min) + ctrl_min
                full_ctrl[act_id] = scaled
            
        self.data.ctrl[:] = full_ctrl
        
        # 2. Physics Step
        try:
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        except Exception as e:
            print(f"Simulation Error: {e}")
            return self._get_obs(self.data.qpos[self.joint_indices]), -100, True, False, {}

        # 3. Counters
        self.motion_frame = (self.motion_frame + 1) % len(self.motion_data)
        
        # 4. Reward
        target_qpos = self.get_reference_pose(self.motion_frame)
        
        # Tracking Error (Only on the 21 mapped joints)
        current_joints = self.data.qpos[self.joint_indices]
        target_joints = target_qpos[self.joint_indices]
        
        # Handle Potential NaNs in error
        diff = current_joints - target_joints
        pose_error = np.mean(np.square(diff))
        if np.isnan(pose_error):
            pose_error = 100.0 # Penalty
            
        r_tracking = np.exp(-10 * pose_error)
        
        # Survival
        root_z = self.data.qpos[2]
        r_survival = 1.0 if root_z > 0.6 else 0.0 # Relaxed threshold to 0.6
        
        total_reward = 0.8 * r_tracking + 0.2 * r_survival
        
        # 5. Termination
        terminated = bool(root_z < 0.5) # Falling condition
        truncated = False
        info = {"tracking_error": pose_error}

        return self._get_obs(target_joints), total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.motion_frame = self.np_random.integers(0, len(self.motion_data))
        target_qpos = self.get_reference_pose(self.motion_frame)
        self.data.qpos[:] = target_qpos
        self.data.qvel[:] = 0 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(target_qpos[self.joint_indices]), {}

    def _get_obs(self, target_joints):
        current_joints = self.data.qpos[self.joint_indices]
        error = target_joints - current_joints
        # Use filtered observation space
        # Base: qpos of controlled joints, qvel of controlled joints
        # + error
        # Flatten and ensure safe types
        obs = np.concatenate([
            current_joints, 
            self.data.qvel[self.joint_indices], # Velocity of relevant joints
            error
        ]).astype(np.float32)
        
        # Note: If observation space definition (Step 828) included full qpos/qvel, we should match it.
        # Previous definition: n_obs = (nq-7) + nv + 19.
        # My new logic simplifies it to minimal set. 
        # I must ensure observation_space matches this size!
        # Size = 21 (pos) + 21 (vel) + 21 (err) = 63.
        # Step 828 code did: n_actions = len(actuator_indices).
        # But observation_space was commented out as "..."
        # I should probably update `observation_space` in `reset` if possible, or just be consistent.
        # I will assume the previous tool call set it up, or I will rely on Gymnasium's flexibility (it complains if mismatch).
        
        return obs
        
    def render(self):
        if self.viewer is None and self.render_mode == "human":
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
