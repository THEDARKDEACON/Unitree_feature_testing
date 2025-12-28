import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import math

class H1RetargetingNode(Node):
    def __init__(self):
        super().__init__('retargeting_node')

        self.declare_parameter('test_mode', True)
        self.test_mode = self.get_parameter('test_mode').value

        # Mapping of H1 Joint Names to their specific controller topics
        # Based on ros_gz_h1_bridge.yaml
        self.joint_topics = {
            'left_hip_yaw_joint': '/h1/left_hip_yaw_joint/cmd_pos',
            'left_hip_pitch_joint': '/h1/left_hip_pitch_joint/cmd_pos',
            'left_hip_roll_joint': '/h1/left_hip_roll_joint/cmd_pos',
            'left_knee_joint': '/h1/left_knee_joint/cmd_pos',
            'left_ankle_pitch_joint': '/h1/left_ankle_pitch_joint/cmd_pos',
            'left_ankle_roll_joint': '/h1/left_ankle_roll_joint/cmd_pos',
            'right_hip_yaw_joint': '/h1/right_hip_yaw_joint/cmd_pos',
            'right_hip_pitch_joint': '/h1/right_hip_pitch_joint/cmd_pos',
            'right_hip_roll_joint': '/h1/right_hip_roll_joint/cmd_pos',
            'right_knee_joint': '/h1/right_knee_joint/cmd_pos',
            'right_ankle_pitch_joint': '/h1/right_ankle_pitch_joint/cmd_pos',
            'right_ankle_roll_joint': '/h1/right_ankle_roll_joint/cmd_pos',
            'torso_joint': '/h1/torso_joint/cmd_pos',
            'left_shoulder_pitch_joint': '/h1/left_shoulder_pitch_joint/cmd_pos',
            'left_shoulder_roll_joint': '/h1/left_shoulder_roll_joint/cmd_pos',
            'left_shoulder_yaw_joint': '/h1/left_shoulder_yaw_joint/cmd_pos',
            'left_elbow_joint': '/h1/left_elbow_joint/cmd_pos',
            'right_shoulder_pitch_joint': '/h1/right_shoulder_pitch_joint/cmd_pos',
            'right_shoulder_roll_joint': '/h1/right_shoulder_roll_joint/cmd_pos',
            'right_shoulder_yaw_joint': '/h1/right_shoulder_yaw_joint/cmd_pos',
            'right_elbow_joint': '/h1/right_elbow_joint/cmd_pos'
        }

        self.publishers_ = {}
        for joint, topic in self.joint_topics.items():
            self.publishers_[joint] = self.create_publisher(Float64, topic, 10)

        # Subscriber for the "Retargeted" Pose from Video/MoCap
        self.create_subscription(JointState, '/reference_joint_states', self.reference_callback, 10)

        if self.test_mode:
            self.get_logger().info("Starting in TEST MODE: Generating Sine Wave Motion")
            self.timer = self.create_timer(0.02, self.test_motion_callback) # 50Hz
            self.start_time = self.get_clock().now().nanoseconds / 1e9

    def reference_callback(self, msg: JointState):
        """
        Receives a full JointState message (e.g. from an offline retargeting script)
        and forwards commands to individual controllers.
        """
        for i, name in enumerate(msg.name):
            if name in self.publishers_:
                cmd = Float64()
                cmd.data = msg.position[i]
                self.publishers_[name].publish(cmd)

    def test_motion_callback(self):
        """
        Generates a simple squat/arm wave motion to verify kinematics.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        t = current_time - self.start_time

        # Simple Squat: Move Knees and Hips
        squat_angle = 0.5 * (1.0 - math.cos(t * 2.0)) # 0 to 1.0 (approx 60 deg)
        
        # Arm Wave
        arm_angle = 0.5 * math.sin(t * 3.0)

        cmd_map = {
            'left_knee_joint': squat_angle,
            'right_knee_joint': squat_angle,
            'left_hip_pitch_joint': -squat_angle * 0.5, # Counter-balance
            'right_hip_pitch_joint': -squat_angle * 0.5,
            'left_shoulder_pitch_joint': arm_angle,
            'right_shoulder_pitch_joint': -arm_angle
        }

        for joint, val in cmd_map.items():
            if joint in self.publishers_:
                msg = Float64()
                msg.data = val
                self.publishers_[joint].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = H1RetargetingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
