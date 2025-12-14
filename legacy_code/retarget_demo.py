"""
Retargeting Module Usage Example

This script demonstrates how the retargeting pipeline works with mock data.
It shows:
1. Creating HumanState from perception outputs
2. Setting up the retargeter with an IK solver
3. Running the retargeting loop
4. Analyzing the quality of generated trajectories

Run this to verify your retargeting setup before connecting real hardware.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import List, Tuple, Tuple

# Import our modules
import sys
sys.path.insert(0, '/home/claude/dynamical_edge')

from core.human_state import (
    Human3DState, DexterHandState, HumanState, EnvObject, 
    fuse_to_human_state
)
from core.recorder import RobotObs, RobotAction, DemoStep, Episode
from core.retargeting import (
    Retargeter, RetargetConfig, GripperMapping,
    pose_to_matrix, matrix_to_pose, pose_error
)
from robot.ik_solver import SimpleIKSolver, IKConfig


# ---------------------------------------------------------------------------
# Mock Robot: A Simple 3-DOF Planar Arm
# ---------------------------------------------------------------------------
# For testing without a real robot URDF, we define a simple planar arm.
# This has 3 revolute joints in a plane, making IK tractable analytically
# but still exercising the full pipeline.

class MockPlanarArm:
    """
    A 3-DOF planar arm for testing the retargeting pipeline.
    
    This is a simplified arm in the XY plane (horizontal), elevated at height Z=0.3.
    
    Joint 0: Base rotation (around Z) - rotates entire arm horizontally
    Joint 1: Shoulder flexion - in the horizontal plane
    Joint 2: Elbow flexion - in the horizontal plane
    
    Link lengths: L1 = 0.35m, L2 = 0.30m, L3 = 0.15m (to EE)
    Total reach: 0.8m from base
    """
    
    def __init__(self):
        self.L1 = 0.35  # Upper arm
        self.L2 = 0.30  # Forearm  
        self.L3 = 0.15  # Hand to EE
        self.base_height = 0.3  # Height above ground
        
        self.n_joints = 3
        self.q_min = np.array([-np.pi, -2.5, -2.5])
        self.q_max = np.array([np.pi, 2.5, 2.5])
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute EE pose for joint angles [q0, q1, q2].
        
        All joints rotate around Z. The arm operates in a horizontal plane
        at fixed height. This is a simplification but makes IK much easier.
        
        Returns 4x4 transform with EE position and orientation.
        """
        q0, q1, q2 = q
        
        # All angles are cumulative rotations around Z in the horizontal plane
        # q0 rotates the whole arm
        # q1 is the shoulder angle relative to q0
        # q2 is the elbow angle relative to q0+q1
        
        angle_shoulder = q0
        angle_elbow = q0 + q1
        angle_wrist = q0 + q1 + q2
        
        # Compute each joint position
        # Shoulder at origin (0, 0, base_height)
        x_elbow = self.L1 * np.cos(angle_shoulder)
        y_elbow = self.L1 * np.sin(angle_shoulder)
        
        x_wrist = x_elbow + self.L2 * np.cos(angle_elbow)
        y_wrist = y_elbow + self.L2 * np.sin(angle_elbow)
        
        x_ee = x_wrist + self.L3 * np.cos(angle_wrist)
        y_ee = y_wrist + self.L3 * np.sin(angle_wrist)
        
        # Construct pose
        T = np.eye(4)
        T[:3, 3] = [x_ee, y_ee, self.base_height]
        
        # Orientation: pointing in the direction of the final link
        R = Rotation.from_euler('z', angle_wrist)
        T[:3, :3] = R.as_matrix()
        
        return T


def create_mock_ik_solver() -> SimpleIKSolver:
    """Create an IK solver for our mock planar arm."""
    arm = MockPlanarArm()
    
    config = IKConfig(
        max_iterations=200,
        tolerance_position=0.005,
        tolerance_rotation=0.5,  # Relaxed - 3 DOF can't control orientation independently
        damping=1e-3,
        step_size=0.3,
    )
    
    return PositionOnlyIKSolver(
        arm=arm,
        config=config
    )


class PositionOnlyIKSolver:
    """
    Position-only IK for the 3-DOF planar arm.
    
    Since we only have 3 DOF and want to track 3D position, we can't
    independently control orientation. This solver only minimizes position error.
    
    For real robots with 6-7 DOF, use the full IK solver with Pinocchio.
    """
    
    def __init__(self, arm: MockPlanarArm, config: IKConfig):
        self.arm = arm
        self.config = config
        self.n_joints = arm.n_joints
        self._q_min = arm.q_min
        self._q_max = arm.q_max
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        return self._q_min.copy()
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        return self._q_max.copy()
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self.arm.forward_kinematics(q)
    
    def _numerical_jacobian_position(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute position-only Jacobian via finite differences."""
        J = np.zeros((3, self.n_joints))
        T0 = self.arm.forward_kinematics(q)
        pos0 = T0[:3, 3]
        
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += eps
            T_plus = self.arm.forward_kinematics(q_plus)
            J[:, i] = (T_plus[:3, 3] - pos0) / eps
        
        return J
    
    def solve(
        self,
        q_init: np.ndarray,
        T_target: np.ndarray,
        max_iterations: int = None,
        tolerance: float = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK for position only (ignores orientation).
        
        Uses damped least squares with iterative updates.
        """
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance_position
        
        q = q_init.copy()
        target_pos = T_target[:3, 3]
        
        # Project target to arm's height plane (arm can only reach z=0.3)
        target_pos_constrained = target_pos.copy()
        target_pos_constrained[2] = self.arm.base_height
        
        for iteration in range(max_iter):
            T_current = self.arm.forward_kinematics(q)
            current_pos = T_current[:3, 3]
            
            # Position error
            e_pos = target_pos_constrained - current_pos
            pos_error = np.linalg.norm(e_pos)
            
            if pos_error < tol:
                return q, True, pos_error
            
            # Compute Jacobian (position only: 3x3)
            J = self._numerical_jacobian_position(q)
            
            # Damped pseudo-inverse
            JJT = J @ J.T
            damping = self.config.damping
            J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(3))
            
            # Update
            dq = J_pinv @ e_pos
            q = q + self.config.step_size * dq
            
            # Clamp to limits
            q = np.clip(q, self._q_min, self._q_max)
        
        # Final error
        T_final = self.arm.forward_kinematics(q)
        final_error = np.linalg.norm(target_pos_constrained - T_final[:3, 3])
        return q, False, final_error


# ---------------------------------------------------------------------------
# Mock Perception: Generate Synthetic Human Motion
# ---------------------------------------------------------------------------

def generate_reaching_trajectory(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_steps: int = 50,
    grasp_at_step: int = 35
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate a simple reaching + grasping trajectory.
    
    Returns list of (hand_position, grasp_aperture) tuples.
    The hand moves from start to end with a minimum-jerk profile,
    and closes the grasp near the end.
    """
    trajectory = []
    
    for i in range(n_steps):
        # Minimum jerk profile: smooth acceleration/deceleration
        t = i / (n_steps - 1)
        s = 10*t**3 - 15*t**4 + 6*t**5  # Normalized position [0, 1]
        
        pos = start_pos + s * (end_pos - start_pos)
        
        # Grasp closes after grasp_at_step
        if i < grasp_at_step:
            grasp = 0.0
        else:
            grasp_progress = (i - grasp_at_step) / (n_steps - grasp_at_step)
            grasp = min(1.0, grasp_progress * 1.5)
        
        trajectory.append((pos, grasp))
    
    return trajectory


def create_mock_human_state(
    hand_position: np.ndarray,
    grasp_aperture: float,
    object_position: np.ndarray,
    timestamp: float
) -> HumanState:
    """
    Create a mock HumanState for testing.
    
    In a real system, this would come from camera + glove fusion.
    Here we synthesize plausible values with the arm at working height.
    """
    # Create mock body pose
    n_keypoints = 17  # Simplified COCO body
    keypoints = np.zeros((n_keypoints, 3))
    
    # Pelvis at origin (human standing at origin)
    keypoints[0] = [0, 0, 0.9]  # Standing height
    
    # Approximate arm chain - human is reaching forward
    shoulder_pos = np.array([0.0, 0.2, 0.3])  # Shoulder at arm working height
    
    # Elbow halfway between shoulder and hand
    elbow_pos = shoulder_pos + 0.5 * (hand_position - shoulder_pos)
    
    keypoints[14] = shoulder_pos                 # r_shoulder
    keypoints[15] = elbow_pos                    # r_elbow  
    keypoints[16] = hand_position                # r_wrist
    
    body = Human3DState(
        timestamp=timestamp,
        keypoints_3d=keypoints,
        keypoint_confidence=np.ones(n_keypoints)
    )
    
    # Create mock hand state
    finger_angles = np.ones((5, 3)) * grasp_aperture * 1.5
    
    hand = DexterHandState(
        timestamp=timestamp,
        side='right',
        finger_angles=finger_angles,
        finger_abduction=np.zeros(4),
        wrist_quat_local=np.array([0, 0, 0, 1]),
    )
    
    # Create object
    obj_pose = np.eye(4)
    obj_pose[:3, 3] = object_position
    
    obj = EnvObject(
        object_id='target_cup',
        class_id=1,
        class_name='cup',
        pose_world=obj_pose
    )
    
    return fuse_to_human_state(
        body, hand, None, [obj],
        primary_object_id='target_cup'
    )


# ---------------------------------------------------------------------------
# Main Demo Script
# ---------------------------------------------------------------------------

def run_retargeting_demo():
    """
    Run a complete retargeting demo with mock data.
    
    This simulates a human reaching for an object and the retargeter
    converting that motion to robot joint trajectories.
    """
    print("=" * 60)
    print("Retargeting Module Demo")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up IK solver and retargeter...")
    ik_solver = create_mock_ik_solver()
    
    config = RetargetConfig(
        T_ee_hand=np.eye(4),  # No correction for mock setup
        workspace_min=np.array([-1.0, -1.0, 0.0]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        max_joint_velocity=2.0,
        gripper_mapping=GripperMapping(strategy='mean')
    )
    
    retargeter = Retargeter(ik_solver, config)
    retargeter.reset()
    
    # Define the task - now in the horizontal plane at the arm's height
    object_position = np.array([0.5, 0.1, 0.3])  # Within reach, at arm height
    hand_start = np.array([0.3, 0.3, 0.3])       # Starting position
    hand_end = object_position + np.array([0.0, 0.0, 0.0])  # Go to object
    
    print(f"   Object at: {object_position}")
    print(f"   Hand start: {hand_start}")
    print(f"   Hand end: {hand_end}")
    
    # Generate human trajectory
    print("\n2. Generating synthetic human reaching trajectory...")
    human_trajectory = generate_reaching_trajectory(hand_start, hand_end, n_steps=50)
    print(f"   Generated {len(human_trajectory)} timesteps")
    
    # Initial robot state
    q_init = np.array([0.0, 0.3, 0.3])  # Reasonable starting config
    
    robot_obs = RobotObs(
        timestamp=0.0,
        joint_positions=q_init,
        joint_velocities=np.zeros(3),
        gripper_position=0.0
    )
    
    # Run retargeting
    print("\n3. Running retargeting loop...")
    episode = Episode(
        episode_id='demo_001',
        task_id='reach_and_grasp',
        env_id='mock_env'
    )
    
    prev_q = q_init
    dt = 0.1  # 10 Hz
    
    for i, (hand_pos, grasp) in enumerate(human_trajectory):
        timestamp = i * dt
        
        # Create mock human state
        human_state = create_mock_human_state(
            hand_pos, grasp, object_position, timestamp
        )
        
        # Update robot observation (in real system, read from robot)
        robot_obs = RobotObs(
            timestamp=timestamp,
            joint_positions=prev_q,
            joint_velocities=np.zeros(3),
            gripper_position=retargeter._prev_gripper
        )
        
        # Retarget
        step = retargeter.human_to_robot_action(
            human=human_state,
            robot_obs=robot_obs,
            prev_q=prev_q,
            episode_id=episode.episode_id,
            t_index=i,
            task_id=episode.task_id,
            env_id=episode.env_id,
            dt=dt
        )
        
        episode.append_step(step)
        prev_q = step.action.joint_position_target
        
        # Progress indicator
        if i % 10 == 0:
            print(f"   Step {i}: q={prev_q.round(3)}, gripper={step.action.gripper_target:.2f}, valid={step.is_valid}")
    
    # Analyze results
    print("\n4. Analyzing results...")
    print(f"   Total steps: {len(episode)}")
    print(f"   Valid steps: {episode.n_valid_steps}")
    print(f"   IK success rate: {retargeter.ik_success_rate:.1%}")
    print(f"   Episode duration: {episode.duration:.1f}s")
    
    # Extract trajectories for plotting
    obs_array = episode.get_obs_array()
    action_array = episode.get_action_array()
    
    print(f"   Observation shape: {obs_array.shape}")
    print(f"   Action shape: {action_array.shape}")
    
    # Compute EE trajectory (via FK)
    ee_positions = []
    arm = MockPlanarArm()
    for step in episode.steps:
        if step.is_valid:
            T = arm.forward_kinematics(step.action.joint_position_target)
            ee_positions.append(T[:3, 3])
    ee_positions = np.array(ee_positions)
    
    # Also get the human hand positions for comparison
    human_positions = np.array([h[0] for h in human_trajectory])
    
    print("\n5. Trajectory comparison:")
    print(f"   Human hand travel distance: {np.sum(np.linalg.norm(np.diff(human_positions, axis=0), axis=1)):.3f}m")
    print(f"   Robot EE travel distance: {np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)):.3f}m")
    
    # Check final error (how close did robot EE get to target)
    final_ee_pos = ee_positions[-1]
    target_pos = hand_end
    final_error = np.linalg.norm(final_ee_pos - target_pos)
    print(f"   Final position error: {final_error*1000:.1f}mm")
    
    # Save episode
    print("\n6. Saving episode...")
    save_path = episode.save('/home/claude/dynamical_edge/demo_episode.npz')
    print(f"   Saved to: {save_path}")
    
    # Verify we can load it back
    loaded = Episode.load(save_path)
    print(f"   Loaded episode with {len(loaded)} steps")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return episode, ee_positions, human_positions


def plot_trajectories(ee_positions: np.ndarray, human_positions: np.ndarray):
    """
    Create a visualization of the human and robot trajectories.
    
    This helps verify that the retargeting is working correctly:
    the robot EE should follow a similar path to the human hand.
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(*human_positions.T, 'b-', label='Human hand', linewidth=2)
    ax1.plot(*ee_positions.T, 'r--', label='Robot EE', linewidth=2)
    ax1.scatter(*human_positions[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(*human_positions[-1], c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Trajectories')
    
    # Position over time
    ax2 = fig.add_subplot(122)
    t = np.arange(len(human_positions)) * 0.1
    
    ax2.plot(t, human_positions[:, 0], 'b-', label='Human X')
    ax2.plot(t, human_positions[:, 2], 'b--', label='Human Z')
    ax2.plot(t[:len(ee_positions)], ee_positions[:, 0], 'r-', label='Robot X')
    ax2.plot(t[:len(ee_positions)], ee_positions[:, 2], 'r--', label='Robot Z')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.set_title('Position vs Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/dynamical_edge/trajectory_plot.png', dpi=150)
    plt.close()
    print("Saved trajectory plot to trajectory_plot.png")


if __name__ == '__main__':
    episode, ee_positions, human_positions = run_retargeting_demo()
    
    # Uncomment to generate plot (requires matplotlib with backend)
    # plot_trajectories(ee_positions, human_positions)
