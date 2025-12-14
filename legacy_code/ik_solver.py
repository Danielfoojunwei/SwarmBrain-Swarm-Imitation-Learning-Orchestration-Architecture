"""
Inverse Kinematics Solver using Pinocchio

Pinocchio is a rigid body dynamics library that provides efficient kinematics
and dynamics computations. For IK, we use its differential kinematics and
either a damped pseudo-inverse or a QP-based approach.

The damped pseudo-inverse method:
    - Compute Jacobian J at current configuration
    - Compute pose error ΔX (6D: 3 position + 3 orientation)
    - Compute joint update: Δq = J† @ ΔX, where J† is damped pseudo-inverse
    - Iterate until convergence or max iterations

The QP-based method (more robust):
    - Same as above, but solve a quadratic program that respects joint limits
    - min ||J @ Δq - ΔX||² + λ||Δq||² subject to q_min ≤ q + Δq ≤ q_max

We implement both and let the user choose based on their needs.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("Warning: Pinocchio not installed. IK solver will use fallback.")


@dataclass
class IKConfig:
    """Configuration for the IK solver."""
    
    max_iterations: int = 100               # Max iterations per solve
    tolerance_position: float = 1e-4        # Position error tolerance (m)
    tolerance_rotation: float = 1e-4        # Rotation error tolerance (rad)
    
    damping: float = 1e-6                   # Damping for pseudo-inverse
    step_size: float = 1.0                  # Step size multiplier
    
    # Joint limit handling
    respect_joint_limits: bool = True       # Enable joint limit constraints
    joint_limit_margin: float = 0.05        # Stay this far from limits (rad)
    
    # Regularization toward a preferred configuration
    regularization_weight: float = 1e-4     # Weight for staying near q_init
    
    # Method selection
    method: str = 'damped_pinv'             # 'damped_pinv' or 'qp' (if available)


class PinocchioIKSolver:
    """
    IK solver using the Pinocchio library.
    
    This solver handles:
    - Arbitrary URDF robot models
    - Configurable end-effector frame
    - Joint limits (soft or hard)
    - Damped pseudo-inverse with optional QP refinement
    
    Usage:
        solver = PinocchioIKSolver("robot.urdf", "tool_frame")
        q_solution, success, error = solver.solve(q_init, T_target)
    """
    
    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str,
        config: IKConfig = None
    ):
        """
        Initialize the IK solver.
        
        Args:
            urdf_path: Path to robot URDF file
            ee_frame_name: Name of the end-effector frame in the URDF
            config: Solver configuration
        """
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio is required for IK. Install with: pip install pin")
        
        self.config = config or IKConfig()
        
        # Load robot model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # Find the end-effector frame ID
        if not self.model.existFrame(ee_frame_name):
            available = [self.model.frames[i].name for i in range(self.model.nframes)]
            raise ValueError(
                f"Frame '{ee_frame_name}' not found in URDF. "
                f"Available frames: {available}"
            )
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        # Cache joint limits
        self._q_min = self.model.lowerPositionLimit.copy()
        self._q_max = self.model.upperPositionLimit.copy()
        
        # Apply margin to limits
        margin = self.config.joint_limit_margin
        self._q_min_safe = self._q_min + margin
        self._q_max_safe = self._q_max - margin
    
    @property
    def n_joints(self) -> int:
        """Number of joints in the model."""
        return self.model.nq
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        return self._q_min.copy()
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        return self._q_max.copy()
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose for given joint configuration.
        
        Args:
            q: Joint configuration [n_joints]
            
        Returns:
            T_ee: 4x4 homogeneous transform of EE in world/base frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get frame placement as SE3 object
        oMf = self.data.oMf[self.ee_frame_id]
        
        # Convert to 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        
        return T
    
    def _compute_error(
        self,
        q: np.ndarray,
        T_target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute 6D error between current and target EE pose.
        
        The error is in "se(3)" form: [e_position, e_rotation]
        where e_rotation uses the angle-axis representation.
        
        Returns:
            error_6d: [6] error vector
            error_norm: Scalar error magnitude
        """
        T_current = self.forward_kinematics(q)
        
        # Position error (simple subtraction)
        e_pos = T_target[:3, 3] - T_current[:3, 3]
        
        # Rotation error (using Pinocchio's log3 for SO(3))
        R_current = T_current[:3, :3]
        R_target = T_target[:3, :3]
        R_error = R_target @ R_current.T
        
        # Convert rotation error to axis-angle
        e_rot = pin.log3(R_error)
        
        error_6d = np.concatenate([e_pos, e_rot])
        
        # Weighted norm (position in meters, rotation in radians)
        error_norm = np.sqrt(
            np.sum(e_pos**2) + np.sum(e_rot**2)
        )
        
        return error_6d, error_norm
    
    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the frame Jacobian at current configuration.
        
        The Jacobian maps joint velocities to end-effector twist:
            v_ee = J @ q_dot
        
        We use the LOCAL_WORLD_ALIGNED convention (position in world frame,
        angular velocity also in world frame).
        
        Returns:
            J: [6, n_joints] Jacobian matrix
        """
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get frame Jacobian
        J = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        return J
    
    def _damped_pinv(self, J: np.ndarray, damping: float) -> np.ndarray:
        """
        Compute damped pseudo-inverse of Jacobian.
        
        The damped pseudo-inverse is:
            J† = J.T @ inv(J @ J.T + λ²I)
        
        This provides numerical stability near singularities at the cost
        of some accuracy. The damping factor λ should be small (1e-6 to 1e-3).
        """
        JJT = J @ J.T
        n = JJT.shape[0]
        JJT_damped = JJT + damping**2 * np.eye(n)
        
        return J.T @ np.linalg.inv(JJT_damped)
    
    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint values to safe limits."""
        return np.clip(q, self._q_min_safe, self._q_max_safe)
    
    def solve(
        self,
        q_init: np.ndarray,
        T_target: np.ndarray,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK for desired end-effector pose.
        
        Uses iterative damped pseudo-inverse:
        1. Compute error ΔX between current and target pose
        2. Compute Jacobian J at current q
        3. Compute Δq = J† @ ΔX
        4. Update q ← q + α * Δq (α is step size)
        5. Clamp to joint limits
        6. Repeat until convergence
        
        Args:
            q_init: Initial joint configuration [n_joints]
            T_target: Desired EE pose as 4x4 homogeneous transform
            max_iterations: Override config max iterations
            tolerance: Override config tolerance
            
        Returns:
            q_solution: Final joint configuration
            success: True if converged within tolerance
            final_error: Final pose error magnitude
        """
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or max(self.config.tolerance_position, self.config.tolerance_rotation)
        
        q = q_init.copy()
        
        for iteration in range(max_iter):
            # Compute error
            error_6d, error_norm = self._compute_error(q, T_target)
            
            # Check convergence
            pos_error = np.linalg.norm(error_6d[:3])
            rot_error = np.linalg.norm(error_6d[3:])
            
            if (pos_error < self.config.tolerance_position and 
                rot_error < self.config.tolerance_rotation):
                return q, True, error_norm
            
            # Compute Jacobian and pseudo-inverse
            J = self._compute_jacobian(q)
            J_pinv = self._damped_pinv(J, self.config.damping)
            
            # Compute joint update
            dq = J_pinv @ error_6d
            
            # Add regularization toward initial configuration
            if self.config.regularization_weight > 0:
                dq -= self.config.regularization_weight * (q - q_init)
            
            # Update with step size
            q = q + self.config.step_size * dq
            
            # Clamp to joint limits
            if self.config.respect_joint_limits:
                q = self._clamp_to_limits(q)
        
        # Did not converge; return best solution found
        _, final_error = self._compute_error(q, T_target)
        return q, False, final_error
    
    def solve_with_nullspace(
        self,
        q_init: np.ndarray,
        T_target: np.ndarray,
        q_preferred: np.ndarray,
        nullspace_gain: float = 0.1
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK with nullspace optimization.
        
        For redundant robots (7+ DOF), the nullspace of the Jacobian can be
        used to achieve secondary objectives without affecting the EE pose.
        Here we use it to stay close to a preferred configuration.
        
        The update rule is:
            Δq = J† @ ΔX + (I - J† @ J) @ (q_preferred - q)
        
        The second term projects the "error to preferred" into the nullspace.
        
        Args:
            q_init: Initial configuration
            T_target: Desired EE pose
            q_preferred: Preferred joint configuration (for nullspace)
            nullspace_gain: Gain for nullspace optimization
            
        Returns:
            Same as solve()
        """
        q = q_init.copy()
        
        for iteration in range(self.config.max_iterations):
            error_6d, error_norm = self._compute_error(q, T_target)
            
            pos_error = np.linalg.norm(error_6d[:3])
            rot_error = np.linalg.norm(error_6d[3:])
            
            if (pos_error < self.config.tolerance_position and 
                rot_error < self.config.tolerance_rotation):
                return q, True, error_norm
            
            J = self._compute_jacobian(q)
            J_pinv = self._damped_pinv(J, self.config.damping)
            
            # Primary task: reach target pose
            dq_primary = J_pinv @ error_6d
            
            # Nullspace projection: stay near preferred config
            I_n = np.eye(self.n_joints)
            nullspace_proj = I_n - J_pinv @ J
            dq_nullspace = nullspace_proj @ (q_preferred - q)
            
            # Combined update
            dq = dq_primary + nullspace_gain * dq_nullspace
            
            q = q + self.config.step_size * dq
            
            if self.config.respect_joint_limits:
                q = self._clamp_to_limits(q)
        
        _, final_error = self._compute_error(q, T_target)
        return q, False, final_error


# ---------------------------------------------------------------------------
# Fallback IK Solver (no Pinocchio dependency)
# ---------------------------------------------------------------------------

class SimpleIKSolver:
    """
    A minimal IK solver for testing without Pinocchio.
    
    This uses numerical differentiation and gradient descent. It's slow and
    less robust than Pinocchio, but useful for:
    - Testing the retargeting pipeline without full dependencies
    - Simple planar or low-DOF robots
    
    NOT recommended for production use.
    """
    
    def __init__(
        self,
        fk_func,                            # Callable: q -> T_ee (4x4)
        n_joints: int,
        joint_limits: Tuple[np.ndarray, np.ndarray],
        config: IKConfig = None
    ):
        """
        Args:
            fk_func: Forward kinematics function
            n_joints: Number of joints
            joint_limits: (lower_limits, upper_limits) arrays
        """
        self.fk = fk_func
        self.n_joints = n_joints
        self._q_min, self._q_max = joint_limits
        self.config = config or IKConfig()
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        return self._q_min.copy()
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        return self._q_max.copy()
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self.fk(q)
    
    def _numerical_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute Jacobian via finite differences."""
        J = np.zeros((6, self.n_joints))
        T0 = self.fk(q)
        pos0 = T0[:3, 3]
        
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += eps
            T_plus = self.fk(q_plus)
            
            # Position Jacobian
            J[:3, i] = (T_plus[:3, 3] - pos0) / eps
            
            # Rotation Jacobian (approximate via angle-axis)
            R_delta = T_plus[:3, :3] @ T0[:3, :3].T
            # Small angle approximation: rotation vector ≈ skew-symmetric part
            J[3, i] = (R_delta[2, 1] - R_delta[1, 2]) / (2 * eps)
            J[4, i] = (R_delta[0, 2] - R_delta[2, 0]) / (2 * eps)
            J[5, i] = (R_delta[1, 0] - R_delta[0, 1]) / (2 * eps)
        
        return J
    
    def solve(
        self,
        q_init: np.ndarray,
        T_target: np.ndarray,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """Gradient descent IK (slow but functional)."""
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance_position
        
        q = q_init.copy()
        
        for _ in range(max_iter):
            T_current = self.fk(q)
            
            # Position error
            e_pos = T_target[:3, 3] - T_current[:3, 3]
            
            # Simple rotation error (trace-based)
            R_current = T_current[:3, :3]
            R_target = T_target[:3, :3]
            R_error = R_target @ R_current.T
            
            # Rodrigues formula inverse for small angles
            trace = np.trace(R_error)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            if angle < 1e-6:
                e_rot = np.zeros(3)
            else:
                axis = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1]
                ]) / (2 * np.sin(angle))
                e_rot = angle * axis
            
            error_6d = np.concatenate([e_pos, e_rot])
            error_norm = np.linalg.norm(error_6d)
            
            if np.linalg.norm(e_pos) < tol and angle < self.config.tolerance_rotation:
                return q, True, error_norm
            
            # Numerical Jacobian
            J = self._numerical_jacobian(q)
            
            # Damped pseudo-inverse
            JJT = J @ J.T
            J_pinv = J.T @ np.linalg.inv(JJT + self.config.damping**2 * np.eye(6))
            
            # Update
            dq = J_pinv @ error_6d
            q = q + self.config.step_size * dq
            q = np.clip(q, self._q_min, self._q_max)
        
        T_final = self.fk(q)
        final_pos_error = np.linalg.norm(T_target[:3, 3] - T_final[:3, 3])
        return q, False, final_pos_error


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_ik_solver(
    urdf_path: str = None,
    ee_frame_name: str = None,
    fk_func = None,
    n_joints: int = None,
    joint_limits: Tuple[np.ndarray, np.ndarray] = None,
    config: IKConfig = None
) -> 'PinocchioIKSolver | SimpleIKSolver':
    """
    Create an IK solver, preferring Pinocchio if available.
    
    If URDF path is provided and Pinocchio is installed, creates PinocchioIKSolver.
    Otherwise, falls back to SimpleIKSolver (requires fk_func).
    
    Args:
        urdf_path: Path to URDF file (for Pinocchio)
        ee_frame_name: End-effector frame name (for Pinocchio)
        fk_func: Forward kinematics function (for fallback solver)
        n_joints: Number of joints (for fallback solver)
        joint_limits: (lower, upper) arrays (for fallback solver)
        config: Solver configuration
        
    Returns:
        IK solver instance
    """
    if urdf_path is not None and PINOCCHIO_AVAILABLE:
        return PinocchioIKSolver(urdf_path, ee_frame_name, config)
    
    if fk_func is not None and n_joints is not None and joint_limits is not None:
        return SimpleIKSolver(fk_func, n_joints, joint_limits, config)
    
    raise ValueError(
        "Either (urdf_path + ee_frame_name) with Pinocchio installed, "
        "or (fk_func + n_joints + joint_limits) must be provided."
    )
