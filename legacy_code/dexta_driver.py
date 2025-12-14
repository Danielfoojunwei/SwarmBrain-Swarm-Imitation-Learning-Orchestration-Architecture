"""
DextaGlove SDK Driver Module

This module interfaces with the Dexta Robotics glove (Dexmo / DextaGlove) for
high-fidelity hand motion capture. The glove provides:

1. Finger joint angles: 3 joints per finger (MCP, PIP, DIP) × 5 fingers = 15 DOF
2. Finger abduction: 4 inter-finger spread angles
3. Wrist orientation: IMU-based quaternion
4. Haptic feedback: Force feedback to fingertips (output, not captured here)
5. Contact sensing: Fingertip force sensors

Why use a glove instead of camera-based hand tracking?

1. Occlusion immunity: Cameras can't see fingers when they're curled or
   occluded by objects. The glove always knows joint angles.

2. Higher precision: Glove gives direct joint angle measurements at ~100Hz.
   Camera-based solutions estimate from images and are less precise for
   fine finger movements.

3. Contact detection: The glove has force sensors on fingertips, so we know
   exactly when and how hard the demonstrator is grasping.

4. IMU orientation: The wrist IMU gives drift-free orientation (with 
   occasional recalibration), better than estimating wrist pose from images.

The tradeoff is that gloves require the demonstrator to wear hardware,
while camera-based tracking is non-intrusive.

SDK Integration:

DextaGlove communicates via USB or Bluetooth. The SDK provides a C/C++ API
that we wrap with ctypes for Python access. The key functions are:

    DextaInit() -> Initialize the SDK
    DextaOpen(device_id) -> Connect to a specific glove
    DextaGetHandData(device_id, &data) -> Read current hand state
    DextaClose(device_id) -> Disconnect
    DextaShutdown() -> Cleanup

The HandData struct contains all sensor readings.

Frame Conventions:

The glove reports wrist orientation in its own IMU frame. This needs to be
transformed to the world frame for fusion with camera data. We do this via
a calibration step where the user holds their hand in a known pose (e.g.,
pointing forward with palm down) to establish the IMU-to-world transform.
"""

import ctypes
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from pathlib import Path
from enum import IntEnum
import logging
import struct

# Import our hand state dataclass
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.human_state import DexterHandState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK C Structure Definitions
# ---------------------------------------------------------------------------

class FingerID(IntEnum):
    """Finger identifiers."""
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class JointID(IntEnum):
    """Joint identifiers within a finger."""
    MCP = 0  # Metacarpophalangeal (base)
    PIP = 1  # Proximal interphalangeal (middle)
    DIP = 2  # Distal interphalangeal (tip)


# C structures for SDK interop
class Vector3(ctypes.Structure):
    """3D vector."""
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
    ]
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


class Quaternion(ctypes.Structure):
    """Quaternion for orientation (x, y, z, w convention)."""
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('w', ctypes.c_float),
    ]
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)


class FingerData(ctypes.Structure):
    """Data for a single finger."""
    _fields_ = [
        ('joint_angles', ctypes.c_float * 3),  # MCP, PIP, DIP in radians
        ('abduction', ctypes.c_float),          # Spread angle in radians
        ('tip_force', ctypes.c_float),          # Fingertip force in Newtons
        ('contact', ctypes.c_bool),             # Is fingertip in contact
    ]


class HandData(ctypes.Structure):
    """
    Complete hand data from the glove.
    
    This structure is filled by DextaGetHandData().
    """
    _fields_ = [
        ('timestamp', ctypes.c_double),         # Timestamp in seconds
        ('is_valid', ctypes.c_bool),            # Data validity flag
        ('hand_side', ctypes.c_int),            # 0=left, 1=right
        ('fingers', FingerData * 5),            # Data for each finger
        ('wrist_orientation', Quaternion),      # Wrist IMU orientation
        ('wrist_angular_velocity', Vector3),    # Wrist angular velocity
    ]


class DeviceInfo(ctypes.Structure):
    """Information about a connected glove device."""
    _fields_ = [
        ('device_id', ctypes.c_int),
        ('serial_number', ctypes.c_char * 64),
        ('firmware_version', ctypes.c_char * 32),
        ('hand_side', ctypes.c_int),
        ('battery_level', ctypes.c_float),
        ('is_connected', ctypes.c_bool),
    ]


# ---------------------------------------------------------------------------
# SDK Wrapper Class
# ---------------------------------------------------------------------------

class DextaSDKError(Exception):
    """Exception for SDK errors."""
    pass


class DextaSDKNotFoundError(DextaSDKError):
    """Raised when the SDK library cannot be found."""
    pass


class DextaSDK:
    """
    Python wrapper for the DextaGlove C SDK.
    
    This class handles:
    - Loading the native SDK library
    - Device discovery and connection
    - Continuous data capture in a background thread
    - Data conversion to our DexterHandState format
    - Calibration for world frame alignment
    
    Usage:
        sdk = DextaSDK()
        sdk.connect()
        
        while True:
            state = sdk.read_state()
            print(f"Finger closure: {state.get_finger_closure()}")
        
        sdk.disconnect()
    
    Thread Safety:
        The read_state() method is thread-safe and returns the latest
        captured data. The capture loop runs in a background thread.
    """
    
    # SDK library paths to try
    LIB_PATHS = [
        '/usr/lib/libdexta.so',
        '/usr/local/lib/libdexta.so',
        './libdexta.so',
        'C:/Program Files/DextaRobotics/SDK/bin/dexta.dll',
    ]
    
    def __init__(
        self,
        lib_path: Optional[str] = None,
        hand_side: str = 'right',
        capture_rate: float = 100.0,  # Hz
    ):
        """
        Initialize the DextaGlove SDK wrapper.
        
        Args:
            lib_path: Path to SDK library (auto-detect if None)
            hand_side: 'left' or 'right' hand glove
            capture_rate: Target data capture rate in Hz
        """
        self.hand_side = hand_side
        self.capture_rate = capture_rate
        self.capture_interval = 1.0 / capture_rate
        
        # SDK state
        self._lib: Optional[ctypes.CDLL] = None
        self._device_id: Optional[int] = None
        self._is_connected = False
        
        # Capture thread state
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._data_lock = threading.Lock()
        self._latest_data: Optional[HandData] = None
        self._data_ready = threading.Event()
        
        # Calibration
        self._T_world_imu: Optional[np.ndarray] = None
        self._calibration_samples: List[Quaternion] = []
        
        # Load the SDK
        self._load_sdk(lib_path)
    
    def _load_sdk(self, lib_path: Optional[str] = None):
        """Load the native SDK library."""
        paths_to_try = [lib_path] if lib_path else self.LIB_PATHS
        
        for path in paths_to_try:
            if path is None:
                continue
            try:
                self._lib = ctypes.CDLL(path)
                logger.info(f"Loaded DextaGlove SDK from: {path}")
                self._setup_functions()
                return
            except OSError:
                continue
        
        # SDK not found - we'll use mock mode
        logger.warning("DextaGlove SDK not found - using mock mode")
        self._lib = None
    
    def _setup_functions(self):
        """Set up function signatures for the SDK."""
        if self._lib is None:
            return
        
        # DextaInit() -> int (0 = success)
        self._lib.DextaInit.restype = ctypes.c_int
        self._lib.DextaInit.argtypes = []
        
        # DextaShutdown() -> void
        self._lib.DextaShutdown.restype = None
        self._lib.DextaShutdown.argtypes = []
        
        # DextaGetDeviceCount() -> int
        self._lib.DextaGetDeviceCount.restype = ctypes.c_int
        self._lib.DextaGetDeviceCount.argtypes = []
        
        # DextaGetDeviceInfo(device_id, &info) -> int
        self._lib.DextaGetDeviceInfo.restype = ctypes.c_int
        self._lib.DextaGetDeviceInfo.argtypes = [ctypes.c_int, ctypes.POINTER(DeviceInfo)]
        
        # DextaOpen(device_id) -> int
        self._lib.DextaOpen.restype = ctypes.c_int
        self._lib.DextaOpen.argtypes = [ctypes.c_int]
        
        # DextaClose(device_id) -> int
        self._lib.DextaClose.restype = ctypes.c_int
        self._lib.DextaClose.argtypes = [ctypes.c_int]
        
        # DextaGetHandData(device_id, &data) -> int
        self._lib.DextaGetHandData.restype = ctypes.c_int
        self._lib.DextaGetHandData.argtypes = [ctypes.c_int, ctypes.POINTER(HandData)]
    
    def discover_devices(self) -> List[DeviceInfo]:
        """
        Discover connected DextaGlove devices.
        
        Returns:
            List of DeviceInfo for each connected glove
        """
        if self._lib is None:
            # Mock mode: return a fake device
            info = DeviceInfo()
            info.device_id = 0
            info.hand_side = 1 if self.hand_side == 'right' else 0
            info.is_connected = True
            info.battery_level = 0.85
            return [info]
        
        devices = []
        count = self._lib.DextaGetDeviceCount()
        
        for i in range(count):
            info = DeviceInfo()
            result = self._lib.DextaGetDeviceInfo(i, ctypes.byref(info))
            if result == 0:
                devices.append(info)
        
        return devices
    
    def connect(self, device_id: int = 0) -> bool:
        """
        Connect to a DextaGlove device.
        
        Args:
            device_id: Device index (0 for first glove)
            
        Returns:
            True if connection successful
        """
        if self._is_connected:
            return True
        
        if self._lib is None:
            # Mock mode
            self._device_id = device_id
            self._is_connected = True
            self._start_capture_thread()
            logger.info(f"Connected to mock DextaGlove (device {device_id})")
            return True
        
        # Initialize SDK
        result = self._lib.DextaInit()
        if result != 0:
            raise DextaSDKError(f"Failed to initialize SDK: error {result}")
        
        # Open device
        result = self._lib.DextaOpen(device_id)
        if result != 0:
            raise DextaSDKError(f"Failed to open device {device_id}: error {result}")
        
        self._device_id = device_id
        self._is_connected = True
        
        # Start capture thread
        self._start_capture_thread()
        
        logger.info(f"Connected to DextaGlove (device {device_id})")
        return True
    
    def disconnect(self):
        """Disconnect from the glove."""
        if not self._is_connected:
            return
        
        # Stop capture thread
        self._stop_capture_thread()
        
        if self._lib is not None and self._device_id is not None:
            self._lib.DextaClose(self._device_id)
            self._lib.DextaShutdown()
        
        self._is_connected = False
        self._device_id = None
        
        logger.info("Disconnected from DextaGlove")
    
    def _start_capture_thread(self):
        """Start the background data capture thread."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _stop_capture_thread(self):
        """Stop the background capture thread."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
    
    def _capture_loop(self):
        """Background thread that continuously captures glove data."""
        while self._running:
            start_time = time.time()
            
            try:
                if self._lib is not None:
                    # Real SDK
                    data = HandData()
                    result = self._lib.DextaGetHandData(
                        self._device_id, 
                        ctypes.byref(data)
                    )
                    
                    if result == 0 and data.is_valid:
                        with self._data_lock:
                            self._latest_data = data
                        self._data_ready.set()
                else:
                    # Mock mode: generate synthetic data
                    data = self._generate_mock_data()
                    with self._data_lock:
                        self._latest_data = data
                    self._data_ready.set()
                    
            except Exception as e:
                logger.error(f"Glove capture error: {e}")
            
            # Maintain target capture rate
            elapsed = time.time() - start_time
            sleep_time = self.capture_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_mock_data(self) -> HandData:
        """Generate synthetic hand data for testing."""
        data = HandData()
        data.timestamp = time.time()
        data.is_valid = True
        data.hand_side = 1 if self.hand_side == 'right' else 0
        
        # Generate plausible finger data
        # Simulate a slow grasp/release cycle
        t = time.time()
        grasp_phase = (np.sin(t * 0.5) + 1) / 2  # 0 to 1, ~2 second cycle
        
        for i in range(5):
            finger = data.fingers[i]
            
            # Joint angles increase with grasp (thumb is different)
            if i == 0:  # Thumb
                base_angles = [0.3, 0.2, 0.1]
            else:
                base_angles = [0.8, 1.0, 0.6]  # Typical flexion limits
            
            for j in range(3):
                # Add some variation per finger
                finger.joint_angles[j] = base_angles[j] * grasp_phase * (1 + 0.1 * i)
            
            # Abduction (fingers spread)
            finger.abduction = 0.1 * (1 - grasp_phase)
            
            # Contact when grasping
            finger.contact = grasp_phase > 0.7
            finger.tip_force = 2.0 * grasp_phase if finger.contact else 0.0
        
        # Wrist orientation (slight rotation)
        wrist_angle = 0.2 * np.sin(t * 0.3)
        data.wrist_orientation.x = np.sin(wrist_angle / 2)
        data.wrist_orientation.y = 0
        data.wrist_orientation.z = 0
        data.wrist_orientation.w = np.cos(wrist_angle / 2)
        
        return data
    
    def read_state(self) -> Optional[DexterHandState]:
        """
        Read the current hand state.
        
        This returns the latest captured data, converted to our
        DexterHandState format.
        
        Returns:
            DexterHandState if data available, None otherwise
        """
        if not self._is_connected:
            return None
        
        # Wait briefly for data if none available yet
        if self._latest_data is None:
            self._data_ready.wait(timeout=0.1)
        
        with self._data_lock:
            if self._latest_data is None:
                return None
            
            data = self._latest_data
        
        # Convert to DexterHandState
        return self._convert_to_hand_state(data)
    
    def _convert_to_hand_state(self, data: HandData) -> DexterHandState:
        """Convert SDK HandData to our DexterHandState format."""
        # Extract finger angles [5 fingers, 3 joints each]
        finger_angles = np.zeros((5, 3))
        for i in range(5):
            for j in range(3):
                finger_angles[i, j] = data.fingers[i].joint_angles[j]
        
        # Extract abduction angles [4 inter-finger gaps]
        # (between thumb-index, index-middle, middle-ring, ring-pinky)
        finger_abduction = np.zeros(4)
        for i in range(4):
            # Abduction is stored per finger; we want inter-finger
            finger_abduction[i] = (data.fingers[i].abduction + 
                                   data.fingers[i+1].abduction) / 2
        
        # Extract fingertip forces
        fingertip_forces = np.array([
            data.fingers[i].tip_force for i in range(5)
        ])
        
        # Get wrist orientation
        wrist_quat = data.wrist_orientation.to_numpy()
        
        # Create state
        state = DexterHandState(
            timestamp=data.timestamp,
            side=self.hand_side,
            finger_angles=finger_angles,
            finger_abduction=finger_abduction,
            wrist_quat_local=wrist_quat,
            fingertip_forces=fingertip_forces,
            T_world_glove_imu=self._T_world_imu,
        )
        
        return state
    
    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    
    def start_calibration(self):
        """
        Start the calibration procedure.
        
        Calibration establishes the transform from the glove's IMU frame
        to the world frame. The user should hold their hand in a known
        pose (e.g., pointing forward with palm down).
        """
        self._calibration_samples = []
        logger.info("Calibration started. Hold hand in reference pose...")
    
    def add_calibration_sample(self):
        """Add a calibration sample at the current pose."""
        if self._latest_data is None:
            logger.warning("No glove data available for calibration")
            return
        
        with self._data_lock:
            quat = self._latest_data.wrist_orientation
            self._calibration_samples.append(Quaternion(quat.x, quat.y, quat.z, quat.w))
        
        logger.info(f"Added calibration sample ({len(self._calibration_samples)} total)")
    
    def finish_calibration(
        self,
        reference_forward: np.ndarray = np.array([1, 0, 0]),
        reference_up: np.ndarray = np.array([0, 0, 1]),
    ) -> np.ndarray:
        """
        Finish calibration and compute the IMU-to-world transform.
        
        The reference vectors define the expected world-frame orientation
        when the user holds the calibration pose.
        
        Args:
            reference_forward: World-frame direction the hand points toward
            reference_up: World-frame "up" direction (palm normal points opposite)
            
        Returns:
            4x4 transform from IMU frame to world frame
        """
        if len(self._calibration_samples) < 3:
            logger.warning("Need at least 3 calibration samples")
            # Use identity as fallback
            self._T_world_imu = np.eye(4)
            return self._T_world_imu
        
        # Average the quaternion samples
        quats = np.array([
            [q.x, q.y, q.z, q.w] for q in self._calibration_samples
        ])
        
        # Ensure all quaternions are in same hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        
        avg_quat = np.mean(quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        
        # Convert to rotation matrix (this is R_imu_hand: IMU orientation in hand frame)
        from scipy.spatial.transform import Rotation
        R_imu = Rotation.from_quat(avg_quat).as_matrix()
        
        # Build the expected world-frame rotation
        # Hand frame: X=forward (fingers), Y=left, Z=up (palm normal)
        z_world = reference_up / np.linalg.norm(reference_up)
        x_world = reference_forward / np.linalg.norm(reference_forward)
        y_world = np.cross(z_world, x_world)
        y_world = y_world / np.linalg.norm(y_world)
        x_world = np.cross(y_world, z_world)  # Reorthogonalize
        
        R_world_hand = np.column_stack([x_world, y_world, z_world])
        
        # Compute R_world_imu = R_world_hand @ R_hand_imu
        # R_hand_imu = inv(R_imu_hand) = R_imu.T
        R_world_imu = R_world_hand @ R_imu.T
        
        # Build 4x4 transform (no translation since IMU is at wrist)
        self._T_world_imu = np.eye(4)
        self._T_world_imu[:3, :3] = R_world_imu
        
        logger.info("Calibration complete")
        logger.info(f"T_world_imu:\n{self._T_world_imu}")
        
        return self._T_world_imu
    
    def set_calibration(self, T_world_imu: np.ndarray):
        """Set the calibration transform directly (e.g., loaded from file)."""
        self._T_world_imu = T_world_imu.copy()
    
    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been performed."""
        return self._T_world_imu is not None
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    # -----------------------------------------------------------------------
    # Context Manager
    # -----------------------------------------------------------------------
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ---------------------------------------------------------------------------
# High-Level Glove Manager
# ---------------------------------------------------------------------------

class GloveManager:
    """
    High-level manager for one or two DextaGloves (bimanual capture).
    
    This class simplifies working with paired gloves for bimanual tasks.
    """
    
    def __init__(self):
        self.right_glove: Optional[DextaSDK] = None
        self.left_glove: Optional[DextaSDK] = None
    
    def connect_right(self, device_id: int = 0) -> bool:
        """Connect to the right hand glove."""
        self.right_glove = DextaSDK(hand_side='right')
        return self.right_glove.connect(device_id)
    
    def connect_left(self, device_id: int = 1) -> bool:
        """Connect to the left hand glove."""
        self.left_glove = DextaSDK(hand_side='left')
        return self.left_glove.connect(device_id)
    
    def connect_both(self) -> Tuple[bool, bool]:
        """Connect to both gloves."""
        right_ok = self.connect_right(0)
        left_ok = self.connect_left(1)
        return right_ok, left_ok
    
    def disconnect(self):
        """Disconnect all gloves."""
        if self.right_glove:
            self.right_glove.disconnect()
        if self.left_glove:
            self.left_glove.disconnect()
    
    def read_both(self) -> Tuple[Optional[DexterHandState], Optional[DexterHandState]]:
        """
        Read state from both gloves.
        
        Returns:
            (right_state, left_state) tuple
        """
        right = self.right_glove.read_state() if self.right_glove else None
        left = self.left_glove.read_state() if self.left_glove else None
        return right, left
    
    def calibrate_both(self):
        """Run calibration for both gloves."""
        if self.right_glove:
            logger.info("Calibrating right glove...")
            self.right_glove.start_calibration()
            for _ in range(10):
                time.sleep(0.1)
                self.right_glove.add_calibration_sample()
            self.right_glove.finish_calibration()
        
        if self.left_glove:
            logger.info("Calibrating left glove...")
            self.left_glove.start_calibration()
            for _ in range(10):
                time.sleep(0.1)
                self.left_glove.add_calibration_sample()
            self.left_glove.finish_calibration()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def visualize_hand_state(state: DexterHandState) -> str:
    """
    Create a text visualization of hand state.
    
    Useful for debugging without graphics.
    """
    lines = [
        f"Hand: {state.side.upper()}",
        f"Timestamp: {state.timestamp:.3f}",
        "",
        "Finger Closure:",
    ]
    
    closure = state.get_finger_closure()
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    for i, name in enumerate(finger_names):
        bar_len = int(closure[i] * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        lines.append(f"  {name:6s}: [{bar}] {closure[i]:.2f}")
    
    lines.append("")
    lines.append(f"Grasp aperture: {state.get_grasp_aperture():.2f}")
    lines.append(f"Has contact: {state.has_contact()}")
    
    if state.fingertip_forces is not None:
        lines.append("")
        lines.append("Fingertip forces (N):")
        for i, name in enumerate(finger_names):
            lines.append(f"  {name:6s}: {state.fingertip_forces[i]:.1f}")
    
    return '\n'.join(lines)
