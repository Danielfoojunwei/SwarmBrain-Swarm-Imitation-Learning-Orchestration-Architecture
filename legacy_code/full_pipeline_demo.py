"""
Full Perception Pipeline Integration

This script demonstrates the complete data flow from hardware sensors through
to the retargeter. It serves as both documentation and a testbed for the
perception pipeline.

Data Flow Architecture:
=======================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         SENSOR LAYER                                    │
    │                                                                         │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐         ┌─────────────┐    │
    │   │ Camera  │    │ Camera  │    │ Camera  │         │ DextaGlove  │    │
    │   │  Front  │    │  Left   │    │  Right  │         │    SDK      │    │
    │   └────┬────┘    └────┬────┘    └────┬────┘         └──────┬──────┘    │
    │        │              │              │                      │           │
    │        ▼              ▼              ▼                      ▼           │
    │   ┌─────────────────────────────────────────┐         ┌───────────┐    │
    │   │      MultiViewCameraRig                 │         │ read_state│    │
    │   │      - Synchronized frame capture       │         └─────┬─────┘    │
    │   │      - Timestamp alignment              │               │          │
    │   └─────────────────┬───────────────────────┘               │          │
    │                     │                                       │          │
    └─────────────────────┼───────────────────────────────────────┼──────────┘
                          │                                       │
                          ▼                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         PERCEPTION LAYER                                │
    │                                                                         │
    │   ┌─────────────────────────────────────────┐         ┌───────────┐    │
    │   │      RTMW3DInference                    │         │DexterHand │    │
    │   │      - 2D detection per view            │         │  State    │    │
    │   │      - Cross-view triangulation         │         └─────┬─────┘    │
    │   │      - Temporal filtering               │               │          │
    │   └─────────────────┬───────────────────────┘               │          │
    │                     │                                       │          │
    │                     ▼                                       │          │
    │               ┌───────────┐                                 │          │
    │               │Human3DState                                 │          │
    │               │(body pose)│                                 │          │
    │               └─────┬─────┘                                 │          │
    │                     │                                       │          │
    │                     └──────────────┬────────────────────────┘          │
    │                                    │                                    │
    │                                    ▼                                    │
    │                         ┌────────────────────┐                          │
    │                         │ fuse_to_human_state│                          │
    │                         │  - Align glove to  │                          │
    │                         │    camera wrist    │                          │
    │                         │  - Detect objects  │                          │
    │                         └─────────┬──────────┘                          │
    │                                   │                                     │
    └───────────────────────────────────┼─────────────────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   HumanState    │
                               │ (unified repr)  │
                               └────────┬────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         RETARGETING LAYER                               │
    │                                                                         │
    │                    ┌──────────────────────────┐                         │
    │                    │       Retargeter         │                         │
    │                    │  - Object-centric xform  │                         │
    │                    │  - IK solving            │                         │
    │                    │  - Gripper mapping       │                         │
    │                    └───────────┬──────────────┘                         │
    │                                │                                        │
    │                                ▼                                        │
    │                         ┌───────────┐                                   │
    │                         │ DemoStep  │                                   │
    │                         └───────────┘                                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘


Usage:
======

For testing with mock sensors (no hardware):
    python full_pipeline_demo.py --mock

For real hardware:
    python full_pipeline_demo.py \\
        --camera rtsp://user:pass@192.168.1.100/stream1 front \\
        --camera rtsp://user:pass@192.168.1.101/stream1 left \\
        --camera rtsp://user:pass@192.168.1.102/stream1 right \\
        --calibration ./calibration/ \\
        --glove right

The calibration directory should contain:
    front_calibration.json
    left_calibration.json
    right_calibration.json

Each calibration file is generated by the camera_calibrate.py script using
a ChArUco board.
"""

import numpy as np
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.cameras import (
    OnvifCamera, MultiViewCameraRig, CameraCalibration,
    SynchronizedFrameSet
)
from perception.pose_mmpose import (
    RTMW3DInference, MultiViewConfig, create_pose_estimator, MockPoseDetector
)
from glove.dexta_driver import DextaSDK, GloveManager, visualize_hand_state
from core.human_state import (
    Human3DState, DexterHandState, HumanState, EnvObject,
    fuse_to_human_state
)
from core.retargeting import Retargeter, RetargetConfig
from core.recorder import RobotObs, Episode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the full perception pipeline."""
    
    # Camera settings
    camera_urls: Dict[str, str] = None  # camera_id -> rtsp_url
    calibration_dir: Optional[Path] = None
    sync_tolerance: float = 0.05  # Max frame time difference (50ms)
    
    # Pose estimation settings
    pose_model_path: Optional[str] = None
    pose_model_type: str = 'mock'  # 'mock', 'onnx', 'tensorrt'
    min_views: int = 2
    
    # Glove settings
    use_glove: bool = True
    glove_hand: str = 'right'
    
    # Object detection (placeholder - would use YOLO/SAM in production)
    object_positions: Dict[str, np.ndarray] = None  # object_id -> position
    
    # Capture settings
    target_fps: float = 15.0
    
    def __post_init__(self):
        if self.camera_urls is None:
            self.camera_urls = {}
        if self.object_positions is None:
            self.object_positions = {}


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class PerceptionPipeline:
    """
    Orchestrates the full perception pipeline from sensors to HumanState.
    
    This class manages:
    1. Multi-view camera capture and synchronization
    2. 3D human pose estimation via RTMW
    3. DextaGlove hand tracking
    4. Fusion into unified HumanState
    
    The output HumanState can be directly fed to the Retargeter.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the perception pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Components (initialized in setup())
        self.camera_rig: Optional[MultiViewCameraRig] = None
        self.pose_estimator: Optional[RTMW3DInference] = None
        self.glove: Optional[DextaSDK] = None
        
        # State
        self._is_running = False
        self._frame_count = 0
        self._timing_history: List[float] = []
    
    def setup(self):
        """
        Initialize all hardware connections and models.
        
        This should be called once before starting capture.
        """
        logger.info("Setting up perception pipeline...")
        
        # Set up cameras
        self._setup_cameras()
        
        # Set up pose estimator
        self._setup_pose_estimator()
        
        # Set up glove
        if self.config.use_glove:
            self._setup_glove()
        
        logger.info("Perception pipeline setup complete")
    
    def _setup_cameras(self):
        """Initialize the multi-view camera rig."""
        logger.info("Setting up cameras...")
        
        self.camera_rig = MultiViewCameraRig(
            sync_tolerance=self.config.sync_tolerance
        )
        
        # Add each camera
        for camera_id, rtsp_url in self.config.camera_urls.items():
            logger.info(f"  Adding camera '{camera_id}': {rtsp_url[:50]}...")
            self.camera_rig.add_camera(camera_id, rtsp_url)
        
        # Load calibrations if available
        if self.config.calibration_dir:
            calib_path = Path(self.config.calibration_dir)
            if calib_path.exists():
                self.camera_rig.load_calibrations(calib_path)
            else:
                logger.warning(f"Calibration directory not found: {calib_path}")
                # Create default calibrations for testing
                for camera_id in self.config.camera_urls:
                    calib = CameraCalibration.create_default(camera_id)
                    self.camera_rig.set_calibration(camera_id, calib)
        else:
            # Create default calibrations for mock mode
            for camera_id in self.config.camera_urls:
                calib = CameraCalibration.create_default(camera_id)
                self.camera_rig.set_calibration(camera_id, calib)
        
        # Start cameras
        self.camera_rig.start()
        
        logger.info(f"  {self.camera_rig.num_cameras} cameras started")
    
    def _setup_pose_estimator(self):
        """Initialize the 3D pose estimator."""
        logger.info("Setting up pose estimator...")
        
        pose_config = MultiViewConfig(
            min_views_for_triangulation=self.config.min_views,
            enable_temporal_filter=True,
            temporal_smoothing_factor=0.3,
        )
        
        self.pose_estimator = create_pose_estimator(
            camera_rig=self.camera_rig,
            model_path=self.config.pose_model_path,
            model_type=self.config.pose_model_type,
            config=pose_config,
        )
        
        logger.info(f"  Pose estimator ready (type: {self.config.pose_model_type})")
    
    def _setup_glove(self):
        """Initialize the DextaGlove."""
        logger.info("Setting up DextaGlove...")
        
        self.glove = DextaSDK(hand_side=self.config.glove_hand)
        
        try:
            self.glove.connect()
            logger.info(f"  Glove connected ({self.config.glove_hand} hand)")
            
            # Run calibration
            logger.info("  Starting glove calibration...")
            logger.info("  >>> Hold hand pointing FORWARD, palm DOWN <<<")
            time.sleep(1.0)
            
            self.glove.start_calibration()
            for i in range(10):
                time.sleep(0.1)
                self.glove.add_calibration_sample()
            self.glove.finish_calibration()
            
            logger.info("  Glove calibration complete")
            
        except Exception as e:
            logger.warning(f"  Glove connection failed: {e}")
            logger.info("  Continuing without glove (will use camera-only tracking)")
            self.glove = None
    
    def capture_human_state(self) -> Optional[HumanState]:
        """
        Capture and fuse a single HumanState from all sensors.
        
        This is the main entry point called in the capture loop.
        
        Returns:
            Fused HumanState, or None if capture failed
        """
        start_time = time.time()
        
        # Get 3D body pose from cameras
        body_state = self.pose_estimator.infer_once()
        
        if body_state is None:
            logger.debug("No body pose detected")
            return None
        
        # Get hand state from glove
        hand_right = None
        hand_left = None
        
        if self.glove is not None:
            hand_state = self.glove.read_state()
            if hand_state is not None:
                if self.config.glove_hand == 'right':
                    hand_right = hand_state
                else:
                    hand_left = hand_state
        
        # Detect objects (placeholder - would use YOLO/SAM)
        objects = self._detect_objects()
        
        # Fuse into HumanState
        human_state = fuse_to_human_state(
            body_state,
            hand_right,
            hand_left,
            objects,
        )
        
        # Track timing
        elapsed = time.time() - start_time
        self._timing_history.append(elapsed)
        if len(self._timing_history) > 100:
            self._timing_history.pop(0)
        
        self._frame_count += 1
        
        return human_state
    
    def _detect_objects(self) -> List[EnvObject]:
        """
        Detect objects in the scene.
        
        This is a placeholder that returns configured static objects.
        In production, this would:
        1. Run YOLO/SAM on camera images
        2. Triangulate object positions from multiple views
        3. Track objects over time
        """
        objects = []
        
        for obj_id, position in self.config.object_positions.items():
            pose = np.eye(4)
            pose[:3, 3] = position
            
            obj = EnvObject(
                object_id=obj_id,
                class_id=1,
                class_name='object',
                pose_world=pose,
            )
            objects.append(obj)
        
        return objects
    
    def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down perception pipeline...")
        
        if self.camera_rig:
            self.camera_rig.stop()
        
        if self.glove:
            self.glove.disconnect()
        
        logger.info("Perception pipeline shutdown complete")
    
    @property
    def fps(self) -> float:
        """Get average capture FPS."""
        if not self._timing_history:
            return 0.0
        mean_time = np.mean(self._timing_history)
        return 1.0 / mean_time if mean_time > 0 else 0.0
    
    def __enter__(self):
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# ---------------------------------------------------------------------------
# Mock Cameras for Testing
# ---------------------------------------------------------------------------

class MockMultiViewCameraRig:
    """
    Mock camera rig that generates synthetic synchronized frames.
    
    This allows testing the full pipeline without real cameras.
    """
    
    def __init__(self, camera_ids: List[str], sync_tolerance: float = 0.05):
        self.camera_ids = camera_ids
        self.sync_tolerance = sync_tolerance
        self.calibrations: Dict[str, CameraCalibration] = {}
        
        # Generate default calibrations for a 3-camera setup
        self._generate_calibrations()
    
    def _generate_calibrations(self):
        """
        Generate calibrations for a realistic 3-camera setup.
        
        Cameras are positioned in an arc around the capture volume:
        - Front: (0, 2, 1.5) looking at origin
        - Left:  (-2, 1, 1.5) looking at origin
        - Right: (2, 1, 1.5) looking at origin
        """
        positions = {
            'front': np.array([0, 2, 1.5]),
            'left': np.array([-2, 1, 1.5]),
            'right': np.array([2, 1, 1.5]),
        }
        
        target = np.array([0, 0, 1.0])  # Look at this point
        
        for cam_id in self.camera_ids:
            pos = positions.get(cam_id, np.array([0, 2, 1.5]))
            
            # Compute camera orientation (look at target)
            z_axis = target - pos
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            up = np.array([0, 0, 1])
            x_axis = np.cross(up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            y_axis = np.cross(z_axis, x_axis)
            
            R = np.column_stack([x_axis, y_axis, z_axis]).T
            
            calib = CameraCalibration.create_default(cam_id)
            calib.R = R
            calib.t = pos
            calib.__post_init__()  # Recompute derived quantities
            
            self.calibrations[cam_id] = calib
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get_synchronized_frames(self) -> SynchronizedFrameSet:
        """Generate synthetic synchronized frames."""
        from perception.cameras import CameraFrame
        
        frames = {}
        timestamp = time.time()
        
        for cam_id in self.camera_ids:
            # Generate a blank frame (pose detector handles this)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            frames[cam_id] = CameraFrame(
                image=image,
                timestamp=timestamp,
                camera_id=cam_id,
                frame_number=0,
            )
        
        return SynchronizedFrameSet(
            frames=frames,
            reference_timestamp=timestamp,
            max_time_delta=0.0,
        )
    
    def get_projection_matrices(self) -> Dict[str, np.ndarray]:
        return {cid: c.projection_matrix for cid, c in self.calibrations.items()}
    
    @property
    def num_cameras(self) -> int:
        return len(self.camera_ids)
    
    def set_calibration(self, camera_id: str, calib: CameraCalibration):
        self.calibrations[camera_id] = calib


def create_mock_config() -> PipelineConfig:
    """Create a configuration for mock/testing mode."""
    return PipelineConfig(
        camera_urls={
            'front': 'mock://front',
            'left': 'mock://left',
            'right': 'mock://right',
        },
        pose_model_type='mock',
        use_glove=True,
        glove_hand='right',
        object_positions={
            'target_object': np.array([0.5, 0.0, 0.3]),
        },
        target_fps=15.0,
    )


# ---------------------------------------------------------------------------
# Demo Main Loop
# ---------------------------------------------------------------------------

def run_pipeline_demo(config: PipelineConfig, duration: float = 10.0):
    """
    Run the perception pipeline demo.
    
    Args:
        config: Pipeline configuration
        duration: How long to run in seconds
    """
    logger.info("=" * 60)
    logger.info("Perception Pipeline Demo")
    logger.info("=" * 60)
    
    # Handle mock mode specially
    use_mock_cameras = all(url.startswith('mock://') for url in config.camera_urls.values())
    
    if use_mock_cameras:
        logger.info("Running in MOCK mode (no real hardware)")
    
    # Set up pipeline
    pipeline = PerceptionPipeline(config)
    
    # Override camera rig with mock if needed
    if use_mock_cameras:
        pipeline.camera_rig = MockMultiViewCameraRig(list(config.camera_urls.keys()))
        pipeline.camera_rig.start()
        
        # Set up mock pose estimator
        from perception.pose_mmpose import MockPoseDetector
        detector = MockPoseDetector({'num_keypoints': 17})
        pipeline.pose_estimator = RTMW3DInference(
            pipeline.camera_rig,
            detector,
            MultiViewConfig(),
        )
        
        # Set up mock glove
        pipeline.glove = DextaSDK(hand_side=config.glove_hand)
        pipeline.glove.connect()
        
    else:
        pipeline.setup()
    
    # Run capture loop
    logger.info(f"\nCapturing for {duration} seconds...")
    logger.info("-" * 60)
    
    frame_interval = 1.0 / config.target_fps
    start_time = time.time()
    captured_states = []
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Capture human state
            human_state = pipeline.capture_human_state()
            
            if human_state is not None:
                captured_states.append(human_state)
                
                # Print status every 15 frames
                if len(captured_states) % 15 == 0:
                    # Get key information
                    wrist_pos = human_state.body.wrist_right_position
                    
                    grasp = 0.0
                    if human_state.hand_right is not None:
                        grasp = human_state.hand_right.get_grasp_aperture()
                    
                    logger.info(
                        f"Frame {len(captured_states):4d} | "
                        f"FPS: {pipeline.fps:5.1f} | "
                        f"Wrist: [{wrist_pos[0]:5.2f}, {wrist_pos[1]:5.2f}, {wrist_pos[2]:5.2f}] | "
                        f"Grasp: {grasp:.2f}"
                    )
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    finally:
        pipeline.shutdown()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete")
    logger.info("=" * 60)
    logger.info(f"Total frames captured: {len(captured_states)}")
    logger.info(f"Average FPS: {pipeline.fps:.1f}")
    
    return captured_states


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Full Perception Pipeline Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mock', action='store_true',
        help='Run in mock mode without real hardware'
    )
    
    parser.add_argument(
        '--camera', nargs=2, action='append', metavar=('URL', 'ID'),
        help='Add camera: --camera rtsp://... front'
    )
    
    parser.add_argument(
        '--calibration', type=Path,
        help='Path to calibration directory'
    )
    
    parser.add_argument(
        '--glove', choices=['left', 'right', 'none'], default='right',
        help='Which hand has the glove'
    )
    
    parser.add_argument(
        '--model', type=str,
        help='Path to pose estimation model (ONNX or TensorRT)'
    )
    
    parser.add_argument(
        '--model-type', choices=['mock', 'onnx', 'tensorrt'], default='mock',
        help='Type of pose estimation model'
    )
    
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='Demo duration in seconds'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    if args.mock:
        config = create_mock_config()
    else:
        camera_urls = {}
        if args.camera:
            for url, cam_id in args.camera:
                camera_urls[cam_id] = url
        
        if not camera_urls:
            logger.error("No cameras specified. Use --mock for testing or --camera to add cameras.")
            return
        
        config = PipelineConfig(
            camera_urls=camera_urls,
            calibration_dir=args.calibration,
            pose_model_path=args.model,
            pose_model_type=args.model_type,
            use_glove=(args.glove != 'none'),
            glove_hand=args.glove if args.glove != 'none' else 'right',
        )
    
    # Run demo
    run_pipeline_demo(config, args.duration)


if __name__ == '__main__':
    main()
