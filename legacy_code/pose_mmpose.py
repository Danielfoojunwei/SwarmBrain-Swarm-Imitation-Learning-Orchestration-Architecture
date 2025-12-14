"""
MMPose / RTMW3D Inference Module

This module wraps MMPose's RTMW (Real-Time Multi-person Wholebody) model for
3D human pose estimation from multiple camera views.

The Pipeline:

1. Run 2D pose detection on each camera view independently
   - RTMW detects body + hands + face keypoints in 2D pixel coordinates
   - Also outputs confidence scores per keypoint

2. Associate detections across views
   - If multiple people are visible, we need to match the same person across cameras
   - Use epipolar geometry constraints or appearance features

3. Triangulate to 3D
   - For each keypoint, triangulate 2D detections from multiple views
   - Use RANSAC to handle outlier detections

4. Filter and smooth
   - Apply temporal filtering to reduce jitter
   - Enforce skeletal constraints (bone lengths, joint limits)

Model Options:

- RTMW3D: RTMPose trained for wholebody (133 keypoints including hands/face)
- RTMPose-X: Larger model, more accurate, slower
- RTMPose-M/L: Medium/Large models, good accuracy/speed tradeoff
- RTMPose-S/T: Small/Tiny models for real-time on edge devices

For Jetson Orin deployment, we'll use TensorRT for inference acceleration.
The model should be exported to ONNX, then compiled to TensorRT engine.

COCO-WholeBody Keypoint Layout (133 keypoints):
    Body: 17 keypoints (COCO format)
    Feet: 6 keypoints
    Face: 68 keypoints  
    Left hand: 21 keypoints
    Right hand: 21 keypoints

For retargeting, we primarily use the body keypoints. Hand keypoints from
the camera are supplementary - the DextaGlove provides higher-fidelity
hand pose data.
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import logging

# Conditional imports for ML frameworks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .cameras import (
    MultiViewCameraRig, SynchronizedFrameSet, CameraCalibration,
    triangulate_points_ransac
)

# Import Human3DState from core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.human_state import Human3DState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keypoint Definitions
# ---------------------------------------------------------------------------

# COCO-WholeBody 133 keypoints
COCO_WHOLEBODY_KEYPOINTS = {
    # Body (0-16)
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
    
    # Feet (17-22)
    'left_big_toe': 17, 'left_small_toe': 18, 'left_heel': 19,
    'right_big_toe': 20, 'right_small_toe': 21, 'right_heel': 22,
    
    # Face starts at 23 (68 keypoints)
    # Left hand starts at 91 (21 keypoints)
    # Right hand starts at 112 (21 keypoints)
}

# Simplified body-only layout (17 keypoints for our Human3DState)
BODY_KEYPOINTS = {
    'pelvis': 0,  # We'll compute this as midpoint of hips
    'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
    'left_hip': 4, 'left_knee': 5, 'left_ankle': 6,
    'spine': 7,  # Computed as midpoint between pelvis and chest
    'chest': 8,  # Computed from shoulders
    'neck': 9,   # Midpoint of shoulders
    'head': 10,  # Nose or midpoint of ears
    'left_shoulder': 11, 'left_elbow': 12, 'left_wrist': 13,
    'right_shoulder': 14, 'right_elbow': 15, 'right_wrist': 16,
}

# Mapping from COCO-WholeBody indices to our body keypoint indices
COCO_TO_BODY_MAPPING = {
    # Direct mappings
    5: 11,   # left_shoulder
    6: 14,   # right_shoulder  
    7: 12,   # left_elbow
    8: 15,   # right_elbow
    9: 13,   # left_wrist
    10: 16,  # right_wrist
    11: 4,   # left_hip
    12: 1,   # right_hip
    13: 5,   # left_knee
    14: 2,   # right_knee
    15: 6,   # left_ankle
    16: 3,   # right_ankle
}

# Skeleton connectivity for visualization
SKELETON_CONNECTIONS = [
    # Torso
    (0, 7), (7, 8), (8, 9), (9, 10),  # Pelvis -> spine -> chest -> neck -> head
    # Left arm
    (8, 11), (11, 12), (12, 13),  # Chest -> l_shoulder -> l_elbow -> l_wrist
    # Right arm
    (8, 14), (14, 15), (15, 16),  # Chest -> r_shoulder -> r_elbow -> r_wrist
    # Left leg
    (0, 4), (4, 5), (5, 6),  # Pelvis -> l_hip -> l_knee -> l_ankle
    # Right leg
    (0, 1), (1, 2), (2, 3),  # Pelvis -> r_hip -> r_knee -> r_ankle
]


# ---------------------------------------------------------------------------
# 2D Pose Detection Result
# ---------------------------------------------------------------------------

@dataclass
class Detection2D:
    """
    A 2D pose detection from a single camera view.
    
    Contains the pixel coordinates and confidence scores for all detected
    keypoints for a single person.
    """
    camera_id: str
    person_id: int                      # 0-indexed person within this frame
    
    keypoints: np.ndarray               # [N_keypoints, 2] pixel coordinates
    confidence: np.ndarray              # [N_keypoints] confidence scores
    bbox: np.ndarray                    # [4] bounding box [x1, y1, x2, y2]
    bbox_score: float                   # Detection confidence for this person
    
    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)
    
    def get_keypoint(self, idx: int) -> Tuple[np.ndarray, float]:
        """Get keypoint position and confidence by index."""
        return self.keypoints[idx], self.confidence[idx]
    
    def get_visible_keypoints(self, threshold: float = 0.3) -> np.ndarray:
        """Get indices of keypoints with confidence above threshold."""
        return np.where(self.confidence > threshold)[0]


# ---------------------------------------------------------------------------
# MMPose Inference Backends
# ---------------------------------------------------------------------------

class PoseDetectorBase:
    """Base class for 2D pose detectors."""
    
    def __init__(self, model_config: dict):
        self.config = model_config
        self.input_size = model_config.get('input_size', (256, 192))  # H, W
        self.num_keypoints = model_config.get('num_keypoints', 133)
    
    def preprocess(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image [H, W, 3]
            bbox: Optional bounding box to crop [x1, y1, x2, y2]
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if bbox is not None:
            # Crop to bounding box
            x1, y1, x2, y2 = map(int, bbox)
            image = image[y1:y2, x1:x2]
        
        # Resize to model input size
        h, w = self.input_size
        image_resized = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_rgb / 255.0 - mean) / std
        
        # Transpose to CHW and add batch dimension
        image_chw = image_normalized.transpose(2, 0, 1).astype(np.float32)
        
        return image_chw[np.newaxis, ...]  # [1, 3, H, W]
    
    def postprocess(
        self,
        heatmaps: np.ndarray,
        original_size: Tuple[int, int],
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert model output heatmaps to keypoint coordinates.
        
        Args:
            heatmaps: [1, num_keypoints, hm_h, hm_w] heatmap output
            original_size: (height, width) of original image
            bbox: Bounding box used for cropping
            
        Returns:
            keypoints: [num_keypoints, 2] pixel coordinates in original image
            confidence: [num_keypoints] confidence scores
        """
        heatmaps = heatmaps[0]  # Remove batch dimension
        num_kpts, hm_h, hm_w = heatmaps.shape
        
        keypoints = np.zeros((num_kpts, 2))
        confidence = np.zeros(num_kpts)
        
        for i in range(num_kpts):
            hm = heatmaps[i]
            
            # Find peak
            flat_idx = np.argmax(hm)
            y, x = np.unravel_index(flat_idx, (hm_h, hm_w))
            
            # Subpixel refinement (quadratic interpolation)
            if 0 < x < hm_w - 1 and 0 < y < hm_h - 1:
                dx = (hm[y, x+1] - hm[y, x-1]) / (2 * (2*hm[y, x] - hm[y, x-1] - hm[y, x+1]) + 1e-6)
                dy = (hm[y+1, x] - hm[y-1, x]) / (2 * (2*hm[y, x] - hm[y-1, x] - hm[y+1, x]) + 1e-6)
                x = x + np.clip(dx, -0.5, 0.5)
                y = y + np.clip(dy, -0.5, 0.5)
            
            # Scale to input size
            x = x / hm_w * self.input_size[1]
            y = y / hm_h * self.input_size[0]
            
            # Scale to original image or bbox coordinates
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x = x / self.input_size[1] * (x2 - x1) + x1
                y = y / self.input_size[0] * (y2 - y1) + y1
            else:
                orig_h, orig_w = original_size
                x = x / self.input_size[1] * orig_w
                y = y / self.input_size[0] * orig_h
            
            keypoints[i] = [x, y]
            confidence[i] = hm[int(round(y * hm_h / self.input_size[0])), 
                              int(round(x * hm_w / self.input_size[1]))]
        
        return keypoints, confidence
    
    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """
        Detect poses in an image.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class ONNXPoseDetector(PoseDetectorBase):
    """
    Pose detector using ONNX Runtime.
    
    This is the most portable backend, working on any platform with ONNX Runtime.
    For Jetson, use the TensorRT execution provider for GPU acceleration.
    """
    
    def __init__(self, model_path: str, model_config: dict):
        super().__init__(model_config)
        
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        
        # Select execution providers (TensorRT > CUDA > CPU)
        providers = []
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        logger.info(f"Loaded ONNX model: {model_path}")
        logger.info(f"  Providers: {self.session.get_providers()}")
    
    def detect(
        self,
        image: np.ndarray,
        bboxes: Optional[List[np.ndarray]] = None,
        camera_id: str = "unknown"
    ) -> List[Detection2D]:
        """
        Detect poses in an image.
        
        Args:
            image: BGR image [H, W, 3]
            bboxes: Optional list of person bounding boxes from detector
            camera_id: Camera identifier for output
            
        Returns:
            List of Detection2D for each detected person
        """
        orig_h, orig_w = image.shape[:2]
        detections = []
        
        if bboxes is None:
            # Full image (single person mode)
            bboxes = [np.array([0, 0, orig_w, orig_h])]
        
        for person_id, bbox in enumerate(bboxes):
            # Preprocess
            input_tensor = self.preprocess(image, bbox)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            heatmaps = outputs[0]  # Assume first output is heatmaps
            
            # Postprocess
            keypoints, confidence = self.postprocess(heatmaps, (orig_h, orig_w), bbox)
            
            detections.append(Detection2D(
                camera_id=camera_id,
                person_id=person_id,
                keypoints=keypoints,
                confidence=confidence,
                bbox=bbox,
                bbox_score=1.0,  # Would come from person detector
            ))
        
        return detections


class TensorRTPoseDetector(PoseDetectorBase):
    """
    Pose detector using TensorRT for maximum performance on NVIDIA GPUs.
    
    This is the recommended backend for Jetson Orin deployment.
    The model must be pre-compiled to a TensorRT engine file.
    """
    
    def __init__(self, engine_path: str, model_config: dict):
        super().__init__(model_config)
        
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not installed")
        
        # Load TensorRT engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        logger.info(f"TensorRT engine loaded successfully")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for input/output."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def detect(
        self,
        image: np.ndarray,
        bboxes: Optional[List[np.ndarray]] = None,
        camera_id: str = "unknown"
    ) -> List[Detection2D]:
        """Detect poses using TensorRT inference."""
        orig_h, orig_w = image.shape[:2]
        detections = []
        
        if bboxes is None:
            bboxes = [np.array([0, 0, orig_w, orig_h])]
        
        for person_id, bbox in enumerate(bboxes):
            # Preprocess
            input_tensor = self.preprocess(image, bbox)
            
            # Copy input to device
            np.copyto(self.inputs[0]['host'], input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]['device'],
                self.inputs[0]['host'],
                self.stream
            )
            
            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            # Copy output to host
            cuda.memcpy_dtoh_async(
                self.outputs[0]['host'],
                self.outputs[0]['device'],
                self.stream
            )
            self.stream.synchronize()
            
            # Reshape output
            heatmaps = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
            
            # Postprocess
            keypoints, confidence = self.postprocess(heatmaps, (orig_h, orig_w), bbox)
            
            detections.append(Detection2D(
                camera_id=camera_id,
                person_id=person_id,
                keypoints=keypoints,
                confidence=confidence,
                bbox=bbox,
                bbox_score=1.0,
            ))
        
        return detections


class MockPoseDetector(PoseDetectorBase):
    """
    Mock pose detector for testing without a real model.
    
    Generates synthetic detections that are geometrically consistent -
    given camera calibrations, the 2D points will triangulate to valid
    3D positions.
    """
    
    def __init__(self, model_config: dict = None):
        super().__init__(model_config or {'num_keypoints': 17})
        
        # Store the "true" 3D skeleton that we project into each view
        self._skeleton_3d = self._create_default_skeleton()
        self._animation_time = 0.0
    
    def _create_default_skeleton(self) -> np.ndarray:
        """
        Create a default 17-keypoint skeleton in 3D.
        
        This represents a person standing at the origin, facing +Y,
        with arms slightly raised.
        """
        # Keypoint positions for a standing person
        # Body is centered at origin, facing +Y direction
        skeleton = np.array([
            [0.0, 0.0, 1.6],      # 0: nose
            [-0.03, 0.02, 1.62],  # 1: left_eye
            [0.03, 0.02, 1.62],   # 2: right_eye
            [-0.08, 0.0, 1.6],    # 3: left_ear
            [0.08, 0.0, 1.6],     # 4: right_ear
            [-0.2, 0.0, 1.4],     # 5: left_shoulder
            [0.2, 0.0, 1.4],      # 6: right_shoulder
            [-0.35, 0.1, 1.2],    # 7: left_elbow
            [0.35, 0.1, 1.2],     # 8: right_elbow
            [-0.4, 0.2, 1.0],     # 9: left_wrist
            [0.4, 0.2, 1.0],      # 10: right_wrist
            [-0.1, 0.0, 1.0],     # 11: left_hip
            [0.1, 0.0, 1.0],      # 12: right_hip
            [-0.1, 0.0, 0.5],     # 13: left_knee
            [0.1, 0.0, 0.5],      # 14: right_knee
            [-0.1, 0.0, 0.0],     # 15: left_ankle
            [0.1, 0.0, 0.0],      # 16: right_ankle
        ])
        
        return skeleton
    
    def _animate_skeleton(self) -> np.ndarray:
        """
        Apply simple animation to the skeleton (arm reaching motion).
        """
        skeleton = self._skeleton_3d.copy()
        
        # Animate right arm in a reaching motion
        t = self._animation_time
        reach_amount = (np.sin(t * 0.5) + 1) / 2  # 0 to 1
        
        # Move right wrist forward and up
        skeleton[10, 1] = 0.2 + reach_amount * 0.3  # Forward
        skeleton[10, 2] = 1.0 + reach_amount * 0.3  # Up
        
        # Adjust elbow accordingly
        skeleton[8, 1] = 0.1 + reach_amount * 0.15
        skeleton[8, 2] = 1.2 + reach_amount * 0.1
        
        self._animation_time += 0.1
        
        return skeleton
    
    def detect(
        self,
        image: np.ndarray,
        bboxes: Optional[List[np.ndarray]] = None,
        camera_id: str = "unknown",
        calibration: Optional['CameraCalibration'] = None,
    ) -> List[Detection2D]:
        """
        Generate mock detections by projecting 3D skeleton into camera view.
        
        If calibration is provided, we project the true 3D points to get
        geometrically consistent 2D detections.
        """
        orig_h, orig_w = image.shape[:2]
        
        # Get animated skeleton
        skeleton_3d = self._animate_skeleton()
        
        if calibration is not None:
            # Project 3D points to 2D using camera calibration
            keypoints = np.zeros((17, 2))
            
            for i in range(17):
                pt_2d = calibration.project_point(skeleton_3d[i])
                keypoints[i] = pt_2d
            
            # Add small noise
            keypoints += np.random.randn(17, 2) * 2.0
            
        else:
            # No calibration - use fake 2D positions
            keypoints = self._generate_mock_skeleton(orig_w, orig_h)
        
        # High confidence for all keypoints
        confidence = np.ones(17) * 0.9 + np.random.rand(17) * 0.1
        
        return [Detection2D(
            camera_id=camera_id,
            person_id=0,
            keypoints=keypoints,
            confidence=confidence,
            bbox=np.array([orig_w*0.2, orig_h*0.1, orig_w*0.8, orig_h*0.9]),
            bbox_score=0.95,
        )]
    
    def _generate_mock_skeleton(self, w: int, h: int) -> np.ndarray:
        """Generate a plausible 17-keypoint skeleton (fallback without calibration)."""
        cx, cy = w / 2, h / 2
        
        keypoints = np.array([
            [cx, cy - h*0.35],         # 0: nose
            [cx - w*0.02, cy - h*0.37], # 1: left_eye
            [cx + w*0.02, cy - h*0.37], # 2: right_eye
            [cx - w*0.05, cy - h*0.35], # 3: left_ear
            [cx + w*0.05, cy - h*0.35], # 4: right_ear
            [cx - w*0.12, cy - h*0.20], # 5: left_shoulder
            [cx + w*0.12, cy - h*0.20], # 6: right_shoulder
            [cx - w*0.18, cy - h*0.05], # 7: left_elbow
            [cx + w*0.18, cy - h*0.05], # 8: right_elbow
            [cx - w*0.22, cy + h*0.10], # 9: left_wrist
            [cx + w*0.22, cy + h*0.10], # 10: right_wrist
            [cx - w*0.08, cy + h*0.05], # 11: left_hip
            [cx + w*0.08, cy + h*0.05], # 12: right_hip
            [cx - w*0.08, cy + h*0.25], # 13: left_knee
            [cx + w*0.08, cy + h*0.25], # 14: right_knee
            [cx - w*0.08, cy + h*0.42], # 15: left_ankle
            [cx + w*0.08, cy + h*0.42], # 16: right_ankle
        ])
        
        noise = np.random.randn(*keypoints.shape) * 3
        return keypoints + noise


# ---------------------------------------------------------------------------
# Multi-View 3D Pose Estimator
# ---------------------------------------------------------------------------

@dataclass
class MultiViewConfig:
    """Configuration for multi-view pose estimation."""
    
    # Triangulation parameters
    min_views_for_triangulation: int = 2    # Minimum cameras seeing a keypoint
    max_reprojection_error: float = 15.0    # Pixels
    ransac_iterations: int = 20
    
    # Temporal filtering
    enable_temporal_filter: bool = True
    temporal_smoothing_factor: float = 0.3  # EMA alpha
    
    # Confidence thresholds
    keypoint_confidence_threshold: float = 0.3
    
    # Person association
    max_epipolar_distance: float = 20.0     # Pixels for cross-view matching


class RTMW3DInference:
    """
    Multi-view 3D human pose estimator using RTMW (or any 2D pose detector).
    
    This class orchestrates:
    1. 2D pose detection on each camera view
    2. Cross-view person association
    3. 3D triangulation of keypoints
    4. Temporal filtering for smooth output
    
    Usage:
        rig = MultiViewCameraRig()
        rig.add_camera("cam1", "rtsp://...")
        rig.add_camera("cam2", "rtsp://...")
        rig.add_camera("cam3", "rtsp://...")
        rig.start()
        
        pose_estimator = RTMW3DInference(rig, detector, config)
        
        while True:
            human_state = pose_estimator.infer_once()
            if human_state:
                print(human_state.keypoints_3d)
    """
    
    def __init__(
        self,
        camera_rig: MultiViewCameraRig,
        detector: PoseDetectorBase,
        config: MultiViewConfig = None,
    ):
        """
        Initialize the 3D pose estimator.
        
        Args:
            camera_rig: Configured multi-view camera rig
            detector: 2D pose detector (shared across views)
            config: Estimation parameters
        """
        self.rig = camera_rig
        self.detector = detector
        self.config = config or MultiViewConfig()
        
        # State for temporal filtering
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_confidence: Optional[np.ndarray] = None
        
        # Performance tracking
        self._inference_times: List[float] = []
    
    def _detect_all_views(
        self,
        frames: SynchronizedFrameSet
    ) -> Dict[str, List[Detection2D]]:
        """
        Run 2D pose detection on all camera views.
        
        Returns:
            Dictionary mapping camera_id -> list of detections
        """
        all_detections = {}
        
        for camera_id in frames.camera_ids:
            frame = frames.get_frame(camera_id)
            if frame is None:
                continue
            
            detections = self.detector.detect(frame.image, camera_id=camera_id)
            all_detections[camera_id] = detections
        
        return all_detections
    
    def _associate_persons_across_views(
        self,
        detections: Dict[str, List[Detection2D]],
    ) -> List[Dict[str, Detection2D]]:
        """
        Match person detections across camera views.
        
        For single-person scenarios (our primary use case), this is simple:
        just take the first detection from each view.
        
        For multi-person scenarios, we'd use:
        - Epipolar geometry constraints
        - Appearance features (ReID)
        - Temporal continuity
        
        Returns:
            List of person associations, each is a dict of camera_id -> Detection2D
        """
        # Simple single-person case: take first detection from each view
        # TODO: Implement proper multi-person association
        
        person_views = {}
        
        for camera_id, dets in detections.items():
            if dets:
                person_views[camera_id] = dets[0]  # First person only
        
        if not person_views:
            return []
        
        return [person_views]
    
    def _triangulate_person(
        self,
        person_views: Dict[str, Detection2D],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate all keypoints for a person from multiple views.
        
        Args:
            person_views: camera_id -> Detection2D for this person
            
        Returns:
            keypoints_3d: [17, 3] array of 3D keypoint positions
            confidence: [17] array of confidence scores
        """
        projection_matrices = self.rig.get_projection_matrices()
        
        # Get calibrations for undistortion
        calibrations = self.rig.calibrations
        
        num_body_keypoints = 17
        keypoints_3d = np.zeros((num_body_keypoints, 3))
        confidence = np.zeros(num_body_keypoints)
        
        # Map from COCO indices to our body indices
        coco_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        for coco_idx in coco_body_indices:
            # Get body keypoint index
            body_idx = COCO_TO_BODY_MAPPING.get(coco_idx, None)
            
            # Collect 2D points from all views
            points_2d = {}
            conf_scores = {}
            
            for camera_id, det in person_views.items():
                if coco_idx >= det.num_keypoints:
                    continue
                
                kpt, conf = det.get_keypoint(coco_idx)
                
                if conf < self.config.keypoint_confidence_threshold:
                    continue
                
                # Undistort the point
                if camera_id in calibrations:
                    kpt = calibrations[camera_id].undistort_points(kpt)
                
                points_2d[camera_id] = kpt
                conf_scores[camera_id] = conf
            
            # Triangulate if we have enough views
            if len(points_2d) >= self.config.min_views_for_triangulation:
                point_3d, error, inliers = triangulate_points_ransac(
                    points_2d,
                    projection_matrices,
                    conf_scores,
                    ransac_iterations=self.config.ransac_iterations,
                    inlier_threshold=self.config.max_reprojection_error,
                )
                
                if point_3d is not None and body_idx is not None:
                    keypoints_3d[body_idx] = point_3d
                    # Confidence is mean of inlier confidences
                    inlier_confs = [conf_scores[cid] for cid, is_in in inliers.items() if is_in]
                    confidence[body_idx] = np.mean(inlier_confs) if inlier_confs else 0.0
        
        # Compute derived keypoints (pelvis, spine, chest, neck, head)
        self._compute_derived_keypoints(keypoints_3d, confidence)
        
        return keypoints_3d, confidence
    
    def _compute_derived_keypoints(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ):
        """
        Compute derived keypoints that aren't directly detected.
        
        Pelvis (0): midpoint of hips
        Spine (7): midpoint of pelvis and chest
        Chest (8): midpoint of shoulders
        Neck (9): between chest and head
        Head (10): from nose or ear midpoint
        """
        l_hip = keypoints[4]
        r_hip = keypoints[1]
        l_shoulder = keypoints[11]
        r_shoulder = keypoints[14]
        
        # Pelvis: midpoint of hips
        if confidence[4] > 0 and confidence[1] > 0:
            keypoints[0] = (l_hip + r_hip) / 2
            confidence[0] = (confidence[4] + confidence[1]) / 2
        
        # Chest: midpoint of shoulders
        if confidence[11] > 0 and confidence[14] > 0:
            keypoints[8] = (l_shoulder + r_shoulder) / 2
            confidence[8] = (confidence[11] + confidence[14]) / 2
        
        # Spine: between pelvis and chest
        if confidence[0] > 0 and confidence[8] > 0:
            keypoints[7] = (keypoints[0] + keypoints[8]) / 2
            confidence[7] = (confidence[0] + confidence[8]) / 2
        
        # Neck: above chest
        if confidence[8] > 0:
            # Estimate neck as 15% of torso height above chest
            torso_vec = keypoints[8] - keypoints[0] if confidence[0] > 0 else np.array([0, 0, 0.3])
            keypoints[9] = keypoints[8] + 0.15 * torso_vec
            confidence[9] = confidence[8] * 0.9
        
        # Head: above neck (we don't have direct nose detection in our 17-point model)
        if confidence[9] > 0:
            keypoints[10] = keypoints[9] + np.array([0, 0, 0.2])
            confidence[10] = confidence[9] * 0.8
    
    def _temporal_filter(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal smoothing to reduce jitter.
        
        Uses exponential moving average (EMA) with confidence-weighted blending.
        """
        if not self.config.enable_temporal_filter or self._prev_keypoints is None:
            self._prev_keypoints = keypoints.copy()
            self._prev_confidence = confidence.copy()
            return keypoints, confidence
        
        alpha = self.config.temporal_smoothing_factor
        
        # Blend with previous frame, weighted by confidence
        for i in range(len(keypoints)):
            if confidence[i] > 0:
                if self._prev_confidence[i] > 0:
                    # Both frames have valid keypoint: smooth
                    keypoints[i] = alpha * keypoints[i] + (1 - alpha) * self._prev_keypoints[i]
                # else: just use current (no previous)
            else:
                if self._prev_confidence[i] > 0:
                    # Current frame missing, use previous with decay
                    keypoints[i] = self._prev_keypoints[i]
                    confidence[i] = self._prev_confidence[i] * 0.9
        
        self._prev_keypoints = keypoints.copy()
        self._prev_confidence = confidence.copy()
        
        return keypoints, confidence
    
    def infer_once(self, person_id: int = 0) -> Optional[Human3DState]:
        """
        Run one inference cycle and return 3D human pose.
        
        This is the main entry point for the perception pipeline.
        
        Args:
            person_id: Which person to track (0 for primary subject)
            
        Returns:
            Human3DState with 3D keypoints, or None if detection failed
        """
        start_time = time.time()
        
        # Get synchronized frames from all cameras
        frames = self.rig.get_synchronized_frames()
        
        if frames is None or frames.num_cameras < 2:
            logger.debug("Insufficient camera frames for 3D estimation")
            return None
        
        # Run 2D detection on all views
        detections = self._detect_all_views(frames)
        
        if not detections:
            logger.debug("No 2D detections in any view")
            return None
        
        # Associate persons across views
        person_associations = self._associate_persons_across_views(detections)
        
        if not person_associations:
            logger.debug("No cross-view person associations")
            return None
        
        # Get requested person (or first if not found)
        if person_id >= len(person_associations):
            person_id = 0
        
        person_views = person_associations[person_id]
        
        # Triangulate to 3D
        keypoints_3d, confidence = self._triangulate_person(person_views)
        
        # Apply temporal filtering
        keypoints_3d, confidence = self._temporal_filter(keypoints_3d, confidence)
        
        # Check if we got enough valid keypoints
        valid_count = np.sum(confidence > 0.1)
        if valid_count < 5:
            logger.debug(f"Too few valid keypoints: {valid_count}")
            return None
        
        # Track timing
        elapsed = time.time() - start_time
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        
        return Human3DState(
            timestamp=frames.reference_timestamp,
            keypoints_3d=keypoints_3d,
            keypoint_confidence=confidence,
        )
    
    def attach_camera(self, camera):
        """Legacy API compatibility - cameras are managed by the rig."""
        pass
    
    @property
    def fps(self) -> float:
        """Get average inference FPS."""
        if not self._inference_times:
            return 0.0
        mean_time = np.mean(self._inference_times)
        return 1.0 / mean_time if mean_time > 0 else 0.0


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_pose_estimator(
    camera_rig: MultiViewCameraRig,
    model_path: Optional[str] = None,
    model_type: str = 'mock',  # 'onnx', 'tensorrt', 'mock'
    config: MultiViewConfig = None,
) -> RTMW3DInference:
    """
    Create a configured 3D pose estimator.
    
    Args:
        camera_rig: Multi-view camera rig
        model_path: Path to model file (ONNX or TensorRT engine)
        model_type: Type of detector to use
        config: Estimation configuration
        
    Returns:
        Configured RTMW3DInference instance
    """
    model_config = {
        'input_size': (256, 192),
        'num_keypoints': 17,  # COCO body keypoints
    }
    
    if model_type == 'mock':
        detector = MockPoseDetector(model_config)
    elif model_type == 'onnx':
        if model_path is None:
            raise ValueError("model_path required for ONNX detector")
        detector = ONNXPoseDetector(model_path, model_config)
    elif model_type == 'tensorrt':
        if model_path is None:
            raise ValueError("model_path required for TensorRT detector")
        detector = TensorRTPoseDetector(model_path, model_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return RTMW3DInference(camera_rig, detector, config)
