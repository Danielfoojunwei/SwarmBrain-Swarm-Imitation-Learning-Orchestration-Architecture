"""
Perception modules for multi-view human pose estimation.

This package contains:
- cameras: ONVIF camera interface and multi-view synchronization
- pose_mmpose: MMPose/RTMW3D wrapper for 3D pose estimation
"""

from .cameras import (
    OnvifCamera,
    MultiViewCameraRig,
    SynchronizedFrameSet,
    CameraCalibration,
    CameraFrame,
    triangulate_point,
    triangulate_points_ransac,
)

from .pose_mmpose import (
    RTMW3DInference,
    MultiViewConfig,
    Detection2D,
    create_pose_estimator,
    MockPoseDetector,
)

__all__ = [
    'OnvifCamera',
    'MultiViewCameraRig',
    'SynchronizedFrameSet',
    'CameraCalibration',
    'CameraFrame',
    'triangulate_point',
    'triangulate_points_ransac',
    'RTMW3DInference',
    'MultiViewConfig',
    'Detection2D',
    'create_pose_estimator',
    'MockPoseDetector',
]
