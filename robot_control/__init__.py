"""
SwarmBrain Robot Control Layer

This module implements the reflex layer for individual humanoid robots.
Each robot runs the Dynamical skill engine locally with role-conditioned
policies derived from imitation learning.

Components:
- skills: Executable skill modules
- policies: Role-conditioned policy networks
- drivers: Hardware interfaces (cameras, gloves, actuators)
- ros2_nodes: ROS 2 nodes for robot control
"""

__version__ = "1.0.0"
