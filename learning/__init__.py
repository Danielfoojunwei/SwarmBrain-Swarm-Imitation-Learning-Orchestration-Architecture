"""
SwarmBrain Learning Layer

This module implements cross-site improvement through federated and swarm learning.
No raw teleoperation data leaves the site - only encrypted model updates are shared.

Components:
- federated_client: Client for participating in federated learning rounds
- secure_aggregation: Dropout-resilient secure aggregation of model updates
- reputation: zkRep-style reputation system using zero-knowledge proofs
- clustering: Dynamic user clustering by task, environment, and network quality
- scheduling: Joint device scheduling and bandwidth allocation
"""

__version__ = "1.0.0"
