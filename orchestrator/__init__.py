"""
SwarmBrain Mission Orchestrator

This module implements the coordination layer for multi-robot missions.
It transforms work orders into task graphs, assigns roles, and launches
coordination primitives.

Components:
- task_planner: Converts work orders to task graphs
- scheduler: Role assignment and task allocation
- coordination_primitives: Handover, mutex, barrier, rendezvous
"""

__version__ = "1.0.0"
