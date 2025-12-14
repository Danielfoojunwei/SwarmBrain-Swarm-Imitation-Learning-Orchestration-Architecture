"""
Unified Metrics Collection

Collects and exposes metrics from:
- SwarmBrain orchestrator (mission progress, task assignments)
- SwarmBridge FL service (training rounds, aggregation)
- CSA Registry (skill versions, deployments)
- Dynamical API (skill executions, robot status)

Metrics are exposed in Prometheus format for Grafana dashboards.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


# SwarmBrain Orchestrator Metrics
missions_total = Counter(
    "swarm brain_missions_total",
    "Total number of missions created",
    ["priority"],
)

tasks_assigned = Counter(
    "swarm brain_tasks_assigned_total",
    "Total number of tasks assigned to robots",
    ["skill_type"],
)

tasks_completed = Counter(
    "swarm brain_tasks_completed_total",
    "Total number of completed tasks",
    ["skill_type", "status"],
)

active_missions = Gauge(
    "swarmbrain_active_missions",
    "Number of active missions",
)

robot_utilization = Gauge(
    "swarmbrain_robot_utilization",
    "Robot utilization percentage",
    ["robot_id"],
)

# Federated Learning Metrics (from SwarmBridge)
fl_rounds_total = Counter(
    "swarmbrain_fl_rounds_total",
    "Total FL rounds initiated",
    ["learning_mode", "privacy_mode"],
)

fl_round_duration = Histogram(
    "swarmbrain_fl_round_duration_seconds",
    "FL round duration in seconds",
    ["learning_mode"],
)

fl_participants = Gauge(
    "swarmbrain_fl_participants",
    "Number of participants in current FL round",
    ["round_id"],
)

fl_aggregation_loss = Gauge(
    "swarmbrain_fl_aggregation_loss",
    "Aggregated loss from FL round",
    ["round_id"],
)

# Skill Registry Metrics (from CSA Registry + Dynamical)
skills_registered = Gauge(
    "swarmbrain_skills_registered",
    "Number of registered skills",
    ["skill_type"],
)

skill_versions = Gauge(
    "swarmbrain_skill_versions",
    "Number of versions for a skill",
    ["skill_id"],
)

skill_executions = Counter(
    "swarmbrain_skill_executions_total",
    "Total skill executions",
    ["skill_id", "robot_id", "status"],
)

# Edge Device Health Metrics (from Dynamical)
edge_device_health = Gauge(
    "swarmbrain_edge_device_health",
    "Edge device health score (0-1)",
    ["device_id"],
)

edge_device_battery = Gauge(
    "swarmbrain_edge_device_battery",
    "Edge device battery level (0-1)",
    ["device_id"],
)

edge_device_latency = Histogram(
    "swarmbrain_edge_device_latency_ms",
    "Edge device network latency in milliseconds",
    ["device_id"],
)


@dataclass
class MetricsConfig:
    """Configuration for unified metrics collection"""
    swarm_bridge_url: str = "http://localhost:8083"
    csa_registry_url: str = "http://localhost:8082"
    dynamical_api_url: str = "http://localhost:8085"
    collection_interval: int = 30  # seconds


class UnifiedMetricsCollector:
    """
    Collects metrics from all swarm services

    Periodically polls:
    - SwarmBridge FL service
    - CSA Registry
    - Dynamical API
    - SwarmBrain orchestrator

    Exposes unified metrics for Prometheus/Grafana.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.running = False
        logger.info("UnifiedMetricsCollector initialized")

    async def start_collection(self):
        """Start periodic metrics collection"""
        import asyncio

        self.running = True
        logger.info("Starting unified metrics collection")

        while self.running:
            try:
                await self._collect_fl_metrics()
                await self._collect_registry_metrics()
                await self._collect_edge_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            await asyncio.sleep(self.config.collection_interval)

    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("Stopped unified metrics collection")

    async def _collect_fl_metrics(self):
        """Collect FL metrics from SwarmBridge"""
        try:
            response = requests.get(
                f"{self.config.swarm_bridge_url}/api/v1/metrics",
                timeout=5,
            )
            response.raise_for_status()

            metrics = response.json()

            # Update FL metrics
            for round_data in metrics.get("active_rounds", []):
                round_id = round_data["round_id"]
                fl_participants.labels(round_id=round_id).set(
                    len(round_data.get("participants", []))
                )

                if "aggregated_loss" in round_data:
                    fl_aggregation_loss.labels(round_id=round_id).set(
                        round_data["aggregated_loss"]
                    )

        except Exception as e:
            logger.warning(f"Failed to collect FL metrics: {e}")

    async def _collect_registry_metrics(self):
        """Collect skill registry metrics"""
        try:
            # Collect from CSA Registry
            response = requests.get(
                f"{self.config.csa_registry_url}/api/v1/metrics",
                timeout=5,
            )
            response.raise_for_status()

            csa_metrics = response.json()
            skills_registered.labels(skill_type="multi_actor").set(
                csa_metrics.get("total_csas", 0)
            )

            # Collect from Dynamical API
            response = requests.get(
                f"{self.config.dynamical_api_url}/api/v1/metrics",
                timeout=5,
            )
            response.raise_for_status()

            dynamical_metrics = response.json()
            skills_registered.labels(skill_type="single_actor").set(
                dynamical_metrics.get("total_skills", 0)
            )

        except Exception as e:
            logger.warning(f"Failed to collect registry metrics: {e}")

    async def _collect_edge_metrics(self):
        """Collect edge device health metrics from Dynamical"""
        try:
            response = requests.get(
                f"{self.config.dynamical_api_url}/api/v1/devices/health",
                timeout=5,
            )
            response.raise_for_status()

            devices = response.json().get("devices", [])

            for device in devices:
                device_id = device["device_id"]

                edge_device_health.labels(device_id=device_id).set(
                    device.get("health_score", 0)
                )

                edge_device_battery.labels(device_id=device_id).set(
                    device.get("battery_level", 0)
                )

                if "latency_ms" in device:
                    edge_device_latency.labels(device_id=device_id).observe(
                        device["latency_ms"]
                    )

        except Exception as e:
            logger.warning(f"Failed to collect edge metrics: {e}")

    def record_mission_created(self, priority: int):
        """Record a new mission creation"""
        missions_total.labels(priority=str(priority)).inc()

    def record_task_assigned(self, skill_type: str):
        """Record a task assignment"""
        tasks_assigned.labels(skill_type=skill_type).inc()

    def record_task_completed(self, skill_type: str, status: str):
        """Record a task completion"""
        tasks_completed.labels(skill_type=skill_type, status=status).inc()

    def set_active_missions(self, count: int):
        """Update active missions gauge"""
        active_missions.set(count)

    def set_robot_utilization(self, robot_id: str, utilization: float):
        """Update robot utilization"""
        robot_utilization.labels(robot_id=robot_id).set(utilization)

    def record_fl_round_started(self, learning_mode: str, privacy_mode: str):
        """Record FL round initiation"""
        fl_rounds_total.labels(
            learning_mode=learning_mode,
            privacy_mode=privacy_mode,
        ).inc()

    def record_fl_round_completed(self, learning_mode: str, duration_seconds: float):
        """Record FL round completion"""
        fl_round_duration.labels(learning_mode=learning_mode).observe(duration_seconds)

    def record_skill_execution(self, skill_id: str, robot_id: str, status: str):
        """Record skill execution"""
        skill_executions.labels(
            skill_id=skill_id,
            robot_id=robot_id,
            status=status,
        ).inc()
