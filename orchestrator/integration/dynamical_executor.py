"""
Dynamical Executor Module

Calls Dynamical API to execute skills (both single-actor and multi-actor CSA roles)
on edge robots. Handles skill deployment, execution monitoring, and result aggregation.

Execution Flow:
SwarmBrain Mission Planner → Dynamical Executor → Dynamical API → Edge Robots
"""

import logging
import requests
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SkillExecutionStatus(str, Enum):
    """Skill execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SkillExecutionRequest:
    """Request to execute a skill on a robot"""
    robot_id: str                        # Target robot ID
    skill_id: str                        # Skill ID (CSA:role or single-actor skill)
    role: Optional[str] = None           # Role for multi-actor skills
    parameters: Dict[str, Any] = None    # Skill-specific parameters
    timeout: float = 30.0                # Maximum execution time (seconds)
    coordination_group_id: Optional[str] = None  # For multi-actor coordination


@dataclass
class SkillExecutionResult:
    """Result of skill execution"""
    execution_id: str
    robot_id: str
    skill_id: str
    status: SkillExecutionStatus
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None


class DynamicalExecutor:
    """
    Executes skills via Dynamical API on edge robots
    """

    def __init__(
        self,
        dynamical_api_url: str = "http://localhost:8085",
        default_timeout: float = 30.0,
        poll_interval: float = 0.5,
    ):
        """
        Args:
            dynamical_api_url: URL of Dynamical API
            default_timeout: Default skill execution timeout
            poll_interval: Status polling interval (seconds)
        """
        self.dynamical_api_url = dynamical_api_url.rstrip("/")
        self.default_timeout = default_timeout
        self.poll_interval = poll_interval

        logger.info(f"Dynamical Executor initialized with API={self.dynamical_api_url}")

    def execute_skill(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        """
        Execute a skill on a robot via Dynamical API

        Args:
            request: Skill execution request

        Returns:
            SkillExecutionResult with execution status and result
        """
        try:
            # Prepare execution payload
            payload = {
                "robot_id": request.robot_id,
                "skill_id": request.skill_id,
                "parameters": request.parameters or {},
                "timeout": request.timeout or self.default_timeout,
            }

            # Add role for multi-actor skills
            if request.role:
                payload["role"] = request.role

            # Add coordination group for multi-actor coordination
            if request.coordination_group_id:
                payload["coordination_group_id"] = request.coordination_group_id

            # Submit execution request
            url = f"{self.dynamical_api_url}/api/v1/skills/execute"
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            execution_data = response.json()
            execution_id = execution_data["execution_id"]

            logger.info(
                f"Started skill execution: robot={request.robot_id}, "
                f"skill={request.skill_id}, execution_id={execution_id}"
            )

            # Poll for completion
            result = self._wait_for_completion(
                execution_id=execution_id,
                timeout=request.timeout or self.default_timeout,
            )

            return result

        except requests.RequestException as e:
            logger.error(f"Failed to execute skill {request.skill_id} on {request.robot_id}: {e}")
            return SkillExecutionResult(
                execution_id="",
                robot_id=request.robot_id,
                skill_id=request.skill_id,
                status=SkillExecutionStatus.FAILED,
                success=False,
                start_time=datetime.utcnow(),
                error_message=str(e),
            )

    def execute_multi_actor_skill(
        self,
        csa_skill_id: str,
        robot_role_assignments: Dict[str, str],
        coordination_group_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, SkillExecutionResult]:
        """
        Execute a multi-actor skill with coordinated role assignments

        Args:
            csa_skill_id: CSA skill ID
            robot_role_assignments: {robot_id: role} mapping
            coordination_group_id: Unique ID for this coordination group
            parameters: Shared parameters for all roles
            timeout: Maximum execution time

        Returns:
            Dictionary of {robot_id: SkillExecutionResult}
        """
        results = {}

        # Create execution requests for each robot
        requests_list = [
            SkillExecutionRequest(
                robot_id=robot_id,
                skill_id=f"{csa_skill_id}:{role}",  # CSA skill with role
                role=role,
                parameters=parameters,
                timeout=timeout,
                coordination_group_id=coordination_group_id,
            )
            for robot_id, role in robot_role_assignments.items()
        ]

        # Execute all roles in parallel (Dynamical handles coordination)
        logger.info(
            f"Executing multi-actor skill: {csa_skill_id} with {len(requests_list)} robots "
            f"(coordination_group={coordination_group_id})"
        )

        # Submit all execution requests
        execution_ids = []
        for req in requests_list:
            try:
                payload = {
                    "robot_id": req.robot_id,
                    "skill_id": req.skill_id,
                    "role": req.role,
                    "parameters": req.parameters or {},
                    "timeout": req.timeout,
                    "coordination_group_id": coordination_group_id,
                }

                url = f"{self.dynamical_api_url}/api/v1/skills/execute"
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()

                exec_data = response.json()
                execution_ids.append((req.robot_id, exec_data["execution_id"]))

                logger.info(f"Started execution for robot {req.robot_id} (role={req.role})")

            except requests.RequestException as e:
                logger.error(f"Failed to start execution for robot {req.robot_id}: {e}")
                results[req.robot_id] = SkillExecutionResult(
                    execution_id="",
                    robot_id=req.robot_id,
                    skill_id=req.skill_id,
                    status=SkillExecutionStatus.FAILED,
                    success=False,
                    start_time=datetime.utcnow(),
                    error_message=str(e),
                )

        # Wait for all executions to complete
        for robot_id, execution_id in execution_ids:
            result = self._wait_for_completion(execution_id, timeout)
            results[robot_id] = result

        # Log overall success
        all_success = all(r.success for r in results.values())
        if all_success:
            logger.info(f"✅ Multi-actor skill {csa_skill_id} completed successfully")
        else:
            failed_robots = [rid for rid, r in results.items() if not r.success]
            logger.warning(f"⚠️ Multi-actor skill {csa_skill_id} failed for robots: {failed_robots}")

        return results

    def get_execution_status(self, execution_id: str) -> Optional[SkillExecutionResult]:
        """
        Get current execution status from Dynamical API

        Args:
            execution_id: Execution ID to query

        Returns:
            SkillExecutionResult or None if not found
        """
        try:
            url = f"{self.dynamical_api_url}/api/v1/skills/executions/{execution_id}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()

            return SkillExecutionResult(
                execution_id=data["execution_id"],
                robot_id=data["robot_id"],
                skill_id=data["skill_id"],
                status=SkillExecutionStatus(data["status"]),
                success=data.get("success", False),
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                duration=data.get("duration"),
                error_message=data.get("error_message"),
                result_data=data.get("result_data"),
            )

        except requests.RequestException as e:
            logger.error(f"Failed to get execution status for {execution_id}: {e}")
            return None

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution

        Args:
            execution_id: Execution ID to cancel

        Returns:
            True if cancellation successful
        """
        try:
            url = f"{self.dynamical_api_url}/api/v1/skills/executions/{execution_id}/cancel"
            response = requests.post(url, timeout=5)
            response.raise_for_status()

            logger.info(f"Cancelled execution: {execution_id}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False

    def list_robot_capabilities(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """
        Query robot capabilities from Dynamical API

        Args:
            robot_id: Robot ID

        Returns:
            Dictionary of capabilities or None
        """
        try:
            url = f"{self.dynamical_api_url}/api/v1/robots/{robot_id}/capabilities"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            capabilities = response.json()
            logger.debug(f"Robot {robot_id} capabilities: {capabilities}")
            return capabilities

        except requests.RequestException as e:
            logger.error(f"Failed to get capabilities for robot {robot_id}: {e}")
            return None

    def check_robot_available(self, robot_id: str) -> bool:
        """
        Check if robot is available for execution

        Args:
            robot_id: Robot ID

        Returns:
            True if available
        """
        try:
            url = f"{self.dynamical_api_url}/api/v1/robots/{robot_id}/status"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            status = response.json()
            is_available = status.get("status") == "idle" and status.get("health") == "healthy"

            logger.debug(f"Robot {robot_id} available: {is_available}")
            return is_available

        except requests.RequestException as e:
            logger.warning(f"Failed to check robot {robot_id} availability: {e}")
            return False

    # --- Private Helper Methods ---

    def _wait_for_completion(self, execution_id: str, timeout: float) -> SkillExecutionResult:
        """
        Poll execution status until completion or timeout

        Args:
            execution_id: Execution ID to monitor
            timeout: Maximum wait time

        Returns:
            SkillExecutionResult
        """
        start_time = time.time()
        last_status = None

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Execution {execution_id} timed out after {elapsed:.1f}s")
                return SkillExecutionResult(
                    execution_id=execution_id,
                    robot_id="",
                    skill_id="",
                    status=SkillExecutionStatus.TIMEOUT,
                    success=False,
                    start_time=datetime.utcnow(),
                    duration=elapsed,
                    error_message=f"Execution timed out after {timeout}s",
                )

            # Poll status
            result = self.get_execution_status(execution_id)
            if not result:
                time.sleep(self.poll_interval)
                continue

            # Check if terminal state
            if result.status in {
                SkillExecutionStatus.COMPLETED,
                SkillExecutionStatus.FAILED,
                SkillExecutionStatus.CANCELLED,
            }:
                logger.info(
                    f"Execution {execution_id} {result.status} "
                    f"(duration={result.duration:.2f}s, success={result.success})"
                )
                return result

            # Log progress
            if result.status != last_status:
                logger.info(f"Execution {execution_id} status: {result.status}")
                last_status = result.status

            time.sleep(self.poll_interval)
