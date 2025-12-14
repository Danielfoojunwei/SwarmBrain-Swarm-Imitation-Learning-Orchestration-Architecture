"""
Industrial Data Integration for Mission Orchestrator

Enhances the orchestrator with industrial data context:
- Real-time SCADA sensor data
- MES work order synchronization
- Equipment availability and status
- Production constraints and priorities

The orchestrator uses industrial data to:
1. Automatically create missions from MES work orders
2. Prioritize tasks based on production schedules
3. Avoid equipment conflicts (check SCADA status)
4. Optimize resource allocation based on real-time conditions
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging

from orchestrator.task_planner.mission_orchestrator import (
    MissionOrchestrator,
    WorkOrder,
    Task,
    TaskGraph
)
from industrial_data.streams.data_aggregator import (
    IndustrialDataAggregator,
    AggregatedData
)
from industrial_data.models.isa95_models import (
    ISA95ResourceMapper,
    ProductionRequest,
    Equipment
)


class IndustrialOrchestrator(MissionOrchestrator):
    """
    Extended orchestrator with industrial data integration.

    Inherits from base MissionOrchestrator and adds industrial
    data awareness for manufacturing environments.
    """

    def __init__(self, industrial_aggregator: IndustrialDataAggregator):
        """
        Initialize industrial orchestrator.

        Args:
            industrial_aggregator: Industrial data aggregator instance
        """
        super().__init__()

        self.industrial_data = industrial_aggregator
        self.logger = logging.getLogger(__name__)

        # Register callback for industrial data updates
        self.industrial_data.register_update_callback(
            self._on_industrial_data_update
        )

        # ISA-95 resource mapper
        self.resource_mapper = ISA95ResourceMapper()

        # Equipment registry (ISA-95 Equipment model)
        self.equipment_registry: Dict[str, Equipment] = {}

        # Auto-sync settings
        self.auto_sync_work_orders = True
        self.auto_sync_interval = 60  # seconds

    def _on_industrial_data_update(self, data: AggregatedData):
        """
        Handle industrial data updates.

        Called automatically when SCADA/MES/IoT data changes.

        Args:
            data: Aggregated industrial data
        """
        self.logger.debug(
            f"Industrial data update: "
            f"{len(data.scada_tags)} SCADA tags, "
            f"{len(data.iot_sensors)} IoT sensors, "
            f"{len(data.work_orders)} work orders"
        )

        # Auto-sync work orders if enabled
        if self.auto_sync_work_orders:
            self._sync_work_orders(data.work_orders)

        # Check equipment availability
        self._update_equipment_status(data.scada_tags)

    def _sync_work_orders(self, mes_work_orders: List[Any]):
        """
        Synchronize MES work orders to orchestrator missions.

        Args:
            mes_work_orders: Work orders from MES
        """
        for mes_wo in mes_work_orders:
            # Skip if already synced
            if mes_wo.work_order_id in self.active_missions:
                continue

            # Convert MES work order to orchestrator work order
            work_order = WorkOrder(
                order_id=mes_wo.work_order_id,
                description=f"Produce {mes_wo.quantity} {mes_wo.unit} of {mes_wo.product_name}",
                tasks=self._convert_mes_operations_to_tasks(mes_wo),
                priority=mes_wo.priority,
                deadline=mes_wo.scheduled_end
            )

            # Create mission
            try:
                task_graph = self.create_mission(work_order)

                self.logger.info(
                    f"Auto-synced MES work order {mes_wo.work_order_id} "
                    f"as mission with {len(task_graph.tasks)} tasks"
                )

            except Exception as e:
                self.logger.error(
                    f"Error syncing work order {mes_wo.work_order_id}: {e}"
                )

    def _convert_mes_operations_to_tasks(self, mes_work_order: Any) -> List[Dict[str, Any]]:
        """
        Convert MES operations to orchestrator tasks.

        Maps manufacturing operations to robot skills.

        Args:
            mes_work_order: MES work order with operations

        Returns:
            List of task specifications
        """
        tasks = []

        for i, operation in enumerate(mes_work_order.operations):
            # Map operation to robot skill
            skill = self._map_operation_to_skill(operation.get('operation_name', ''))

            task = {
                'id': f"task_{i+1}",
                'skill': skill,
                'role': 'worker',
                'dependencies': [f"task_{i}"] if i > 0 else [],
                'metadata': {
                    'mes_operation_id': operation.get('operation_id'),
                    'work_center': operation.get('work_center'),
                    'standard_time': operation.get('standard_time'),
                    'resources': operation.get('resources', [])
                }
            }

            tasks.append(task)

        return tasks

    def _map_operation_to_skill(self, operation_name: str) -> str:
        """
        Map MES operation name to robot skill.

        Args:
            operation_name: Name of operation from MES

        Returns:
            Robot skill name
        """
        operation_lower = operation_name.lower()

        # Define mapping rules
        mappings = {
            'pick': 'grasp',
            'grasp': 'grasp',
            'place': 'place',
            'put': 'place',
            'assemble': 'manipulate',
            'attach': 'manipulate',
            'inspect': 'inspect',
            'check': 'inspect',
            'measure': 'inspect',
            'transport': 'navigate',
            'move': 'navigate',
            'deliver': 'navigate',
            'screw': 'manipulate',
            'tighten': 'manipulate',
            'weld': 'manipulate',
            'glue': 'manipulate',
            'paint': 'manipulate',
        }

        for keyword, skill in mappings.items():
            if keyword in operation_lower:
                return skill

        # Default fallback
        return 'generic'

    def _update_equipment_status(self, scada_tags: Dict[str, Any]):
        """
        Update equipment availability based on SCADA data.

        Args:
            scada_tags: Current SCADA tag values
        """
        # Example: Check if equipment is available
        # In real implementation, parse specific tags for equipment status

        for tag_key, tag_value in scada_tags.items():
            # Example: If tag indicates machine fault, mark robot unavailable
            if 'fault' in tag_value.name.lower() and tag_value.value:
                # Extract equipment ID from tag name
                # This is application-specific
                self.logger.warning(
                    f"Equipment fault detected: {tag_value.name}"
                )

    def check_equipment_availability(self, equipment_id: str) -> bool:
        """
        Check if equipment is available for task assignment.

        Uses real-time SCADA data to verify equipment status.

        Args:
            equipment_id: Equipment/robot identifier

        Returns:
            True if equipment is available
        """
        # Check robot registry
        if equipment_id not in self.robot_registry:
            return False

        robot_info = self.robot_registry[equipment_id]

        # Check basic availability
        if robot_info['status'] != 'available':
            return False

        # Check SCADA for equipment faults
        # Example: Check specific SCADA tags for this equipment
        fault_tag = self.industrial_data.get_tag_value(
            'factory_scada',
            f"{equipment_id}_fault"
        )

        if fault_tag and fault_tag.value:
            self.logger.warning(
                f"Equipment {equipment_id} has active fault"
            )
            return False

        # Check battery level for mobile robots
        battery_tag = self.industrial_data.get_tag_value(
            'factory_scada',
            f"{equipment_id}_battery"
        )

        if battery_tag and battery_tag.value < 20.0:
            self.logger.warning(
                f"Equipment {equipment_id} has low battery: {battery_tag.value}%"
            )
            return False

        return True

    async def assign_tasks_with_industrial_context(
        self,
        mission_id: str
    ) -> Dict[str, str]:
        """
        Assign tasks with industrial data context.

        Enhanced version of assign_tasks that considers:
        - Real-time equipment status from SCADA
        - Production priorities from MES
        - Resource availability

        Args:
            mission_id: ID of mission to assign tasks for

        Returns:
            Mapping of task_id to robot_id
        """
        task_graph = self.active_missions.get(mission_id)
        if not task_graph:
            raise ValueError(f'Mission {mission_id} not found')

        ready_tasks = task_graph.get_ready_tasks()
        assignments = {}

        for task in ready_tasks:
            # Find available robots with capability
            available_robots = [
                rid for rid in self.robot_registry.keys()
                if (task.skill in self.robot_registry[rid]['capabilities']
                    and self.check_equipment_availability(rid))
            ]

            if not available_robots:
                self.logger.warning(
                    f"No available robots for task {task.task_id} ({task.skill})"
                )
                continue

            # Prioritize based on work center preference from MES
            preferred_robot = self._select_preferred_robot(
                available_robots,
                task.metadata.get('work_center')
            )

            # Assign task
            task.status = 'assigned'
            task.assigned_robot = preferred_robot
            self.robot_registry[preferred_robot]['status'] = 'busy'
            self.robot_registry[preferred_robot]['current_task'] = task.task_id
            assignments[task.task_id] = preferred_robot

            self.logger.info(
                f'Assigned task {task.task_id} ({task.skill}) to robot {preferred_robot}'
            )

        return assignments

    def _select_preferred_robot(
        self,
        available_robots: List[str],
        preferred_work_center: Optional[str]
    ) -> str:
        """
        Select preferred robot based on work center affinity.

        Args:
            available_robots: List of available robot IDs
            preferred_work_center: Preferred work center from MES

        Returns:
            Selected robot ID
        """
        # If work center specified, prefer robots in that work center
        if preferred_work_center:
            for robot_id in available_robots:
                robot_metadata = self.robot_registry[robot_id].get('metadata', {})
                if robot_metadata.get('work_center') == preferred_work_center:
                    return robot_id

        # Otherwise, return first available
        return available_robots[0]

    def get_production_context(self) -> Dict[str, Any]:
        """
        Get current production context for decision making.

        Returns:
            Dictionary with production metrics and status
        """
        aggregated_data = self.industrial_data.get_aggregated_data()

        return {
            'timestamp': aggregated_data.timestamp.isoformat(),
            'active_work_orders': len(aggregated_data.work_orders),
            'scada_tags_count': len(aggregated_data.scada_tags),
            'iot_sensors_count': len(aggregated_data.iot_sensors),
            'available_robots': sum(
                1 for rid in self.robot_registry.keys()
                if self.check_equipment_availability(rid)
            ),
            'total_robots': len(self.robot_registry)
        }


# Example usage
async def example_industrial_orchestrator():
    """Example of using industrial orchestrator."""

    # Initialize industrial data aggregator
    aggregator = IndustrialDataAggregator()

    # Add data sources
    await aggregator.add_opcua_source(
        source_id="factory_scada",
        endpoint_url="opc.tcp://plc.factory.local:4840",
        tags=[],  # Add your tags
        username="scada_user",
        password="scada_pass"
    )

    aggregator.add_mqtt_source(
        source_id="warehouse_iot",
        broker_host="mqtt.factory.local",
        topics=["factory/+/+/+", "warehouse/+/+/+"]
    )

    aggregator.add_mes_source(
        source_id="production_mes",
        mes_base_url="http://mes.factory.local",
        api_key="your_api_key"
    )

    # Initialize orchestrator
    orchestrator = IndustrialOrchestrator(aggregator)

    # Register robots
    orchestrator.register_robot(
        robot_id="robot_001",
        capabilities=["grasp", "place", "navigate", "inspect"],
        metadata={'work_center': 'WC-100'}
    )

    # Start aggregator
    await aggregator.start()

    # Orchestrator will now automatically sync MES work orders
    # and use real-time SCADA/IoT data for task assignment
