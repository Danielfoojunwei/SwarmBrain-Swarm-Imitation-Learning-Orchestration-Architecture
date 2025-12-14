"""
MES (Manufacturing Execution System) Data Collector

Integrates with MES systems to synchronize:
- Work orders and production schedules
- Material tracking and inventory
- Quality data and inspections
- Equipment status and maintenance
- Production KPIs and metrics

Supports common MES platforms:
- SAP MES/MII
- Siemens Opcenter
- Rockwell FactoryTalk
- Dassault DELMIA
- Custom REST/SOAP APIs
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import requests
import logging


class WorkOrderStatus(Enum):
    """Work order status states."""
    CREATED = "created"
    RELEASED = "released"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class MaterialStatus(Enum):
    """Material status states."""
    AVAILABLE = "available"
    RESERVED = "reserved"
    IN_USE = "in_use"
    CONSUMED = "consumed"
    SCRAPPED = "scrapped"


@dataclass
class WorkOrder:
    """
    Manufacturing work order from MES.

    Follows ISA-95 work definition model.
    """
    work_order_id: str
    product_id: str
    product_name: str
    quantity: int
    unit: str
    status: WorkOrderStatus
    priority: int
    scheduled_start: datetime
    scheduled_end: datetime
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    production_line: Optional[str] = None
    operations: List[Dict[str, Any]] = field(default_factory=list)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionOperation:
    """A single operation within a work order."""
    operation_id: str
    operation_number: int
    operation_name: str
    work_center: str
    standard_time: float  # minutes
    actual_time: Optional[float] = None
    status: str = "pending"
    resources: List[str] = field(default_factory=list)
    instructions: str = ""


@dataclass
class Material:
    """Material/component used in production."""
    material_id: str
    material_name: str
    quantity: float
    unit: str
    status: MaterialStatus
    lot_number: Optional[str] = None
    location: Optional[str] = None
    expiry_date: Optional[datetime] = None


@dataclass
class QualityInspection:
    """Quality inspection result."""
    inspection_id: str
    work_order_id: str
    timestamp: datetime
    inspector: str
    result: str  # "pass", "fail", "conditional"
    measurements: Dict[str, float]
    defects: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ProductionMetrics:
    """Production KPIs and metrics."""
    timestamp: datetime
    production_line: str
    units_produced: int
    units_scrapped: int
    oee: float  # Overall Equipment Effectiveness (0-100%)
    availability: float  # 0-100%
    performance: float  # 0-100%
    quality: float  # 0-100%
    downtime_minutes: float
    cycle_time_avg: float  # minutes
    throughput_rate: float  # units/hour


class MESCollector:
    """
    Collector for Manufacturing Execution System data.

    Polls MES for work orders, production data, and quality metrics.
    """

    def __init__(
        self,
        mes_base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        poll_interval: int = 60  # seconds
    ):
        """
        Initialize MES collector.

        Args:
            mes_base_url: Base URL of MES REST API
            api_key: API key for authentication
            username: Username for basic auth
            password: Password for basic auth
            poll_interval: How often to poll MES (seconds)
        """
        self.mes_base_url = mes_base_url.rstrip('/')
        self.poll_interval = poll_interval

        self.session = requests.Session()

        # Set authentication
        if api_key:
            self.session.headers['X-API-Key'] = api_key
        elif username and password:
            self.session.auth = (username, password)

        self.logger = logging.getLogger(__name__)

        # Cache
        self.work_orders: Dict[str, WorkOrder] = {}
        self.materials: Dict[str, Material] = {}

    def get_work_orders(
        self,
        status: Optional[WorkOrderStatus] = None,
        production_line: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[WorkOrder]:
        """
        Retrieve work orders from MES.

        Args:
            status: Filter by work order status
            production_line: Filter by production line
            start_date: Filter by scheduled start date
            end_date: Filter by scheduled end date

        Returns:
            List of work orders
        """
        try:
            params = {}
            if status:
                params['status'] = status.value
            if production_line:
                params['production_line'] = production_line
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()

            response = self.session.get(
                f"{self.mes_base_url}/api/v1/work_orders",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            work_orders = []
            for wo_data in data.get('work_orders', []):
                work_order = WorkOrder(
                    work_order_id=wo_data['id'],
                    product_id=wo_data['product_id'],
                    product_name=wo_data['product_name'],
                    quantity=wo_data['quantity'],
                    unit=wo_data['unit'],
                    status=WorkOrderStatus(wo_data['status']),
                    priority=wo_data.get('priority', 1),
                    scheduled_start=datetime.fromisoformat(wo_data['scheduled_start']),
                    scheduled_end=datetime.fromisoformat(wo_data['scheduled_end']),
                    actual_start=datetime.fromisoformat(wo_data['actual_start']) if wo_data.get('actual_start') else None,
                    actual_end=datetime.fromisoformat(wo_data['actual_end']) if wo_data.get('actual_end') else None,
                    production_line=wo_data.get('production_line'),
                    operations=wo_data.get('operations', []),
                    materials=wo_data.get('materials', []),
                    metadata=wo_data.get('metadata', {})
                )

                work_orders.append(work_order)
                self.work_orders[work_order.work_order_id] = work_order

            self.logger.info(f"Retrieved {len(work_orders)} work orders from MES")
            return work_orders

        except requests.RequestException as e:
            self.logger.error(f"Error retrieving work orders from MES: {e}")
            raise

    def update_work_order_status(
        self,
        work_order_id: str,
        status: WorkOrderStatus,
        actual_quantity: Optional[int] = None
    ) -> bool:
        """
        Update work order status in MES.

        Args:
            work_order_id: Work order ID
            status: New status
            actual_quantity: Actual quantity produced

        Returns:
            True if update successful
        """
        try:
            payload = {
                'status': status.value,
                'timestamp': datetime.utcnow().isoformat()
            }

            if actual_quantity is not None:
                payload['actual_quantity'] = actual_quantity

            response = self.session.put(
                f"{self.mes_base_url}/api/v1/work_orders/{work_order_id}/status",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            self.logger.info(f"Updated work order {work_order_id} status to {status.value}")
            return True

        except requests.RequestException as e:
            self.logger.error(f"Error updating work order status: {e}")
            return False

    def get_materials(
        self,
        material_ids: Optional[List[str]] = None,
        location: Optional[str] = None
    ) -> List[Material]:
        """
        Retrieve material inventory from MES.

        Args:
            material_ids: Specific material IDs to retrieve
            location: Filter by storage location

        Returns:
            List of materials
        """
        try:
            params = {}
            if material_ids:
                params['material_ids'] = ','.join(material_ids)
            if location:
                params['location'] = location

            response = self.session.get(
                f"{self.mes_base_url}/api/v1/materials",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            materials = []
            for mat_data in data.get('materials', []):
                material = Material(
                    material_id=mat_data['id'],
                    material_name=mat_data['name'],
                    quantity=mat_data['quantity'],
                    unit=mat_data['unit'],
                    status=MaterialStatus(mat_data['status']),
                    lot_number=mat_data.get('lot_number'),
                    location=mat_data.get('location'),
                    expiry_date=datetime.fromisoformat(mat_data['expiry_date']) if mat_data.get('expiry_date') else None
                )

                materials.append(material)
                self.materials[material.material_id] = material

            return materials

        except requests.RequestException as e:
            self.logger.error(f"Error retrieving materials from MES: {e}")
            raise

    def submit_quality_inspection(self, inspection: QualityInspection) -> bool:
        """
        Submit quality inspection results to MES.

        Args:
            inspection: Quality inspection data

        Returns:
            True if submission successful
        """
        try:
            payload = {
                'inspection_id': inspection.inspection_id,
                'work_order_id': inspection.work_order_id,
                'timestamp': inspection.timestamp.isoformat(),
                'inspector': inspection.inspector,
                'result': inspection.result,
                'measurements': inspection.measurements,
                'defects': inspection.defects,
                'notes': inspection.notes
            }

            response = self.session.post(
                f"{self.mes_base_url}/api/v1/quality/inspections",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            self.logger.info(f"Submitted quality inspection {inspection.inspection_id}")
            return True

        except requests.RequestException as e:
            self.logger.error(f"Error submitting quality inspection: {e}")
            return False

    def get_production_metrics(
        self,
        production_line: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[ProductionMetrics]:
        """
        Retrieve production metrics/KPIs for a production line.

        Args:
            production_line: Production line identifier
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Production metrics or None if not available
        """
        try:
            params = {
                'production_line': production_line,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }

            response = self.session.get(
                f"{self.mes_base_url}/api/v1/metrics/production",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            metrics = ProductionMetrics(
                timestamp=datetime.fromisoformat(data['timestamp']),
                production_line=data['production_line'],
                units_produced=data['units_produced'],
                units_scrapped=data['units_scrapped'],
                oee=data['oee'],
                availability=data['availability'],
                performance=data['performance'],
                quality=data['quality'],
                downtime_minutes=data['downtime_minutes'],
                cycle_time_avg=data['cycle_time_avg'],
                throughput_rate=data['throughput_rate']
            )

            return metrics

        except requests.RequestException as e:
            self.logger.error(f"Error retrieving production metrics: {e}")
            return None

    def get_equipment_status(self, equipment_id: str) -> Dict[str, Any]:
        """
        Get current equipment status from MES.

        Args:
            equipment_id: Equipment/machine identifier

        Returns:
            Equipment status data
        """
        try:
            response = self.session.get(
                f"{self.mes_base_url}/api/v1/equipment/{equipment_id}/status",
                timeout=30
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            self.logger.error(f"Error retrieving equipment status: {e}")
            raise

    def sync_work_orders_to_orchestrator(
        self,
        orchestrator_url: str,
        status_filter: Optional[WorkOrderStatus] = None
    ) -> int:
        """
        Synchronize MES work orders to SwarmBrain orchestrator.

        Args:
            orchestrator_url: URL of SwarmBrain orchestrator API
            status_filter: Only sync work orders with this status

        Returns:
            Number of work orders synchronized
        """
        try:
            # Get work orders from MES
            work_orders = self.get_work_orders(status=status_filter)

            # Convert to orchestrator format
            synced_count = 0
            for wo in work_orders:
                # Convert MES work order to orchestrator mission format
                mission_payload = {
                    'order_id': wo.work_order_id,
                    'description': f"Production of {wo.quantity} {wo.unit} of {wo.product_name}",
                    'tasks': self._convert_operations_to_tasks(wo.operations),
                    'priority': wo.priority,
                    'metadata': {
                        'mes_work_order_id': wo.work_order_id,
                        'product_id': wo.product_id,
                        'production_line': wo.production_line,
                        'scheduled_start': wo.scheduled_start.isoformat(),
                        'scheduled_end': wo.scheduled_end.isoformat()
                    }
                }

                # Send to orchestrator
                response = requests.post(
                    f"{orchestrator_url}/api/v1/missions",
                    json=mission_payload,
                    timeout=30
                )

                if response.status_code == 200:
                    synced_count += 1
                    self.logger.info(f"Synced work order {wo.work_order_id} to orchestrator")

            self.logger.info(f"Synchronized {synced_count} work orders to orchestrator")
            return synced_count

        except Exception as e:
            self.logger.error(f"Error synchronizing work orders to orchestrator: {e}")
            return 0

    def _convert_operations_to_tasks(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MES operations to orchestrator tasks."""
        tasks = []

        for i, op in enumerate(operations):
            task = {
                'id': f"task_{i+1}",
                'skill': self._map_operation_to_skill(op.get('operation_name', '')),
                'role': 'worker',
                'dependencies': [f"task_{i}"] if i > 0 else [],
                'metadata': {
                    'operation_id': op.get('operation_id'),
                    'work_center': op.get('work_center'),
                    'standard_time': op.get('standard_time')
                }
            }
            tasks.append(task)

        return tasks

    def _map_operation_to_skill(self, operation_name: str) -> str:
        """Map MES operation name to robot skill."""
        operation_lower = operation_name.lower()

        if 'pick' in operation_lower or 'grasp' in operation_lower:
            return 'grasp'
        elif 'place' in operation_lower or 'put' in operation_lower:
            return 'place'
        elif 'assemble' in operation_lower:
            return 'manipulate'
        elif 'inspect' in operation_lower or 'check' in operation_lower:
            return 'inspect'
        elif 'transport' in operation_lower or 'move' in operation_lower:
            return 'navigate'
        else:
            return 'generic'
