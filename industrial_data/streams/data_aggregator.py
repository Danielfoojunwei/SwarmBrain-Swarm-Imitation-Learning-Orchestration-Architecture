"""
Industrial Data Aggregator

Aggregates data from multiple industrial sources:
- SCADA (OPC UA tags, real-time sensor data)
- MES (work orders, production schedules)
- MQTT (IoT sensors, edge devices)
- Time-series databases (historical trends)

Provides unified data access for SwarmBrain orchestrator.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging

from industrial_data.protocols.opcua_client import OPCUAClient, OPCUATag, TagValue
from industrial_data.protocols.mqtt_client import MQTTClient, MQTTMessage
from industrial_data.collectors.mes_collector import MESCollector, WorkOrder
from industrial_data.models.isa95_models import Equipment, ProductionRequest
from industrial_data.analytics import OEECalculator, EquipmentHealthMonitor


@dataclass
class IndustrialDataSource:
    """Configuration for an industrial data source."""
    source_id: str
    source_type: str  # "opcua", "mqtt", "mes", "timeseries"
    endpoint: str
    enabled: bool = True
    config: Dict[str, Any] = None


@dataclass
class AggregatedData:
    """Aggregated industrial data for a time window."""
    timestamp: datetime
    scada_tags: Dict[str, TagValue]
    iot_sensors: Dict[str, Any]
    work_orders: List[WorkOrder]
    production_metrics: Dict[str, Any]
    equipment_status: Dict[str, Any]


class IndustrialDataAggregator:
    """
    Aggregates data from all industrial systems.

    Provides unified data access for SwarmBrain orchestrator to make
    informed task planning decisions based on real-time factory state.
    """

    def __init__(self):
        """Initialize data aggregator."""
        self.logger = logging.getLogger(__name__)

        # Data source clients
        self.opcua_clients: Dict[str, OPCUAClient] = {}
        self.mqtt_clients: Dict[str, MQTTClient] = {}
        self.mes_collectors: Dict[str, MESCollector] = {}

        # Latest data cache
        self.scada_data: Dict[str, TagValue] = {}
        self.iot_data: Dict[str, Any] = {}
        self.work_orders: List[WorkOrder] = []

        # Analytics modules
        self.oee_calculator = OEECalculator()
        self.health_monitor = EquipmentHealthMonitor()

        # Callbacks for data updates
        self.update_callbacks: List[Callable[[AggregatedData], None]] = []

        self.running = False

    async def add_opcua_source(
        self,
        source_id: str,
        endpoint_url: str,
        tags: List[OPCUATag],
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Add an OPC UA data source (SCADA/PLC).

        Args:
            source_id: Unique identifier for this source
            endpoint_url: OPC UA server endpoint
            tags: Tags to subscribe to
            username: Optional authentication username
            password: Optional authentication password
        """
        try:
            client = OPCUAClient(
                endpoint_url=endpoint_url,
                username=username,
                password=password
            )

            await client.connect()

            # Subscribe to tags
            await client.subscribe_tags(
                tags,
                lambda values: self._on_scada_update(source_id, values)
            )

            self.opcua_clients[source_id] = client
            self.logger.info(f"Added OPC UA source: {source_id}")

        except Exception as e:
            self.logger.error(f"Error adding OPC UA source {source_id}: {e}")
            raise

    def add_mqtt_source(
        self,
        source_id: str,
        broker_host: str,
        topics: List[str],
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Add an MQTT data source (IoT sensors).

        Args:
            source_id: Unique identifier for this source
            broker_host: MQTT broker hostname
            topics: Topics to subscribe to
            broker_port: MQTT broker port
            username: Optional authentication username
            password: Optional authentication password
        """
        try:
            client = MQTTClient(
                broker_host=broker_host,
                broker_port=broker_port,
                client_id=f"swarm_{source_id}",
                username=username,
                password=password
            )

            client.connect()

            # Subscribe to topics
            client.subscribe_multiple(
                topics,
                lambda msg: self._on_iot_update(source_id, msg)
            )

            self.mqtt_clients[source_id] = client
            self.logger.info(f"Added MQTT source: {source_id}")

        except Exception as e:
            self.logger.error(f"Error adding MQTT source {source_id}: {e}")
            raise

    def add_mes_source(
        self,
        source_id: str,
        mes_base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        poll_interval: int = 60
    ):
        """
        Add a MES data source.

        Args:
            source_id: Unique identifier for this source
            mes_base_url: Base URL of MES API
            api_key: Optional API key
            username: Optional username
            password: Optional password
            poll_interval: Polling interval in seconds
        """
        try:
            collector = MESCollector(
                mes_base_url=mes_base_url,
                api_key=api_key,
                username=username,
                password=password,
                poll_interval=poll_interval
            )

            self.mes_collectors[source_id] = collector
            self.logger.info(f"Added MES source: {source_id}")

        except Exception as e:
            self.logger.error(f"Error adding MES source {source_id}: {e}")
            raise

    def _on_scada_update(self, source_id: str, tag_values: List[TagValue]):
        """Handle SCADA tag updates."""
        for tag_value in tag_values:
            key = f"{source_id}:{tag_value.name}"
            self.scada_data[key] = tag_value

            self.logger.debug(
                f"SCADA update: {tag_value.name} = {tag_value.value}"
            )

        # Trigger aggregation callback
        self._trigger_callbacks()

    def _on_iot_update(self, source_id: str, message: MQTTMessage):
        """Handle IoT sensor updates via MQTT."""
        key = f"{source_id}:{message.topic}"
        self.iot_data[key] = {
            'topic': message.topic,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'qos': message.qos
        }

        self.logger.debug(
            f"IoT update: {message.topic} = {message.payload}"
        )

        # Trigger aggregation callback
        self._trigger_callbacks()

    async def poll_mes_data(self):
        """Poll MES systems for work orders and production data."""
        for source_id, collector in self.mes_collectors.items():
            try:
                # Get active work orders
                work_orders = collector.get_work_orders()
                self.work_orders.extend(work_orders)

                self.logger.info(
                    f"Polled MES {source_id}: {len(work_orders)} work orders"
                )

            except Exception as e:
                self.logger.error(f"Error polling MES {source_id}: {e}")

    def get_aggregated_data(self) -> AggregatedData:
        """Get current aggregated data from all sources."""
        # Calculate production metrics using OEE calculator
        production_metrics = self.oee_calculator.calculate(
            scada_data=self.scada_data,
            work_orders=self.work_orders
        )

        # Get equipment health metrics
        equipment_status = self.health_monitor.get_health_metrics(
            sensor_data=self.iot_data
        )

        return AggregatedData(
            timestamp=datetime.utcnow(),
            scada_tags=self.scada_data.copy(),
            iot_sensors=self.iot_data.copy(),
            work_orders=self.work_orders.copy(),
            production_metrics=production_metrics,
            equipment_status=equipment_status
        )

    def get_tag_value(self, source_id: str, tag_name: str) -> Optional[TagValue]:
        """Get current value of a specific SCADA tag."""
        key = f"{source_id}:{tag_name}"
        return self.scada_data.get(key)

    def get_sensor_value(self, source_id: str, topic: str) -> Optional[Any]:
        """Get current value of a specific IoT sensor."""
        key = f"{source_id}:{topic}"
        sensor_data = self.iot_data.get(key)
        return sensor_data['payload'] if sensor_data else None

    def get_active_work_orders(self) -> List[WorkOrder]:
        """Get all active work orders from MES."""
        return [
            wo for wo in self.work_orders
            if wo.status.value in ['released', 'in_progress']
        ]

    def register_update_callback(self, callback: Callable[[AggregatedData], None]):
        """
        Register a callback to be called when data is updated.

        Args:
            callback: Function to call with aggregated data
        """
        self.update_callbacks.append(callback)

    def _trigger_callbacks(self):
        """Trigger all registered callbacks with current data."""
        if not self.update_callbacks:
            return

        aggregated_data = self.get_aggregated_data()

        for callback in self.update_callbacks:
            try:
                callback(aggregated_data)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")

    async def start(self):
        """Start data aggregation."""
        self.running = True
        self.logger.info("Started industrial data aggregator")

        # Start MES polling loop
        while self.running:
            await self.poll_mes_data()
            await asyncio.sleep(60)  # Poll every minute

    async def stop(self):
        """Stop data aggregation and disconnect all clients."""
        self.running = False

        # Disconnect OPC UA clients
        for source_id, client in self.opcua_clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting OPC UA {source_id}: {e}")

        # Disconnect MQTT clients
        for source_id, client in self.mqtt_clients.items():
            try:
                client.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting MQTT {source_id}: {e}")

        self.logger.info("Stopped industrial data aggregator")


# Example configuration
EXAMPLE_INDUSTRIAL_CONFIG = {
    "sources": [
        {
            "source_id": "factory_scada",
            "source_type": "opcua",
            "endpoint": "opc.tcp://plc.factory.local:4840",
            "enabled": True,
            "config": {
                "username": "scada_user",
                "tags": [
                    {"node_id": "ns=2;i=1001", "name": "ProductionCount"},
                    {"node_id": "ns=2;i=1002", "name": "MachineSpeed"},
                    {"node_id": "ns=2;i=1003", "name": "Temperature"}
                ]
            }
        },
        {
            "source_id": "warehouse_iot",
            "source_type": "mqtt",
            "endpoint": "mqtt.factory.local",
            "enabled": True,
            "config": {
                "broker_port": 1883,
                "topics": [
                    "factory/line1/temperature/+",
                    "factory/line1/vibration/+",
                    "warehouse/robot/battery/+"
                ]
            }
        },
        {
            "source_id": "production_mes",
            "source_type": "mes",
            "endpoint": "http://mes.factory.local",
            "enabled": True,
            "config": {
                "api_key": "your_api_key_here",
                "poll_interval": 60
            }
        }
    ]
}
