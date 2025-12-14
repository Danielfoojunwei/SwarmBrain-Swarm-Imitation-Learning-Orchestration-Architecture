"""
MQTT Client for Industrial IoT Integration

MQTT is widely used for industrial IoT sensors, edge devices, and
real-time data streaming in smart factories.

Supports:
- Sensor data collection (temperature, vibration, pressure, etc.)
- Edge device communication
- Real-time event streaming
- Sparkplug B protocol for industrial MQTT
- QoS levels for reliable messaging
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("paho-mqtt not installed. Install with: pip install paho-mqtt")


class QoS(Enum):
    """MQTT Quality of Service levels."""
    AT_MOST_ONCE = 0  # Fire and forget
    AT_LEAST_ONCE = 1  # Acknowledged delivery
    EXACTLY_ONCE = 2  # Assured delivery


@dataclass
class MQTTMessage:
    """MQTT message received from broker."""
    topic: str
    payload: Any
    timestamp: datetime
    qos: int
    retained: bool


class MQTTClient:
    """
    MQTT client for industrial IoT sensor data collection.

    Connects to MQTT broker (e.g., Mosquitto, HiveMQ, AWS IoT)
    and subscribes to sensor topics.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        ca_cert_path: Optional[str] = None
    ):
        """
        Initialize MQTT client.

        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port (1883 for plain, 8883 for TLS)
            client_id: Unique client identifier
            username: Optional username for authentication
            password: Optional password for authentication
            use_tls: Use TLS/SSL encryption
            ca_cert_path: Path to CA certificate for TLS
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt is required. Install with: pip install paho-mqtt")

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"swarm_mqtt_{datetime.utcnow().timestamp()}"

        self.client = mqtt.Client(client_id=self.client_id)

        # Set authentication
        if username and password:
            self.client.username_pw_set(username, password)

        # Set TLS
        if use_tls and ca_cert_path:
            self.client.tls_set(ca_certs=ca_cert_path)

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.logger = logging.getLogger(__name__)
        self.connected = False

        # Message handlers
        self.topic_handlers: Dict[str, Callable[[MQTTMessage], None]] = {}
        self.default_handler: Optional[Callable[[MQTTMessage], None]] = None

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnect from MQTT broker: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        """Callback when message received."""
        try:
            # Try to parse JSON payload
            try:
                payload = json.loads(msg.payload.decode())
            except:
                payload = msg.payload.decode()

            message = MQTTMessage(
                topic=msg.topic,
                payload=payload,
                timestamp=datetime.utcnow(),
                qos=msg.qos,
                retained=msg.retain
            )

            # Route to appropriate handler
            if msg.topic in self.topic_handlers:
                self.topic_handlers[msg.topic](message)
            elif self.default_handler:
                self.default_handler(message)
            else:
                self.logger.debug(f"Unhandled message on topic {msg.topic}: {payload}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()  # Start background thread for message processing
            self.logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.info("Disconnected from MQTT broker")

    def subscribe(
        self,
        topic: str,
        callback: Callable[[MQTTMessage], None],
        qos: QoS = QoS.AT_LEAST_ONCE
    ):
        """
        Subscribe to a topic.

        Args:
            topic: MQTT topic (supports wildcards: +, #)
            callback: Function to call when message received
            qos: Quality of Service level
        """
        self.topic_handlers[topic] = callback
        self.client.subscribe(topic, qos=qos.value)
        self.logger.info(f"Subscribed to topic: {topic} (QoS {qos.value})")

    def subscribe_multiple(
        self,
        topics: List[str],
        callback: Callable[[MQTTMessage], None],
        qos: QoS = QoS.AT_LEAST_ONCE
    ):
        """Subscribe to multiple topics with same callback."""
        for topic in topics:
            self.subscribe(topic, callback, qos)

    def publish(
        self,
        topic: str,
        payload: Any,
        qos: QoS = QoS.AT_LEAST_ONCE,
        retain: bool = False
    ):
        """
        Publish a message to a topic.

        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON encoded if dict/list)
            qos: Quality of Service level
            retain: Retain message on broker
        """
        try:
            # Encode payload
            if isinstance(payload, (dict, list)):
                payload_bytes = json.dumps(payload).encode()
            elif isinstance(payload, str):
                payload_bytes = payload.encode()
            else:
                payload_bytes = str(payload).encode()

            self.client.publish(topic, payload_bytes, qos=qos.value, retain=retain)
            self.logger.debug(f"Published to {topic}: {payload}")

        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            raise

    def set_default_handler(self, callback: Callable[[MQTTMessage], None]):
        """Set default handler for messages without specific handler."""
        self.default_handler = callback


class SparkplugB:
    """
    Sparkplug B protocol implementation for industrial MQTT.

    Sparkplug B is an industrial IoT protocol specification that defines:
    - How to organize MQTT topics
    - Payload formatting (Protocol Buffers)
    - Birth/death certificates
    - Metrics and data types
    """

    @staticmethod
    def build_topic(
        group_id: str,
        message_type: str,
        edge_node_id: str,
        device_id: Optional[str] = None
    ) -> str:
        """
        Build Sparkplug B topic.

        Args:
            group_id: Logical grouping of nodes
            message_type: NBIRTH, NDEATH, DBIRTH, DDEATH, NDATA, DDATA, NCMD, DCMD
            edge_node_id: Edge node identifier
            device_id: Optional device identifier

        Returns:
            Formatted Sparkplug B topic
        """
        if device_id:
            return f"spBv1.0/{group_id}/{message_type}/{edge_node_id}/{device_id}"
        else:
            return f"spBv1.0/{group_id}/{message_type}/{edge_node_id}"

    @staticmethod
    def parse_topic(topic: str) -> Dict[str, str]:
        """Parse Sparkplug B topic into components."""
        parts = topic.split('/')
        if len(parts) < 4:
            raise ValueError(f"Invalid Sparkplug B topic: {topic}")

        result = {
            'namespace': parts[0],
            'group_id': parts[1],
            'message_type': parts[2],
            'edge_node_id': parts[3]
        }

        if len(parts) == 5:
            result['device_id'] = parts[4]

        return result


# Example sensor topics for industrial IoT
EXAMPLE_SENSOR_TOPICS = [
    "factory/line1/temperature/sensor01",
    "factory/line1/vibration/motor01",
    "factory/line1/pressure/valve01",
    "factory/line2/vision/camera01",
    "warehouse/robot/battery/robot001",
    "warehouse/robot/position/robot001",
    "quality/inspection/gate01",
]

# Example Sparkplug B topics
EXAMPLE_SPARKPLUG_TOPICS = [
    "spBv1.0/factory/NDATA/line1/plc01",  # Node data
    "spBv1.0/factory/DDATA/line1/sensor01",  # Device data
    "spBv1.0/warehouse/NBIRTH/zone1/robot001",  # Node birth
]
