# Industrial Data Integration Guide

## Overview

SwarmBrain integrates with industrial data systems commonly found in manufacturing facilities:

- **SCADA** (Supervisory Control and Data Acquisition)
- **MES** (Manufacturing Execution System)
- **IoT Sensors** (MQTT-based)
- **PLCs** (Programmable Logic Controllers)
- **Time-series Databases** (InfluxDB)

This integration enables SwarmBrain to:
1. **Auto-sync work orders** from MES to robot missions
2. **Monitor equipment status** via SCADA in real-time
3. **Collect sensor data** from factory IoT devices
4. **Make informed decisions** based on production context
5. **Follow ISA-95 standards** for manufacturing integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Factory Floor                             │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐│
│  │  PLC 1   │   │  PLC 2   │   │ Sensor 1 │   │ Sensor 2 ││
│  │ (OPC UA) │   │ (Modbus) │   │  (MQTT)  │   │  (MQTT)  ││
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘│
│       │              │              │              │       │
└───────┼──────────────┼──────────────┼──────────────┼───────┘
        │              │              │              │
        │              │              └──────┬───────┘
        │              │                     │
        │              │              ┌──────▼──────┐
        │              │              │ MQTT Broker │
        │              │              │ (Mosquitto) │
        │              │              └──────┬──────┘
        │              │                     │
    ┌───▼──────────────▼─────────────────────▼───────┐
    │       Industrial Data Aggregator                │
    │                                                 │
    │  • OPC UA Client (SCADA/PLC)                   │
    │  • MQTT Client (IoT Sensors)                   │
    │  • MES Collector (Work Orders)                 │
    │  • InfluxDB Writer (Time-series)               │
    └───────────────────┬─────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ SwarmBrain Orchestrator│
            │                        │
            │ • Auto-sync work orders│
            │ • Real-time equipment  │
            │   status monitoring    │
            │ • Context-aware task   │
            │   assignment           │
            └────────────────────────┘
```

---

## Supported Protocols

### 1. OPC UA (SCADA/PLC)

**Use Case**: Connect to PLCs, SCADA systems, industrial equipment

**Features**:
- Real-time tag subscriptions
- Bulk read/write operations
- Historical data access
- Alarm and event handling
- Secure communication (TLS + user auth)

**Example**:
```python
from industrial_data.protocols.opcua_client import OPCUAClient, OPCUATag, DataType

# Connect to PLC
client = OPCUAClient(
    endpoint_url="opc.tcp://plc.factory.local:4840",
    username="opcua_user",
    password="password"
)

await client.connect()

# Define tags
tags = [
    OPCUATag(
        node_id="ns=2;i=1001",
        name="ProductionCount",
        data_type=DataType.INT32
    ),
    OPCUATag(
        node_id="ns=2;i=1002",
        name="MachineSpeed",
        data_type=DataType.DOUBLE
    )
]

# Subscribe to real-time updates
await client.subscribe_tags(tags, callback=on_tag_update)
```

### 2. MQTT (IoT Sensors)

**Use Case**: Connect to IoT sensors, edge devices, wireless sensors

**Features**:
- Topic-based pub/sub
- QoS levels (0, 1, 2)
- Sparkplug B support
- WebSocket support
- Retained messages

**Example**:
```python
from industrial_data.protocols.mqtt_client import MQTTClient, QoS

# Connect to MQTT broker
client = MQTTClient(
    broker_host="mqtt.factory.local",
    broker_port=1883,
    username="mqtt_user",
    password="password"
)

client.connect()

# Subscribe to sensor topics
topics = [
    "factory/line1/temperature/+",
    "factory/line1/vibration/+",
    "warehouse/robot/battery/+"
]

client.subscribe_multiple(topics, callback=on_sensor_data, qos=QoS.AT_LEAST_ONCE)
```

### 3. MES API (Work Orders)

**Use Case**: Sync production schedules, work orders, quality data

**Features**:
- Work order retrieval and updates
- Material inventory tracking
- Quality inspection submission
- Production metrics (OEE, throughput)
- Equipment status queries

**Example**:
```python
from industrial_data.collectors.mes_collector import MESCollector, WorkOrderStatus

# Connect to MES
collector = MESCollector(
    mes_base_url="http://mes.factory.local",
    api_key="your_api_key"
)

# Get active work orders
work_orders = collector.get_work_orders(
    status=WorkOrderStatus.RELEASED,
    production_line="Line1"
)

# Sync to orchestrator
collector.sync_work_orders_to_orchestrator(
    orchestrator_url="http://orchestrator:8000"
)
```

---

## ISA-95 Compliance

SwarmBrain follows ISA-95 (IEC 62264) standards for manufacturing integration.

### Equipment Model

```python
from industrial_data.models.isa95_models import Equipment, EquipmentLevel

# Map robot to ISA-95 Equipment
robot_equipment = Equipment(
    equipment_id="robot_001",
    equipment_name="Collaborative Robot #1",
    equipment_level=EquipmentLevel.WORK_UNIT,
    capabilities=["grasp", "place", "inspect"]
)
```

### Production Request Model

```python
from industrial_data.models.isa95_models import ProductionRequest, ProductionRequestState

# MES work order → ISA-95 Production Request
production_request = ProductionRequest(
    request_id="WO-2025-001",
    description="Assemble 100 units of Product A",
    state=ProductionRequestState.RELEASED,
    segment_id="assembly_line_1",
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(hours=4),
    priority=75
)
```

---

## Industrial Data Aggregator

The aggregator collects data from all industrial sources and provides unified access.

### Setup

```python
from industrial_data.streams.data_aggregator import IndustrialDataAggregator

aggregator = IndustrialDataAggregator()

# Add OPC UA source (SCADA)
await aggregator.add_opcua_source(
    source_id="factory_scada",
    endpoint_url="opc.tcp://plc.factory.local:4840",
    tags=plc_tags,
    username="scada_user",
    password="scada_pass"
)

# Add MQTT source (IoT)
aggregator.add_mqtt_source(
    source_id="warehouse_iot",
    broker_host="mqtt.factory.local",
    topics=["factory/+/+/+", "warehouse/+/+/+"]
)

# Add MES source
aggregator.add_mes_source(
    source_id="production_mes",
    mes_base_url="http://mes.factory.local",
    api_key="your_api_key"
)

# Start aggregation
await aggregator.start()
```

### Access Data

```python
# Get aggregated data
data = aggregator.get_aggregated_data()

# Get specific tag value
production_count = aggregator.get_tag_value("factory_scada", "ProductionCount")

# Get sensor value
temperature = aggregator.get_sensor_value("warehouse_iot", "factory/line1/temperature/sensor01")

# Get active work orders
work_orders = aggregator.get_active_work_orders()
```

---

## Orchestrator Integration

The orchestrator automatically consumes industrial data for intelligent task planning.

### Auto-Sync Work Orders

```python
from orchestrator.industrial_integration import IndustrialOrchestrator

# Initialize with industrial data
orchestrator = IndustrialOrchestrator(industrial_aggregator)

# Enable auto-sync (default: True)
orchestrator.auto_sync_work_orders = True

# Work orders from MES are automatically converted to missions
# No manual intervention required!
```

### Equipment Availability Check

```python
# Orchestrator checks SCADA before assigning tasks
available = orchestrator.check_equipment_availability("robot_001")

# Checks:
# - Robot registry status
# - SCADA fault tags
# - Battery level (for mobile robots)
# - Other real-time conditions
```

### Context-Aware Task Assignment

```python
# Assign tasks with industrial context
assignments = await orchestrator.assign_tasks_with_industrial_context(mission_id)

# Considers:
# - Real-time equipment status (SCADA)
# - Production priorities (MES)
# - Work center affinity
# - Resource availability
```

---

## Docker Deployment

### Basic Setup (MQTT only)

```bash
docker-compose up -d mqtt-broker orchestrator
```

### Full Industrial Stack

```bash
docker-compose --profile industrial up -d
```

This starts:
- ✅ MQTT Broker (Eclipse Mosquitto)
- ✅ OPC UA Server Simulator
- ✅ InfluxDB (time-series database)
- ✅ Chronograf (InfluxDB visualization)
- ✅ Industrial Data Aggregator

### Access Points

- **MQTT Broker**: `localhost:1883` (plain), `localhost:9001` (WebSocket)
- **OPC UA Server**: `opc.tcp://localhost:50000`
- **InfluxDB**: `http://localhost:8086` (admin/adminpassword)
- **Chronograf**: `http://localhost:8888`

---

## Configuration

### MQTT Broker Config

Create `config/mosquitto/mosquitto.conf`:

```conf
listener 1883
allow_anonymous true

listener 9001
protocol websockets

persistence true
persistence_location /mosquitto/data/

log_dest file /mosquitto/log/mosquitto.log
log_type all
```

### Industrial Sources Config

Create `config/industrial/sources.yml`:

```yaml
sources:
  - source_id: factory_scada
    type: opcua
    endpoint: opc.tcp://plc.factory.local:4840
    username: scada_user
    password: ${SCADA_PASSWORD}
    tags:
      - node_id: "ns=2;i=1001"
        name: "ProductionCount"
        data_type: "INT32"

  - source_id: warehouse_iot
    type: mqtt
    broker_host: mqtt.factory.local
    broker_port: 1883
    topics:
      - "factory/+/+/+"
      - "warehouse/+/+/+"

  - source_id: production_mes
    type: mes
    endpoint: http://mes.factory.local
    api_key: ${MES_API_KEY}
    poll_interval: 60
```

---

## Production Deployment

### Security Considerations

1. **Use TLS for all connections**:
   - OPC UA: Enable security policy (Basic256Sha256)
   - MQTT: Use port 8883 with TLS
   - MES API: HTTPS only

2. **Network segmentation**:
   - Separate network for industrial systems
   - Firewall rules limiting access
   - VPN for remote access

3. **Authentication**:
   - Strong credentials for all systems
   - Rotate API keys regularly
   - Use certificate-based auth where possible

### High Availability

- Deploy industrial aggregator as multiple replicas
- Use MQTT broker clustering (HiveMQ, EMQX)
- InfluxDB enterprise for replication
- Load balancer for MES API calls

---

## Troubleshooting

### OPC UA Connection Issues

```bash
# Test OPC UA connection
python -c "from asyncua import Client; import asyncio; asyncio.run(Client('opc.tcp://localhost:50000').connect())"
```

### MQTT Connection Issues

```bash
# Test MQTT publish
mosquitto_pub -h localhost -p 1883 -t "test/topic" -m "Hello"

# Test MQTT subscribe
mosquitto_sub -h localhost -p 1883 -t "test/#"
```

### Check Aggregator Logs

```bash
docker logs swarm_industrial_aggregator -f
```

---

## References

- [ISA-95 Standard](https://www.isa.org/standards-and-publications/isa-standards/isa-standards-committees/isa95)
- [OPC UA Specification](https://opcfoundation.org/about/opc-technologies/opc-ua/)
- [MQTT Protocol](https://mqtt.org/)
- [Sparkplug B Specification](https://sparkplug.eclipse.org/)
- [InfluxDB Documentation](https://docs.influxdata.com/influxdb/)
