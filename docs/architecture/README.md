# SwarmBrain Architecture

## Overview

SwarmBrain implements a three-layer architecture for privacy-preserving multi-robot coordination:

1. **Reflex Layer**: Local robot control and skill execution
2. **Coordination Layer**: Multi-robot task orchestration
3. **Learning Layer**: Cross-site federated improvement

## System Architecture Diagram

```
                           ┌─────────────────────────────────────┐
                           │         SwarmBrain Cloud            │
                           │                                     │
                           │  ┌──────────┐    ┌──────────────┐  │
                           │  │FL Server │    │ Orchestrator │  │
                           │  │          │    │              │  │
                           │  │• Secure  │    │• Planning    │  │
                           │  │  Agg     │◄───┤• Scheduling  │  │
                           │  │• FHE     │    │• Coord       │  │
                           │  └────▲─────┘    └──────▲───────┘  │
                           └───────┼─────────────────┼──────────┘
                                   │                 │
                    ┌──────────────┴─────────────────┴────────────────┐
                    │          Encrypted Model Updates                │
                    │          Task Assignments & Coordination        │
                    └──────────────┬─────────────────┬────────────────┘
                                   │                 │
         ┌─────────────────────────┼─────────────────┼─────────────────┐
         │                         │                 │                 │
    ┌────▼──────┐            ┌────▼──────┐     ┌────▼──────┐     ┌────▼──────┐
    │  Site A   │            │  Site B   │     │  Site C   │     │  Site D   │
    │  Factory  │            │  Warehouse│     │  Hospital │     │  Lab      │
    │           │            │           │     │           │     │           │
    │ Robot 1-3 │            │ Robot 4-6 │     │ Robot 7-9 │     │Robot 10-12│
    │           │            │           │     │           │     │           │
    │• Perceive │            │• Perceive │     │• Perceive │     │• Perceive │
    │• Retarget │            │• Retarget │     │• Retarget │     │• Retarget │
    │• Execute  │            │• Execute  │     │• Execute  │     │• Execute  │
    │• Learn    │            │• Learn    │     │• Learn    │     │• Learn    │
    │• Encrypt  │            │• Encrypt  │     │• Encrypt  │     │• Encrypt  │
    └───────────┘            └───────────┘     └───────────┘     └───────────┘
         ▲                        ▲                  ▲                 ▲
         │                        │                  │                 │
    Human Demo              Human Demo          Human Demo       Human Demo
    (stays local)           (stays local)       (stays local)    (stays local)
```

## Layer Details

### 1. Reflex Layer (Edge)

**Purpose**: Real-time robot control with privacy preservation

**Components**:
- **Perception**: Camera + IMU + proprioception → HumanState
- **Retargeting**: Human pose → Robot joint angles (IK solver)
- **Skill Execution**: Role-conditioned policies (trained via IL)
- **Local Training**: On-device model updates
- **Encryption**: N2HE-LWE or OpenFHE CKKS

**Key Properties**:
- 100 Hz control loop
- No raw data transmission
- Local skill library
- ROS 2 DDS communication

**Technology Stack**:
- ROS 2 Humble
- PyTorch (policies)
- MMPose (perception)
- Custom IK solver

### 2. Coordination Layer (Cloud/Edge Gateway)

**Purpose**: Multi-robot mission planning and execution

**Components**:
- **Task Planner**: Work order → DAG task graph
- **Scheduler**: Task assignment based on capabilities
- **Coordination Primitives**:
  - Handover: Object transfer between robots
  - Mutex: Shared resource locking
  - Barrier: Synchronization point
  - Rendezvous: Spatial meeting point

**Key Properties**:
- NetworkX for graph operations
- Behavior tree workflow engine
- Event-driven architecture
- FastAPI REST + gRPC

**Technology Stack**:
- Python 3.10
- NetworkX (task graphs)
- FastAPI (API)
- RabbitMQ (messaging)
- PostgreSQL (state)

### 3. Learning Layer (Cloud)

**Purpose**: Cross-site federated learning without data sharing

**Components**:
- **Federated Client/Server**: Flower framework
- **Secure Aggregation**: Dropout-resilient protocol
- **Dynamic Clustering**: Group by task/env/network
- **Device Scheduling**: RL-based selection + bandwidth allocation
- **Reputation System**: zkRep proofs for weighting

**Key Properties**:
- Encrypted gradient aggregation
- Adaptive clustering
- Fault-tolerant (handles dropouts)
- Fair contribution weighting

**Technology Stack**:
- Flower (FL framework)
- PyTorch (models)
- OpenFHE (FHE)
- Circom/snarkjs (ZKP)
- LSTM (scheduler)

## Data Flow

### Training Loop

```
1. Human demonstrates task at Site A
2. Edge device captures multi-modal data
3. HumanState → Retarget → Robot trajectory
4. Store trajectory locally
5. Train policy on local data
6. Compute model update (gradient)
7. Encrypt update with CKKS
8. Send encrypted update to FL server
9. FL server aggregates (homomorphically or after decryption)
10. Broadcast global model
11. Robots update local policies
```

### Mission Execution Loop

```
1. Work order arrives at orchestrator
2. Planner creates task graph (DAG)
3. Scheduler assigns tasks to robots
4. Robots receive task assignments via ROS 2
5. Robots execute skills locally
6. Coordination primitives handle interactions
7. Status updates flow back to orchestrator
8. Mission completion triggers learning round
```

## Communication Patterns

### Robot ↔ Robot (ROS 2 DDS)

- **Topics**: `/robot/{id}/status`, `/robot/{id}/task`, `/swarm/coordination`
- **QoS**: Reliable, transient local
- **Namespace**: Per-fleet isolation
- **Discovery**: Automatic via DDS

### Robot ↔ Orchestrator (REST/gRPC)

- **REST**: Mission status, robot registration
- **gRPC**: Streaming coordination commands
- **Auth**: TLS + API keys

### Robot ↔ FL Server (Flower gRPC)

- **Protocol**: Flower gRPC
- **Encryption**: TLS + payload encryption
- **Aggregation**: Server-side secure aggregation

## Security Model

### Threat Model

**Assumptions**:
- Honest-but-curious server (aggregator)
- Untrusted network
- Trusted edge devices
- Malicious robots possible

**Protections**:
1. **Data Privacy**: FHE encryption of model updates
2. **Communication Security**: TLS for all network traffic
3. **Integrity**: zkRep proofs prevent malicious updates
4. **Availability**: Dropout-resilient aggregation

### Privacy Guarantees

- **Raw Data**: Never leaves edge device
- **Trajectories**: Encrypted before transmission
- **Model Updates**: Encrypted aggregation
- **Identities**: Hidden via zkRep proofs

## Scalability

### Horizontal Scaling

- **Robots**: 1,000+ per fleet
- **Fleets**: Multiple isolated namespaces
- **Sites**: Geographically distributed
- **Learning Rounds**: Asynchronous participation

### Performance Targets

- **Control Loop**: 100 Hz (robot)
- **FL Round**: <5 minutes (100 robots)
- **Task Assignment**: <1 second
- **Coordination Latency**: <100ms

## Deployment Topologies

### Single-Site Development

```
Docker Compose:
- 3 robot simulators
- 1 orchestrator
- 1 FL server
- Supporting services (DB, Redis, RabbitMQ)
```

### Multi-Site Production

```
Kubernetes:
- Edge: K3s on Jetson Orin
- Cloud: EKS/GKE/AKS
- Networking: WireGuard VPN
- Storage: S3/GCS for models
- Monitoring: Prometheus + Grafana
```

## Technology Choices Rationale

### Why ROS 2?

- DDS for reliable real-time communication
- QoS tuning for different use cases
- Namespace isolation for multi-fleet
- Active ecosystem and tooling

### Why Flower?

- Framework-agnostic (PyTorch, TF, NumPy)
- Customizable aggregation strategies
- Simulation mode for testing
- Active development and community

### Why OpenFHE?

- Fastest FHE library (6-37x speedup vs. alternatives)
- Multiple schemes (BGV, BFV, CKKS, FHEW)
- Post-quantum secure
- Production-ready C++ with Python bindings

### Why NetworkX?

- Standard graph library for Python
- Efficient DAG operations
- Rich algorithms (critical path, topological sort)
- Serializable graphs

## Future Enhancements

### v1.1
- Full OpenFHE integration for encrypted validation
- Circom circuits for zkRep proofs
- Open-RMF integration for multi-vendor fleets

### v2.0
- Hierarchical federated learning (site → region → global)
- Cross-modal skill transfer (vision ↔ tactile)
- Online learning (continual adaptation)
- Advanced visualization (3D mission replay)

## References

- [ROS 2 Design](https://design.ros2.org/)
- [Flower Documentation](https://flower.dev/docs/)
- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [Circom Documentation](https://docs.circom.io/)
