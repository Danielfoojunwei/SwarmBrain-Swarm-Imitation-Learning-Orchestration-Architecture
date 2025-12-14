# SwarmBrain: Privacy-Preserving Multi-Robot Coordination System

![SwarmBrain Architecture](docs/architecture/swarm_architecture.png)

**SwarmBrain** is a production-ready, privacy-preserving swarm coordination system for humanoid robots. It enables multi-robot collaboration through federated learning, secure aggregation, and intelligent task orchestration—all without sharing raw teleoperation data.

## 🌟 Key Features

### Three-Layer Architecture

1. **Reflex Layer** (Robot Control)
   - Local skill execution via role-conditioned policies
   - ROS 2-based multi-robot communication
   - Real-time control at 100 Hz
   - Zero raw data transmission

2. **Coordination Layer** (Mission Orchestrator)
   - Work order → task graph conversion
   - Multi-robot role assignment
   - Coordination primitives (handover, mutex, barrier, rendezvous)
   - Behavior tree workflow modeling

3. **Learning Layer** (Cross-Site Improvement)
   - Federated learning with Flower framework
   - Dropout-resilient secure aggregation
   - Dynamic user clustering
   - Joint device scheduling & bandwidth allocation
   - zkRep-style reputation system

### Privacy & Security

- **Homomorphic Encryption** (OpenFHE): Validate updates without decryption
- **Zero-Knowledge Proofs**: Prove reputation without revealing identity
- **Secure Aggregation**: Shamir secret sharing + seed-homomorphic PRG
- **No Raw Data Sharing**: Only encrypted model updates leave the edge

### Production-Ready Features

- **Containerized Deployment**: Docker Compose for all services
- **CI/CD Pipeline**: GitHub Actions with linting, testing, security scans
- **Monitoring**: Prometheus + Grafana dashboards
- **Scalable Architecture**: Microservices with message queues
- **Comprehensive Testing**: Unit, integration, and performance tests

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SWARM BRAIN                              │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │
│  │   Robot 1     │  │   Robot 2     │  │   Robot N     │     │
│  │  (Edge Node)  │  │  (Edge Node)  │  │  (Edge Node)  │     │
│  │               │  │               │  │               │     │
│  │ • Perception  │  │ • Perception  │  │ • Perception  │     │
│  │ • Retargeting │  │ • Retargeting │  │ • Retargeting │     │
│  │ • Skill Exec  │  │ • Skill Exec  │  │ • Skill Exec  │     │
│  │ • Local Train │  │ • Local Train │  │ • Local Train │     │
│  │ • Encryption  │  │ • Encryption  │  │ • Encryption  │     │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘     │
│          │                  │                  │             │
│          └──────────────────┼──────────────────┘             │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │  Orchestrator   │                       │
│                    │                 │                       │
│                    │ • Task Planning │                       │
│                    │ • Role Assignment│                       │
│                    │ • Coordination  │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│          ┌──────────────────┼──────────────────┐            │
│          │                  │                  │            │
│   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐     │
│   │ FL Server   │   │ Clustering  │   │ Reputation  │     │
│   │             │   │             │   │  (zkRep)    │     │
│   │ • Secure    │   │ • Dynamic   │   │             │     │
│   │   Aggregation│   │   Grouping │   │ • ZK Proofs │     │
│   │ • FHE       │   │ • Scheduling│   │ • Weighting │     │
│   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### Robot Control (`/robot_control`)
- **ROS 2 Nodes**: Multi-robot communication via DDS
- **Skills**: Executable task modules (grasp, navigate, manipulate)
- **Policies**: Role-conditioned neural networks trained via IL
- **Drivers**: Hardware interfaces (cameras, gloves, actuators)

#### Orchestrator (`/orchestrator`)
- **Task Planner**: Converts work orders to DAG task graphs
- **Scheduler**: Assigns tasks to robots based on capabilities
- **Coordination Primitives**: Handover, mutex, barrier, rendezvous
- **Workflow Engine**: Behavior trees for complex missions

#### Learning (`/learning`)
- **Federated Client**: Flower-based FL participation
- **Secure Aggregation**: Dropout-resilient protocol
- **Clustering**: Dynamic grouping by task/environment/network
- **Scheduling**: RL-based device selection and bandwidth allocation
- **Reputation**: zkRep proofs for contribution weighting

#### Crypto (`/crypto`)
- **FHE**: OpenFHE integration for encrypted validation
- **ZKP**: Zero-knowledge reputation proofs
- **Utils**: Key management, secure randomness

---

## 📦 Installation

### Prerequisites

- **OS**: Ubuntu 22.04 (recommended) or later
- **Python**: 3.9+ (3.10 recommended)
- **Docker**: 24.0+ with Docker Compose
- **ROS 2**: Humble Hawksbill (for robot nodes)
- **Hardware**:
  - Min: 8GB RAM, 4 CPU cores
  - Recommended: 16GB RAM, 8 CPU cores, NVIDIA GPU (for ML)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/swarm-brain.git
cd swarm-brain

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f orchestrator
```

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/swarm-brain.git
cd swarm-brain

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ROS 2 (Ubuntu)
sudo apt update
sudo apt install ros-humble-desktop-full
source /opt/ros/humble/setup.bash

# Build ROS 2 workspace
colcon build
source install/setup.bash
```

### Option 3: Production Deployment

See [Deployment Guide](docs/guides/deployment.md) for Kubernetes setup.

---

## 🚀 Quick Start

### 1. Start the System

```bash
# Start all services
docker-compose up -d

# Verify all containers are running
docker-compose ps
```

### 2. Access Web Interfaces

- **Orchestrator API**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **RabbitMQ Management**: http://localhost:15672 (swarm/swarm)
- **Prometheus**: http://localhost:9090

### 3. Register Robots

```python
import requests

# Register a robot with orchestrator
response = requests.post('http://localhost:8000/api/v1/robots', json={
    'robot_id': 'robot_001',
    'capabilities': ['grasp', 'navigate', 'manipulate'],
    'location': {'lat': 37.7749, 'lon': -122.4194}
})

print(response.json())
```

### 4. Create a Mission

```python
# Create a work order
work_order = {
    'order_id': 'wo_001',
    'description': 'Collaborative assembly task',
    'tasks': [
        {
            'id': 'task_1',
            'skill': 'navigate',
            'role': 'leader',
            'dependencies': []
        },
        {
            'id': 'task_2',
            'skill': 'grasp',
            'role': 'follower',
            'dependencies': ['task_1']
        },
        {
            'id': 'task_3',
            'skill': 'manipulate',
            'role': 'leader',
            'dependencies': ['task_2'],
            'coordination': 'handover'
        }
    ],
    'priority': 1
}

response = requests.post('http://localhost:8000/api/v1/missions', json=work_order)
mission = response.json()
print(f"Mission created: {mission['mission_id']}")
```

### 5. Monitor Progress

```python
# Get mission status
response = requests.get(f'http://localhost:8000/api/v1/missions/{mission_id}/status')
status = response.json()

print(f"Total tasks: {status['total_tasks']}")
print(f"Completed: {status['status_breakdown']['completed']}")
print(f"In progress: {status['status_breakdown']['in_progress']}")
```

### 6. Start Federated Learning

```bash
# On robot edge devices
python -m learning.federated_client.fl_client \
    --server-address fl-server:8080 \
    --robot-id robot_001 \
    --data-path /data/demonstrations

# Server will aggregate updates automatically
```

---

## 🛠️ Development

### Repository Structure

```
swarm-brain/
├── robot_control/          # Robot control layer
│   ├── skills/            # Executable skill modules
│   ├── policies/          # Neural network policies
│   ├── drivers/           # Hardware interfaces
│   └── ros2_nodes/        # ROS 2 nodes
├── orchestrator/          # Coordination layer
│   ├── task_planner/      # Mission planning
│   ├── scheduler/         # Task assignment
│   └── coordination_primitives/  # Handover, mutex, etc.
├── learning/              # Learning layer
│   ├── federated_client/  # FL client/server
│   ├── secure_aggregation/  # Dropout-resilient aggregation
│   ├── clustering/        # Dynamic clustering
│   ├── scheduling/        # Device scheduling
│   └── reputation/        # zkRep system
├── crypto/                # Cryptographic layer
│   ├── fhe/              # Homomorphic encryption
│   ├── zkp/              # Zero-knowledge proofs
│   └── utils/            # Crypto utilities
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── performance/     # Benchmarks
├── docs/                # Documentation
├── config/              # Configuration files
├── docker/              # Dockerfiles
└── legacy_code/         # Original Dynamical.ai code

```

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=robot_control --cov=orchestrator --cov=learning --cov=crypto

# Specific module
pytest tests/unit/learning/test_clustering.py -v
```

### Code Quality

```bash
# Format code
black robot_control orchestrator learning crypto

# Sort imports
isort robot_control orchestrator learning crypto

# Lint
flake8 robot_control orchestrator learning crypto

# Type check
mypy robot_control orchestrator learning crypto
```

### Adding a New Skill

1. Create skill module in `robot_control/skills/`
2. Implement skill interface
3. Add to skill registry
4. Train policy via imitation learning
5. Write tests
6. Update documentation

See [Skill Development Guide](docs/guides/skill_development.md).

---

## 🧪 Testing

### Test Coverage

- **Unit Tests**: 85%+ coverage target
- **Integration Tests**: Critical paths
- **Performance Tests**: Latency, throughput benchmarks
- **Security Tests**: Vulnerability scanning

### Running the Full Test Suite

```bash
# All tests
pytest

# Specific test categories
pytest tests/unit          # Fast, isolated tests
pytest tests/integration   # System interaction tests
pytest tests/performance   # Benchmarks

# Generate HTML coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

---

## 📚 Documentation

Full documentation is available at [docs/](docs/):

- **[Architecture Overview](docs/architecture/README.md)**: System design and components
- **[API Reference](docs/api/README.md)**: REST and gRPC endpoints
- **[Development Guide](docs/guides/development.md)**: Contributing and extending
- **[Deployment Guide](docs/guides/deployment.md)**: Production deployment
- **[Security Guide](docs/guides/security.md)**: Privacy and security best practices

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

SwarmBrain builds on research from:

- **NTU**: Dynamic clustering, secure aggregation, joint scheduling
- **OpenFHE**: Homomorphic encryption library
- **Flower**: Federated learning framework
- **ROS 2**: Robot Operating System
- **Open-RMF**: Multi-fleet coordination

See [CITATIONS.md](CITATIONS.md) for full references.

---

## 📞 Contact

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/swarm-brain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/swarm-brain/discussions)

---

## 🗺️ Roadmap

### v1.0 (Current)
- ✅ Core three-layer architecture
- ✅ ROS 2 integration
- ✅ Federated learning with Flower
- ✅ Secure aggregation
- ✅ Dynamic clustering
- ✅ Docker deployment

### v1.1 (Q2 2025)
- [ ] OpenFHE full integration
- [ ] zkRep circuit implementation (Circom)
- [ ] Open-RMF integration
- [ ] Multi-site deployment
- [ ] Enhanced monitoring

### v2.0 (Q4 2025)
- [ ] Hierarchical federated learning
- [ ] Cross-modal skill transfer
- [ ] Real-world deployment case studies
- [ ] Mobile app for mission control
- [ ] Advanced visualization

---

**Built with ❤️ by the SwarmBrain Team**
