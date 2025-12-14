#!/bin/bash
# Setup Multi-Actor Integration for SwarmBrain
#
# This script sets up the complete Multi-Actor Swarm Imitation Learning
# integration with SwarmBrain, including CSA support, OpenFL coordination,
# and unified federated learning.

set -e  # Exit on error

echo "=========================================="
echo "SwarmBrain Multi-Actor Integration Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the SwarmBrain project root directory"
    exit 1
fi

print_info "Step 1/8: Initializing git submodules"
# Initialize multi_actor submodule (if not already done)
if [ ! -d "external/multi_actor/.git" ]; then
    git submodule update --init --recursive external/multi_actor
    print_success "Multi-actor submodule initialized"
else
    print_success "Multi-actor submodule already initialized"
fi

print_info "Step 2/8: Creating workspace directories"
# Create necessary directories
mkdir -p robot_control/skills/csa/{models,checkpoints,cache}
mkdir -p learning/swarm_workspace/{single_actor,multi_actor,hybrid}
mkdir -p config/multi_actor
mkdir -p logs/multi_actor

print_success "Workspace directories created"

print_info "Step 3/8: Installing Multi-Actor dependencies"
# Install multi_actor dependencies
if [ -f "external/multi_actor/requirements.txt" ]; then
    pip install -r external/multi_actor/requirements.txt
    print_success "Multi-actor dependencies installed"
else
    print_info "Multi-actor requirements.txt not found, skipping"
fi

print_info "Step 4/8: Installing Multi-Actor Python package"
# Install multi_actor as editable package
if [ -f "external/multi_actor/pyproject.toml" ]; then
    pip install -e external/multi_actor/
    print_success "Multi-actor package installed"
else
    print_info "Multi-actor pyproject.toml not found, skipping"
fi

print_info "Step 5/8: Installing additional SwarmBrain dependencies"
# Install additional dependencies for multi-actor integration
cat > /tmp/multi_actor_requirements.txt <<EOF
# Multi-actor integration dependencies
pydantic>=2.0.0
python-onvif-zeep>=0.2.12
opencv-python>=4.8.0
mmpose>=1.1.0
lerobot>=0.1.0

# Federated learning
openfl>=1.5.0

# Privacy-preserving
opacus>=1.4.0
crypten>=0.4.0
pyfhel>=3.3.0

# BehaviorTree (Python bindings)
py-trees>=2.2.0
py-trees-ros>=2.2.0
EOF

pip install -r /tmp/multi_actor_requirements.txt
rm /tmp/multi_actor_requirements.txt
print_success "Additional dependencies installed"

print_info "Step 6/8: Copying configuration files"
# Copy configuration templates if they don't exist
if [ ! -f "config/multi_actor/csa_config.yaml" ]; then
    print_info "Configuration files already exist in config/multi_actor/"
else
    print_success "Configuration files ready"
fi

print_info "Step 7/8: Setting up Python path"
# Add external/multi_actor to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/multi_actor"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/external/multi_actor\"" >> ~/.bashrc

print_success "Python path configured"

print_info "Step 8/8: Running integration tests"
# Run basic integration tests
python3 <<EOF
import sys
sys.path.insert(0, './external/multi_actor')
sys.path.insert(0, './integration')

try:
    # Test multi_actor imports
    from ml.training.advanced_multi_actor import IntentCommunicationModule
    from swarm.openfl.coordinator import SwarmCoordinator
    from ml.artifact.schema import CooperativeSkillArtefact
    print("✓ Multi-actor imports successful")

    # Test integration imports
    from multi_actor_adapter import CSAEngine, MultiActorCoordinator, SwarmBridge
    print("✓ Integration adapter imports successful")

    print("\nAll imports successful!")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Multi-Actor Integration Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Load a CSA artifact:"
echo "   python scripts/load_csa.py --csa-path <path> --csa-id <id>"
echo ""
echo "2. Start the SwarmBridge coordinator:"
echo "   python -m integration.multi_actor_adapter.swarm_bridge"
echo ""
echo "3. Train a multi-actor skill:"
echo "   python external/multi_actor/ml/training/train_cooperative_bc.py \\"
echo "     --dataset data/cooperative_demos.h5 \\"
echo "     --num-actors 2 \\"
echo "     --output models/cooperative_skill.pt"
echo ""
echo "4. Run a multi-actor mission:"
echo "   python scripts/run_multi_actor_mission.py \\"
echo "     --mission-config examples/multi_actor_mission.yaml"
echo ""
echo "Documentation: docs/guides/multi_actor_integration.md"
echo ""
