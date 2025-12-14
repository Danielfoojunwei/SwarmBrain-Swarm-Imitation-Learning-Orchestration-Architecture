#!/bin/bash
#
# Setup Script for Dynamical.ai Integration
#
# This script sets up the dynamical_2 repository as a git submodule
# and configures the integration with SwarmBrain.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Dynamical.ai Integration Setup${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Step 1: Initialize git submodule
echo -e "${YELLOW}Step 1: Setting up dynamical_2 submodule...${NC}"

if [ ! -d "external/dynamical_2" ]; then
    echo "Cloning dynamical_2 repository..."
    git submodule add https://github.com/Danielfoojunwei/dynamical_2.git external/dynamical_2 || true
    git submodule update --init --recursive
    echo -e "${GREEN}✓ dynamical_2 cloned successfully${NC}"
else
    echo "dynamical_2 already exists, updating..."
    git submodule update --remote external/dynamical_2
    echo -e "${GREEN}✓ dynamical_2 updated${NC}"
fi

# Step 2: Install dynamical_2 dependencies
echo -e "${YELLOW}Step 2: Installing dynamical_2 dependencies...${NC}"

if [ -f "external/dynamical_2/requirements.txt" ]; then
    pip install -r external/dynamical_2/requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Warning: requirements.txt not found in dynamical_2${NC}"
fi

# Step 3: Create model directories
echo -e "${YELLOW}Step 3: Creating model directories...${NC}"

mkdir -p robot_control/skills/dynamical/models
mkdir -p robot_control/skills/dynamical/checkpoints
mkdir -p robot_control/skills/dynamical/demonstrations
mkdir -p config/dynamical

echo -e "${GREEN}✓ Directories created${NC}"

# Step 4: Create default configuration
echo -e "${YELLOW}Step 4: Creating default configuration...${NC}"

cat > config/dynamical/skills.yaml <<EOF
# Dynamical.ai Skill Configuration

skills:
  - name: pick_and_place
    policy_type: diffusion
    model_path: robot_control/skills/dynamical/models/pick_and_place.pt
    input_modalities:
      - rgb
      - depth
      - proprioception
    action_dim: 7
    observation_dim: 1024
    enabled: true

  - name: assembly
    policy_type: act
    model_path: robot_control/skills/dynamical/models/assembly.pt
    input_modalities:
      - rgb
      - proprioception
    action_dim: 14
    observation_dim: 512
    enabled: true

  - name: inspection
    policy_type: transformer
    model_path: robot_control/skills/dynamical/models/inspection.pt
    input_modalities:
      - rgb
      - depth
    action_dim: 7
    observation_dim: 2048
    enabled: false

# Training configuration
training:
  workspace_dir: /workspace/training
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  device: cuda

# Federated learning integration
federated_learning:
  enabled: true
  server_address: fl-server:8080
  sync_interval: 3600  # seconds
EOF

echo -e "${GREEN}✓ Configuration created${NC}"

# Step 5: Test import
echo -e "${YELLOW}Step 5: Testing integration...${NC}"

python3 -c "
import sys
sys.path.insert(0, 'external/dynamical_2')
try:
    from integration.dynamical_adapter.skill_engine import DynamicalSkillEngine
    print('✓ Skill engine import successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Integration test passed${NC}"
else
    echo -e "${RED}✗ Integration test failed${NC}"
    exit 1
fi

# Step 6: Create symlink for legacy code
echo -e "${YELLOW}Step 6: Linking legacy code...${NC}"

if [ -d "legacy_code" ]; then
    ln -sf "$(pwd)/legacy_code" "$(pwd)/external/dynamical_2/legacy_integration"
    echo -e "${GREEN}✓ Legacy code linked${NC}"
fi

# Done
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Next steps:"
echo "1. Train your first skill:"
echo "   python scripts/train_skill.py --skill pick_and_place --data demonstrations/"
echo ""
echo "2. Load skill in robot:"
echo "   python scripts/load_skill.py --robot robot_001 --skill pick_and_place"
echo ""
echo "3. Start federated learning:"
echo "   docker-compose --profile fl up -d"
echo ""
