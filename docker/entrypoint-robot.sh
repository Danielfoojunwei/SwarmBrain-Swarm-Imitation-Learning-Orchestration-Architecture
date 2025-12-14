#!/bin/bash
set -e

# Source ROS 2 setup
source /opt/ros/${ROS_DISTRO}/setup.bash

# Wait for orchestrator to be ready
echo "Waiting for orchestrator to be ready..."
while ! nc -z orchestrator 8000; do
  sleep 1
done
echo "Orchestrator is ready!"

# Wait for FL server to be ready
echo "Waiting for FL server to be ready..."
while ! nc -z fl-server 8080; do
  sleep 1
done
echo "FL server is ready!"

# Execute the main command
exec "$@"
