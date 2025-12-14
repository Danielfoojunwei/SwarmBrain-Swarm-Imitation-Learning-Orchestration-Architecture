# SwarmBrain Robot Controller Dockerfile
# Based on ROS 2 Humble with Python 3.10

FROM osrf/ros:humble-desktop-full

# Metadata
LABEL maintainer="SwarmBrain Team"
LABEL description="Robot controller node for SwarmBrain multi-robot system"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ROS 2 Python packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-nav2-msgs \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY robot_control /workspace/robot_control
COPY crypto /workspace/crypto
COPY interfaces /workspace/interfaces
COPY config /workspace/config

# Source ROS 2 setup
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Expose ports (if needed for debugging)
EXPOSE 11311

# Set entrypoint to source ROS and run robot controller
COPY docker/entrypoint-robot.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "robot_control.ros2_nodes.robot_controller"]
