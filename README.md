# ROS 2 Autonomous Navigation and Learning-Based Control

A complete ROS 2 navigation project integrating robot simulation, map-based localization, custom A* planning, PID control, and a DDPG controller implemented without an external reinforcement-learning library.

## Workspace packages

- `robot_description`: robot URDF, meshes, Gazebo worlds, RViz configurations, bridges, IMU and odometry support.
- `nav_control`: map server/AMCL launch, A* path planning, scan conversion, odometry TF, PID following, DDPG training, and RL following.

## Requirements

Ubuntu with ROS 2 Humble or compatible, Gazebo Sim, RViz2, Python 3, C++17, Colcon, `nav2_map_server`, `nav2_amcl`, `nav2_lifecycle_manager`, `ros_gz_sim`, `ros_gz_bridge`, `tf2_ros`, and standard ROS 2 message packages.

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo apt install -y ros-<ros_distro>-rviz2 ros-<ros_distro>-xacro
sudo apt install -y ros-<ros_distro>-robot-state-publisher ros-<ros_distro>-joint-state-publisher
sudo apt install -y ros-<ros_distro>-ros-gz-sim ros-<ros_distro>-ros-gz-bridge
sudo apt install -y ros-<ros_distro>-nav2-map-server ros-<ros_distro>-nav2-amcl ros-<ros_distro>-nav2-lifecycle-manager
```

## Clone and build

```bash
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws/src
git clone https://github.com/Narjes-Gh2024/Final_Exam_Robotic-Technology.git
cd ~/robotics_ws
source /opt/ros/<ros_distro>/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Robot simulation

Display the robot model:

```bash
ros2 launch robot_description display.launch.py
```

Start Gazebo, spawn the robot, bridge simulator topics, publish TF, and open RViz:

```bash
ros2 launch robot_description gazebo.launch.py
```

## Localization and navigation

Start the map server and AMCL-based localization stack:

```bash
ros2 launch nav_control localization.launch.py
```

The navigation package includes:

- Map server and AMCL localization
- Occupancy-grid handling
- Custom A* global planner
- TF conversion and odometry support
- Laser-scan conversion
- PID and RL trajectory following

## Executables

```bash
ros2 run nav_control path_planner
ros2 run nav_control pid_follower
ros2 run nav_control rl_follower
ros2 run nav_control odom_tf
ros2 run nav_control scan_converter
```

## Learning-based control

The package contains:

- `ddpg.py`: actor-critic DDPG implementation.
- `rl_env.py`: path-following environment.
- `rl_train.py`: training workflow.
- `rl_follower.py`: inference and trajectory following.
- `pid_follower.py`: classical baseline for comparison.

Train the controller from the package environment:

```bash
python3 -m nav_control.rl_train
```

Use the trained controller through ROS 2 after building and sourcing the workspace:

```bash
ros2 run nav_control rl_follower
```

## Interfaces

Inspect available nodes and topics:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /scan
ros2 topic echo /odom
ros2 topic echo /cmd_vel
ros2 run tf2_tools view_frames
```

The main navigation frames are `map -> odom -> base_link`. The planner publishes a `nav_msgs/Path`; controllers consume path/pose information and publish velocity commands.

## Configuration

- `nav_control/maps/`: map files.
- `nav_control/config/map_server_params.yaml`: map-server configuration.
- `nav_control/config/amcl_params.yaml`: AMCL parameters.
- `robot_description/config/`: simulator bridge and robot parameters.
- `robot_description/launch/`: RViz and Gazebo launch files.
- `Videos/`: PID and RL tracking demonstrations.

## Troubleshooting

Source ROS and the workspace in every terminal. If a package is not found, rebuild with `colcon build --symlink-install`. If AMCL does not converge, verify that the map, laser scan, and TF chain are available. If velocity commands do not move the robot, inspect `/cmd_vel`, the bridge configuration, and the motor command node.

## Author

**Narjes Ghazanfari** — B.Sc. Aerospace Engineering, Sharif University of Technology.

Research interests: autonomous robotics, dynamical modeling, feedback control, and trajectory tracking.
