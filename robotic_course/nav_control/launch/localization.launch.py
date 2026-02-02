import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from pathlib import Path


def generate_launch_description():
    robot_dir = get_package_share_directory('robot_description')
    nav_dir = get_package_share_directory('nav_control')

    world = os.path.join(robot_dir, 'world', 'depot.sdf')
    urdf = os.path.join(robot_dir, 'src', 'description', 'robot.urdf')
    rviz_cfg = os.path.join(robot_dir, 'rviz', 'config.rviz')
    bridge_cfg = os.path.join(robot_dir, 'config', 'gz_bridge.yaml')
    
    map_yaml = os.path.join(nav_dir, 'maps', 'my_map.yaml')
    amcl_cfg = os.path.join(nav_dir, 'config', 'amcl_params.yaml')

    with open(urdf, 'r') as f:
        robot_desc = f.read()

    gz_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            os.path.join(robot_dir, 'world'),
            str(Path(robot_dir).parent.resolve())
        ])
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -v 4 ', world]}.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_cfg,
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'use_sim_time': True}, {'robot_description': robot_desc}]
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'robot', '-topic', '/robot_description', '-x', '0', '-y', '0', '-z', '0.9'],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen'
    )

    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['--frame-id', 'base_link', '--child-frame-id', 'robot/base_link/rplidar_c1_sensor',
                   '--x', '0', '--y', '0', '--z', '0.4', '--roll', '0', '--pitch', '0', '--yaw', '0'],
        parameters=[{'use_sim_time': True}]
    )

    odom_tf = Node(
        package='nav_control',
        executable='odom_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        output='screen',
        parameters=[{'use_sim_time': True}, {'yaml_filename': map_yaml}, {'frame_id': 'map'}, {'topic_name': '/map'}]
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        output='screen',
        parameters=[amcl_cfg, {'use_sim_time': True}],
        remappings=[('/odom', '/wheel_encoder/odom')]
    )

    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': True}, {'autostart': True}, {'node_names': ['map_server', 'amcl']}]
    )

    planner = Node(
        package='nav_control',
        executable='path_planner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    scan_conv = Node(
        package='nav_control',
        executable='scan_converter',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        gz_path,
        gz_sim,
        bridge,
        robot_state_pub,
        spawn,
        lidar_tf,
        rviz,
        odom_tf,
        scan_conv,
        map_server,
        amcl,
        lifecycle,
        planner,
    ])
