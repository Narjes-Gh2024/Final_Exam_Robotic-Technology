import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt, atan2, pi
import os


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        return torch.tanh(self.out(x))


class RLFollower(Node):
    def __init__(self):
        super().__init__('rl_follower')

        default_model = '/home/narjes/project_ws/src/robotic_course/nav_control/nav_control/models/ddpg_final.pth'

        
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('max_v', 0.5)
        self.declare_parameter('max_w', 1.0)
        self.declare_parameter('goal_dist', 0.4)
        self.declare_parameter('rate', 10.0)
        
        self.model_path = self.get_parameter('model_path').value
        self.max_v = self.get_parameter('max_v').value
        self.max_w = self.get_parameter('max_w').value
        self.goal_dist = self.get_parameter('goal_dist').value
        rate = self.get_parameter('rate').value
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.path = []
        self.wp_idx = 0
        self.active = False
        self.has_pose = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(5, 2).to(self.device)
        self.load_model()
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_pose, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.on_path, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.plan_client = self.create_client(GetPlan, '/plan_path')
        self.start_srv = self.create_service(Empty, '/start_rl', self.on_start)
        self.stop_srv = self.create_service(Empty, '/stop_rl', self.on_stop)
        self.timer = self.create_timer(1.0 / rate, self.control)
        
        self.get_logger().info('RL Follower Ready')
    
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.actor.load_state_dict(ckpt['actor'])
                self.actor.eval()
                self.get_logger().info('Model loaded')
            except Exception as e:
                self.get_logger().error(f'Load failed: {e}')
        else:
            self.get_logger().warn('Model not found')
    
    def on_pose(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.has_pose = True
    
    def on_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.wp_idx = 0
        self.active = True
        cmd = Twist()
        cmd.linear.x = 0.1
        self.cmd_pub.publish(cmd)
    
    def on_goal(self, msg):
        if not self.plan_client.wait_for_service(timeout_sec=1.0):
            return
        req = GetPlan.Request()
        req.goal = msg
        self.plan_client.call_async(req)
    
    def on_start(self, req, res):
        self.active = True
        return res
    
    def on_stop(self, req, res):
        self.active = False
        self.stop()
        return res
    
    def control(self):
        if not self.active or not self.path or not self.has_pose:
            return
        
        if self.wp_idx >= len(self.path):
            self.active = False
            self.stop()
            return
        
        tx, ty = self.path[self.wp_idx]
        dist = sqrt((tx - self.x)**2 + (ty - self.y)**2)
        
        if dist < self.goal_dist:
            self.wp_idx += 1
            return
        
        state = self.get_state()
        action = self.get_action(state)
        self.apply(action)
    
    def get_state(self):
        tx, ty = self.path[self.wp_idx]
        
        cte = self.cross_track_error()
        heading = atan2(ty - self.y, tx - self.x)
        herr = self.wrap(heading - self.yaw)
        dist = sqrt((tx - self.x)**2 + (ty - self.y)**2)
        
        return np.array([
            np.clip(cte / 1.5, -1, 1),
            herr / pi,
            np.clip(dist / 5.0, 0, 1),
            self.v / self.max_v,
            self.w / self.max_w
        ], dtype=np.float32)
    
    def cross_track_error(self):
        if len(self.path) < 2 or self.wp_idx == 0:
            if self.path:
                tx, ty = self.path[self.wp_idx]
                return sqrt((tx - self.x)**2 + (ty - self.y)**2)
            return 0.0
        
        p1 = np.array(self.path[max(0, self.wp_idx - 1)])
        p2 = np.array(self.path[self.wp_idx])
        robot = np.array([self.x, self.y])
        
        line = p2 - p1
        length = np.linalg.norm(line)
        
        if length < 0.001:
            return np.linalg.norm(robot - p1)
        
        unit = line / length
        proj = np.clip(np.dot(robot - p1, unit), 0, length)
        closest = p1 + proj * unit
        
        return np.linalg.norm(robot - closest)
    
    def wrap(self, a):
        while a > pi:
            a -= 2 * pi
        while a < -pi:
            a += 2 * pi
        return a
    
    def get_action(self, state):
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.actor(t).cpu().numpy().flatten()
    
    def apply(self, action):
        cmd = Twist()
        v = float(action[0]) * self.max_v
        w = float(action[1]) * self.max_w
        
        if abs(w) > 0.5 and abs(v) < 0.15:
            v = 0.2
            w = np.clip(w, -0.5, 0.5)
        
        cmd.linear.x = v
        cmd.angular.z = w
        self.v = v
        self.w = w
        self.cmd_pub.publish(cmd)
    
    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.v = 0.0
        self.w = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = RLFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
