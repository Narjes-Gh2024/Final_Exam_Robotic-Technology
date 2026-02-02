import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty


class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.limit = abs(float(limit))
        self.integral = 0.0
        self.prev = 0.0
        self.started = False

    def reset(self):
        self.integral = 0.0
        self.prev = 0.0
        self.started = False

    def step(self, err, dt):
        if dt <= 0.0:
            return 0.0
        if not self.started:
            self.prev = err
            self.started = True
        
        self.integral += err * dt
        self.integral = max(-self.limit, min(self.integral, self.limit))
        
        deriv = (err - self.prev) / dt
        self.prev = err
        
        return self.kp * err + self.ki * self.integral + self.kd * deriv


class PIDFollower(Node):
    def __init__(self):
        super().__init__('pid_follower')
        
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('wp_thresh', 0.3)
        self.declare_parameter('kp_v', 0.8)
        self.declare_parameter('ki_v', 0.0)
        self.declare_parameter('kd_v', 0.1)
        self.declare_parameter('kp_w', 2.5)
        self.declare_parameter('ki_w', 0.0)
        self.declare_parameter('kd_w', 0.2)
        self.declare_parameter('max_v', 0.5)
        self.declare_parameter('max_w', 1.0)
        self.declare_parameter('slow_angle', 0.6)

        self.dt = float(self.get_parameter('dt').value)
        self.wp_thresh = float(self.get_parameter('wp_thresh').value)
        self.max_v = float(self.get_parameter('max_v').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.slow_angle = float(self.get_parameter('slow_angle').value)

        self.pid_v = PID(
            self.get_parameter('kp_v').value,
            self.get_parameter('ki_v').value,
            self.get_parameter('kd_v').value,
            limit=0.7
        )
        self.pid_w = PID(
            self.get_parameter('kp_w').value,
            self.get_parameter('ki_w').value,
            self.get_parameter('kd_w').value,
            limit=1.0
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.has_pose = False
        self.path = []
        self.wp_idx = 0
        self.active = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_pose, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.on_path, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.plan_client = self.create_client(GetPlan, '/plan_path')
        self.start_srv = self.create_service(Empty, '/start_pid', self.on_start)
        self.stop_srv = self.create_service(Empty, '/stop_pid', self.on_stop)
        self.timer = self.create_timer(self.dt, self.control)

    def wrap(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def on_pose(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.has_pose = True

    def on_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.wp_idx = 0
        self.pid_v.reset()
        self.pid_w.reset()
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
        if not self.active or not self.has_pose or not self.path:
            return

        if self.wp_idx >= len(self.path):
            self.active = False
            self.stop()
            return

        wx, wy = self.path[self.wp_idx]
        dx = wx - self.x
        dy = wy - self.y
        dist = math.hypot(dx, dy)

        if dist < self.wp_thresh:
            self.wp_idx += 1
            self.pid_v.reset()
            self.pid_w.reset()
            return

        target_yaw = math.atan2(dy, dx)
        heading_err = self.wrap(target_yaw - self.yaw)

        w = self.pid_w.step(heading_err, self.dt)
        v = self.pid_v.step(dist, self.dt)

        if abs(heading_err) > self.slow_angle:
            v *= 0.2

        v = max(-self.max_v, min(self.max_v, v))
        w = max(-self.max_w, min(self.max_w, w))

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PIDFollower()
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
