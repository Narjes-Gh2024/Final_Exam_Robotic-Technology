import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA
import numpy as np
import heapq
from math import sqrt


class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        
        self.declare_parameter('robot_length', 0.4)
        self.declare_parameter('robot_width', 0.46)
        self.declare_parameter('safety_margin', 1.15)
        
        length = self.get_parameter('robot_length').value
        width = self.get_parameter('robot_width').value
        margin = self.get_parameter('safety_margin').value
        
        diagonal = sqrt((length/2)**2 + (width/2)**2)
        self.radius = margin * diagonal
        
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_pose, 10)
        self.srv = self.create_service(GetPlan, '/plan_path', self.on_plan_request)
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        
        self.map = None
        self.inflated = None
        self.pose = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        
    
    def on_map(self, msg):
        self.map = msg
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.inflate_obstacles()
    
    def inflate_obstacles(self):
        if not self.map:
            return
        
        w = self.map.info.width
        h = self.map.info.height
        res = self.map.info.resolution
        
        grid = np.array(self.map.data).reshape((h, w))
        cells = int(np.ceil(self.radius / res))
        
        self.inflated = grid.copy()
        
        obstacles = np.where(grid > 50)
        unknown = np.where(grid < 0)
        
        for oy, ox in zip(obstacles[0], obstacles[1]):
            for dy in range(-cells, cells + 1):
                for dx in range(-cells, cells + 1):
                    if dx*dx + dy*dy <= cells*cells:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if self.inflated[ny, nx] < 50:
                                self.inflated[ny, nx] = 99
        
        for oy, ox in zip(unknown[0], unknown[1]):
            for dy in range(-cells, cells + 1):
                for dx in range(-cells, cells + 1):
                    if dx*dx + dy*dy <= cells*cells:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if self.inflated[ny, nx] == 0:
                                self.inflated[ny, nx] = 50
    
    def on_pose(self, msg):
        self.pose = msg.pose.pose
    
    def on_plan_request(self, req, res):
        if not self.map or self.inflated is None or not self.pose:
            res.plan = Path()
            return res
        
        sx, sy = self.pose.position.x, self.pose.position.y
        gx, gy = req.goal.pose.position.x, req.goal.pose.position.y
        
        start = self.to_grid(sx, sy)
        goal = self.to_grid(gx, gy)
        
        path = self.find_path(start[0], start[1], goal[0], goal[1])
        
        if not path:
            res.plan = Path()
            return res
        
        msg = self.make_path_msg(path)
        self.path_pub.publish(msg)
        res.plan = msg
        
        return res
    
    def to_grid(self, wx, wy):
        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y
        return (int((wx - ox) / res), int((wy - oy) / res))
    
    def to_world(self, gx, gy):
        res = self.map.info.resolution
        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y
        return (gx * res + ox + res/2, gy * res + oy + res/2)
    
    def valid(self, x, y):
        w = self.map.info.width
        h = self.map.info.height
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return self.inflated[y, x] < 50
    
    def neighbors(self, x, y):
        dirs = [(1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0),
                (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]
        result = []
        for dx, dy, cost in dirs:
            nx, ny = x + dx, y + dy
            if self.valid(nx, ny):
                if abs(dx) == 1 and abs(dy) == 1:
                    if not self.valid(x+dx, y) or not self.valid(x, y+dy):
                        continue
                result.append((nx, ny, cost))
        return result
    
    def find_nearest_valid(self, x, y):
        for r in range(1, 20):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = x + dx, y + dy
                        if self.valid(nx, ny):
                            return nx, ny
        return None, None
    
    def find_path(self, sx, sy, gx, gy):
        if not self.valid(sx, sy):
            sx, sy = self.find_nearest_valid(sx, sy)
            if sx is None:
                return None
        
        if not self.valid(gx, gy):
            gx, gy = self.find_nearest_valid(gx, gy)
            if gx is None:
                return None
        
        cnt = 0
        heap = [(0, cnt, sx, sy)]
        cnt += 1
        
        g = {(sx, sy): 0}
        parent = {}
        closed = set()
        
        while heap:
            _, _, cx, cy = heapq.heappop(heap)
            
            if cx == gx and cy == gy:
                path = [(cx, cy)]
                while (cx, cy) in parent:
                    cx, cy = parent[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return path
            
            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))
            
            for nx, ny, cost in self.neighbors(cx, cy):
                if (nx, ny) in closed:
                    continue
                
                ng = g[(cx, cy)] + cost
                
                if (nx, ny) not in g or ng < g[(nx, ny)]:
                    parent[(nx, ny)] = (cx, cy)
                    g[(nx, ny)] = ng
                    f = ng + sqrt((gx-nx)**2 + (gy-ny)**2)
                    heapq.heappush(heap, (f, cnt, nx, ny))
                    cnt += 1
        
        return None
    
    def make_path_msg(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for gx, gy in path:
            pose = PoseStamped()
            pose.header = msg.header
            wx, wy = self.to_world(gx, gy)
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
