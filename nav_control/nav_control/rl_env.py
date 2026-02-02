import numpy as np
from math import sqrt, atan2, pi


class PathEnv:
    def __init__(self):
        self.observation_shape = 5
        self.num_actions = 2
        self.observation_space = type('obj', (object,), {'n': self.observation_shape})()
        self.action_space = type('obj', (object,), {'n': self.num_actions})()
        self.n_active_features = 1

        self.max_v = 0.5
        self.max_w = 1.0
        self.dt = 0.1
        self.goal_thresh = 0.4
        self.collision_thresh = 0.3
        self.max_cte = 1.5

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0

        self.path = []
        self.wp_idx = 0
        self.obstacles = []
        self.steps = 0
        self.max_steps = 300
        self.prev_dist = 0.0

    def set_path(self, path):
        self.path = path
        self.wp_idx = 0

    def set_obstacles(self, obs):
        self.obstacles = obs

    def set_robot_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0
        self.wp_idx = 0
        self.v = 0.0
        self.w = 0.0

        if self.path:
            tx, ty = self.path[0]
            self.prev_dist = sqrt((tx - self.x)**2 + (ty - self.y)**2)

        return self.get_state(), {}

    def step(self, action):
        self.steps += 1

        action = action.cpu().numpy().flatten()
        self.v = float(action[0]) * self.max_v
        self.w = float(action[1]) * self.max_w

        self.yaw += self.w * self.dt
        self.yaw = self.wrap(self.yaw)
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt

        reward, done, trunc = self.calc_reward()
        state = self.get_state()

        return state, reward, done, trunc, {'wp_idx': self.wp_idx}

    def get_state(self):
        if not self.path:
            return np.zeros(self.observation_shape, dtype=np.float32)

        if self.wp_idx >= len(self.path):
            self.wp_idx = len(self.path) - 1

        tx, ty = self.path[self.wp_idx]
        cte = self.cross_track_error()
        herr = self.heading_error()
        dist = sqrt((tx - self.x)**2 + (ty - self.y)**2)

        return np.array([
            np.clip(cte / self.max_cte, -1, 1),
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

    def heading_error(self):
        if not self.path:
            return 0.0
        tx, ty = self.path[self.wp_idx]
        target = atan2(ty - self.y, tx - self.x)
        return self.wrap(target - self.yaw)

    def calc_reward(self):
        done = False
        trunc = False
        reward = 0.0

        if not self.path:
            return 0.0, True, False

        tx, ty = self.path[self.wp_idx]
        dist = sqrt((tx - self.x)**2 + (ty - self.y)**2)

        progress = self.prev_dist - dist
        reward += progress * 10.0
        self.prev_dist = dist

        cte = self.cross_track_error()
        reward -= cte * 0.5

        herr = abs(self.heading_error())
        reward -= herr * 0.3

        if abs(self.w) > 0.5 and herr < 0.3:
            reward -= abs(self.w) * 0.5

        if self.v > 0.1:
            reward += 0.1
        elif self.v < 0:
            reward -= 0.2

        if dist < self.goal_thresh:
            reward += 10.0
            self.wp_idx += 1

            if self.wp_idx < len(self.path):
                nx, ny = self.path[self.wp_idx]
                self.prev_dist = sqrt((nx - self.x)**2 + (ny - self.y)**2)

            if self.wp_idx >= len(self.path):
                reward += 100.0
                done = True

        if cte > self.max_cte:
            reward -= 50.0
            done = True

        for ox, oy, r in self.obstacles:
            d = sqrt((ox - self.x)**2 + (oy - self.y)**2)
            if d < (r + self.collision_thresh):
                reward -= 100.0
                done = True
                break

        if self.steps >= self.max_steps:
            trunc = True

        return reward, done, trunc

    def wrap(self, a):
        while a > pi:
            a -= 2 * pi
        while a < -pi:
            a += 2 * pi
        return a
