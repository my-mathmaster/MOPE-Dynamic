import pygame
import math
import heapq
import json
import random
import time
import numpy as np
from collections import deque

# 配置参数
CELL_SIZE = 20
GRID_WIDTH = 40
GRID_HEIGHT = 30
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.parent = None
        self.neighbors = []

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class PathPlanner:
    def __init__(self):
        self.grid = [[Node(x, y) for y in range(GRID_HEIGHT)] for x in range(GRID_WIDTH)]
        self.static_obstacles = set()
        self.dynamic_obstacles = []
        self.start = None
        self.end = None
        self.targets = []  # 多个目标点（中继点）
        self.path = []
        self.search_nodes = []
        self.algorithm = 'A*'  # 可选'A*'或'DWA'
        self.performance_stats = {
            'path_length': 0,
            'computation_time': 0,
            'collisions': 0,
            'last_update': time.time()
        }
        self.setup_neighbors()

        # 动态窗口法参数
        self.config = {
            'max_vel': 2.0,
            'max_rot': math.pi / 4,
            'acc_lim': 1.0,
            'rot_lim': math.pi / 2,
            'predict_time': 2.0,
            'dt': 0.1,
            'robot_radius': 0.5,
            'vel_step': 0.5,
            'rot_step': math.pi / 12,
            'weights': {
                'goal': 0.4,
                'clearance': 0.3,
                'velocity': 0.2,
                'smoothness': 0.1
            }
        }

        # 机器人状态 (x, y, theta, v, w)
        self.robot_state = [1, 1, 0, 0, 0]

        # 多目标优化权重
        self.weights = {
            'length': 0.4,
            'time': 0.3,
            'energy': 0.3
        }

    def setup_neighbors(self):
        """为每个网格节点建立邻居关系"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]

        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        self.grid[x][y].neighbors.append(self.grid[nx][ny])

    def heuristic(self, node1, node2):
        """启发函数（欧几里得距离）"""
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def a_star(self):
        """A*算法实现"""
        start_time = time.time()
        if not self.start or not self.end:
            return []

        # 初始化
        for row in self.grid:
            for node in row:
                node.g = float('inf')
                node.h = self.heuristic(node, self.end)
                node.parent = None

        self.start.g = 0
        self.search_nodes = []

        # 优先队列
        open_set = [(self.start.g + self.start.h, id(self.start), self.start)]
        heapq.heapify(open_set)
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[2]
            self.search_nodes.append((current.x, current.y))

            if current == self.end:
                # 重建路径
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent

                self.performance_stats['computation_time'] = time.time() - start_time
                self.performance_stats['path_length'] = len(path)
                return path[::-1]

            closed_set.add((current.x, current.y))

            for neighbor in current.neighbors:
                if (neighbor.x, neighbor.y) in closed_set:
                    continue

                # 碰撞检测
                if (neighbor.x, neighbor.y) in self.static_obstacles or \
                        any(self.distance(neighbor.x, neighbor.y, ox, oy) <= radius
                            for ox, oy, radius, _, _, _ in self.dynamic_obstacles):
                    continue

                tentative_g = current.g + self.heuristic(current, neighbor)

                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current

                    # 检查是否在open_set中
                    in_open = any(item[2] == neighbor for item in open_set)
                    if not in_open:
                        heapq.heappush(open_set,
                                       (neighbor.g + neighbor.h, id(neighbor), neighbor))

        self.performance_stats['computation_time'] = time.time() - start_time
        return []  # 未找到路径

    def distance(self, x1, y1, x2, y2):
        """计算两点之间的距离"""
        return math.hypot(x1 - x2, y1 - y2)

    def distance_to_robot(self, x, y):
        """计算点到机器人的距离"""
        return math.hypot(x - self.robot_state[0], y - self.robot_state[1])

    def dwa_plan(self, target_pos):
        """动态窗口法（DWA）实现"""
        start_time = time.time()
        best_score = -float('inf')
        best_vel = None
        best_traj = []

        # 获取当前机器人状态
        x, y, theta, v, w = self.robot_state

        # 动态窗口计算
        min_vel = max(0, v - self.config['acc_lim'] * self.config['dt'])
        max_vel = min(self.config['max_vel'], v + self.config['acc_lim'] * self.config['dt'])
        min_rot = max(-self.config['max_rot'], w - self.config['rot_lim'] * self.config['dt'])
        max_rot = min(self.config['max_rot'], w + self.config['rot_lim'] * self.config['dt'])

        # 采样速度空间
        v_samples = np.arange(min_vel, max_vel, self.config['vel_step'])
        w_samples = np.arange(min_rot, max_rot, self.config['rot_step'])

        # 添加零速度和当前速度
        if 0 not in v_samples:
            v_samples = np.append(v_samples, 0)
        if v not in v_samples:
            v_samples = np.append(v_samples, v)
        if w not in w_samples:
            w_samples = np.append(w_samples, w)

        for v_sample in v_samples:
            for w_sample in w_samples:
                # 模拟轨迹
                traj = self.simulate_trajectory(x, y, theta, v_sample, w_sample)

                if not traj:
                    continue

                # 计算评价函数
                goal_score = self.calculate_goal_score(traj, target_pos)
                clearance_score = self.calculate_clearance_score(traj)
                velocity_score = v_sample / self.config['max_vel']
                smoothness_score = self.calculate_smoothness_score(traj)

                # 综合评分
                weights = self.config['weights']
                score = (weights['goal'] * goal_score +
                         weights['clearance'] * clearance_score +
                         weights['velocity'] * velocity_score +
                         weights['smoothness'] * smoothness_score)

                if score > best_score:
                    best_score = score
                    best_vel = (v_sample, w_sample)
                    best_traj = traj

        # 更新性能统计
        self.performance_stats['computation_time'] = time.time() - start_time

        # 更新机器人状态（只返回轨迹，实际状态在仿真中更新）
        return best_traj, best_vel

    def calculate_goal_score(self, traj, target_pos):
        """计算目标得分"""
        # 轨迹终点距离目标的位置
        end_x, end_y = traj[-1][0], traj[-1][1]
        dist = self.distance(end_x, end_y, target_pos[0], target_pos[1])
        return 1.0 / (dist + 0.1)

    def calculate_clearance_score(self, traj):
        """计算安全距离得分"""
        min_clearance = float('inf')

        for point in traj:
            x, y = point[0], point[1]

            # 静态障碍物
            for ox, oy in self.static_obstacles:
                dist = self.distance(x, y, ox, oy)
                if dist < min_clearance:
                    min_clearance = dist

            # 动态障碍物
            for ox, oy, radius, _, _, _ in self.dynamic_obstacles:
                dist = self.distance(x, y, ox, oy) - radius
                if dist < min_clearance:
                    min_clearance = dist

        # 避免除零错误
        min_clearance = max(min_clearance, 0.01)
        return min(1.0, min_clearance / 5.0)

    def calculate_smoothness_score(self, traj):
        """计算轨迹平滑度得分"""
        if len(traj) < 3:
            return 1.0

        angles = []
        for i in range(1, len(traj) - 1):
            x1, y1 = traj[i - 1][0], traj[i - 1][1]
            x2, y2 = traj[i][0], traj[i][1]
            x3, y3 = traj[i + 1][0], traj[i + 1][1]

            vec1 = (x2 - x1, y2 - y1)
            vec2 = (x3 - x2, y3 - y2)

            # 计算角度变化
            dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = math.hypot(vec1[0], vec1[1])
            mag2 = math.hypot(vec2[0], vec2[1])

            if mag1 > 0 and mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                angle = math.acos(max(-1, min(1, cos_angle)))
                angles.append(angle)

        if not angles:
            return 1.0

        avg_angle = sum(angles) / len(angles)
        return 1.0 / (avg_angle + 0.1)

    def simulate_trajectory(self, x, y, theta, v, w):
        """模拟给定速度的轨迹"""
        traj = []
        dt = self.config['dt']
        predict_time = self.config['predict_time']
        steps = int(predict_time / dt)

        current_x, current_y, current_theta = x, y, theta

        for _ in range(steps):
            # 更新位置和朝向
            current_x += v * math.cos(current_theta) * dt
            current_y += v * math.sin(current_theta) * dt
            current_theta += w * dt

            # 规范化角度
            current_theta = current_theta % (2 * math.pi)

            # 检查边界
            if not (0 <= current_x < GRID_WIDTH and 0 <= current_y < GRID_HEIGHT):
                return None

            # 检查碰撞
            if self.check_collision(current_x, current_y):
                return None

            traj.append((current_x, current_y, current_theta))

        return traj

    def check_collision(self, x, y):
        """检查给定位置是否与障碍物碰撞"""
        # 静态障碍物
        if (int(x), int(y)) in self.static_obstacles:
            return True

        # 动态障碍物
        for ox, oy, radius, _, _, _ in self.dynamic_obstacles:
            if self.distance(x, y, ox, oy) <= radius + self.config['robot_radius']:
                self.performance_stats['collisions'] += 1
                return True

        return False

    def optimize_path(self):
        """多目标优化（简单加权优化示例）"""
        if not self.targets:
            return []

        # 使用简单的贪心算法解决TSP问题
        targets = [self.start] + self.targets + [self.end]
        unvisited = set(targets[1:-1])  # 排除起点和终点
        current = targets[0]
        optimized_path = [current]

        while unvisited:
            # 找到最近的目标点
            nearest = None
            min_dist = float('inf')

            for target in unvisited:
                dist = self.heuristic(current, target)
                if dist < min_dist:
                    min_dist = dist
                    nearest = target

            if nearest:
                # 规划到最近目标点的路径
                self.end = nearest
                segment = self.a_star()
                if segment:
                    # 转换为节点坐标
                    segment_nodes = [self.grid[x][y] for x, y in segment]
                    optimized_path.extend(segment_nodes[1:])  # 跳过起点

                current = nearest
                unvisited.remove(nearest)

        # 规划到最终终点的路径
        self.end = targets[-1]
        segment = self.a_star()
        if segment:
            segment_nodes = [self.grid[x][y] for x, y in segment]
            optimized_path.extend(segment_nodes[1:])

        # 转换为坐标序列
        path_coords = [(node.x, node.y) for node in optimized_path]

        # 计算路径指标
        path_length = len(path_coords)
        time_estimate = path_length * 0.1  # 假设每个单元移动时间
        energy_estimate = sum(self.distance(path_coords[i][0], path_coords[i][1],
                                            path_coords[i + 1][0], path_coords[i + 1][1])
                              for i in range(len(path_coords) - 1))

        # 更新性能统计
        self.performance_stats['path_length'] = path_length
        self.performance_stats['energy'] = energy_estimate
        self.performance_stats['time'] = time_estimate

        return path_coords

    def add_static_obstacle(self, x, y):
        """添加静态障碍物"""
        self.static_obstacles.add((x, y))

    def remove_static_obstacle(self, x, y):
        """移除静态障碍物"""
        if (x, y) in self.static_obstacles:
            self.static_obstacles.remove((x, y))

    def add_dynamic_obstacle(self, x, y, radius, movement_type='linear', speed=0.1, angle=0):
        """添加动态障碍物"""
        self.dynamic_obstacles.append([x, y, radius, movement_type, speed, angle])

    def update_dynamic_obstacles(self):
        """更新动态障碍物的位置"""
        for obstacle in self.dynamic_obstacles:
            x, y, radius, movement_type, speed, angle = obstacle

            if movement_type == 'linear':
                # 线性运动
                x += math.cos(angle) * speed
                y += math.sin(angle) * speed

                # 边界反弹
                if x <= radius or x >= GRID_WIDTH - radius:
                    angle = math.pi - angle
                if y <= radius or y >= GRID_HEIGHT - radius:
                    angle = -angle

            elif movement_type == 'circular':
                # 圆周运动
                center_x, center_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                orbit_radius = 10  # 圆周半径
                angle += speed
                x = center_x + orbit_radius * math.cos(angle)
                y = center_y + orbit_radius * math.sin(angle)

            elif movement_type == 'random':
                # 随机运动
                if random.random() < 0.05:  # 5%的几率改变方向
                    angle = random.uniform(0, 2 * math.pi)
                x += math.cos(angle) * speed
                y += math.sin(angle) * speed

                # 边界反弹
                if x <= radius or x >= GRID_WIDTH - radius:
                    angle = math.pi - angle
                if y <= radius or y >= GRID_HEIGHT - radius:
                    angle = -angle

            obstacle[0] = x
            obstacle[1] = y
            obstacle[5] = angle

    def update_robot_state(self, v, w):
        """更新机器人状态"""
        x, y, theta, _, _ = self.robot_state
        dt = self.config['dt']

        # 更新位置和朝向
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt

        # 规范化角度
        theta = theta % (2 * math.pi)

        # 更新状态
        self.robot_state = [x, y, theta, v, w]

        # 检查是否到达目标点
        if self.targets:
            target = self.targets[0]
            if self.distance(x, y, target.x, target.y) < 0.5:
                self.targets.pop(0)

    def load_config(self, config_file):
        """从JSON配置文件加载场景"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # 重置场景
            self.static_obstacles = set()
            self.dynamic_obstacles = []
            self.targets = []

            # 设置起点
            start_x, start_y = config.get('start', [1, 1])
            self.start = self.grid[start_x][start_y]
            self.robot_state = [start_x, start_y, 0, 0, 0]

            # 设置终点
            end_x, end_y = config.get('end', [GRID_WIDTH - 2, GRID_HEIGHT - 2])
            self.end = self.grid[end_x][end_y]

            # 设置中继点
            for target in config.get('targets', []):
                tx, ty = target
                self.targets.append(self.grid[tx][ty])

            # 添加静态障碍物
            for obstacle in config.get('static_obstacles', []):
                ox, oy = obstacle
                self.add_static_obstacle(ox, oy)

            # 添加动态障碍物
            for dob in config.get('dynamic_obstacles', []):
                dx, dy = dob.get('position', [10, 10])
                radius = dob.get('radius', 1)
                movement = dob.get('movement', 'linear')
                speed = dob.get('speed', 0.5)
                angle = dob.get('angle', 0)
                self.add_dynamic_obstacle(dx, dy, radius, movement, speed, angle)

            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False


class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("动态多目标路径规划仿真")
        self.clock = pygame.time.Clock()
        self.planner = PathPlanner()
        self.running = True
        self.paused = False
        self.font = pygame.font.Font(None, 24)
        self.show_search = True  # 显示搜索过程
        self.show_trajectory = True  # 显示DWA轨迹
        self.config_files = ["scene1.json", "scene2.json", "scene3.json"]
        self.current_scene = 0

        # 初始化测试场景
        self.init_test_scenes()
        self.planner.load_config(self.config_files[self.current_scene])

    def init_test_scenes(self):
        """创建测试场景配置文件"""
        # 场景1: 简单静态环境
        scene1 = {
            "start": [1, 1],
            "end": [GRID_WIDTH - 2, GRID_HEIGHT - 2],
            "static_obstacles": [[x, 15] for x in range(10, 30)],
            "dynamic_obstacles": []
        }

        # 场景2: 动态障碍物环境
        # 场景2: 动态障碍物环境
        scene2 = {
            "start": [1, 1],
            "end": [GRID_WIDTH - 2, GRID_HEIGHT - 2],
            "static_obstacles": [[x, 10] for x in range(5, 35)],
            "dynamic_obstacles": [
                {"position": [10, 20], "movement": "linear", "angle": 0.785, "speed": 0.5},  # 明确设置速度为 0.1
                {"position": [30, 15], "movement": "circular", "speed": 0.05}  # 明确设置速度为 0.1
            ]
        }

        # 场景3: 多目标点环境
        # 场景3: 多目标点环境
        scene3 = {
            "start": [1, 1],
            "end": [GRID_WIDTH - 2, GRID_HEIGHT - 2],
            "targets": [[10, 10], [30, 5], [20, 25]],
            "static_obstacles": [[x, 15] for x in range(5, 35)],
            "dynamic_obstacles": [
                {"position": [15, 20], "movement": "random", "speed": 0.5},  # 明确设置速度为 0.1
                {"position": [25, 10], "movement": "circular", "speed": 0.05}  # 明确设置速度为 0.1
            ]
        }

        # 保存场景文件
        with open("scene1.json", "w") as f:
            json.dump(scene1, f)

        with open("scene2.json", "w") as f:
            json.dump(scene2, f)

        with open("scene3.json", "w") as f:
            json.dump(scene3, f)

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    # 重新规划路径
                    self.replan_path()

                elif event.key == pygame.K_a:
                    self.planner.algorithm = 'A*' if self.planner.algorithm == 'DWA' else 'DWA'
                    self.replan_path()

                elif event.key == pygame.K_s:
                    self.show_search = not self.show_search

                elif event.key == pygame.K_t:
                    self.show_trajectory = not self.show_trajectory

                elif event.key == pygame.K_n:
                    # 下一个场景
                    self.current_scene = (self.current_scene + 1) % len(self.config_files)
                    self.planner.load_config(self.config_files[self.current_scene])
                    self.replan_path()

                elif event.key == pygame.K_o:
                    # 优化路径（多目标点）
                    if self.planner.targets:
                        self.planner.path = self.planner.optimize_path()

            # 鼠标左键添加障碍物
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键添加
                    x, y = event.pos
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        if (grid_x, grid_y) != (self.planner.start.x, self.planner.start.y) and \
                                (grid_x, grid_y) != (self.planner.end.x, self.planner.end.y):
                            self.planner.add_static_obstacle(grid_x, grid_y)
                            self.replan_path()

                elif event.button == 3:  # 右键移除
                    x, y = event.pos
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    self.planner.remove_static_obstacle(grid_x, grid_y)
                    self.replan_path()

                elif event.button == 2:  # 中键添加目标点
                    x, y = event.pos
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        new_target = self.planner.grid[grid_x][grid_y]
                        if new_target not in self.planner.targets and new_target != self.planner.start and new_target != self.planner.end:
                            self.planner.targets.append(new_target)
                            self.replan_path()

    def replan_path(self):
        """重新规划路径"""
        self.planner.path = []
        self.planner.search_nodes = []

        if self.planner.algorithm == 'A*':
            self.planner.path = self.planner.a_star()
        else:
            if self.planner.targets:
                target_pos = (self.planner.targets[0].x, self.planner.targets[0].y)
            else:
                target_pos = (self.planner.end.x, self.planner.end.y)

            trajectory, (v, w) = self.planner.dwa_plan(target_pos)
            if trajectory:
                # 转换为网格坐标
                self.planner.path = [(int(x), int(y)) for x, y, _ in trajectory]
                self.planner.update_robot_state(v, w)

    def draw_grid(self):
        """绘制网格"""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))

        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_WIDTH, y))

    def draw_elements(self):
        """绘制元素"""
        # 绘制静态障碍物
        for x, y in self.planner.static_obstacles:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GRAY, rect)

        # 绘制动态障碍物及其轨迹
        for ox, oy, radius, movement, _, angle in self.planner.dynamic_obstacles:
            center = (int(ox * CELL_SIZE), int(oy * CELL_SIZE))
            pygame.draw.circle(self.screen, RED, center, int(radius * CELL_SIZE))

            # 绘制运动方向指示器
            dir_x = int(ox * CELL_SIZE + math.cos(angle) * 15)
            dir_y = int(oy * CELL_SIZE + math.sin(angle) * 15)
            pygame.draw.line(self.screen, ORANGE, center, (dir_x, dir_y), 2)

        # 绘制搜索节点（带透明度效果）
        if self.show_search:
            for x, y in self.planner.search_nodes:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE))
                s.set_alpha(64)
                s.fill(BLUE)
                self.screen.blit(s, (x * CELL_SIZE, y * CELL_SIZE))

        # 绘制路径
        if len(self.planner.path) > 1:
            points = [(int(x * CELL_SIZE + CELL_SIZE / 2),
                       int(y * CELL_SIZE + CELL_SIZE / 2))
                      for x, y in self.planner.path]
            pygame.draw.lines(self.screen, GREEN, False, points, 3)

        # 绘制DWA轨迹
        if self.show_trajectory and self.planner.algorithm == 'DWA':
            # 模拟当前轨迹
            x, y, theta, v, w = self.planner.robot_state
            trajectory = self.planner.simulate_trajectory(x, y, theta, v, w)

            if trajectory:
                points = [(int(x * CELL_SIZE + CELL_SIZE / 2),
                           int(y * CELL_SIZE + CELL_SIZE / 2))
                          for x, y, _ in trajectory]
                pygame.draw.lines(self.screen, PURPLE, False, points, 1)

                # 绘制轨迹点
                for i, (x, y, _) in enumerate(trajectory):
                    if i % 5 == 0:  # 每5个点绘制一个
                        pos = (int(x * CELL_SIZE), int(y * CELL_SIZE))  # ←← 补上右括号
                        pygame.draw.circle(self.screen, PINK, pos, 2)

                        # 绘制起点和终点
                        if self.planner.start:
                            rect = pygame.Rect(self.planner.start.x * CELL_SIZE,
                                               self.planner.start.y * CELL_SIZE,
                                               CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(self.screen, BLUE, rect)

                        if self.planner.end:
                            rect = pygame.Rect(self.planner.end.x * CELL_SIZE,
                                               self.planner.end.y * CELL_SIZE,
                                               CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(self.screen, YELLOW, rect)

                        # 绘制目标点（中继点）
                        for i, target in enumerate(self.planner.targets):
                            rect = pygame.Rect(target.x * CELL_SIZE,
                                               target.y * CELL_SIZE,
                                               CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(self.screen, ORANGE, rect)

                        # 绘制目标点编号
                        text = self.font.render(str(i + 1), True, BLACK)
                        self.screen.blit(text, (target.x * CELL_SIZE + 5, target.y * CELL_SIZE + 3))

                        # 绘制机器人
                        if self.planner.algorithm == 'DWA':
                            rx, ry, theta, _, _ = self.planner.robot_state
                        center = (int(rx * CELL_SIZE), int(ry * CELL_SIZE))
                        pygame.draw.circle(self.screen, CYAN, center,
                                           int(self.planner.config['robot_radius'] * CELL_SIZE))

                        # 绘制朝向
                        front_x = int(rx * CELL_SIZE + math.cos(theta) * 10)
                        front_y = int(ry * CELL_SIZE + math.sin(theta) * 10)
                        pygame.draw.line(self.screen, BLACK, center, (front_x, front_y), 2)

                        # 绘制信息面板
                        self.draw_info_panel()

    def draw_info_panel(self):
        """绘制信息面板"""
        # 半透明背景
        info_surface = pygame.Surface((250, 150))
        info_surface.set_alpha(180)
        info_surface.fill(WHITE)
        self.screen.blit(info_surface, (10, 10))

        # 算法信息
        algo_text = self.font.render(f"算法: {self.planner.algorithm}", True, BLACK)
        self.screen.blit(algo_text, (20, 15))

        # 场景信息
        scene_text = self.font.render(f"场景: {self.current_scene + 1}/{len(self.config_files)}", True, BLACK)
        self.screen.blit(scene_text, (20, 40))

        # 性能数据
        stats = self.planner.performance_stats
        length_text = self.font.render(f"路径长度: {stats['path_length']}", True, BLACK)
        time_text = self.font.render(f"计算时间: {stats['computation_time']:.4f}s", True, BLACK)
        collision_text = self.font.render(f"碰撞次数: {stats['collisions']}", True, BLACK)

        self.screen.blit(length_text, (20, 65))
        self.screen.blit(time_text, (20, 90))
        self.screen.blit(collision_text, (20, 115))

        # 控制提示
        controls = [
            "空格: 暂停/继续",
            "A: 切换算法",
            "N: 下一场景",
            "O: 优化路径",
            "S: 搜索显示",
            "T: 轨迹显示",
            "左键: 添加障碍",
            "右键: 移除障碍",
            "中键: 添加目标"
        ]

        for i, text in enumerate(controls):
            hint = self.font.render(text, True, BLACK)
            self.screen.blit(hint, (WINDOW_WIDTH - 180, 15 + i * 20))

    def run(self):
        """主循环"""
        last_update = time.time()

        while self.running:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time

            self.handle_events()

            if not self.paused:
                # 更新动态障碍物
                self.planner.update_dynamic_obstacles()

                # 对于DWA算法，需要连续规划
                if self.planner.algorithm == 'DWA':
                    if self.planner.targets:
                        target_pos = (self.planner.targets[0].x, self.planner.targets[0].y)
                    else:
                        target_pos = (self.planner.end.x, self.planner.end.y)

                    trajectory, (v, w) = self.planner.dwa_plan(target_pos)
                    if trajectory:
                        # 转换为网格坐标
                        self.planner.path = [(int(x), int(y)) for x, y, _ in trajectory]
                        self.planner.update_robot_state(v, w)

                # 定期更新A*路径（如果环境变化）
                elif current_time - self.planner.performance_stats['last_update'] > 0.1:
                    self.replan_path()
                    self.planner.performance_stats['last_update'] = current_time

            # 绘制
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_elements()
            pygame.display.flip()

            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()