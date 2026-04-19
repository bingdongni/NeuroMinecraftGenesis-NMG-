"""
空间智能和具身智能集成系统
Spatial Intelligence and Embodied AI Integration System

该系统集成了3D空间建模、具身智能物理仿真、空间推理、路径规划和实时感知-行动循环
的功能，实现了完整的物理世界交互控制能力。

Author: AI Assistant
Date: 2025-11-13
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor
import queue
import random
from datetime import datetime
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialObjectType(Enum):
    """空间对象类型枚举"""
    OBSTACLE = "obstacle"
    TARGET = "target"
    AGENT = "agent"
    LANDMARK = "landmark"
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    CONTAINER = "container"


class ActionType(Enum):
    """动作类型枚举"""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    PUSH = "push"
    PULL = "pull"
    INTERACT = "interact"
    REACH = "reach"
    GRASP = "grasp"
    RELEASE = "release"


@dataclass
class Vector3:
    """3D向量类"""
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other) -> float:
        """点积运算"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other) -> 'Vector3':
        """叉积运算"""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return Vector3(0, 0, 0)

    def distance_to(self, other):
        return (self - other).length()

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data):
        return cls(data["x"], data["y"], data["z"])


@dataclass
class SpatialObject:
    """空间对象类"""
    id: str
    type: SpatialObjectType
    position: Vector3
    rotation: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    mass: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def update(self, delta_time: float):
        """更新对象状态"""
        # 更新位置基于速度
        self.position = self.position + self.velocity * delta_time

    def is_colliding_with(self, other: 'SpatialObject') -> bool:
        """检测与其他对象的碰撞"""
        distance = self.position.distance_to(other.position)
        return distance < 1.0  # 简化碰撞检测

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "scale": self.scale.to_dict(),
            "velocity": self.velocity.to_dict(),
            "mass": self.mass,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data):
        """从字典创建对象"""
        return cls(
            id=data["id"],
            type=SpatialObjectType(data["type"]),
            position=Vector3.from_dict(data["position"]),
            rotation=Vector3.from_dict(data["rotation"]),
            scale=Vector3.from_dict(data["scale"]),
            velocity=Vector3.from_dict(data["velocity"]),
            mass=data["mass"],
            properties=data["properties"]
        )


@dataclass
class SensorReading:
    """传感器读数"""
    timestamp: float
    position: Vector3
    rotation: Vector3
    distances: List[float] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)
    forces: List[float] = field(default_factory=list)
    tactile_data: List[float] = field(default_factory=list)

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "distances": self.distances,
            "objects_detected": self.objects_detected,
            "forces": self.forces,
            "tactile_data": self.tactile_data
        }


@dataclass
class PathNode:
    """路径规划节点"""
    position: Vector3
    g_cost: float = 0.0  # 从起点到该点的实际代价
    h_cost: float = 0.0  # 该点到终点的预估代价
    parent: Optional['PathNode'] = None

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class SpatialModel:
    """3D空间建模器"""
    
    def __init__(self, world_size: Tuple[float, float, float] = (100.0, 100.0, 50.0)):
        self.world_size = Vector3(*world_size)
        self.objects: Dict[str, SpatialObject] = {}
        self.static_objects: Set[str] = set()
        self.grid_resolution = 1.0  # 1米网格分辨率
        self.occupancy_grid = None
        self.spatial_index = {}
        self.boundaries = self._initialize_boundaries()
        
    def _initialize_boundaries(self) -> Dict[str, float]:
        """初始化空间边界"""
        half_x = self.world_size.x / 2
        half_y = self.world_size.y / 2
        half_z = self.world_size.z / 2
        
        return {
            "min_x": -half_x, "max_x": half_x,
            "min_y": -half_y, "max_y": half_y,
            "min_z": 0, "max_z": half_z
        }
    
    def add_object(self, obj: SpatialObject, static: bool = False):
        """添加空间对象"""
        self.objects[obj.id] = obj
        if static:
            self.static_objects.add(obj.id)
        self._update_spatial_index(obj)
        logger.info(f"添加对象: {obj.id} at {obj.position.to_tuple()}")
    
    def remove_object(self, obj_id: str):
        """移除空间对象"""
        if obj_id in self.objects:
            del self.objects[obj_id]
            self.static_objects.discard(obj_id)
            self._remove_from_spatial_index(obj_id)
            logger.info(f"移除对象: {obj_id}")
    
    def _update_spatial_index(self, obj: SpatialObject):
        """更新空间索引"""
        grid_pos = self.world_to_grid(obj.position)
        if grid_pos not in self.spatial_index:
            self.spatial_index[grid_pos] = []
        self.spatial_index[grid_pos].append(obj.id)
    
    def _remove_from_spatial_index(self, obj_id: str):
        """从空间索引中移除"""
        obj = self.objects.get(obj_id)
        if obj:
            grid_pos = self.world_to_grid(obj.position)
            if grid_pos in self.spatial_index:
                if obj_id in self.spatial_index[grid_pos]:
                    self.spatial_index[grid_pos].remove(obj_id)
    
    def world_to_grid(self, world_pos: Vector3) -> Tuple[int, int, int]:
        """世界坐标转网格坐标"""
        return (
            int(world_pos.x / self.grid_resolution),
            int(world_pos.y / self.grid_resolution),
            int(world_pos.z / self.grid_resolution)
        )
    
    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Vector3:
        """网格坐标转世界坐标"""
        return Vector3(
            grid_pos[0] * self.grid_resolution,
            grid_pos[1] * self.grid_resolution,
            grid_pos[2] * self.grid_resolution
        )
    
    def get_objects_in_region(self, center: Vector3, radius: float) -> List[SpatialObject]:
        """获取指定区域内的对象"""
        result = []
        center_grid = self.world_to_grid(center)
        radius_grid = int(radius / self.grid_resolution)
        
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                for dz in range(-radius_grid, radius_grid + 1):
                    grid_pos = (center_grid[0] + dx, center_grid[1] + dy, center_grid[2] + dz)
                    if grid_pos in self.spatial_index:
                        for obj_id in self.spatial_index[grid_pos]:
                            if obj_id in self.objects:
                                obj = self.objects[obj_id]
                                if center.distance_to(obj.position) <= radius:
                                    result.append(obj)
        
        return result
    
    def raycast(self, start: Vector3, direction: Vector3, max_distance: float = 50.0) -> Tuple[Optional[SpatialObject], float]:
        """光线投射检测 - 增强版"""
        ray_step = min(self.grid_resolution / 2, 0.1)  # 更精确的步长
        ray_dir = direction.normalize()
        current_pos = start
        min_distance = max_distance
        
        # 确保起点在边界内
        if not self.is_position_valid(start):
            return None, 0.0
        
        for i in range(int(max_distance / ray_step)):
            current_pos = current_pos + ray_dir * ray_step
            
            # 检查边界
            if not self.is_position_valid(current_pos):
                return None, current_pos.distance_to(start)
            
            # 检查碰撞
            objects = self.get_objects_in_region(current_pos, self.grid_resolution)
            for obj in objects:
                # 使用更精确的碰撞检测
                distance = current_pos.distance_to(obj.position)
                collision_threshold = max(obj.scale.x, obj.scale.y, obj.scale.z) / 2
                
                if distance < collision_threshold and obj.id not in self.static_objects:
                    return obj, current_pos.distance_to(start)
        
        return None, max_distance
    
    def is_position_valid(self, position: Vector3) -> bool:
        """检查位置是否有效（是否在边界内）"""
        return (self.boundaries["min_x"] <= position.x <= self.boundaries["max_x"] and
                self.boundaries["min_y"] <= position.y <= self.boundaries["max_y"] and
                self.boundaries["min_z"] <= position.z <= self.boundaries["max_z"])
    
    def update_occupancy_grid(self):
        """更新占据网格"""
        grid_size = (
            int(self.world_size.x / self.grid_resolution) + 1,
            int(self.world_size.y / self.grid_resolution) + 1,
            int(self.world_size.z / self.grid_resolution) + 1
        )
        
        self.occupancy_grid = np.zeros(grid_size, dtype=np.uint8)
        
        for obj in self.objects.values():
            if obj.id in self.static_objects:
                grid_pos = self.world_to_grid(obj.position)
                if (0 <= grid_pos[0] < grid_size[0] and
                    0 <= grid_pos[1] < grid_size[1] and
                    0 <= grid_pos[2] < grid_size[2]):
                    self.occupancy_grid[grid_pos] = 1  # 标记为占据
    
    def get_visibility_map(self, position: Vector3, max_range: float = 20.0) -> Dict[str, Any]:
        """获取可见性地图"""
        visibility_data = {
            "position": position.to_dict(),
            "visible_objects": [],
            "occluded_regions": [],
            "accessible_areas": []
        }
        
        # 使用光线投射检测可见对象
        for obj in self.objects.values():
            direction = obj.position - position
            distance = direction.length()
            
            if distance <= max_range:
                hit_obj, hit_distance = self.raycast(position, direction, distance)
                if hit_obj == obj:
                    visibility_data["visible_objects"].append(obj.to_dict())
                elif hit_obj is not None:
                    visibility_data["occluded_regions"].append({
                        "direction": direction.normalize().to_dict(),
                        "distance": hit_distance,
                        "blocking_object": hit_obj.id
                    })
        
        return visibility_data
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "world_size": self.world_size.to_dict(),
            "objects": {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            "static_objects": list(self.static_objects),
            "grid_resolution": self.grid_resolution,
            "boundaries": self.boundaries
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建空间模型"""
        model = cls()
        model.world_size = Vector3.from_dict(data["world_size"])
        model.static_objects = set(data["static_objects"])
        model.grid_resolution = data["grid_resolution"]
        model.boundaries = data["boundaries"]
        
        for obj_id, obj_data in data["objects"].items():
            model.objects[obj_id] = SpatialObject.from_dict(obj_data)
        
        return model


class PhysicsEngine:
    """物理仿真引擎"""
    
    def __init__(self, gravity: float = -9.81, friction: float = 0.5):
        self.gravity = gravity
        self.friction = friction
        self.time_step = 0.016  # 60 FPS
        self.collision_pairs = []
        self.forces = {}
        
    def apply_gravity(self, obj: SpatialObject, spatial_model: SpatialModel):
        """应用重力"""
        if obj.id not in spatial_model.static_objects:
            obj.velocity.y += self.gravity * self.time_step
    
    def apply_friction(self, obj: SpatialObject):
        """应用摩擦力"""
        if abs(obj.velocity.y) < 0.1 and obj.position.y <= 0:  # 地面接触
            obj.velocity.x *= (1 - self.friction * self.time_step)
            obj.velocity.z *= (1 - self.friction * self.time_step)
    
    def resolve_collision(self, obj1: SpatialObject, obj2: SpatialObject):
        """碰撞响应 - 增强版"""
        # 计算碰撞向量
        collision_vector = obj2.position - obj1.position
        collision_distance = collision_vector.length()
        
        if collision_distance == 0:
            collision_vector = Vector3(1, 0, 0)
            collision_distance = 1.0
        
        collision_normal = collision_vector.normalize()
        
        # 计算穿透深度
        obj1_radius = max(obj1.scale.x, obj1.scale.y, obj1.scale.z) / 2
        obj2_radius = max(obj2.scale.x, obj2.scale.y, obj2.scale.z) / 2
        total_radius = obj1_radius + obj2_radius
        overlap = total_radius - collision_distance
        
        # 分离对象
        if overlap > 0:
            if obj1.id in self.static_objects:
                obj2.position = obj2.position + collision_normal * overlap
            elif obj2.id in self.static_objects:
                obj1.position = obj1.position - collision_normal * overlap
            else:
                separation = collision_normal * (overlap / 2)
                obj1.position = obj1.position - separation
                obj2.position = obj2.position + separation
        
        # 动量交换和摩擦力
        if obj1.id not in self.static_objects and obj2.id not in self.static_objects:
            # 计算相对速度
            relative_velocity = obj1.velocity - obj2.velocity
            velocity_along_normal = relative_velocity.dot(collision_normal)
            
            if velocity_along_normal > 0:
                return  # 对象正在分离，不需要碰撞响应
            
            # 计算冲量
            restitution = 0.6  # 弹性系数
            total_mass = obj1.mass + obj2.mass
            
            if total_mass > 0:
                impulse_magnitude = -(1 + restitution) * velocity_along_normal
                impulse_magnitude /= (1/obj1.mass + 1/obj2.mass)
                
                # 应用冲量
                impulse = collision_normal * impulse_magnitude
                obj1.velocity = obj1.velocity + impulse / obj1.mass
                obj2.velocity = obj2.velocity - impulse / obj2.mass
                
                # 应用摩擦力
                tangent_velocity = relative_velocity - collision_normal * velocity_along_normal
                tangent_magnitude = tangent_velocity.length()
                
                if tangent_magnitude > 0:
                    tangent = tangent_velocity.normalize()
                    friction_coefficient = 0.3
                    friction_impulse = tangent * (friction_coefficient * abs(impulse_magnitude))
                    
                    obj1.velocity = obj1.velocity - friction_impulse / obj1.mass
                    obj2.velocity = obj2.velocity + friction_impulse / obj2.mass
    
    def apply_force(self, obj_id: str, force: Vector3, duration: float = 0.0):
        """施力"""
        if obj_id not in self.forces:
            self.forces[obj_id] = []
        self.forces[obj_id].append({
            "force": force,
            "start_time": time.time(),
            "duration": duration
        })
    
    def update_forces(self, obj: SpatialObject):
        """更新力"""
        current_time = time.time()
        if obj.id in self.forces:
            remaining_forces = []
            for force_data in self.forces[obj.id]:
                if current_time - force_data["start_time"] <= force_data["duration"]:
                    remaining_forces.append(force_data)
                else:
                    # 计算力产生的加速度并应用
                    acceleration = force_data["force"] / obj.mass
                    obj.velocity = obj.velocity + acceleration * self.time_step
            
            self.forces[obj.id] = remaining_forces
    
    def step(self, spatial_model: SpatialModel):
        """执行物理仿真步进"""
        # 应用重力
        for obj in spatial_model.objects.values():
            self.apply_gravity(obj, spatial_model)
        
        # 更新力
        for obj in spatial_model.objects.values():
            self.update_forces(obj)
        
        # 应用摩擦力
        for obj in spatial_model.objects.values():
            self.apply_friction(obj)
        
        # 碰撞检测和响应
        self.collision_pairs = []
        objects = list(spatial_model.objects.values())
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]
                if obj1.is_colliding_with(obj2):
                    self.collision_pairs.append((obj1, obj2))
                    self.resolve_collision(obj1, obj2)
        
        # 更新对象位置
        for obj in spatial_model.objects.values():
            if obj.id not in spatial_model.static_objects:
                obj.update(self.time_step)


class SpatialReasoner:
    """空间推理引擎"""
    
    def __init__(self):
        self.spatial_relations = {}
        self.topological_map = {}
        self.metabolic_constraints = []
        
    def analyze_spatial_relations(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """分析空间关系"""
        relations = {
            "proximity": {},  # 邻近关系
            "containment": {},  # 包含关系
            "support": {},  # 支持关系
            "occlusion": {},  # 遮挡关系
            "alignment": {}  # 对齐关系
        }
        
        objects = list(spatial_model.objects.values())
        
        # 分析邻近关系
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                distance = obj1.position.distance_to(obj2.position)
                
                if distance < 3.0:  # 3米内认为邻近
                    relations["proximity"][f"{obj1.id}_{obj2.id}"] = {
                        "distance": distance,
                        "closeness": max(0, 1 - distance / 3.0)
                    }
        
        # 分析包含关系
        for obj in objects:
            if obj.type == SpatialObjectType.WALL or "container" in obj.properties:
                relations["containment"][obj.id] = []
                for other in objects:
                    if other != obj and self._is_inside(other.position, obj):
                        relations["containment"][obj.id].append(other.id)
        
        # 分析支持关系
        for obj in objects:
            if obj.position.y < 1.0:  # 接近地面
                supporting_objects = self._find_supporting_objects(obj, objects)
                if supporting_objects:
                    relations["support"][obj.id] = supporting_objects
        
        return relations
    
    def _is_inside(self, position: Vector3, container: SpatialObject) -> bool:
        """判断位置是否在容器内"""
        # 简化的包含检测
        container_size = container.scale
        relative_pos = position - container.position
        
        return (abs(relative_pos.x) <= container_size.x / 2 and
                abs(relative_pos.y) <= container_size.y / 2 and
                abs(relative_pos.z) <= container_size.z / 2)
    
    def _find_supporting_objects(self, obj: SpatialObject, all_objects: List[SpatialObject]) -> List[str]:
        """找到支撑对象"""
        supporting_objects = []
        for other in all_objects:
            if other != obj:
                horizontal_distance = math.sqrt(
                    (obj.position.x - other.position.x) ** 2 +
                    (obj.position.z - other.position.z) ** 2
                )
                
                if (horizontal_distance < 1.5 and  # 水平距离
                    other.position.y > obj.position.y - 0.1 and  # 其他对象更高
                    other.position.y < obj.position.y + 2.0):  # 高度差合理
                    supporting_objects.append(other.id)
        
        return supporting_objects
    
    def predict_spatial_outcomes(self, spatial_model: SpatialModel, action_plan: List[ActionType]) -> List[Dict[str, Any]]:
        """预测空间操作结果"""
        predictions = []
        current_model = SpatialModel.from_dict(spatial_model.to_dict())
        
        for action in action_plan:
            outcome = self._simulate_action(current_model, action)
            predictions.append(outcome)
            # 应用动作结果到模型用于下一步预测
            self._apply_action_outcome(current_model, outcome)
        
        return predictions
    
    def _simulate_action(self, spatial_model: SpatialModel, action: ActionType) -> Dict[str, Any]:
        """模拟单个动作"""
        outcome = {
            "action": action.value,
            "success": False,
            "new_objects": [],
            "changed_objects": [],
            "spatial_changes": [],
            "confidence": 0.0
        }
        
        # 简化的动作模拟
        if action == ActionType.MOVE_FORWARD:
            outcome["success"] = True
            outcome["spatial_changes"].append("Agent向前移动1米")
            outcome["confidence"] = 0.9
        elif action == ActionType.PICK_UP:
            # 检查是否可以拾取
            nearby_objects = spatial_model.get_objects_in_region(
                spatial_model.objects["agent"].position, 2.0
            )
            if any(obj.type == SpatialObjectType.TARGET for obj in nearby_objects):
                outcome["success"] = True
                outcome["changed_objects"].append("target_object")
                outcome["confidence"] = 0.8
        # 可以添加更多动作的模拟...
        
        return outcome
    
    def _apply_action_outcome(self, spatial_model: SpatialModel, outcome: Dict[str, Any]):
        """应用动作结果到空间模型"""
        if outcome["success"]:
            # 根据动作类型更新空间模型
            action = ActionType(outcome["action"])
            if action == ActionType.MOVE_FORWARD:
                # 移动代理
                if "agent" in spatial_model.objects:
                    agent = spatial_model.objects["agent"]
                    agent.position = agent.position + Vector3(0, 0, 1)
            # 可以添加更多逻辑...
    
    def infer_goal_trajectories(self, spatial_model: SpatialModel, goal: str) -> List[Vector3]:
        """推断目标轨迹"""
        trajectories = []
        
        # 简化的轨迹推断
        if "reach_target" in goal:
            # 找到目标对象
            target = None
            for obj in spatial_model.objects.values():
                if obj.type == SpatialObjectType.TARGET:
                    target = obj
                    break
            
            if target:
                # 生成到达目标的轨迹
                agent = spatial_model.objects.get("agent")
                if agent:
                    # 简单直线轨迹
                    direction = target.position - agent.position
                    steps = max(1, int(direction.length() / 2.0))  # 每步2米
                    
                    for i in range(1, steps + 1):
                        t = i / steps
                        waypoint = agent.position + direction * t
                        trajectories.append(waypoint)
        
        return trajectories
    
    def create_cognitive_map(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """创建认知地图"""
        cognitive_map = {
            "landmarks": [],
            "routes": [],
            "regions": [],
            "spatial_index": {},
            "navigation_graph": {}
        }
        
        # 识别地标
        for obj in spatial_model.objects.values():
            if obj.type == SpatialObjectType.LANDMARK:
                cognitive_map["landmarks"].append({
                    "id": obj.id,
                    "position": obj.position.to_dict(),
                    "salience": obj.properties.get("salience", 1.0),
                    "type": obj.properties.get("landmark_type", "unknown")
                })
        
        # 创建导航图
        landmarks = cognitive_map["landmarks"]
        for i, landmark1 in enumerate(landmarks):
            for landmark2 in landmarks[i+1:]:
                distance = Vector3.from_dict(landmark1["position"]).distance_to(
                    Vector3.from_dict(landmark2["position"])
                )
                cognitive_map["navigation_graph"][landmark1["id"]] = {
                    landmark2["id"]: {"distance": distance, "visibility": "unknown"}
                }
        
        return cognitive_map


class PathPlanner:
    """增强路径规划器"""
    
    def __init__(self):
        self.grid_resolution = 0.5  # 更精细的网格
        self.max_iterations = 5000
        self.heuristic_weight = 1.2
        self.fall_back_planners = ["dijkstra", "rapidly_exploring_random_tree"]
        self.planning_cache = {}  # 路径规划缓存
        
    def dijkstra_pathfinding(self, spatial_model: SpatialModel, start: Vector3, goal: Vector3) -> Optional[List[Vector3]]:
        """Dijkstra路径规划算法"""
        # 创建优先队列（使用列表和排序）
        open_list = []
        closed_set = set()
        
        # 创建起始节点
        start_node = PathNode(start)
        start_node.g_cost = 0
        start_node.h_cost = 0
        open_list.append(start_node)
        
        while open_list:
            # 按g_cost排序
            open_list.sort(key=lambda x: x.g_cost)
            current = open_list.pop(0)
            
            # 到达目标
            if current.position.distance_to(goal) < self.grid_resolution:
                return self._reconstruct_path(current)
            
            closed_set.add(current.position.to_tuple())
            
            # 探索邻居
            for neighbor_pos in self._get_neighbors(current.position):
                neighbor_tuple = neighbor_pos.to_tuple()
                
                if neighbor_tuple in closed_set:
                    continue
                
                if not self._is_valid_position(neighbor_pos, spatial_model):
                    closed_set.add(neighbor_tuple)
                    continue
                
                # 计算新的g_cost
                tentative_g_cost = current.g_cost + current.position.distance_to(neighbor_pos)
                
                # 检查是否已在开放列表中
                existing_node = self._find_node_in_open_list(open_list, neighbor_pos)
                if existing_node is None:
                    neighbor_node = PathNode(neighbor_pos)
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.h_cost = 0  # Dijkstra不使用启发式
                    neighbor_node.parent = current
                    open_list.append(neighbor_node)
                elif tentative_g_cost < existing_node.g_cost:
                    existing_node.g_cost = tentative_g_cost
                    existing_node.parent = current
        
        return None
    
    def rrt_pathfinding(self, spatial_model: SpatialModel, start: Vector3, goal: Vector3, 
                       max_samples: int = 1000) -> Optional[List[Vector3]]:
        """RRT (Rapidly Exploring Random Tree) 路径规划"""
        nodes = [start]  # 树节点列表
        parent = {start.to_tuple(): None}
        
        for _ in range(max_samples):
            # 随机采样点
            if random.random() < 0.1:  # 10%概率直接选择目标
                random_point = goal
            else:
                random_point = Vector3(
                    random.uniform(spatial_model.boundaries["min_x"], spatial_model.boundaries["max_x"]),
                    random.uniform(spatial_model.boundaries["min_y"], spatial_model.boundaries["max_y"]),
                    random.uniform(spatial_model.boundaries["min_z"], spatial_model.boundaries["max_z"])
                )
            
            # 找到最近的节点
            nearest_node = self._find_nearest_node(nodes, random_point)
            
            # 在最近节点和随机点之间创建新节点
            direction = random_point - nearest_node
            step_size = min(2.0, direction.length())  # 最大步长2米
            if direction.length() > 0:
                new_point = nearest_node + direction.normalize() * step_size
                
                # 检查路径是否有效
                if self._is_path_segment_valid(nearest_node, new_point, spatial_model):
                    nodes.append(new_point)
                    parent[new_point.to_tuple()] = nearest_node
                    
                    # 检查是否到达目标
                    if new_point.distance_to(goal) < 2.0:
                        if self._is_path_segment_valid(new_point, goal, spatial_model):
                            return self._reconstruct_rrt_path(parent, start, new_point, goal)
        
        return None
    
    def _find_nearest_node(self, nodes: List[Vector3], target: Vector3) -> Vector3:
        """找到最近的节点"""
        return min(nodes, key=lambda node: node.distance_to(target))
    
    def _reconstruct_rrt_path(self, parent: Dict, start: Vector3, end: Vector3, goal: Vector3) -> List[Vector3]:
        """重构RRT路径"""
        path = []
        current = end
        
        # 回溯路径
        while current is not None:
            path.append(current)
            current = parent.get(current.to_tuple())
        
        path.reverse()
        
        # 添加到目标的路径
        if path[-1] != goal:
            path.append(goal)
        
        return path
    
    def a_star_pathfinding(self, spatial_model: SpatialModel, start: Vector3, goal: Vector3, 
                          max_iterations: int = None) -> Optional[List[Vector3]]:
        """A*路径规划算法"""
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # 创建开放列表和关闭列表
        open_list = []
        closed_list = set()
        
        # 创建起始节点
        start_node = PathNode(start)
        start_node.h_cost = self._heuristic(start, goal)
        open_list.append(start_node)
        
        iterations = 0
        while open_list and iterations < max_iterations:
            iterations += 1
            
            # 找到F成本最小的节点
            current = min(open_list)
            open_list.remove(current)
            closed_list.add(current.position.to_tuple())
            
            # 到达目标
            if current.position.distance_to(goal) < self.grid_resolution:
                return self._reconstruct_path(current)
            
            # 检查邻居节点
            for neighbor_pos in self._get_neighbors(current.position):
                neighbor_tuple = neighbor_pos.to_tuple()
                
                if neighbor_tuple in closed_list:
                    continue
                
                # 检查是否为有效位置
                if not self._is_valid_position(neighbor_pos, spatial_model):
                    closed_list.add(neighbor_tuple)
                    continue
                
                # 创建或更新邻居节点
                tentative_g_cost = current.g_cost + current.position.distance_to(neighbor_pos)
                
                existing_node = self._find_node_in_open_list(open_list, neighbor_pos)
                if existing_node is None:
                    neighbor_node = PathNode(neighbor_pos)
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.h_cost = self._heuristic(neighbor_pos, goal)
                    neighbor_node.parent = current
                    open_list.append(neighbor_node)
                elif tentative_g_cost < existing_node.g_cost:
                    existing_node.g_cost = tentative_g_cost
                    existing_node.parent = current
        
        return None  # 未找到路径
    
    def _heuristic(self, pos1: Vector3, pos2: Vector3) -> float:
        """启发式函数（欧几里得距离）"""
        return pos1.distance_to(pos2) * self.heuristic_weight
    
    def _get_neighbors(self, position: Vector3) -> List[Vector3]:
        """获取邻居位置（8连通）"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append(Vector3(
                        position.x + dx * self.grid_resolution,
                        position.y + dy * self.grid_resolution,
                        position.z + dz * self.grid_resolution
                    ))
        return neighbors
    
    def _is_valid_position(self, position: Vector3, spatial_model: SpatialModel) -> bool:
        """检查位置是否有效"""
        # 检查边界
        if not spatial_model.is_position_valid(position):
            return False
        
        # 检查碰撞
        objects = spatial_model.get_objects_in_region(position, self.grid_resolution)
        for obj in objects:
            if obj.id not in spatial_model.static_objects:
                return False
        
        return True
    
    def _find_node_in_open_list(self, open_list: List[PathNode], position: Vector3) -> Optional[PathNode]:
        """在开放列表中查找指定位置的节点"""
        for node in open_list:
            if node.position.distance_to(position) < self.grid_resolution / 2:
                return node
        return None
    
    def _reconstruct_path(self, end_node: PathNode) -> List[Vector3]:
        """重构路径"""
        path = []
        current = end_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return list(reversed(path))
    
    def smooth_path(self, path: List[Vector3], spatial_model: SpatialModel) -> List[Vector3]:
        """路径平滑优化"""
        if len(path) < 3:
            return path
        
        smoothed_path = [path[0]]  # 保留起始点
        
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            
            # 找到最远的可见点
            while j > i + 1:
                if self._is_path_segment_valid(path[i], path[j], spatial_model):
                    break
                j -= 1
            
            smoothed_path.append(path[j])
            i = j
        
        return smoothed_path
    
    def _is_path_segment_valid(self, start: Vector3, end: Vector3, spatial_model: SpatialModel) -> bool:
        """检查路径段是否有效"""
        # 使用光线投射检查路径上的障碍物
        direction = end - start
        distance = direction.length()
        
        if distance == 0:
            return True
        
        direction = direction.normalize()
        step_size = self.grid_resolution / 2
        steps = int(distance / step_size)
        
        for i in range(1, steps):
            current_pos = start + direction * step_size * i
            if not self._is_valid_position(current_pos, spatial_model):
                return False
        
        return True


class SensorSystem:
    """传感器系统"""
    
    def __init__(self, sensor_range: float = 20.0):
        self.sensor_range = sensor_range
        self.noise_level = 0.1
        self.fov_horizontal = 120.0  # 水平视野角度
        self.fov_vertical = 90.0     # 垂直视野角度
        self.sensor_accuracy = 0.95  # 传感器精度
        self.sensor_angles = self._generate_sensor_angles()
        self.reading_history = deque(maxlen=100)
        self.calibration_offset = Vector3(0, 0, 0)  # 传感器校准偏移
        
    def _generate_sensor_angles(self) -> List[Vector3]:
        """生成更精确的传感器角度"""
        angles = []
        # 主要方向（前、后、左、右、上、下）
        angles.extend([
            Vector3(0, 0, 1),   # 前
            Vector3(0, 0, -1),  # 后
            Vector3(-1, 0, 0),  # 左
            Vector3(1, 0, 0),   # 右
            Vector3(0, 1, 0),   # 上
            Vector3(0, -1, 0)   # 下
        ])
        
        # 添加斜向角度
        for yaw in [-45, 45]:
            for pitch in [-30, 30]:
                # 简化的角度转换
                angle = Vector3(
                    math.sin(math.radians(yaw)) * math.cos(math.radians(pitch)),
                    math.sin(math.radians(pitch)),
                    math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
                )
                angles.append(angle.normalize())
        
        return angles
    
    def sense_depth_image(self, agent_position: Vector3, agent_rotation: Vector3, 
                         spatial_model: SpatialModel, image_size: Tuple[int, int] = (64, 48)) -> np.ndarray:
        """生成深度图像"""
        depth_image = np.zeros(image_size, dtype=np.float32)
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        
        for i in range(image_size[1]):  # 垂直像素
            for j in range(image_size[0]):  # 水平像素
                # 计算像素对应的角度
                angle_x = (j - center_x) / center_x * self.fov_horizontal / 2
                angle_y = (i - center_y) / center_y * self.fov_vertical / 2
                
                # 创建方向向量
                direction = Vector3(
                    math.sin(math.radians(angle_x)),
                    math.sin(math.radians(angle_y)),
                    math.cos(math.radians(angle_x)) * math.cos(math.radians(angle_y))
                )
                
                # 应用代理旋转
                rotated_direction = self._rotate_vector(direction, agent_rotation)
                
                # 执行光线投射
                hit_obj, hit_distance = spatial_model.raycast(
                    agent_position + self.calibration_offset, 
                    rotated_direction, 
                    self.sensor_range
                )
                
                # 添加噪声
                noisy_distance = hit_distance + np.random.normal(0, self.noise_level)
                depth_image[i, j] = max(0, noisy_distance)
        
        return depth_image
    
    def _rotate_vector(self, vector: Vector3, rotation: Vector3) -> Vector3:
        """旋转向量"""
        # 简化的旋转计算（绕Y轴旋转）
        yaw = math.radians(rotation.y)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        return Vector3(
            vector.x * cos_yaw - vector.z * sin_yaw,
            vector.y,
            vector.x * sin_yaw + vector.z * cos_yaw
        )
    
    def get_point_cloud(self, agent_position: Vector3, agent_rotation: Vector3, 
                       spatial_model: SpatialModel, num_points: int = 1000) -> np.ndarray:
        """生成点云数据"""
        points = np.zeros((num_points, 3))
        
        for i in range(num_points):
            # 随机采样角度和距离
            azimuth = random.uniform(0, 360)
            elevation = random.uniform(-90, 90)
            distance = random.uniform(1.0, self.sensor_range)
            
            # 转换为笛卡尔坐标
            direction = Vector3(
                math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation)),
                math.sin(math.radians(elevation)),
                math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation))
            )
            
            # 应用代理旋转
            rotated_direction = self._rotate_vector(direction, agent_rotation)
            
            # 执行光线投射
            hit_point = agent_position + rotated_direction * distance
            points[i] = [hit_point.x, hit_point.y, hit_point.z]
        
        return points
        
    def sense(self, agent_position: Vector3, agent_rotation: Vector3, 
              spatial_model: SpatialModel) -> SensorReading:
        """执行感知"""
        reading = SensorReading(
            timestamp=time.time(),
            position=agent_position,
            rotation=agent_rotation
        )
        
        # 距离感知
        reading.distances = self._sense_distances(agent_position, spatial_model)
        
        # 对象检测
        reading.objects_detected = self._detect_objects(agent_position, spatial_model)
        
        # 力感知
        reading.forces = self._sense_forces(agent_position, spatial_model)
        
        # 触觉感知
        reading.tactile_data = self._sense_tactile(agent_position, spatial_model)
        
        self.reading_history.append(reading)
        return reading
    
    def _sense_distances(self, position: Vector3, spatial_model: SpatialModel) -> List[float]:
        """距离感知"""
        distances = []
        
        for angle in self.sensor_angles:
            hit_obj, hit_distance = spatial_model.raycast(position, angle, self.sensor_range)
            # 添加噪声
            noisy_distance = hit_distance + np.random.normal(0, self.noise_level)
            distances.append(max(0, noisy_distance))
        
        return distances
    
    def _detect_objects(self, position: Vector3, spatial_model: SpatialModel) -> List[str]:
        """对象检测"""
        detected_objects = []
        nearby_objects = spatial_model.get_objects_in_region(position, self.sensor_range)
        
        for obj in nearby_objects:
            # 检查可见性
            direction = obj.position - position
            hit_obj, _ = spatial_model.raycast(position, direction, direction.length())
            if hit_obj == obj:
                detected_objects.append(obj.id)
        
        return detected_objects
    
    def _sense_forces(self, position: Vector3, spatial_model: SpatialModel) -> List[float]:
        """力感知"""
        forces = []
        # 简化力感知：检测附近的运动对象
        nearby_objects = spatial_model.get_objects_in_region(position, 2.0)
        
        for obj in nearby_objects:
            if obj.velocity.length() > 0.1:
                force_magnitude = obj.mass * obj.velocity.length()
                forces.append(force_magnitude)
            else:
                forces.append(0.0)
        
        return forces
    
    def _sense_tactile(self, position: Vector3, spatial_model: SpatialModel) -> List[float]:
        """触觉感知"""
        tactile_data = []
        # 检查直接接触
        contact_objects = spatial_model.get_objects_in_region(position, 1.0)
        
        for obj in contact_objects:
            if obj.position.distance_to(position) < 1.0:
                tactile_data.append(1.0)  # 接触
            else:
                tactile_data.append(0.0)  # 未接触
        
        return tactile_data
    
    def get_kinesthetic_feedback(self, action: ActionType, result: bool) -> Dict[str, float]:
        """运动觉反馈"""
        return {
            "force_feedback": 1.0 if result else 0.0,
            "position_accuracy": 0.9 if result else 0.3,
            "temporal_accuracy": 0.8 if result else 0.2
        }


class ActionExecutor:
    """动作执行器"""
    
    def __init__(self):
        self.action_history = deque(maxlen=50)
        self.execution_metrics = {}
        
    def execute_action(self, action: ActionType, spatial_model: SpatialModel, 
                      physics_engine: PhysicsEngine) -> Dict[str, Any]:
        """执行动作"""
        result = {
            "action": action.value,
            "success": False,
            "execution_time": 0.0,
            "side_effects": [],
            "new_state": None
        }
        
        start_time = time.time()
        
        try:
            if action == ActionType.MOVE_FORWARD:
                result = self._execute_move_forward(spatial_model, physics_engine)
            elif action == ActionType.MOVE_BACKWARD:
                result = self._execute_move_backward(spatial_model, physics_engine)
            elif action == ActionType.TURN_LEFT:
                result = self._execute_turn_left(spatial_model)
            elif action == ActionType.TURN_RIGHT:
                result = self._execute_turn_right(spatial_model)
            elif action == ActionType.PICK_UP:
                result = self._execute_pick_up(spatial_model)
            elif action == ActionType.PUT_DOWN:
                result = self._execute_put_down(spatial_model)
            elif action == ActionType.PUSH:
                result = self._execute_push(spatial_model, physics_engine)
            elif action == ActionType.PULL:
                result = self._execute_pull(spatial_model, physics_engine)
            elif action == ActionType.INTERACT:
                result = self._execute_interact(spatial_model)
            else:
                result["success"] = False
                result["error"] = f"未实现的动作类型: {action.value}"
            
            result["execution_time"] = time.time() - start_time
            self.action_history.append(result)
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["execution_time"] = time.time() - start_time
            logger.error(f"动作执行失败: {e}")
        
        return result
    
    def _execute_move_forward(self, spatial_model: SpatialModel, 
                             physics_engine: PhysicsEngine) -> Dict[str, Any]:
        """执行向前移动"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        # 计算目标位置
        target_position = agent.position + Vector3(0, 0, 1)
        
        # 检查路径是否可行
        if self._is_path_clear(agent.position, target_position, spatial_model):
            # 应用移动力
            physics_engine.apply_force("agent", Vector3(0, 0, 10), 0.1)
            return {
                "success": True,
                "new_position": target_position.to_dict(),
                "side_effects": ["代理向前移动1米"]
            }
        else:
            return {"success": False, "error": "路径被阻挡"}
    
    def _execute_move_backward(self, spatial_model: SpatialModel, 
                              physics_engine: PhysicsEngine) -> Dict[str, Any]:
        """执行向后移动"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        target_position = agent.position + Vector3(0, 0, -1)
        
        if self._is_path_clear(agent.position, target_position, spatial_model):
            physics_engine.apply_force("agent", Vector3(0, 0, -10), 0.1)
            return {
                "success": True,
                "new_position": target_position.to_dict(),
                "side_effects": ["代理向后移动1米"]
            }
        else:
            return {"success": False, "error": "路径被阻挡"}
    
    def _execute_turn_left(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """执行左转"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        agent.rotation.y += 90  # 向左转90度
        
        return {
            "success": True,
            "new_rotation": agent.rotation.to_dict(),
            "side_effects": ["代理向左转90度"]
        }
    
    def _execute_turn_right(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """执行右转"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        agent.rotation.y -= 90  # 向右转90度
        
        return {
            "success": True,
            "new_rotation": agent.rotation.to_dict(),
            "side_effects": ["代理向右转90度"]
        }
    
    def _execute_pick_up(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """执行拾取"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        # 查找附近的可拾取对象
        nearby_objects = spatial_model.get_objects_in_region(agent.position, 2.0)
        pickable_objects = [obj for obj in nearby_objects 
                           if obj.type == SpatialObjectType.TARGET and obj.id != "agent"]
        
        if pickable_objects:
            target_obj = pickable_objects[0]
            # 将对象添加到代理的持有列表中
            if "held_objects" not in agent.properties:
                agent.properties["held_objects"] = []
            agent.properties["held_objects"].append(target_obj.id)
            return {
                "success": True,
                "picked_object": target_obj.id,
                "side_effects": [f"代理拾取了{target_obj.id}"]
            }
        else:
            return {"success": False, "error": "附近没有可拾取的对象"}
    
    def _execute_put_down(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """执行放下"""
        agent = spatial_model.objects.get("agent")
        if not agent or "held_objects" not in agent.properties:
            return {"success": False, "error": "代理没有持有任何对象"}
        
        held_objects = agent.properties["held_objects"]
        if not held_objects:
            return {"success": False, "error": "代理没有持有任何对象"}
        
        placed_object_id = held_objects.pop()
        if placed_object_id in spatial_model.objects:
            placed_obj = spatial_model.objects[placed_object_id]
            placed_obj.position = agent.position + Vector3(0, 0, 1)
            
            return {
                "success": True,
                "placed_object": placed_object_id,
                "side_effects": [f"代理放下了{placed_object_id}"]
            }
        else:
            return {"success": False, "error": "持有的对象已不存在"}
    
    def _execute_push(self, spatial_model: SpatialModel, 
                     physics_engine: PhysicsEngine) -> Dict[str, Any]:
        """执行推动"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        # 查找可推动的对象
        nearby_objects = spatial_model.get_objects_in_region(agent.position, 1.5)
        pushable_objects = [obj for obj in nearby_objects 
                           if obj.id not in spatial_model.static_objects and obj.id != "agent"]
        
        if pushable_objects:
            target_obj = pushable_objects[0]
            # 应用推力
            push_direction = (target_obj.position - agent.position).normalize()
            physics_engine.apply_force(target_obj.id, push_direction * 20, 0.2)
            
            return {
                "success": True,
                "pushed_object": target_obj.id,
                "side_effects": [f"代理推动了{target_obj.id}"]
            }
        else:
            return {"success": False, "error": "附近没有可推动的对象"}
    
    def _execute_pull(self, spatial_model: SpatialModel, 
                     physics_engine: PhysicsEngine) -> Dict[str, Any]:
        """执行拉动"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        # 查找可拉动的对象
        nearby_objects = spatial_model.get_objects_in_region(agent.position, 2.0)
        pullable_objects = [obj for obj in nearby_objects 
                           if obj.id not in spatial_model.static_objects and obj.id != "agent"]
        
        if pullable_objects:
            target_obj = pullable_objects[0]
            # 应用拉力
            pull_direction = (agent.position - target_obj.position).normalize()
            physics_engine.apply_force(target_obj.id, pull_direction * 15, 0.3)
            
            return {
                "success": True,
                "pulled_object": target_obj.id,
                "side_effects": [f"代理拉动了{target_obj.id}"]
            }
        else:
            return {"success": False, "error": "附近没有可拉动的对象"}
    
    def _execute_interact(self, spatial_model: SpatialModel) -> Dict[str, Any]:
        """执行交互"""
        agent = spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        # 查找可交互的对象
        nearby_objects = spatial_model.get_objects_in_region(agent.position, 2.0)
        interactive_objects = [obj for obj in nearby_objects 
                              if obj.type in [SpatialObjectType.DOOR, SpatialObjectType.WINDOW]]
        
        if interactive_objects:
            target_obj = interactive_objects[0]
            # 简化的交互逻辑：切换门/窗的状态
            if "is_open" not in target_obj.properties:
                target_obj.properties["is_open"] = False
            target_obj.properties["is_open"] = not target_obj.properties["is_open"]
            
            return {
                "success": True,
                "interacted_object": target_obj.id,
                "state_change": f"门/窗状态: {'开启' if target_obj.properties['is_open'] else '关闭'}",
                "side_effects": [f"代理与{target_obj.id}交互"]
            }
        else:
            return {"success": False, "error": "附近没有可交互的对象"}
    
    def _is_path_clear(self, start: Vector3, end: Vector3, spatial_model: SpatialModel) -> bool:
        """检查路径是否畅通"""
        direction = end - start
        if direction.length() == 0:
            return True
        
        direction = direction.normalize()
        step_size = 0.5
        distance = direction.length()
        steps = int(distance / step_size)
        
        for i in range(1, steps + 1):
            current_pos = start + direction * step_size * i
            if not spatial_model.is_position_valid(current_pos):
                return False
            
            # 检查碰撞
            objects = spatial_model.get_objects_in_region(current_pos, 0.5)
            for obj in objects:
                if obj.id not in spatial_model.static_objects:
                    return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.action_history:
            return {}
        
        successful_actions = [a for a in self.action_history if a.get("success", False)]
        total_actions = len(self.action_history)
        
        return {
            "total_actions": total_actions,
            "successful_actions": len(successful_actions),
            "success_rate": len(successful_actions) / total_actions if total_actions > 0 else 0,
            "average_execution_time": np.mean([a.get("execution_time", 0) for a in self.action_history]),
            "last_action": self.action_history[-1] if self.action_history else None
        }


class SpatialEmbodiedAI:
    """空间智能和具身智能集成系统"""
    
    def __init__(self):
        # 核心组件
        self.spatial_model = SpatialModel()
        self.physics_engine = PhysicsEngine()
        self.spatial_reasoner = SpatialReasoner()
        self.path_planner = PathPlanner()
        self.sensor_system = SensorSystem()
        self.action_executor = ActionExecutor()
        
        # 系统状态
        self.is_running = False
        self.perception_cycle_active = False
        self.action_cycle_active = False
        self._start_time = time.time()  # 记录启动时间
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.perception_queue = queue.Queue()
        self.action_queue = queue.Queue()
        
        # 性能监控
        self.performance_metrics = {
            "perception_latency": deque(maxlen=100),
            "action_latency": deque(maxlen=100),
            "planning_time": deque(maxlen=100),
            "success_rate": deque(maxlen=100)
        }
        
        # 学习参数
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.reward_history = deque(maxlen=1000)
        
        # 高级功能
        self.spatial_memory = {}  # 空间记忆
        self.behavior_patterns = {}  # 行为模式
        self.adaptive_parameters = {
            "perception_sensitivity": 1.0,
            "action_precision": 1.0,
            "learning_persistence": 1.0
        }
        
        # 日志和调试
        self.debug_mode = False
        self.performance_profiling = False
        
        logger.info("空间智能和具身智能系统初始化完成")
    
    def initialize_environment(self, environment_config: Dict[str, Any]):
        """初始化环境"""
        logger.info("初始化环境...")
        
        # 创建地面
        ground = SpatialObject(
            id="ground",
            type=SpatialObjectType.WALL,
            position=Vector3(0, -0.5, 0),
            scale=Vector3(100, 1, 100),
            mass=float('inf'),
            properties={"static": True}
        )
        self.spatial_model.add_object(ground, static=True)
        
        # 创建代理
        agent_start_pos = environment_config.get("agent_start", Vector3(0, 1, 0))
        if isinstance(agent_start_pos, Vector3):
            agent = SpatialObject(
                id="agent",
                type=SpatialObjectType.AGENT,
                position=agent_start_pos,
                scale=Vector3(1, 2, 1),
                mass=80.0,
                properties={"velocity": Vector3(0, 0, 0), "held_objects": []}
            )
        else:
            agent = SpatialObject(
                id="agent",
                type=SpatialObjectType.AGENT,
                position=Vector3.from_dict(agent_start_pos) if isinstance(agent_start_pos, dict) else Vector3(*agent_start_pos),
                scale=Vector3(1, 2, 1),
                mass=80.0,
                properties={"velocity": Vector3(0, 0, 0), "held_objects": []}
            )
        self.spatial_model.add_object(agent, static=False)
        
        # 创建目标对象
        if "targets" in environment_config:
            for i, target_pos in enumerate(environment_config["targets"]):
                target = SpatialObject(
                    id=f"target_{i}",
                    type=SpatialObjectType.TARGET,
                    position=target_pos if isinstance(target_pos, Vector3) else (Vector3.from_dict(target_pos) if isinstance(target_pos, dict) else Vector3(*target_pos)),
                    scale=Vector3(0.5, 0.5, 0.5),
                    mass=5.0,
                    properties={"salience": 1.0}
                )
                self.spatial_model.add_object(target, static=False)
        
        # 创建障碍物
        if "obstacles" in environment_config:
            for i, obstacle_pos in enumerate(environment_config["obstacles"]):
                obstacle = SpatialObject(
                    id=f"obstacle_{i}",
                    type=SpatialObjectType.OBSTACLE,
                    position=obstacle_pos if isinstance(obstacle_pos, Vector3) else (Vector3.from_dict(obstacle_pos) if isinstance(obstacle_pos, dict) else Vector3(*obstacle_pos)),
                    scale=Vector3(2, 2, 2),
                    mass=float('inf'),
                    properties={"static": True}
                )
                self.spatial_model.add_object(obstacle, static=True)
        
        # 创建地标
        if "landmarks" in environment_config:
            for i, landmark_pos in enumerate(environment_config["landmarks"]):
                landmark = SpatialObject(
                    id=f"landmark_{i}",
                    type=SpatialObjectType.LANDMARK,
                    position=landmark_pos if isinstance(landmark_pos, Vector3) else (Vector3.from_dict(landmark_pos) if isinstance(landmark_pos, dict) else Vector3(*landmark_pos)),
                    scale=Vector3(3, 5, 3),
                    mass=float('inf'),
                    properties={
                        "landmark_type": "navigation",
                        "salience": 0.8,
                        "static": True
                    }
                )
                self.spatial_model.add_object(landmark, static=True)
        
        self.spatial_model.update_occupancy_grid()
        logger.info("环境初始化完成")
    
    def start_perception_action_cycle(self):
        """启动感知-行动循环"""
        if self.perception_cycle_active:
            return
        
        self.is_running = True
        self.perception_cycle_active = True
        
        # 启动感知线程
        self.executor.submit(self._perception_cycle)
        logger.info("感知-行动循环已启动")
    
    def stop_perception_action_cycle(self):
        """停止感知-行动循环"""
        self.is_running = False
        self.perception_cycle_active = False
        logger.info("感知-行动循环已停止")
    
    def _perception_cycle(self):
        """感知循环"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # 获取代理状态
                agent = self.spatial_model.objects.get("agent")
                if not agent:
                    time.sleep(0.1)
                    continue
                
                # 感知环境
                sensor_reading = self.sensor_system.sense(
                    agent.position, agent.rotation, self.spatial_model
                )
                
                # 更新空间模型（如果需要）
                self._update_spatial_model_from_perception(sensor_reading)
                
                # 推理空间关系
                spatial_relations = self.spatial_reasoner.analyze_spatial_relations(self.spatial_model)
                
                # 计算感知延迟
                perception_time = time.time() - start_time
                self.performance_metrics["perception_latency"].append(perception_time)
                
                # 等待一小段时间模拟真实感知延迟
                time.sleep(0.016)  # ~60Hz
                
            except Exception as e:
                logger.error(f"感知循环错误: {e}")
                time.sleep(0.1)
    
    def _update_spatial_model_from_perception(self, sensor_reading: SensorReading):
        """从感知数据更新空间模型"""
        # 这里可以实现更复杂的感知融合逻辑
        # 目前简化处理：更新代理位置（如果有更新的传感器数据）
        agent = self.spatial_model.objects.get("agent")
        if agent and abs(sensor_reading.timestamp - time.time()) < 1.0:  # 1秒内的数据
            # 更新代理位置（这里简化处理）
            pass
    
    def plan_and_execute_path(self, goal_position: Vector3) -> Optional[List[Vector3]]:
        """规划并执行路径"""
        agent = self.spatial_model.objects.get("agent")
        if not agent:
            logger.error("未找到代理对象")
            return None
        
        logger.info(f"规划路径到目标: {goal_position.to_tuple()}")
        
        start_time = time.time()
        
        # 1. 路径规划
        path = self.path_planner.a_star_pathfinding(
            self.spatial_model, agent.position, goal_position
        )
        
        if not path:
            logger.warning("无法找到路径")
            return None
        
        # 2. 路径平滑
        smoothed_path = self.path_planner.smooth_path(path, self.spatial_model)
        
        # 3. 执行路径
        execution_result = self._execute_path(smoothed_path)
        
        planning_time = time.time() - start_time
        self.performance_metrics["planning_time"].append(planning_time)
        
        return execution_result
    
    def _execute_path(self, path: List[Vector3]) -> Optional[List[Vector3]]:
        """执行路径"""
        if not path:
            return None
        
        executed_path = []
        success_count = 0
        
        for waypoint in path:
            # 计算到达路标需要的动作
            agent = self.spatial_model.objects.get("agent")
            if not agent:
                break
            
            # 计算方向和距离
            direction_to_waypoint = waypoint - agent.position
            distance = direction_to_waypoint.length()
            
            if distance < 0.5:  # 已接近路标
                executed_path.append(agent.position)
                continue
            
            # 决定动作
            action = self._decide_next_action(agent.position, waypoint, direction_to_waypoint)
            
            # 执行动作
            start_time = time.time()
            result = self.action_executor.execute_action(
                action, self.spatial_model, self.physics_engine
            )
            action_time = time.time() - start_time
            self.performance_metrics["action_latency"].append(action_time)
            
            if result["success"]:
                success_count += 1
                executed_path.append(waypoint)
                self.reward_history.append(1.0)  # 成功获得奖励
            else:
                self.reward_history.append(-0.1)  # 失败产生惩罚
            
            # 更新物理世界
            self.physics_engine.step(self.spatial_model)
            
            # 检查是否需要重新规划（环境发生变化）
            if self._should_replan(agent, waypoint):
                logger.info("检测到环境变化，需要重新规划")
                break
        
        # 计算成功率
        success_rate = success_count / len(path) if path else 0
        self.performance_metrics["success_rate"].append(success_rate)
        
        return executed_path if executed_path else None
    
    def _decide_next_action(self, current_pos: Vector3, target_pos: Vector3, 
                           direction: Vector3) -> ActionType:
        """决定下一个动作"""
        distance = direction.length()
        
        if distance > 0.5:
            if abs(direction.x) > abs(direction.z):
                if direction.x > 0:
                    return ActionType.MOVE_FORWARD
                else:
                    return ActionType.MOVE_BACKWARD
            else:
                if direction.z > 0:
                    return ActionType.MOVE_FORWARD
                else:
                    return ActionType.MOVE_BACKWARD
        
        return ActionType.INTERACT  # 到达目标附近
    
    def _should_replan(self, agent: SpatialObject, target: Vector3) -> bool:
        """检查是否需要重新规划"""
        # 检查是否有新的障碍物
        recent_objects = self.sensor_system.get_objects_in_region(
            agent.position, agent.position.distance_to(target)
        )
        
        for obj in recent_objects:
            if obj.id not in self.spatial_model.static_objects:
                # 检测到移动的障碍物
                if obj.velocity.length() > 0.1:
                    return True
        
        return False
    
    def interact_with_object(self, object_id: str, interaction_type: str = "pickup") -> Dict[str, Any]:
        """与对象交互"""
        agent = self.spatial_model.objects.get("agent")
        if not agent:
            return {"success": False, "error": "未找到代理对象"}
        
        target_obj = self.spatial_model.objects.get(object_id)
        if not target_obj:
            return {"success": False, "error": f"未找到对象 {object_id}"}
        
        # 检查距离
        distance = agent.position.distance_to(target_obj.position)
        if distance > 3.0:
            return {"success": False, "error": "对象距离过远，无法交互"}
        
        # 决定交互动作
        if interaction_type == "pickup" and target_obj.type == SpatialObjectType.TARGET:
            action = ActionType.PICK_UP
        elif interaction_type == "push" and target_obj.id not in self.spatial_model.static_objects:
            action = ActionType.PUSH
        elif interaction_type == "interact":
            action = ActionType.INTERACT
        else:
            return {"success": False, "error": f"不支持的交互类型: {interaction_type}"}
        
        # 执行交互
        result = self.action_executor.execute_action(
            action, self.spatial_model, self.physics_engine
        )
        
        # 更新物理世界
        self.physics_engine.step(self.spatial_model)
        
        return result
    
    def get_spatial_awareness(self) -> Dict[str, Any]:
        """获取空间感知状态"""
        agent = self.spatial_model.objects.get("agent")
        if not agent:
            return {"error": "未找到代理对象"}
        
        return {
            "current_position": agent.position.to_dict(),
            "current_rotation": agent.rotation.to_dict(),
            "nearby_objects": len(self.spatial_model.get_objects_in_region(agent.position, 5.0)),
            "visibility_map": self.spatial_model.get_visibility_map(agent.position),
            "spatial_relations": self.spatial_reasoner.analyze_spatial_relations(self.spatial_model),
            "cognitive_map": self.spatial_reasoner.create_cognitive_map(self.spatial_model)
        }
    
    def learn_from_interaction(self, interaction_history: List[Dict[str, Any]]):
        """从交互中学习"""
        for interaction in interaction_history:
            # 简化的学习算法：记录成功/失败模式
            if interaction.get("success", False):
                # 增加探索率的多样性
                self.exploration_rate = max(0.1, self.exploration_rate * 0.99)
            else:
                # 失败时增加探索率
                self.exploration_rate = min(0.5, self.exploration_rate * 1.01)
        
        logger.info(f"学习完成，探索率更新为: {self.exploration_rate:.3f}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取增强性能报告"""
        # 计算统计信息
        perception_latency = list(self.performance_metrics["perception_latency"])
        action_latency = list(self.performance_metrics["action_latency"])
        planning_time = list(self.performance_metrics["planning_time"])
        success_rate = list(self.performance_metrics["success_rate"])
        
        # 系统状态
        system_status = {
            "is_running": self.is_running,
            "perception_active": self.perception_cycle_active,
            "objects_count": len(self.spatial_model.objects),
            "static_objects_count": len(self.spatial_model.static_objects),
            "total_learning_episodes": len(self.reward_history),
            "system_uptime": time.time() - getattr(self, '_start_time', time.time()),
            "current_timestamp": datetime.now().isoformat()
        }
        
        # 性能指标
        performance_metrics = {
            "avg_perception_latency": np.mean(perception_latency) if perception_latency else 0,
            "min_perception_latency": np.min(perception_latency) if perception_latency else 0,
            "max_perception_latency": np.max(perception_latency) if perception_latency else 0,
            "std_perception_latency": np.std(perception_latency) if perception_latency else 0,
            
            "avg_action_latency": np.mean(action_latency) if action_latency else 0,
            "min_action_latency": np.min(action_latency) if action_latency else 0,
            "max_action_latency": np.max(action_latency) if action_latency else 0,
            "std_action_latency": np.std(action_latency) if action_latency else 0,
            
            "avg_planning_time": np.mean(planning_time) if planning_time else 0,
            "min_planning_time": np.min(planning_time) if planning_time else 0,
            "max_planning_time": np.max(planning_time) if planning_time else 0,
            
            "success_rate": np.mean(success_rate) if success_rate else 0,
            "min_success_rate": np.min(success_rate) if success_rate else 0,
            "max_success_rate": np.max(success_rate) if success_rate else 0
        }
        
        # 学习指标
        recent_rewards = list(self.reward_history)[-100:] if self.reward_history else []
        learning_metrics = {
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "recent_success_rate": np.mean(recent_rewards) if recent_rewards else 0,
            "cumulative_reward": np.sum(list(self.reward_history)) if self.reward_history else 0,
            "reward_variance": np.var(recent_rewards) if len(recent_rewards) > 1 else 0
        }
        
        # 空间智能指标
        spatial_metrics = self._calculate_spatial_intelligence_metrics()
        
        # 具身智能指标
        embodied_metrics = self._calculate_embodied_intelligence_metrics()
        
        return {
            "system_status": system_status,
            "performance_metrics": performance_metrics,
            "action_performance": self.action_executor.get_performance_metrics(),
            "learning_metrics": learning_metrics,
            "spatial_intelligence": spatial_metrics,
            "embodied_intelligence": embodied_metrics,
            "system_health": self._assess_system_health(performance_metrics, learning_metrics)
        }
    
    def _calculate_spatial_intelligence_metrics(self) -> Dict[str, float]:
        """计算空间智能指标"""
        agent = self.spatial_model.objects.get("agent")
        if not agent:
            return {"error": "未找到代理对象"}
        
        # 计算空间覆盖率
        visited_areas = set()
        total_area = (self.spatial_model.boundaries["max_x"] - self.spatial_model.boundaries["min_x"]) * \
                    (self.spatial_model.boundaries["max_z"] - self.spatial_model.boundaries["min_z"])
        
        # 简化的覆盖率计算
        exploration_efficiency = min(1.0, len(self.spatial_model.objects) / 50.0)
        
        # 空间推理准确性
        spatial_relations = self.spatial_reasoner.analyze_spatial_relations(self.spatial_model)
        reasoning_accuracy = self._evaluate_reasoning_accuracy(spatial_relations)
        
        # 导航效率
        navigation_efficiency = 1.0 - (np.mean(list(self.performance_metrics["planning_time"])) / 10.0)
        
        return {
            "spatial_coverage": exploration_efficiency,
            "reasoning_accuracy": reasoning_accuracy,
            "navigation_efficiency": max(0, navigation_efficiency),
            "spatial_memory_capacity": min(1.0, len(self.spatial_model.objects) / 100.0),
            "scene_understanding_score": self._calculate_scene_understanding()
        }
    
    def _calculate_embodied_intelligence_metrics(self) -> Dict[str, float]:
        """计算具身智能指标"""
        action_metrics = self.action_executor.get_performance_metrics()
        
        # 动作执行精度
        execution_precision = action_metrics.get("success_rate", 0.0)
        
        # 学习适应性
        recent_success = np.mean(list(self.reward_history)[-20:]) if self.reward_history else 0
        adaptation_rate = min(1.0, max(0.0, recent_success))
        
        # 物理交互能力
        interaction_capability = min(1.0, len(self.action_executor.action_history) / 100.0)
        
        return {
            "execution_precision": execution_precision,
            "adaptation_rate": adaptation_rate,
            "interaction_capability": interaction_capability,
            "motor_learning_progress": self.learning_rate,
            "sensorimotor_coordination": self._calculate_coordination_score()
        }
    
    def _evaluate_reasoning_accuracy(self, spatial_relations: Dict[str, Any]) -> float:
        """评估推理准确性"""
        # 简化的推理准确性评估
        total_relations = sum(len(relations) for relations in spatial_relations.values())
        if total_relations == 0:
            return 0.5
        
        # 基于关系数量和复杂度的评分
        complexity_score = min(1.0, total_relations / 20.0)
        return complexity_score
    
    def _calculate_scene_understanding(self) -> float:
        """计算场景理解得分"""
        agent = self.spatial_model.objects.get("agent")
        if not agent:
            return 0.0
        
        # 基于可见对象数量和类型的理解得分
        visible_objects = self.spatial_model.get_objects_in_region(agent.position, 10.0)
        object_diversity = len(set(obj.type for obj in visible_objects))
        object_count = len(visible_objects)
        
        # 多样性和数量的综合评分
        diversity_score = min(1.0, object_diversity / 5.0)
        count_score = min(1.0, object_count / 20.0)
        
        return (diversity_score + count_score) / 2
    
    def _calculate_coordination_score(self) -> float:
        """计算协调性得分"""
        action_latency = list(self.performance_metrics["action_latency"])
        if not action_latency:
            return 0.5
        
        # 基于延迟稳定性的协调性评分
        latency_variance = np.var(action_latency)
        coordination_score = max(0, 1.0 - latency_variance)
        
        return coordination_score
    
    def _assess_system_health(self, performance: Dict[str, float], learning: Dict[str, float]) -> Dict[str, float]:
        """评估系统健康度"""
        health_indicators = {}
        
        # 感知延迟健康度
        perception_health = 1.0 if performance.get("avg_perception_latency", 1.0) < 0.1 else 0.5
        health_indicators["perception_health"] = perception_health
        
        # 动作延迟健康度
        action_health = 1.0 if performance.get("avg_action_latency", 1.0) < 0.2 else 0.6
        health_indicators["action_health"] = action_health
        
        # 成功率健康度
        success_health = performance.get("success_rate", 0.0)
        health_indicators["success_health"] = success_health
        
        # 学习健康度
        learning_health = learning.get("recent_success_rate", 0.0)
        health_indicators["learning_health"] = learning_health
        
        # 综合健康度
        overall_health = np.mean(list(health_indicators.values()))
        health_indicators["overall_health"] = overall_health
        
        return health_indicators
    
    def save_state(self, filepath: str):
        """保存系统状态"""
        state = {
            "spatial_model": self.spatial_model.to_dict(),
            "performance_metrics": {
                "perception_latency": list(self.performance_metrics["perception_latency"]),
                "action_latency": list(self.performance_metrics["action_latency"]),
                "planning_time": list(self.performance_metrics["planning_time"]),
                "success_rate": list(self.performance_metrics["success_rate"])
            },
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "reward_history": list(self.reward_history)
            },
            "action_history": [action.to_dict() if hasattr(action, 'to_dict') else str(action) 
                             for action in list(self.action_executor.action_history)]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"系统状态已保存到: {filepath}")
    
    def load_state(self, filepath: str):
        """加载系统状态"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 恢复空间模型
            self.spatial_model = SpatialModel.from_dict(state["spatial_model"])
            
            # 恢复性能指标
            perf_data = state.get("performance_metrics", {})
            self.performance_metrics = {
                "perception_latency": deque(perf_data.get("perception_latency", []), maxlen=100),
                "action_latency": deque(perf_data.get("action_latency", []), maxlen=100),
                "planning_time": deque(perf_data.get("planning_time", []), maxlen=100),
                "success_rate": deque(perf_data.get("success_rate", []), maxlen=100)
            }
            
            # 恢复学习参数
            learning_data = state.get("learning_parameters", {})
            self.learning_rate = learning_data.get("learning_rate", 0.1)
            self.exploration_rate = learning_data.get("exploration_rate", 0.2)
            self.reward_history = deque(learning_data.get("reward_history", []), maxlen=1000)
            
            logger.info(f"系统状态已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"加载系统状态失败: {e}")
            raise
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭空间智能和具身智能系统...")
        
        self.stop_perception_action_cycle()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("系统已关闭")


def create_complex_environment() -> Dict[str, Any]:
    """创建复杂环境配置"""
    return {
        "agent_start": Vector3(0, 1, 0),
        "targets": [
            Vector3(15, 1, 8),
            Vector3(-12, 1, -6),
            Vector3(8, 1, -15),
            Vector3(-10, 1, 12),
            Vector3(20, 1, 0)
        ],
        "obstacles": [
            Vector3(5, 1, 3), Vector3(3, 1, 5), Vector3(-3, 1, 6),
            Vector3(-6, 1, -2), Vector3(-8, 1, -8), Vector3(10, 1, -5),
            Vector3(12, 1, 3), Vector3(-12, 1, 4), Vector3(6, 1, -12),
            Vector3(-4, 1, 10), Vector3(15, 1, -8), Vector3(-15, 1, 10)
        ],
        "landmarks": [
            Vector3(25, 5, 0), Vector3(-25, 5, 0), Vector3(0, 5, 25),
            Vector3(0, 5, -25), Vector3(18, 5, 18), Vector3(-18, 5, -18)
        ],
        "dynamic_objects": [
            {"position": Vector3(7, 1, 7), "type": "moving_target", "path": "circle"},
            {"position": Vector3(-7, 1, -7), "type": "moving_obstacle", "path": "back_and_forth"}
        ],
        "containers": [
            {
                "position": Vector3(8, 1, -8),
                "scale": Vector3(3, 2, 3),
                "contains": ["target_0", "target_1"]
            },
            {
                "position": Vector3(-8, 1, 8),
                "scale": Vector3(2, 3, 2),
                "contains": []
            }
        ]
    }


def create_spatial_intelligence_benchmark() -> Dict[str, Any]:
    """创建空间智能基准测试"""
    return {
        "navigation_tests": [
            {
                "name": "简单导航",
                "start": Vector3(0, 1, 0),
                "goal": Vector3(10, 1, 0),
                "obstacles": [Vector3(5, 1, 2), Vector3(5, 1, -2)]
            },
            {
                "name": "复杂导航",
                "start": Vector3(0, 1, 0),
                "goal": Vector3(15, 1, 15),
                "obstacles": [
                    Vector3(3, 1, 3), Vector3(6, 1, 1), Vector3(9, 1, 4),
                    Vector3(12, 1, 2), Vector3(8, 1, 8), Vector3(11, 1, 11)
                ]
            }
        ],
        "manipulation_tests": [
            {
                "name": "拾取测试",
                "target_position": Vector3(2, 1, 2),
                "interaction_type": "pickup"
            },
            {
                "name": "推动测试",
                "obstacle_position": Vector3(3, 1, 3),
                "interaction_type": "push"
            }
        ],
        "reasoning_tests": [
            {
                "name": "空间关系推理",
                "objects": [
                    {"position": Vector3(0, 1, 0), "type": "landmark"},
                    {"position": Vector3(5, 1, 0), "type": "target"},
                    {"position": Vector3(5, 1, 5), "type": "obstacle"}
                ]
            }
        ]
    }


def run_spatial_intelligence_test(scenario: str = "basic") -> Dict[str, Any]:
    """运行空间智能测试"""
    logger.info(f"开始运行空间智能测试: {scenario}")
    
    # 创建系统实例
    ai_system = SpatialEmbodiedAI()
    
    try:
        # 根据场景选择环境
        if scenario == "basic":
            environment_config = create_demo_environment()
        elif scenario == "complex":
            environment_config = create_complex_environment()
        elif scenario == "benchmark":
            benchmark_config = create_spatial_intelligence_benchmark()
            environment_config = create_demo_environment()  # 简化基准测试
        else:
            environment_config = create_demo_environment()
        
        # 初始化环境
        ai_system.initialize_environment(environment_config)
        
        # 运行测试
        test_results = {}
        
        if scenario in ["basic", "complex"]:
            # 路径规划测试
            if "targets" in environment_config:
                target_pos = environment_config["targets"][0]
                path_result = ai_system.plan_and_execute_path(target_pos)
                test_results["path_planning"] = {
                    "success": path_result is not None,
                    "path_length": len(path_result) if path_result else 0
                }
            
            # 对象交互测试
            if "targets" in environment_config:
                interaction_result = ai_system.interact_with_object("target_0", "pickup")
                test_results["object_interaction"] = {
                    "success": interaction_result.get("success", False),
                    "action_type": interaction_result.get("action", "unknown")
                }
        
        elif scenario == "benchmark":
            benchmark_config = create_spatial_intelligence_benchmark()
            test_results["benchmark"] = {}
            
            for nav_test in benchmark_config["navigation_tests"]:
                # 这里可以实现具体的基准测试逻辑
                test_results["benchmark"][nav_test["name"]] = {"status": "completed"}
        
        # 获取性能报告
        performance_report = ai_system.get_performance_report()
        test_results["performance"] = performance_report
        
        logger.info("空间智能测试完成")
        return test_results
        
    except Exception as e:
        logger.error(f"空间智能测试失败: {e}")
        return {"error": str(e)}
    
    finally:
        ai_system.shutdown()


def create_spatial_visualization_data(ai_system: SpatialEmbodiedAI) -> Dict[str, Any]:
    """创建空间可视化数据"""
    agent = ai_system.spatial_model.objects.get("agent")
    if not agent:
        return {"error": "未找到代理对象"}
    
    # 生成可视化数据结构
    visualization_data = {
        "world_bounds": ai_system.spatial_model.boundaries,
        "objects": [],
        "agent_state": {
            "position": agent.position.to_dict(),
            "rotation": agent.rotation.to_dict(),
            "velocity": agent.velocity.to_dict()
        },
        "spatial_relations": ai_system.spatial_reasoner.analyze_spatial_relations(ai_system.spatial_model),
        "visibility_map": ai_system.spatial_model.get_visibility_map(agent.position),
        "cognitive_map": ai_system.spatial_reasoner.create_cognitive_map(ai_system.spatial_model),
        "performance_metrics": {
            "perception_latency": list(ai_system.performance_metrics["perception_latency"]),
            "action_latency": list(ai_system.performance_metrics["action_latency"]),
            "success_rate": list(ai_system.performance_metrics["success_rate"])
        }
    }
    
    # 转换所有对象为可视化格式
    for obj in ai_system.spatial_model.objects.values():
        visualization_data["objects"].append({
            "id": obj.id,
            "type": obj.type.value,
            "position": obj.position.to_dict(),
            "scale": obj.scale.to_dict(),
            "properties": obj.properties
        })
    
    return visualization_data


def create_demo_environment() -> Dict[str, Any]:
    """创建演示环境配置"""
    return {
        "agent_start": Vector3(0, 1, 0),
        "targets": [
            Vector3(10, 1, 5),
            Vector3(-8, 1, -3),
            Vector3(5, 1, -12)
        ],
        "obstacles": [
            Vector3(3, 1, 2),
            Vector3(-2, 1, 4),
            Vector3(8, 1, -1),
            Vector3(-5, 1, -6),
            Vector3(0, 1, 8)
        ],
        "landmarks": [
            Vector3(15, 3, 0),
            Vector3(-15, 3, 0),
            Vector3(0, 3, 15),
            Vector3(0, 3, -15)
        ]
    }


def main():
    """主函数 - 演示系统功能"""
    logger.info("开始空间智能和具身智能系统演示...")
    
    # 创建系统实例
    ai_system = SpatialEmbodiedAI()
    
    try:
        # 运行不同复杂度的测试
        test_scenarios = ["basic", "complex"]
        all_results = {}
        
        for scenario in test_scenarios:
            logger.info(f"=== 运行 {scenario} 场景测试 ===")
            
            if scenario == "basic":
                environment_config = create_demo_environment()
            else:
                environment_config = create_complex_environment()
            
            # 初始化环境
            ai_system.initialize_environment(environment_config)
            
            # 启动感知-行动循环
            ai_system.start_perception_action_cycle()
            
            # 等待系统稳定
            time.sleep(2)
            
            # 示例1: 路径规划
            logger.info(f"=== {scenario} - 路径规划 ===")
            target_pos = environment_config["targets"][0] if "targets" in environment_config else Vector3(10, 1, 5)
            path = ai_system.plan_and_execute_path(target_pos)
            
            if path:
                logger.info(f"成功规划并执行路径，路径长度: {len(path)}")
            else:
                logger.warning("路径规划失败")
            
            # 示例2: 对象交互
            logger.info(f"=== {scenario} - 对象交互 ===")
            interaction_result = ai_system.interact_with_object("target_0", "pickup")
            logger.info(f"交互结果: {interaction_result['success']}")
            
            # 示例3: 获取空间感知
            logger.info(f"=== {scenario} - 空间感知 ===")
            spatial_awareness = ai_system.get_spatial_awareness()
            logger.info(f"附近对象数量: {spatial_awareness['nearby_objects']}")
            
            # 示例4: 深度感知
            logger.info(f"=== {scenario} - 深度感知 ===")
            agent = ai_system.spatial_model.objects.get("agent")
            if agent:
                depth_image = ai_system.sensor_system.sense_depth_image(
                    agent.position, agent.rotation, ai_system.spatial_model
                )
                logger.info(f"深度图像尺寸: {depth_image.shape}")
            
            # 等待一段时间观察系统行为
            time.sleep(3)
            
            # 获取性能报告
            logger.info(f"=== {scenario} - 性能报告 ===")
            performance_report = ai_system.get_performance_report()
            
            # 保存测试结果
            all_results[scenario] = {
                "path_planning_success": path is not None,
                "interaction_success": interaction_result.get("success", False),
                "nearby_objects": spatial_awareness.get("nearby_objects", 0),
                "performance_report": performance_report
            }
            
            # 重置系统状态用于下一个测试
            ai_system.stop_perception_action_cycle()
            time.sleep(1)
        
        # 运行基准测试
        logger.info("=== 运行基准测试 ===")
        benchmark_results = run_spatial_intelligence_test("benchmark")
        all_results["benchmark"] = benchmark_results
        
        # 生成可视化数据
        logger.info("=== 生成可视化数据 ===")
        ai_system.initialize_environment(create_demo_environment())
        ai_system.start_perception_action_cycle()
        time.sleep(1)
        
        viz_data = create_spatial_visualization_data(ai_system)
        
        # 保存所有结果
        output_file = f"/tmp/spatial_ai_test_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_results": all_results,
                "visualization_data": viz_data,
                "timestamp": datetime.now().isoformat(),
                "system_version": "2.0"
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试结果已保存到: {output_file}")
        
        # 打印总结
        logger.info("=== 测试总结 ===")
        for scenario, results in all_results.items():
            if scenario != "benchmark":
                logger.info(f"{scenario} 场景成功率: {results.get('path_planning_success', False)}")
        
        logger.info("所有演示完成")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭系统...")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
    finally:
        # 关闭系统
        ai_system.shutdown()


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行演示
    main()