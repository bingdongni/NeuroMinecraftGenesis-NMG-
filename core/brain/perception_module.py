#!/usr/bin/env python3
"""
升级版多模态感知融合和世界模型构建系统

该模块实现了完整的升级版感知系统，整合多模态感知融合、世界模型构建和预测、
空间智能和导航、因果推理和关系建模以及环境建模和更新功能。

主要功能模块：
1. 多模态感知融合：整合视觉、音频、触觉、空间等多种感知模态
2. 世界模型构建和预测：动态构建环境模型并进行未来状态预测
3. 空间智能和导航：基于空间认知的智能导航和路径规划
4. 因果推理和关系建模：发现和建模对象间的因果关系
5. 环境建模和更新：实时环境状态建模和动态更新

作者: NeuroMinecraftGenesis
创建时间: 2025-11-13
版本: 2.0 (升级版)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import threading
import time
import logging
import queue
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# 核心数据结构定义
# =========================

@dataclass
class MultimodalPerception:
    """多模态感知数据"""
    timestamp: float
    modality_type: str
    raw_data: Dict[str, Any]
    processed_features: np.ndarray
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    spatial_info: Optional['SpatialInfo'] = None
    temporal_context: Optional['TemporalContext'] = None


@dataclass
class SpatialInfo:
    """空间信息"""
    position: np.ndarray  # 3D位置
    orientation: np.ndarray  # 3D朝向
    bounds: np.ndarray  # 边界框
    depth_map: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    surface_normals: Optional[np.ndarray] = None


@dataclass
class TemporalContext:
    """时间上下文"""
    sequence_id: str
    time_window: float
    motion_vector: np.ndarray
    predicted_motion: Optional[np.ndarray] = None
    stability_score: float = 0.0


@dataclass
class WorldObject:
    """世界对象"""
    object_id: str
    position: np.ndarray
    attributes: Dict[str, Any]
    confidence: float
    modality_sources: List[str]
    first_seen: float
    last_seen: float
    spatial_info: Optional[SpatialInfo] = None
    temporal_history: List[TemporalContext] = field(default_factory=list)
    causal_relationships: Dict[str, 'CausalRelation'] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """因果关系"""
    cause_object: str
    effect_object: str
    relation_type: str
    strength: float
    temporal_lag: float
    confidence: float
    evidence_count: int = 0


@dataclass
class NavigationPath:
    """导航路径"""
    waypoints: List[np.ndarray]
    costs: List[float]
    total_cost: float
    estimated_time: float
    alternative_paths: List[List[np.ndarray]] = field(default_factory=list)


# =========================
# 高级多模态融合引擎
# =========================

class AdvancedMultimodalFusion(nn.Module):
    """高级多模态融合引擎"""
    
    def __init__(self, input_dims: Dict[str, int], fusion_dim: int = 1024):
        super().__init__()
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        
        # 为每种模态创建专门的编码器
        self.modality_encoders = nn.ModuleDict()
        self.attention_layers = nn.ModuleDict()
        
        for modality, dim in input_dims.items():
            # 模态特定编码器
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, fusion_dim // len(input_dims))
            )
            
            # 注意力机制
            self.attention_layers[modality] = nn.MultiheadAttention(
                embed_dim=fusion_dim // len(input_dims),
                num_heads=8,
                batch_first=True
            )
        
        # 跨模态注意力 - 使用融合维度
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, fusion_dim // 2)
        )
        
        # 门控机制
        self.gate_weights = nn.Parameter(torch.ones(len(input_dims)))
        
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        encoded_features = []
        
        # 对每种模态进行编码
        for modality, features in modality_features.items():
            if modality in self.modality_encoders:
                # 确保输入是2D
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                elif features.dim() == 1:
                    features = features.unsqueeze(0)
                
                # 编码
                encoded = self.modality_encoders[modality](features)
                
                # 确保输出是1D张量
                if encoded.dim() > 1:
                    encoded = encoded.view(-1)
                
                encoded_features.append(encoded)
        
        if not encoded_features:
            return torch.zeros(self.fusion_dim)
        
        # 拼接所有编码特征
        concatenated = torch.cat(encoded_features)
        
        # 调整到目标维度
        if concatenated.size(0) < self.fusion_dim:
            # 填充零
            padding = torch.zeros(self.fusion_dim - concatenated.size(0))
            concatenated = torch.cat([concatenated, padding])
        else:
            # 截断
            concatenated = concatenated[:self.fusion_dim]
        
        # 简单的注意力加权
        weights = F.softmax(concatenated[:len(encoded_features)], dim=0)
        weighted = concatenated * weights.sum()
        
        # 最终融合
        output = self.fusion_network(weighted.unsqueeze(0))
        
        return output.squeeze(0)


# =========================
# 世界模型构建和预测系统
# =========================

class WorldModelPredictor:
    """世界模型预测器"""
    
    def __init__(self, model_dim: int = 512):
        self.model_dim = model_dim
        self.object_states = {}
        self.environmental_dynamics = {}
        self.prediction_horizon = 10.0  # 预测时间范围（秒）
        self.update_frequency = 10.0    # 更新频率（Hz）
        
        # 时间序列模型
        self.lstm_model = self._create_lstm_model()
        self.transformer_model = self._create_transformer_model()
        
        # 物理约束模型
        self.physics_constraints = self._load_physics_constraints()
        
    def _create_lstm_model(self) -> nn.LSTM:
        """创建LSTM预测模型"""
        return nn.LSTM(
            input_size=self.model_dim,
            hidden_size=self.model_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.1
        )
    
    def _create_transformer_model(self) -> nn.Transformer:
        """创建Transformer预测模型"""
        return nn.Transformer(
            d_model=self.model_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
    
    def _load_physics_constraints(self) -> Dict[str, Any]:
        """加载物理约束"""
        return {
            'gravity': 9.81,
            'friction_coefficient': 0.3,
            'elasticity': 0.5,
            'max_velocity': 10.0,
            'collision_threshold': 0.1
        }
    
    def update_world_state(self, objects: List[WorldObject], environment_state: Dict[str, Any]):
        """更新世界状态"""
        current_time = time.time()
        
        for obj in objects:
            self.object_states[obj.object_id] = {
                'position': obj.position.copy(),
                'velocity': self._estimate_velocity(obj),
                'acceleration': self._estimate_acceleration(obj),
                'timestamp': current_time,
                'attributes': obj.attributes.copy()
            }
        
        self.environmental_dynamics = environment_state.copy()
        
    def predict_future_states(self, time_horizon: float = None) -> Dict[str, Any]:
        """预测未来状态"""
        if time_horizon is None:
            time_horizon = self.prediction_horizon
        
        predictions = {}
        
        for obj_id, state in self.object_states.items():
            # 基于LSTM的短期预测
            lstm_pred = self._lstm_predict(obj_id, time_horizon)
            
            # 基于Transformer的长期预测
            transformer_pred = self._transformer_predict(obj_id, time_horizon)
            
            # 物理约束校正
            physics_pred = self._apply_physics_constraints(obj_id, time_horizon)
            
            # 融合预测结果
            predictions[obj_id] = self._fuse_predictions(
                lstm_pred, transformer_pred, physics_pred
            )
        
        return predictions
    
    def _estimate_velocity(self, obj: WorldObject) -> np.ndarray:
        """估算对象速度"""
        if len(obj.temporal_history) < 2:
            return np.zeros(3)
        
        history = sorted(obj.temporal_history, key=lambda x: x.time_window)
        latest = history[-1]
        previous = history[-2]
        
        dt = latest.time_window - previous.time_window
        if dt <= 0:
            return np.zeros(3)
        
        displacement = latest.motion_vector
        return displacement / dt
    
    def _estimate_acceleration(self, obj: WorldObject) -> np.ndarray:
        """估算对象加速度"""
        if len(obj.temporal_history) < 3:
            return np.zeros(3)
        
        history = sorted(obj.temporal_history, key=lambda x: x.time_window)
        velocities = []
        
        for i in range(1, len(history)):
            dt = history[i].time_window - history[i-1].time_window
            if dt > 0:
                vel = history[i].motion_vector / dt
                velocities.append(vel)
        
        if len(velocities) < 2:
            return np.zeros(3)
        
        dv = velocities[-1] - velocities[-2]
        dt = history[-1].time_window - history[-2].time_window
        return dv / dt if dt > 0 else np.zeros(3)
    
    def _lstm_predict(self, obj_id: str, time_horizon: float) -> np.ndarray:
        """LSTM预测"""
        # 这里应该使用真实的时间序列数据
        # 简化实现：基于当前速度和加速度的线性预测
        if obj_id not in self.object_states:
            return np.zeros(3)
        
        state = self.object_states[obj_id]
        velocity = state.get('velocity', np.zeros(3))
        acceleration = state.get('acceleration', np.zeros(3))
        
        # 简单运动学预测
        t = np.linspace(0, time_horizon, 10)
        predicted_positions = []
        
        for dt in t:
            position = state['position'] + velocity * dt + 0.5 * acceleration * dt**2
            predicted_positions.append(position)
        
        return np.array(predicted_positions)
    
    def _transformer_predict(self, obj_id: str, time_horizon: float) -> np.ndarray:
        """Transformer预测"""
        # 简化实现：返回LSTM预测结果
        return self._lstm_predict(obj_id, time_horizon)
    
    def _apply_physics_constraints(self, obj_id: str, time_horizon: float) -> np.ndarray:
        """应用物理约束"""
        if obj_id not in self.object_states:
            return np.zeros(3)
        
        state = self.object_states[obj_id]
        
        # 应用重力
        gravity = np.array([0, -self.physics_constraints['gravity'], 0])
        
        # 应用摩擦力（简化）
        velocity = state.get('velocity', np.zeros(3))
        friction = -velocity * self.physics_constraints['friction_coefficient']
        
        # 速度限制
        total_force = gravity + friction
        max_accel = self.physics_constraints['max_velocity'] / time_horizon
        
        # 约束校正
        if np.linalg.norm(total_force) > max_accel:
            total_force = total_force / np.linalg.norm(total_force) * max_accel
        
        return state['position'] + 0.5 * total_force * time_horizon**2
    
    def _fuse_predictions(self, lstm_pred: np.ndarray, transformer_pred: np.ndarray, 
                         physics_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """融合预测结果"""
        # 加权平均
        weights = [0.4, 0.3, 0.3]  # LSTM, Transformer, Physics
        
        if lstm_pred.size > 0 and transformer_pred.size > 0:
            fused = weights[0] * lstm_pred + weights[1] * transformer_pred + weights[2] * physics_pred
        elif lstm_pred.size > 0:
            fused = 0.7 * lstm_pred + 0.3 * physics_pred
        else:
            fused = physics_pred
        
        return {
            'predicted_positions': fused,
            'confidence': min(0.9, len(fused) / 10.0),  # 基于预测步数计算置信度
            'prediction_method': 'hybrid_fusion'
        }


# =========================
# 空间智能和导航系统
# =========================

class SpatialIntelligenceNavigator:
    """空间智能导航器"""
    
    def __init__(self, grid_resolution: float = 0.1):
        self.grid_resolution = grid_resolution
        self.occupancy_grid = None
        self.spatial_graph = nx.Graph()
        self.landmarks = {}
        self.navigation_waypoints = []
        
        # A*路径规划参数
        self.heuristic_weights = {
            'distance': 1.0,
            'visibility': 0.5,
            'complexity': 0.3
        }
        
        # 空间记忆系统
        self.spatial_memory = SpatialMemory()
        
    def build_spatial_representation(self, point_cloud: np.ndarray, 
                                   objects: List[WorldObject]) -> Dict[str, Any]:
        """构建空间表示"""
        # 构建占据栅格地图
        self._build_occupancy_grid(point_cloud, objects)
        
        # 构建空间图
        self._build_spatial_graph(objects)
        
        # 识别地标
        self._identify_landmarks(objects)
        
        # 生成导航路径点
        self._generate_navigation_waypoints()
        
        return {
            'occupancy_grid': self.occupancy_grid,
            'spatial_graph': self.spatial_graph,
            'landmarks': self.landmarks,
            'navigation_waypoints': self.navigation_waypoints
        }
    
    def _build_occupancy_grid(self, point_cloud: np.ndarray, objects: List[WorldObject]):
        """构建占据栅格地图"""
        if len(point_cloud) == 0:
            return
        
        # 确保点云是numpy数组
        if isinstance(point_cloud, dict):
            # 如果point_cloud是字典，尝试提取实际的点云数据
            if 'point_cloud' in point_cloud:
                point_cloud = point_cloud['point_cloud']
            else:
                return
        
        if not isinstance(point_cloud, np.ndarray) or len(point_cloud) == 0:
            return
        
        # 计算点云边界
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        
        # 创建栅格
        grid_size = np.ceil((max_coords - min_coords) / self.grid_resolution).astype(int)
        
        # 初始化栅格（0=未知，1=占据，2=空闲）
        self.occupancy_grid = np.zeros(grid_size, dtype=np.int32)
        
        # 填充占据的栅格
        for point in point_cloud:
            grid_coords = np.floor((point - min_coords) / self.grid_resolution).astype(int)
            if all(0 <= coord < size for coord, size in zip(grid_coords, grid_size)):
                grid_idx = tuple(grid_coords)
                if grid_idx[0] < self.occupancy_grid.shape[0] and grid_idx[1] < self.occupancy_grid.shape[1]:
                    self.occupancy_grid[grid_idx] = 1
        
        # 根据对象更新占据状态
        for obj in objects:
            if obj.spatial_info:
                grid_coords = np.floor(
                    (obj.spatial_info.position - min_coords) / self.grid_resolution
                ).astype(int)
                if all(0 <= coord < size for coord, size in zip(grid_coords, grid_size)):
                    self.occupancy_grid[tuple(grid_coords)] = 1
        
        # 标记空闲区域（简化版本）
        # 将未占据且不太接近占据点的区域标记为自由空间
        if self.occupancy_grid is not None and len(self.occupancy_grid.shape) == 2:
            # 计算占据点距离
            occupied_coords = np.where(self.occupancy_grid == 1)
            if len(occupied_coords[0]) > 0:
                # 简单的距离计算
                for i in range(grid_size[0]):
                    for j in range(grid_size[1]):
                        if self.occupancy_grid[i, j] == 0:
                            # 计算到最近占据点的距离
                            min_dist = float('inf')
                            for k in range(len(occupied_coords[0])):
                                dist = np.sqrt((i - occupied_coords[0][k])**2 + (j - occupied_coords[1][k])**2)
                                min_dist = min(min_dist, dist)
                            
                            if min_dist > 3:  # 3格距离
                                self.occupancy_grid[i, j] = 2
    
    def _build_spatial_graph(self, objects: List[WorldObject]):
        """构建空间关系图"""
        self.spatial_graph.clear()
        
        # 添加对象节点
        for obj in objects:
            self.spatial_graph.add_node(
                obj.object_id,
                position=obj.position,
                type=obj.attributes.get('class', 'unknown'),
                spatial_info=obj.spatial_info
            )
        
        # 添加边（基于空间关系）
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                distance = np.linalg.norm(obj1.position - obj2.position)
                
                if distance < 10.0:  # 10米内的对象建立连接
                    self.spatial_graph.add_edge(
                        obj1.object_id,
                        obj2.object_id,
                        weight=distance,
                        relation_type='spatial_proximity'
                    )
        
        # 添加基于功能关系的边
        self._add_functional_edges(objects)
    
    def _add_functional_edges(self, objects: List[WorldObject]):
        """添加功能关系边"""
        # 基于对象类别的功能关系
        functional_rules = {
            ('person', 'chair'): 'sits_on',
            ('person', 'table'): 'interacts_with',
            ('car', 'road'): 'travels_on',
            ('person', 'door'): 'passes_through',
            ('book', 'shelf'): 'stored_in'
        }
        
        for obj1 in objects:
            for obj2 in objects:
                if obj1.object_id != obj2.object_id:
                    class1 = obj1.attributes.get('class', 'unknown')
                    class2 = obj2.attributes.get('class', 'unknown')
                    
                    relation = functional_rules.get((class1, class2))
                    if relation:
                        self.spatial_graph.add_edge(
                            obj1.object_id,
                            obj2.object_id,
                            weight=1.0,
                            relation_type=relation,
                            is_functional=True
                        )
    
    def _identify_landmarks(self, objects: List[WorldObject]):
        """识别地标"""
        # 基于对象特征识别地标
        landmark_criteria = {
            'building': 10.0,      # 建筑类对象
            'tree': 5.0,          # 树木类对象
            'sign': 3.0,          # 标志类对象
            'fountain': 8.0,      # 喷泉等特殊对象
            'crossroad': 7.0      # 十字路口等
        }
        
        for obj in objects:
            obj_type = obj.attributes.get('class', 'unknown')
            landmark_value = landmark_criteria.get(obj_type, 1.0)
            
            # 基于空间特征评估地标价值
            if obj.spatial_info:
                spatial_complexity = self._assess_spatial_complexity(obj)
                landmark_value *= spatial_complexity
            
            if landmark_value > 5.0:
                self.landmarks[obj.object_id] = {
                    'position': obj.position,
                    'type': obj_type,
                    'value': landmark_value,
                    'visibility': self._assess_visibility(obj),
                    'uniqueness': self._assess_uniqueness(obj, objects)
                }
    
    def _assess_spatial_complexity(self, obj: WorldObject) -> float:
        """评估空间复杂性"""
        if not obj.spatial_info or obj.spatial_info.bounds is None:
            return 1.0
        
        bounds = obj.spatial_info.bounds
        volume = np.prod(bounds[1] - bounds[0])
        surface_area = 2 * np.sum((bounds[1] - bounds[0]) * [bounds[1][1] - bounds[0][1], 
                                                              bounds[1][2] - bounds[0][2], 
                                                              bounds[1][0] - bounds[0][0]])
        
        # 复杂性与体积和表面积相关
        complexity = (volume * surface_area) ** (1/6)
        return min(complexity, 10.0)
    
    def _assess_visibility(self, obj: WorldObject) -> float:
        """评估可见性"""
        # 简化：基于位置和遮挡
        if not obj.spatial_info:
            return 0.5
        
        position = obj.spatial_info.position
        # 基于高度和周围物体评估可见性
        height_score = min(position[1] / 3.0, 1.0)  # 3米为满分
        return height_score
    
    def _assess_uniqueness(self, obj: WorldObject, all_objects: List[WorldObject]) -> float:
        """评估独特性"""
        obj_type = obj.attributes.get('class', 'unknown')
        similar_objects = [o for o in all_objects 
                          if o.attributes.get('class', 'unknown') == obj_type]
        
        # 越稀少越独特
        uniqueness = 1.0 / max(len(similar_objects), 1)
        return min(uniqueness, 1.0)
    
    def _generate_navigation_waypoints(self):
        """生成导航路径点"""
        self.navigation_waypoints.clear()
        
        # 从地标生成导航点
        for landmark_id, landmark_info in self.landmarks.items():
            position = landmark_info['position']
            waypoint = {
                'id': f"wp_{landmark_id}",
                'position': position,
                'type': 'landmark',
                'connections': []
            }
            self.navigation_waypoints.append(waypoint)
        
        # 添加自由空间中的路径点
        if self.occupancy_grid is not None:
            free_points = np.argwhere(self.occupancy_grid == 2)
            
            # 均匀采样导航点
            sample_step = max(1, len(free_points) // 100)  # 最多100个导航点
            sampled_points = free_points[::sample_step]
            
            for i, grid_point in enumerate(sampled_points):
                # 转换回世界坐标
                # 这里需要知道栅格到世界坐标的转换
                waypoint = {
                    'id': f"wp_free_{i}",
                    'position': grid_point.astype(float),
                    'type': 'free_space',
                    'connections': []
                }
                self.navigation_waypoints.append(waypoint)
    
    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                  constraints: Dict[str, Any] = None) -> NavigationPath:
        """路径规划"""
        if constraints is None:
            constraints = {}
        
        # A*路径规划
        path = self._astar_planning(start_pos, goal_pos, constraints)
        
        # 路径优化
        optimized_path = self._optimize_path(path)
        
        # 生成替代路径
        alternative_paths = self._generate_alternative_paths(start_pos, goal_pos, constraints)
        
        # 计算路径成本
        costs = self._calculate_path_costs(optimized_path, constraints)
        total_cost = sum(costs)
        
        # 估算时间
        estimated_time = self._estimate_travel_time(optimized_path, constraints)
        
        return NavigationPath(
            waypoints=optimized_path,
            costs=costs,
            total_cost=total_cost,
            estimated_time=estimated_time,
            alternative_paths=alternative_paths
        )
    
    def _astar_planning(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                       constraints: Dict[str, Any]) -> List[np.ndarray]:
        """A*路径规划算法"""
        # 简化实现：返回直线距离
        # 在实际应用中，这里应该实现完整的A*算法
        
        # 检查路径是否被阻挡
        if self._is_path_blocked(start_pos, goal_pos, constraints):
            # 如果直线被阻挡，寻找绕行路径
            return self._find_detour_path(start_pos, goal_pos, constraints)
        
        # 返回直线路径（简化）
        return [start_pos, goal_pos]
    
    def _is_path_blocked(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                        constraints: Dict[str, Any]) -> bool:
        """检查路径是否被阻挡"""
        if self.occupancy_grid is None:
            return False
        
        # 简化检查：采样路径上的点
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = (1 - t) * start_pos + t * goal_pos
            
            if self._is_position_occupied(sample_pos):
                return True
        
        return False
    
    def _is_position_occupied(self, position: np.ndarray) -> bool:
        """检查位置是否被占据"""
        if self.occupancy_grid is None:
            return False
        
        # 转换位置到栅格坐标
        # 这里需要实现坐标转换逻辑
        return False  # 简化实现
    
    def _find_detour_path(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                         constraints: Dict[str, Any]) -> List[np.ndarray]:
        """寻找绕行路径"""
        # 简化实现：返回L形路径
        midpoint = np.array([
            start_pos[0],
            goal_pos[1],
            start_pos[2]
        ])
        
        return [start_pos, midpoint, goal_pos]
    
    def _optimize_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """路径优化"""
        if len(path) <= 2:
            return path
        
        optimized_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            current = path[i]
            
            # 移除不必要的中间点
            if not self._should_keep_waypoint(optimized_path[-1], current, path[i + 1]):
                continue
            
            optimized_path.append(current)
        
        optimized_path.append(path[-1])
        return optimized_path
    
    def _should_keep_waypoint(self, prev: np.ndarray, current: np.ndarray, 
                            next: np.ndarray) -> bool:
        """判断是否应该保留路径点"""
        # 计算角度
        v1 = current - prev
        v2 = next - current
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return True
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # 如果角度太小，可能可以移除这个点
        return angle > np.pi / 6  # 30度阈值
    
    def _generate_alternative_paths(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                                  constraints: Dict[str, Any]) -> List[List[np.ndarray]]:
        """生成替代路径"""
        alternatives = []
        
        # 生成几个不同的替代路径
        for offset in [2.0, -2.0, 4.0, -4.0]:
            alt_midpoint = np.array([
                (start_pos[0] + goal_pos[0]) / 2 + offset,
                (start_pos[1] + goal_pos[1]) / 2,
                (start_pos[2] + goal_pos[2]) / 2 + offset
            ])
            
            alt_path = [start_pos, alt_midpoint, goal_pos]
            alternatives.append(alt_path)
        
        return alternatives
    
    def _calculate_path_costs(self, path: List[np.ndarray], 
                            constraints: Dict[str, Any]) -> List[float]:
        """计算路径成本"""
        costs = []
        
        for i in range(len(path) - 1):
            segment_cost = np.linalg.norm(path[i + 1] - path[i])
            
            # 应用约束成本
            if constraints.get('avoid_occupied', True):
                if self._is_path_blocked(path[i], path[i + 1], constraints):
                    segment_cost *= 10  # 大幅增加成本
            
            costs.append(segment_cost)
        
        return costs
    
    def _estimate_travel_time(self, path: List[np.ndarray], 
                            constraints: Dict[str, Any]) -> float:
        """估算旅行时间"""
        total_distance = sum(
            np.linalg.norm(path[i + 1] - path[i]) 
            for i in range(len(path) - 1)
        )
        
        # 默认速度（米/秒）
        default_speed = constraints.get('speed', 1.0)
        
        return total_distance / default_speed


class SpatialMemory:
    """空间记忆系统"""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.episodic_memory = deque(maxlen=memory_size)
        self.semantic_memory = {}
        self.spatial_anchors = {}
        
    def store_episode(self, location: np.ndarray, timestamp: float, 
                     context: Dict[str, Any]):
        """存储情节记忆"""
        episode = {
            'location': location.copy(),
            'timestamp': timestamp,
            'context': context.copy(),
            'importance': self._calculate_importance(location, context)
        }
        self.episodic_memory.append(episode)
    
    def retrieve_similar_episodes(self, query_location: np.ndarray, 
                                 query_context: Dict[str, Any],
                                 similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """检索相似情节"""
        similar_episodes = []
        
        for episode in self.episodic_memory:
            location_similarity = self._calculate_location_similarity(
                query_location, episode['location']
            )
            context_similarity = self._calculate_context_similarity(
                query_context, episode['context']
            )
            
            combined_similarity = 0.6 * location_similarity + 0.4 * context_similarity
            
            if combined_similarity >= similarity_threshold:
                similar_episodes.append(episode)
        
        return similar_episodes
    
    def _calculate_importance(self, location: np.ndarray, context: Dict[str, Any]) -> float:
        """计算记忆重要性"""
        # 基于新颖性、频率等因素计算重要性
        novelty_score = 1.0  # 简化实现
        frequency_score = 1.0 / max(len(self.episodic_memory), 1)
        
        return novelty_score + frequency_score
    
    def _calculate_location_similarity(self, loc1: np.ndarray, loc2: np.ndarray) -> float:
        """计算位置相似度"""
        distance = np.linalg.norm(loc1 - loc2)
        max_distance = 10.0  # 10米为完全不相似
        
        return max(0, 1 - distance / max_distance)
    
    def _calculate_context_similarity(self, ctx1: Dict[str, Any], 
                                    ctx2: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        # 简化的上下文相似度计算
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            if ctx1[key] == ctx2[key]:
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
        
        return np.mean(similarity_scores)


# =========================
# 因果推理和关系建模系统
# =========================

class CausalReasoningEngine:
    """因果推理引擎"""
    
    def __init__(self, min_evidence_threshold: int = 3):
        self.min_evidence_threshold = min_evidence_threshold
        self.causal_graph = nx.DiGraph()
        self.temporal_sequences = defaultdict(list)
        self.causal_patterns = {}
        self.intervention_history = []
        
        # 预定义的因果关系模式
        self.causal_templates = {
            'physical_contact': {
                'description': '物理接触导致的因果关系',
                'conditions': ['spatial_proximity', 'temporal_alignment'],
                'strength_modifier': 0.8
            },
            'force_transmission': {
                'description': '力的传递',
                'conditions': ['object_connection', 'force_direction'],
                'strength_modifier': 0.9
            },
            'functional_dependency': {
                'description': '功能性依赖',
                'conditions': ['functional_role', 'operational_sequence'],
                'strength_modifier': 0.7
            },
            'temporal_sequence': {
                'description': '时间序列因果',
                'conditions': ['temporal_order', 'repeated_observation'],
                'strength_modifier': 0.6
            }
        }
    
    def discover_causal_relationships(self, observations: List[MultimodalPerception],
                                    objects: List[WorldObject]) -> Dict[str, CausalRelation]:
        """发现因果关系"""
        # 收集时间序列数据
        self._collect_temporal_sequences(observations, objects)
        
        # 应用因果发现算法
        discovered_causes = self._apply_causal_discovery_algorithms()
        
        # 验证和过滤因果关系
        validated_causes = self._validate_causal_relationships(discovered_causes)
        
        return validated_causes
    
    def _collect_temporal_sequences(self, observations: List[MultimodalPerception],
                                  objects: List[WorldObject]):
        """收集时间序列数据"""
        # 按时间排序观察
        sorted_observations = sorted(observations, key=lambda x: x.timestamp)
        
        # 收集每个对象的时间序列
        for obj in objects:
            obj_observations = [obs for obs in sorted_observations 
                              if obs.metadata.get('object_id') == obj.object_id]
            
            if obj_observations:
                self.temporal_sequences[obj.object_id] = obj_observations
        
        # 收集对象间的时间关系
        self._collect_inter_object_temporal_relations(objects)
    
    def _collect_inter_object_temporal_relations(self, objects: List[WorldObject]):
        """收集对象间时间关系"""
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    relation_key = f"{obj1.object_id}->{obj2.object_id}"
                    
                    # 分析时间依赖关系
                    temporal_lag = self._analyze_temporal_lag(obj1, obj2)
                    correlation_strength = self._calculate_correlation_strength(obj1, obj2)
                    
                    if temporal_lag is not None and correlation_strength > 0.3:
                        self.temporal_sequences[relation_key] = {
                            'temporal_lag': temporal_lag,
                            'correlation': correlation_strength,
                            'obj1_id': obj1.object_id,
                            'obj2_id': obj2.object_id
                        }
    
    def _analyze_temporal_lag(self, obj1: WorldObject, obj2: WorldObject) -> Optional[float]:
        """分析时间延迟"""
        if not obj1.temporal_history or not obj2.temporal_history:
            return None
        
        # 简化实现：分析运动序列的相关性
        history1 = sorted(obj1.temporal_history, key=lambda x: x.time_window)
        history2 = sorted(obj2.temporal_history, key=lambda x: x.time_window)
        
        if len(history1) < 2 or len(history2) < 2:
            return None
        
        # 计算交叉相关
        motions1 = np.array([h.motion_vector for h in history1])
        motions2 = np.array([h.motion_vector for h in history2])
        
        # 简化的延迟计算
        max_lag = min(5, len(motions1), len(motions2))
        best_lag = 0
        best_correlation = 0
        
        for lag in range(max_lag):
            if lag < len(motions1) and lag < len(motions2):
                correlation = np.corrcoef(motions1[:-lag], motions2[lag:])[0, 1]
                if not np.isnan(correlation) and abs(correlation) > best_correlation:
                    best_correlation = abs(correlation)
                    best_lag = lag
        
        return best_lag * 0.1 if best_correlation > 0.3 else None  # 假设每个时间步0.1秒
    
    def _calculate_correlation_strength(self, obj1: WorldObject, obj2: WorldObject) -> float:
        """计算相关强度"""
        if not obj1.temporal_history or not obj2.temporal_history:
            return 0.0
        
        history1 = sorted(obj1.temporal_history, key=lambda x: x.time_window)
        history2 = sorted(obj2.temporal_history, key=lambda x: x.time_window)
        
        if len(history1) < 2 or len(history2) < 2:
            return 0.0
        
        # 计算运动向量的相关性
        motions1 = np.array([h.motion_vector for h in history1])
        motions2 = np.array([h.motion_vector for h in history2])
        
        min_len = min(len(motions1), len(motions2))
        if min_len == 0:
            return 0.0
        
        motions1 = motions1[:min_len]
        motions2 = motions2[:min_len]
        
        try:
            correlation_matrix = np.corrcoef(motions1.T, motions2.T)
            # 提取运动相关性
            correlation = np.mean(correlation_matrix[:3, 3:6])
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _apply_causal_discovery_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """应用因果发现算法"""
        discovered_causes = {}
        
        # PC算法（简化实现）
        pc_causes = self._pc_algorithm()
        discovered_causes.update(pc_causes)
        
        # GES算法（简化实现）
        ges_causes = self._ges_algorithm()
        discovered_causes.update(ges_causes)
        
        # 基于时间序列的因果发现
        temporal_causes = self._temporal_causal_discovery()
        discovered_causes.update(temporal_causes)
        
        return discovered_causes
    
    def _pc_algorithm(self) -> Dict[str, Dict[str, Any]]:
        """PC算法"""
        causes = {}
        
        # 简化实现：基于空间邻近性和时间顺序的启发式方法
        variables = list(self.temporal_sequences.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                # 检查是否存在因果关系
                independence_score = self._test_independence(var1, var2)
                
                if independence_score < 0.1:  # 强依赖
                    # 确定方向
                    direction = self._determine_causal_direction(var1, var2)
                    
                    relation_key = f"{direction['cause']}->{direction['effect']}"
                    causes[relation_key] = {
                        'cause': direction['cause'],
                        'effect': direction['effect'],
                        'strength': 1 - independence_score,
                        'method': 'pc_algorithm',
                        'confidence': min(independence_score + 0.5, 1.0)
                    }
        
        return causes
    
    def _ges_algorithm(self) -> Dict[str, Dict[str, Any]]:
        """GES算法"""
        causes = {}
        
        # 简化实现：贪心搜索
        variables = list(self.temporal_sequences.keys())
        
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    # 计算加入边的得分改善
                    score_improvement = self._calculate_score_improvement(var1, var2)
                    
                    if score_improvement > 0.1:
                        relation_key = f"{var1}->{var2}"
                        causes[relation_key] = {
                            'cause': var1,
                            'effect': var2,
                            'strength': score_improvement,
                            'method': 'ges_algorithm',
                            'confidence': min(score_improvement, 1.0)
                        }
        
        return causes
    
    def _temporal_causal_discovery(self) -> Dict[str, Dict[str, Any]]:
        """时间序列因果发现"""
        causes = {}
        
        for seq_key, seq_data in self.temporal_sequences.items():
            if isinstance(seq_data, dict) and 'temporal_lag' in seq_data:
                obj1_id = seq_data['obj1_id']
                obj2_id = seq_data['obj2_id']
                temporal_lag = seq_data['temporal_lag']
                correlation = seq_data['correlation']
                
                if temporal_lag is not None and correlation > 0.3:
                    relation_key = f"{obj1_id}->{obj2_id}"
                    causes[relation_key] = {
                        'cause': obj1_id,
                        'effect': obj2_id,
                        'strength': correlation,
                        'temporal_lag': temporal_lag,
                        'method': 'temporal_analysis',
                        'confidence': min(correlation, 1.0)
                    }
        
        return causes
    
    def _test_independence(self, var1: str, var2: str) -> float:
        """测试独立性"""
        if var1 not in self.temporal_sequences or var2 not in self.temporal_sequences:
            return 1.0
        
        seq1 = self.temporal_sequences[var1]
        seq2 = self.temporal_sequences[var2]
        
        # 计算互信息
        return self._calculate_mutual_information(var1, var2)
    
    def _calculate_mutual_information(self, var1: str, var2: str) -> float:
        """计算互信息"""
        # 简化实现：返回相关性的负值
        if var1 in self.temporal_sequences and var2 in self.temporal_sequences:
            data1 = self._extract_sequence_values(var1)
            data2 = self._extract_sequence_values(var2)
            
            if len(data1) > 1 and len(data2) > 1:
                correlation = np.corrcoef(data1, data2)[0, 1]
                return 1 - abs(correlation) if not np.isnan(correlation) else 1.0
        
        return 1.0
    
    def _extract_sequence_values(self, var: str) -> np.ndarray:
        """提取序列值"""
        seq_data = self.temporal_sequences[var]
        
        if isinstance(seq_data, list):
            # 如果是观察序列
            return np.array([getattr(obs, 'confidence', 0.0) for obs in seq_data])
        elif isinstance(seq_data, dict) and 'correlation' in seq_data:
            # 如果是关系数据
            return np.array([seq_data['correlation']])
        else:
            return np.array([0.0])
    
    def _determine_causal_direction(self, var1: str, var2: str) -> Dict[str, str]:
        """确定因果方向"""
        # 简化实现：基于时间戳
        # 实际应用中应该使用更复杂的因果方向判断方法
        
        if var1 in self.temporal_sequences and var2 in self.temporal_sequences:
            time1 = self._get_average_timestamp(var1)
            time2 = self._get_average_timestamp(var2)
            
            if time1 < time2:
                return {'cause': var1, 'effect': var2}
            else:
                return {'cause': var2, 'effect': var1}
        
        return {'cause': var1, 'effect': var2}
    
    def _get_average_timestamp(self, var: str) -> float:
        """获取平均时间戳"""
        seq_data = self.temporal_sequences[var]
        
        if isinstance(seq_data, list):
            if seq_data:
                return np.mean([getattr(obs, 'timestamp', 0.0) for obs in seq_data])
        elif isinstance(seq_data, dict) and 'timestamp' in seq_data:
            return seq_data['timestamp']
        
        return 0.0
    
    def _calculate_score_improvement(self, var1: str, var2: str) -> float:
        """计算得分改善"""
        # 简化实现：基于变量间相关性
        correlation = self._calculate_correlation_strength_obj(var1, var2)
        return max(0, correlation - 0.3)  # 基础相关性阈值
    
    def _calculate_correlation_strength_obj(self, obj1_id: str, obj2_id: str) -> float:
        """计算对象间相关性强度"""
        # 这里需要根据对象ID查找对应的对象并计算相关性
        # 简化实现
        return 0.5
    
    def _validate_causal_relationships(self, discovered_causes: Dict[str, Dict[str, Any]]) -> Dict[str, CausalRelation]:
        """验证因果关系"""
        validated = {}
        
        for relation_key, cause_info in discovered_causes.items():
            # 应用最小证据阈值
            evidence_count = self._count_evidence(cause_info)
            
            if evidence_count >= self.min_evidence_threshold:
                # 创建CausalRelation对象
                causal_relation = CausalRelation(
                    cause_object=cause_info['cause'],
                    effect_object=cause_info['effect'],
                    relation_type=self._infer_relation_type(cause_info),
                    strength=cause_info['strength'],
                    temporal_lag=cause_info.get('temporal_lag', 0.0),
                    confidence=cause_info['confidence'],
                    evidence_count=evidence_count
                )
                
                validated[relation_key] = causal_relation
        
        return validated
    
    def _count_evidence(self, cause_info: Dict[str, Any]) -> int:
        """计算证据数量"""
        # 简化实现：基于置信度和强度计算
        base_score = int(cause_info.get('confidence', 0.0) * 5)
        strength_bonus = int(cause_info.get('strength', 0.0) * 3)
        
        return base_score + strength_bonus
    
    def _infer_relation_type(self, cause_info: Dict[str, Any]) -> str:
        """推断关系类型"""
        method = cause_info.get('method', '')
        
        if method == 'temporal_analysis':
            return 'temporal_sequence'
        elif method == 'pc_algorithm':
            return 'statistical_dependency'
        elif method == 'ges_algorithm':
            return 'functional_dependency'
        else:
            return 'unknown'
    
    def predict_intervention_outcomes(self, intervention: Dict[str, Any]) -> Dict[str, float]:
        """预测干预结果"""
        target = intervention.get('target_object')
        action = intervention.get('action')
        
        if not target or target not in self.causal_graph:
            return {}
        
        # 找到目标对象的因果影响链
        downstream_effects = self._find_downstream_effects(target)
        
        # 预测干预结果
        predicted_outcomes = {}
        
        for effect_obj, causal_rel in downstream_effects.items():
            # 基于因果强度和关系类型预测影响
            impact_strength = causal_rel.strength * causal_rel.confidence
            
            if causal_rel.relation_type == 'physical_contact':
                impact_strength *= 0.8
            elif causal_rel.relation_type == 'temporal_sequence':
                impact_strength *= 0.6
            elif causal_rel.relation_type == 'functional_dependency':
                impact_strength *= 0.9
            
            predicted_outcomes[effect_obj] = impact_strength
        
        return predicted_outcomes
    
    def _find_downstream_effects(self, target_object: str) -> Dict[str, CausalRelation]:
        """找到下游影响"""
        downstream = {}
        
        for relation_key, causal_rel in self.causal_graph.edges(data=True):
            if causal_rel.get('cause_object') == target_object:
                downstream[causal_rel['effect_object']] = causal_rel
        
        return downstream
    
    def update_causal_model(self, new_observations: List[MultimodalPerception]):
        """更新因果模型"""
        # 基于新观察更新因果图
        for observation in new_observations:
            # 更新相关的因果关系
            self._update_causal_relationships(observation)
        
        # 重新验证因果关系
        self._revalidate_causal_relationships()
    
    def _update_causal_relationships(self, observation: MultimodalPerception):
        """更新因果关系"""
        # 简化的在线更新逻辑
        obj_id = observation.metadata.get('object_id')
        if obj_id:
            # 更新相关的因果关系强度
            pass  # 实际实现需要复杂的更新逻辑
    
    def _revalidate_causal_relationships(self):
        """重新验证因果关系"""
        # 清理低置信度的因果关系
        to_remove = []
        
        for relation_key, causal_rel in self.causal_graph.edges(data=True):
            if causal_rel.get('confidence', 0.0) < 0.3:
                to_remove.append(relation_key)
        
        for relation_key in to_remove:
            self.causal_graph.remove_edge(*relation_key.split('->'))


# =========================
# 环境建模和更新系统
# =========================

class EnvironmentModeler:
    """环境建模和更新系统"""
    
    def __init__(self, update_rate: float = 1.0):
        self.update_rate = update_rate  # 更新频率（Hz）
        self.environment_state = {}
        self.change_history = deque(maxlen=1000)
        self.predicted_changes = {}
        self.environmental_rules = {}
        self.stability_metrics = {}
        
        # 环境模型组件
        self.static_environment = StaticEnvironmentModel()
        self.dynamic_environment = DynamicEnvironmentModel()
        self.interaction_model = EnvironmentInteractionModel()
        
    def initialize_environment_model(self, initial_objects: List[WorldObject],
                                   spatial_info: Dict[str, Any]):
        """初始化环境模型"""
        # 构建静态环境模型
        self.static_environment.build_static_map(initial_objects, spatial_info)
        
        # 初始化动态环境模型
        self.dynamic_environment.initialize()
        
        # 建立环境交互模型
        self.interaction_model.build_interaction_graph(initial_objects)
        
        # 设置环境规则
        self._initialize_environment_rules()
        
        # 初始化环境状态
        self.environment_state = {
            'static_objects': {obj.object_id: obj for obj in initial_objects},
            'dynamic_objects': {},
            'environmental_conditions': {},
            'last_update': time.time(),
            'model_version': '2.0'
        }
    
    def update_environment_state(self, new_observations: List[MultimodalPerception],
                               objects: List[WorldObject],
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """更新环境状态"""
        update_start_time = time.time()
        
        # 识别环境变化
        detected_changes = self._detect_environmental_changes(objects)
        
        # 更新静态环境模型
        static_changes = self.static_environment.update_static_elements(objects)
        
        # 更新动态环境模型
        dynamic_changes = self.dynamic_environment.update_dynamic_state(
            new_observations, objects
        )
        
        # 更新交互模型
        interaction_changes = self.interaction_model.update_interactions(objects)
        
        # 合并所有变化
        total_changes = self._merge_changes(
            detected_changes, static_changes, dynamic_changes, interaction_changes
        )
        
        # 预测未来变化
        self.predicted_changes = self._predict_future_changes(total_changes)
        
        # 更新稳定性指标
        self.stability_metrics = self._update_stability_metrics(total_changes)
        
        # 记录变化历史
        self.change_history.append({
            'timestamp': update_start_time,
            'changes': total_changes,
            'predicted_changes': self.predicted_changes,
            'stability_metrics': self.stability_metrics
        })
        
        # 更新环境状态
        self._apply_changes_to_state(total_changes)
        
        # 清理过期数据
        self._cleanup_old_data()
        
        update_duration = time.time() - update_start_time
        
        return {
            'environment_state': self.environment_state,
            'detected_changes': total_changes,
            'predicted_changes': self.predicted_changes,
            'stability_metrics': self.stability_metrics,
            'update_duration': update_duration,
            'num_changes': len(total_changes)
        }
    
    def _detect_environmental_changes(self, current_objects: List[WorldObject]) -> Dict[str, Any]:
        """检测环境变化"""
        changes = {
            'new_objects': [],
            'removed_objects': [],
            'modified_objects': [],
            'spatial_changes': [],
            'temporal_changes': []
        }
        
        previous_objects = self.environment_state.get('static_objects', {})
        current_object_dict = {obj.object_id: obj for obj in current_objects}
        
        # 检测新对象
        for obj_id, obj in current_object_dict.items():
            if obj_id not in previous_objects:
                changes['new_objects'].append(obj)
        
        # 检测移除的对象
        for obj_id in previous_objects:
            if obj_id not in current_object_dict:
                changes['removed_objects'].append(obj_id)
        
        # 检测修改的对象
        for obj_id in current_object_dict:
            if obj_id in previous_objects:
                if self._has_object_changed(current_object_dict[obj_id], previous_objects[obj_id]):
                    changes['modified_objects'].append(current_object_dict[obj_id])
        
        # 检测空间变化
        spatial_changes = self._detect_spatial_changes(current_objects)
        changes['spatial_changes'] = spatial_changes
        
        # 检测时间变化
        temporal_changes = self._detect_temporal_changes(current_objects)
        changes['temporal_changes'] = temporal_changes
        
        return changes
    
    def _has_object_changed(self, current_obj: WorldObject, previous_obj: WorldObject) -> bool:
        """判断对象是否发生变化"""
        # 位置变化
        position_change = np.linalg.norm(current_obj.position - previous_obj.position)
        
        # 属性变化
        attributes_changed = current_obj.attributes != previous_obj.attributes
        
        # 置信度变化
        confidence_change = abs(current_obj.confidence - previous_obj.confidence)
        
        # 空间信息变化
        spatial_change = False
        if current_obj.spatial_info and previous_obj.spatial_info:
            spatial_change = np.linalg.norm(
                current_obj.spatial_info.position - previous_obj.spatial_info.position
            ) > 0.1
        
        return (position_change > 0.5 or attributes_changed or 
                confidence_change > 0.2 or spatial_change)
    
    def _detect_spatial_changes(self, objects: List[WorldObject]) -> List[Dict[str, Any]]:
        """检测空间变化"""
        spatial_changes = []
        
        for obj in objects:
            if obj.spatial_info:
                # 检查边界变化
                if hasattr(obj, 'previous_spatial_info') and obj.previous_spatial_info:
                    current_bounds = obj.spatial_info.bounds
                    previous_bounds = obj.previous_spatial_info.bounds
                    
                    if not np.allclose(current_bounds, previous_bounds, atol=0.1):
                        spatial_changes.append({
                            'object_id': obj.object_id,
                            'change_type': 'bounds_change',
                            'current_bounds': current_bounds,
                            'previous_bounds': previous_bounds,
                            'magnitude': np.linalg.norm(current_bounds - previous_bounds)
                        })
        
        return spatial_changes
    
    def _detect_temporal_changes(self, objects: List[WorldObject]) -> List[Dict[str, Any]]:
        """检测时间变化"""
        temporal_changes = []
        
        for obj in objects:
            # 检查运动模式变化
            if len(obj.temporal_history) >= 2:
                recent_motions = obj.temporal_history[-5:]  # 最近5个时间点
                motion_vectors = np.array([h.motion_vector for h in recent_motions])
                
                # 计算运动模式变化
                if len(motion_vectors) > 1:
                    motion_variance = np.var(motion_vectors, axis=0)
                    total_variance = np.sum(motion_variance)
                    
                    if total_variance > 1.0:  # 运动模式显著变化
                        temporal_changes.append({
                            'object_id': obj.object_id,
                            'change_type': 'motion_pattern_change',
                            'motion_variance': total_variance,
                            'timestamp': time.time()
                        })
        
        return temporal_changes
    
    def _merge_changes(self, *change_sets) -> Dict[str, Any]:
        """合并变化"""
        merged = {
            'new_objects': [],
            'removed_objects': [],
            'modified_objects': [],
            'spatial_changes': [],
            'temporal_changes': []
        }
        
        for changes in change_sets:
            if isinstance(changes, dict):
                for change_type in merged.keys():
                    if change_type in changes:
                        merged[change_type].extend(changes[change_type])
        
        return merged
    
    def _predict_future_changes(self, current_changes: Dict[str, Any]) -> Dict[str, Any]:
        """预测未来变化"""
        predictions = {}
        
        # 基于当前变化模式预测
        change_pattern = self._analyze_change_pattern(current_changes)
        
        # 预测新对象出现
        new_object_predictions = self._predict_new_objects(current_changes)
        predictions['new_object_predictions'] = new_object_predictions
        
        # 预测对象移动
        movement_predictions = self._predict_object_movements(current_changes)
        predictions['movement_predictions'] = movement_predictions
        
        # 预测环境条件变化
        condition_predictions = self._predict_environmental_conditions()
        predictions['condition_predictions'] = condition_predictions
        
        return predictions
    
    def _analyze_change_pattern(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """分析变化模式"""
        pattern = {
            'change_frequency': 0.0,
            'change_types': {},
            'spatial_pattern': {},
            'temporal_pattern': {}
        }
        
        # 分析变化频率
        recent_changes = list(self.change_history)[-10:]  # 最近10次变化
        if recent_changes:
            pattern['change_frequency'] = len(recent_changes) / 10.0  # 每时间单位的频率
        
        # 分析变化类型分布
        for change_type in ['new_objects', 'removed_objects', 'modified_objects']:
            count = len(changes.get(change_type, []))
            pattern['change_types'][change_type] = count
        
        return pattern
    
    def _predict_new_objects(self, changes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测新对象"""
        predictions = []
        
        # 基于历史模式预测
        if self.change_history:
            recent_changes = list(self.change_history)[-5:]
            
            # 统计最近出现的新对象类型
            object_types = defaultdict(int)
            for change_record in recent_changes:
                for new_obj in change_record['changes'].get('new_objects', []):
                    obj_type = new_obj.attributes.get('class', 'unknown')
                    object_types[obj_type] += 1
            
            # 预测最可能的新对象类型
            if object_types:
                most_common_type = max(object_types.items(), key=lambda x: x[1])[0]
                
                # 预测出现位置（基于空间模式）
                predicted_locations = self._predict_object_locations(most_common_type)
                
                for location in predicted_locations:
                    predictions.append({
                        'predicted_type': most_common_type,
                        'predicted_location': location,
                        'confidence': 0.6,
                        'time_horizon': 5.0  # 5秒内
                    })
        
        return predictions
    
    def _predict_object_locations(self, object_type: str) -> List[np.ndarray]:
        """预测对象位置"""
        # 基于历史出现位置预测
        if self.change_history:
            recent_positions = []
            
            for change_record in self.change_history[-5:]:
                for new_obj in change_record['changes'].get('new_objects', []):
                    if new_obj.attributes.get('class') == object_type:
                        recent_positions.append(new_obj.position)
            
            if recent_positions:
                # 计算位置分布
                positions = np.array(recent_positions)
                centroid = np.mean(positions, axis=0)
                
                # 预测位置在历史分布附近
                return [centroid]
        
        # 默认预测当前位置周围
        return [np.zeros(3)]
    
    def _predict_object_movements(self, changes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测对象移动"""
        predictions = []
        
        for modified_obj in changes.get('modified_objects', []):
            if len(modified_obj.temporal_history) >= 2:
                # 基于历史运动模式预测
                history = sorted(modified_obj.temporal_history, key=lambda x: x.time_window)
                recent_motions = np.array([h.motion_vector for h in history[-5:]])
                
                if len(recent_motions) > 1:
                    # 计算运动趋势
                    motion_trend = np.mean(recent_motions, axis=0)
                    
                    # 预测下一位置
                    current_position = modified_obj.position
                    predicted_position = current_position + motion_trend
                    
                    predictions.append({
                        'object_id': modified_obj.object_id,
                        'current_position': current_position,
                        'predicted_position': predicted_position,
                        'predicted_velocity': motion_trend,
                        'confidence': 0.7,
                        'time_horizon': 2.0
                    })
        
        return predictions
    
    def _predict_environmental_conditions(self) -> List[Dict[str, Any]]:
        """预测环境条件变化"""
        predictions = []
        
        # 基于历史数据预测环境变化
        if len(self.change_history) >= 5:
            recent_conditions = []
            
            for change_record in list(self.change_history)[-5:]:
                if 'environmental_conditions' in change_record:
                    recent_conditions.append(change_record['environmental_conditions'])
            
            if recent_conditions:
                # 简单的趋势预测
                prediction = {
                    'predicted_conditions': recent_conditions[-1],  # 保持当前条件
                    'confidence': 0.5,
                    'time_horizon': 10.0
                }
                predictions.append(prediction)
        
        return predictions
    
    def _update_stability_metrics(self, changes: Dict[str, Any]) -> Dict[str, float]:
        """更新稳定性指标"""
        metrics = {}
        
        # 环境稳定性分数
        total_changes = sum(len(change_list) for change_list in changes.values() if isinstance(change_list, list))
        metrics['stability_score'] = max(0, 1 - total_changes / 10.0)  # 10个变化为完全不稳定
        
        # 空间稳定性
        spatial_change_count = len(changes.get('spatial_changes', []))
        metrics['spatial_stability'] = max(0, 1 - spatial_change_count / 5.0)
        
        # 时间稳定性
        temporal_change_count = len(changes.get('temporal_changes', []))
        metrics['temporal_stability'] = max(0, 1 - temporal_change_count / 3.0)
        
        # 预测可靠性
        if self.predicted_changes:
            prediction_accuracy = self._evaluate_prediction_accuracy()
            metrics['prediction_reliability'] = prediction_accuracy
        else:
            metrics['prediction_reliability'] = 0.5
        
        return metrics
    
    def _evaluate_prediction_accuracy(self) -> float:
        """评估预测准确性"""
        # 简化实现：基于历史预测准确性
        if len(self.change_history) < 2:
            return 0.5
        
        correct_predictions = 0
        total_predictions = 0
        
        for change_record in self.change_history[-3:]:  # 检查最近3次预测
            predicted_changes = change_record.get('predicted_changes', {})
            actual_changes = change_record.get('changes', {})
            
            # 简化评估：新对象预测准确性
            predicted_new_objects = predicted_changes.get('new_object_predictions', [])
            actual_new_objects = actual_changes.get('new_objects', [])
            
            total_predictions += len(predicted_new_objects)
            correct_predictions += min(len(predicted_new_objects), len(actual_new_objects))
        
        return correct_predictions / max(total_predictions, 1)
    
    def _apply_changes_to_state(self, changes: Dict[str, Any]):
        """将变化应用到状态"""
        # 更新静态对象
        for new_obj in changes.get('new_objects', []):
            self.environment_state['static_objects'][new_obj.object_id] = new_obj
        
        for removed_obj_id in changes.get('removed_objects', []):
            if removed_obj_id in self.environment_state['static_objects']:
                del self.environment_state['static_objects'][removed_obj_id]
        
        # 更新时间戳
        self.environment_state['last_update'] = time.time()
        
        # 更新版本号
        current_version = self.environment_state.get('model_version', '1.0')
        version_parts = current_version.split('.')
        if len(version_parts) == 2:
            patch_version = int(version_parts[1]) + 1
            self.environment_state['model_version'] = f"{version_parts[0]}.{patch_version}"
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        current_time = time.time()
        
        # 清理超过1小时的变化历史
        while self.change_history and current_time - self.change_history[0]['timestamp'] > 3600:
            self.change_history.popleft()
        
        # 清理过期预测
        expired_predictions = []
        for pred_key, prediction in self.predicted_changes.items():
            if isinstance(pred_key, (int, float)) and current_time - pred_key > 300:
                expired_predictions.append(pred_key)
        
        for pred_time in expired_predictions:
            del self.predicted_changes[pred_time]
    
    def _initialize_environment_rules(self):
        """初始化环境规则"""
        self.environmental_rules = {
            'conservation_rules': [
                'Objects do not spontaneously appear/disappear',
                'Total object count is constrained by physical space',
                'Object properties change gradually over time'
            ],
            'causality_rules': [
                'Changes have causes in the environment',
                'Spatial proximity enables causal relationships',
                'Temporal ordering constrains causality'
            ],
            'stability_rules': [
                'Environment tends toward stability',
                'Drastic changes are rare',
                'Stable patterns repeat over time'
            ]
        }


class StaticEnvironmentModel:
    """静态环境模型"""
    
    def __init__(self):
        self.static_objects = {}
        self.spatial_layout = {}
        self.topological_map = {}
        
    def build_static_map(self, objects: List[WorldObject], spatial_info: Dict[str, Any]):
        """构建静态地图"""
        for obj in objects:
            self.static_objects[obj.object_id] = obj
            
        # 构建空间布局
        self._build_spatial_layout(objects, spatial_info)
        
        # 构建拓扑地图
        self._build_topological_map(objects)
    
    def _build_spatial_layout(self, objects: List[WorldObject], spatial_info: Dict[str, Any]):
        """构建空间布局"""
        self.spatial_layout = {
            'objects': {obj.object_id: obj.position for obj in objects},
            'regions': self._identify_spatial_regions(objects),
            'obstacles': self._identify_obstacles(objects),
            'navigation_surfaces': self._identify_navigation_surfaces(objects)
        }
    
    def _build_topological_map(self, objects: List[WorldObject]):
        """构建拓扑地图"""
        self.topological_map = {
            'nodes': {obj.object_id: obj.position for obj in objects},
            'edges': self._calculate_spatial_connections(objects),
            'regions': self._create_topological_regions(objects)
        }
    
    def _identify_spatial_regions(self, objects: List[WorldObject]) -> Dict[str, List[str]]:
        """识别空间区域"""
        # 简化的区域识别
        regions = {
            'central_area': [],
            'peripheral_area': [],
            'boundary_area': []
        }
        
        if objects:
            # 计算中心位置
            positions = np.array([obj.position for obj in objects])
            center = np.mean(positions, axis=0)
            
            for obj in objects:
                distance_to_center = np.linalg.norm(obj.position - center)
                
                if distance_to_center < 2.0:
                    regions['central_area'].append(obj.object_id)
                elif distance_to_center < 5.0:
                    regions['peripheral_area'].append(obj.object_id)
                else:
                    regions['boundary_area'].append(obj.object_id)
        
        return regions
    
    def _identify_obstacles(self, objects: List[WorldObject]) -> List[str]:
        """识别障碍物"""
        obstacles = []
        
        for obj in objects:
            obj_type = obj.attributes.get('class', '').lower()
            if obj_type in ['wall', 'building', 'tree', 'rock', 'fence']:
                obstacles.append(obj.object_id)
        
        return obstacles
    
    def _identify_navigation_surfaces(self, objects: List[WorldObject]) -> List[str]:
        """识别导航表面"""
        surfaces = []
        
        for obj in objects:
            obj_type = obj.attributes.get('class', '').lower()
            if obj_type in ['ground', 'floor', 'road', 'path', 'sidewalk']:
                surfaces.append(obj.object_id)
        
        return surfaces
    
    def _calculate_spatial_connections(self, objects: List[WorldObject]) -> List[Tuple[str, str, float]]:
        """计算空间连接"""
        connections = []
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                distance = np.linalg.norm(obj1.position - obj2.position)
                if distance < 10.0:  # 10米内认为有连接
                    connections.append((obj1.object_id, obj2.object_id, distance))
        
        return connections
    
    def _create_topological_regions(self, objects: List[WorldObject]) -> Dict[str, Dict[str, Any]]:
        """创建拓扑区域"""
        regions = {}
        
        # 基于聚类创建区域
        if len(objects) > 3:
            positions = np.array([obj.position for obj in objects])
            
            # 简单的K-means聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(5, len(objects)), random_state=42)
            labels = kmeans.fit_predict(positions)
            
            for i, obj in enumerate(objects):
                region_id = f"region_{labels[i]}"
                if region_id not in regions:
                    regions[region_id] = {
                        'objects': [],
                        'centroid': kmeans.cluster_centers_[labels[i]],
                        'size': 0
                    }
                
                regions[region_id]['objects'].append(obj.object_id)
                regions[region_id]['size'] += 1
        
        return regions
    
    def update_static_elements(self, objects: List[WorldObject]) -> Dict[str, Any]:
        """更新静态元素"""
        changes = {
            'new_static_objects': [],
            'removed_static_objects': [],
            'modified_static_objects': []
        }
        
        current_objects = {obj.object_id: obj for obj in objects}
        
        # 检测新的静态对象
        for obj_id, obj in current_objects.items():
            if obj_id not in self.static_objects:
                changes['new_static_objects'].append(obj)
                self.static_objects[obj_id] = obj
        
        # 检测移除的静态对象
        for obj_id in list(self.static_objects.keys()):
            if obj_id not in current_objects:
                changes['removed_static_objects'].append(obj_id)
                del self.static_objects[obj_id]
        
        # 检测修改的静态对象
        for obj_id, obj in current_objects.items():
            if obj_id in self.static_objects:
                existing_obj = self.static_objects[obj_id]
                if not np.allclose(obj.position, existing_obj.position, atol=0.01):
                    changes['modified_static_objects'].append(obj)
                    self.static_objects[obj_id] = obj
        
        # 更新空间布局
        self._build_spatial_layout(objects, {})
        self._build_topological_map(objects)
        
        return changes


class DynamicEnvironmentModel:
    """动态环境模型"""
    
    def __init__(self):
        self.dynamic_objects = {}
        self.motion_patterns = {}
        self.interaction_patterns = {}
        self.temporal_dynamics = {}
    
    def initialize(self):
        """初始化动态环境模型"""
        # 动态环境模型初始化逻辑
        pass
    
    def update_dynamic_state(self, observations: List[MultimodalPerception],
                           objects: List[WorldObject]) -> Dict[str, Any]:
        """更新动态状态"""
        changes = {
            'motion_updates': [],
            'interaction_updates': [],
            'pattern_changes': []
        }
        
        # 更新动态对象状态
        for obj in objects:
            if len(obj.temporal_history) > 0:
                self.dynamic_objects[obj.object_id] = obj
        
        # 更新运动模式
        motion_changes = self._update_motion_patterns(objects)
        changes['motion_updates'] = motion_changes
        
        # 更新交互模式
        interaction_changes = self._update_interaction_patterns(objects)
        changes['interaction_updates'] = interaction_changes
        
        # 更新时间动态
        temporal_changes = self._update_temporal_dynamics(observations)
        changes['temporal_changes'] = temporal_changes
        
        return changes
    
    def _update_motion_patterns(self, objects: List[WorldObject]) -> List[Dict[str, Any]]:
        """更新运动模式"""
        motion_changes = []
        
        for obj in objects:
            if len(obj.temporal_history) >= 3:
                # 分析运动模式
                recent_history = obj.temporal_history[-5:]
                motion_vectors = np.array([h.motion_vector for h in recent_history])
                
                # 计算运动特征
                velocity_magnitude = np.linalg.norm(motion_vectors.mean(axis=0))
                acceleration_magnitude = np.linalg.norm(np.diff(motion_vectors, axis=0).mean(axis=0)) if len(motion_vectors) > 1 else 0
                
                motion_pattern = {
                    'object_id': obj.object_id,
                    'velocity': velocity_magnitude,
                    'acceleration': acceleration_magnitude,
                    'direction': motion_vectors.mean(axis=0),
                    'stability': 1.0 / (1.0 + np.var(motion_vectors))
                }
                
                self.motion_patterns[obj.object_id] = motion_pattern
                motion_changes.append(motion_pattern)
        
        return motion_changes
    
    def _update_interaction_patterns(self, objects: List[WorldObject]) -> List[Dict[str, Any]]:
        """更新交互模式"""
        interaction_changes = []
        
        # 检测对象间距离变化
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                distance = np.linalg.norm(obj1.position - obj2.position)
                
                # 基于距离变化的交互检测
                if distance < 2.0:  # 2米内认为有交互
                    interaction = {
                        'object1': obj1.object_id,
                        'object2': obj2.object_id,
                        'type': 'proximity_interaction',
                        'strength': 1.0 / (distance + 0.1),
                        'timestamp': time.time()
                    }
                    
                    interaction_key = f"{obj1.object_id}-{obj2.object_id}"
                    self.interaction_patterns[interaction_key] = interaction
                    interaction_changes.append(interaction)
        
        return interaction_changes
    
    def _update_temporal_dynamics(self, observations: List[MultimodalPerception]) -> Dict[str, Any]:
        """更新时间动态"""
        temporal_changes = {}
        
        # 分析观察频率
        if len(observations) > 1:
            timestamps = [obs.timestamp for obs in observations]
            time_diffs = np.diff(timestamps)
            
            temporal_changes['observation_frequency'] = 1.0 / np.mean(time_diffs)
            temporal_changes['temporal_consistency'] = 1.0 / (1.0 + np.var(time_diffs))
        
        return temporal_changes


class EnvironmentInteractionModel:
    """环境交互模型"""
    
    def __init__(self):
        self.interaction_graph = nx.Graph()
        self.interaction_history = deque(maxlen=100)
        self.interaction_rules = {}
    
    def build_interaction_graph(self, objects: List[WorldObject]):
        """构建交互图"""
        self.interaction_graph.clear()
        
        # 添加节点
        for obj in objects:
            self.interaction_graph.add_node(
                obj.object_id,
                position=obj.position,
                type=obj.attributes.get('class', 'unknown')
            )
        
        # 添加边
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                interaction_strength = self._calculate_interaction_strength(obj1, obj2)
                if interaction_strength > 0.1:
                    self.interaction_graph.add_edge(
                        obj1.object_id,
                        obj2.object_id,
                        weight=interaction_strength
                    )
    
    def _calculate_interaction_strength(self, obj1: WorldObject, obj2: WorldObject) -> float:
        """计算交互强度"""
        # 基于距离和功能关系的交互强度计算
        distance = np.linalg.norm(obj1.position - obj2.position)
        
        # 距离衰减
        distance_factor = max(0, 1 - distance / 5.0)  # 5米为影响范围
        
        # 功能关系增强
        functional_enhancement = self._assess_functional_relation(obj1, obj2)
        
        return distance_factor * (1 + functional_enhancement)
    
    def _assess_functional_relation(self, obj1: WorldObject, obj2: WorldObject) -> float:
        """评估功能关系"""
        type1 = obj1.attributes.get('class', '').lower()
        type2 = obj2.attributes.get('class', '').lower()
        
        # 预定义的功能关系
        functional_pairs = {
            ('person', 'chair'): 0.8,
            ('person', 'table'): 0.6,
            ('car', 'road'): 0.9,
            ('book', 'shelf'): 0.7
        }
        
        pair1 = (type1, type2)
        pair2 = (type2, type1)
        
        return functional_pairs.get(pair1, functional_pairs.get(pair2, 0.0))
    
    def update_interactions(self, objects: List[WorldObject]) -> Dict[str, Any]:
        """更新交互"""
        changes = {
            'new_interactions': [],
            'interaction_updates': [],
            'interaction_removals': []
        }
        
        # 重新构建交互图
        old_edges = set(self.interaction_graph.edges())
        self.build_interaction_graph(objects)
        new_edges = set(self.interaction_graph.edges())
        
        # 检测新的交互
        for edge in new_edges - old_edges:
            changes['new_interactions'].append(edge)
        
        # 检测移除的交互
        for edge in old_edges - new_edges:
            changes['interaction_removals'].append(edge)
        
        # 更新交互历史
        current_interactions = list(new_edges)
        self.interaction_history.append({
            'timestamp': time.time(),
            'interactions': current_interactions
        })
        
        return changes


# =========================
# 主感知模块控制器
# =========================

class AdvancedPerceptionModule:
    """升级版感知模块主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个子系统
        self.fusion_engine = self._initialize_fusion_engine()
        self.world_predictor = WorldModelPredictor()
        self.spatial_navigator = SpatialIntelligenceNavigator()
        self.causal_engine = CausalReasoningEngine()
        self.environment_modeler = EnvironmentModeler()
        
        # 数据管理
        self.perception_buffer = deque(maxlen=1000)
        self.processed_perceptions = deque(maxlen=500)
        self.active_objects = {}
        
        # 系统状态
        self.is_running = False
        self.processing_threads = []
        self.update_thread = None
        
        # 性能监控
        self.performance_metrics = {
            'processing_latency': [],
            'fusion_accuracy': [],
            'prediction_accuracy': [],
            'memory_usage': []
        }
        
        logger.info("升级版感知模块初始化完成")
    
    def _initialize_fusion_engine(self) -> AdvancedMultimodalFusion:
        """初始化融合引擎"""
        input_dims = {
            'visual': self.config.get('visual_dim', 512),
            'audio': self.config.get('audio_dim', 128),
            'spatial': self.config.get('spatial_dim', 256),
            'temporal': self.config.get('temporal_dim', 64)
        }
        
        fusion_dim = self.config.get('fusion_dim', 1024)
        return AdvancedMultimodalFusion(input_dims, fusion_dim)
    
    def start_perception_system(self):
        """启动感知系统"""
        logger.info("启动升级版感知系统...")
        
        self.is_running = True
        
        # 启动各子系统线程
        perception_thread = threading.Thread(target=self._perception_processing_loop)
        prediction_thread = threading.Thread(target=self._prediction_loop)
        causal_thread = threading.Thread(target=self._causal_inference_loop)
        environment_thread = threading.Thread(target=self._environment_update_loop)
        
        self.processing_threads = [perception_thread, prediction_thread, causal_thread, environment_thread]
        
        for thread in self.processing_threads:
            thread.start()
        
        logger.info("升级版感知系统启动完成")
    
    def stop_perception_system(self):
        """停止感知系统"""
        logger.info("停止升级版感知系统...")
        
        self.is_running = False
        
        # 等待线程结束
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        logger.info("升级版感知系统已停止")
    
    def process_multimodal_input(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态输入"""
        start_time = time.time()
        
        # 创建多模态感知对象
        perception_data = self._create_perception_data(raw_inputs)
        
        # 特征融合
        fused_features = self._fuse_multimodal_features(perception_data)
        
        # 更新世界模型
        self.world_predictor.update_world_state(
            list(self.active_objects.values()),
            raw_inputs.get('environment', {})
        )
        
        # 构建空间表示
        if 'spatial' in raw_inputs:
            spatial_representation = self.spatial_navigator.build_spatial_representation(
                raw_inputs['spatial'], list(self.active_objects.values())
            )
        else:
            spatial_representation = {}
        
        # 因果推理
        causal_relations = self.causal_engine.discover_causal_relationships(
            list(self.perception_buffer), list(self.active_objects.values())
        )
        
        # 更新环境模型
        environment_update = self.environment_modeler.update_environment_state(
            list(self.perception_buffer), list(self.active_objects.values())
        )
        
        processing_time = time.time() - start_time
        
        # 构建输出结果
        result = {
            'timestamp': time.time(),
            'fused_features': fused_features.tolist() if hasattr(fused_features, 'tolist') else fused_features,
            'world_state': self.world_predictor.predict_future_states(),
            'spatial_representation': spatial_representation,
            'causal_relations': {k: asdict(v) for k, v in causal_relations.items()},
            'environment_state': environment_update['environment_state'],
            'processing_time': processing_time,
            'confidence_score': self._calculate_confidence_score(perception_data)
        }
        
        # 存储处理结果
        self.processed_perceptions.append(result)
        
        # 更新性能指标
        self.performance_metrics['processing_latency'].append(processing_time)
        self.performance_metrics['memory_usage'].append(self._estimate_memory_usage())
        
        return result
    
    def _create_perception_data(self, raw_inputs: Dict[str, Any]) -> List[MultimodalPerception]:
        """创建感知数据"""
        perception_data = []
        
        # 处理视觉输入
        if 'visual' in raw_inputs:
            visual_perception = MultimodalPerception(
                timestamp=time.time(),
                modality_type='visual',
                raw_data=raw_inputs['visual'],
                processed_features=np.random.rand(512),  # 实际应该使用CNN特征提取
                confidence=0.8,
                metadata={'source': 'camera'}
            )
            perception_data.append(visual_perception)
        
        # 处理音频输入
        if 'audio' in raw_inputs:
            audio_perception = MultimodalPerception(
                timestamp=time.time(),
                modality_type='audio',
                raw_data=raw_inputs['audio'],
                processed_features=np.random.rand(128),  # 实际应该使用音频特征提取
                confidence=0.7,
                metadata={'source': 'microphone'}
            )
            perception_data.append(audio_perception)
        
        # 处理空间输入
        if 'spatial' in raw_inputs:
            spatial_perception = MultimodalPerception(
                timestamp=time.time(),
                modality_type='spatial',
                raw_data=raw_inputs['spatial'],
                processed_features=np.random.rand(256),  # 实际应该使用点云特征提取
                confidence=0.9,
                metadata={'source': 'lidar'}
            )
            perception_data.append(spatial_perception)
        
        # 存储到缓冲区
        self.perception_buffer.extend(perception_data)
        
        return perception_data
    
    def _fuse_multimodal_features(self, perception_data: List[MultimodalPerception]) -> torch.Tensor:
        """融合多模态特征"""
        if not perception_data:
            return torch.zeros(1, self.config.get('fusion_dim', 1024))
        
        # 按模态组织特征
        modality_features = {}
        for perception in perception_data:
            modality = perception.modality_type
            if modality not in modality_features:
                modality_features[modality] = []
            modality_features[modality].append(perception.processed_features)
        
        # 转换为张量并送入融合网络
        with torch.no_grad():
            tensor_inputs = {}
            for modality, features in modality_features.items():
                if features:
                    # 拼接特征
                    combined_features = np.concatenate(features)
                    tensor_inputs[modality] = torch.tensor(
                        combined_features, 
                        dtype=torch.float32
                    ).unsqueeze(0)
            
            fused_output = self.fusion_engine(tensor_inputs)
            return fused_output.squeeze(0)
    
    def _calculate_confidence_score(self, perception_data: List[MultimodalPerception]) -> float:
        """计算置信度分数"""
        if not perception_data:
            return 0.0
        
        confidences = [p.confidence for p in perception_data]
        
        # 计算加权平均置信度
        weights = [p.modality_type == 'spatial' and 0.4 or 0.3 for p in perception_data]
        weighted_confidence = np.average(confidences, weights=weights)
        
        return weighted_confidence
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def _perception_processing_loop(self):
        """感知处理循环"""
        while self.is_running:
            try:
                # 处理缓冲区中的感知数据
                if len(self.perception_buffer) > 10:
                    # 每10个感知数据处理一次
                    batch_perceptions = []
                    for _ in range(min(10, len(self.perception_buffer))):
                        if self.perception_buffer:
                            batch_perceptions.append(self.perception_buffer.popleft())
                    
                    # 批量处理
                    if batch_perceptions:
                        self._process_perception_batch(batch_perceptions)
                
                time.sleep(0.1)  # 10Hz处理频率
                
            except Exception as e:
                logger.error(f"感知处理循环错误: {e}")
                time.sleep(1)
    
    def _process_perception_batch(self, perceptions: List[MultimodalPerception]):
        """批量处理感知数据"""
        # 更新活动对象
        for perception in perceptions:
            if perception.modality_type == 'visual' and 'objects' in perception.raw_data:
                for obj_data in perception.raw_data['objects']:
                    obj_id = f"obj_{obj_data.get('class', 'unknown')}_{len(self.active_objects)}"
                    
                    world_obj = WorldObject(
                        object_id=obj_id,
                        position=np.array(obj_data.get('position', [0, 0, 0])),
                        attributes=obj_data,
                        confidence=perception.confidence,
                        modality_sources=[perception.modality_type],
                        first_seen=perception.timestamp,
                        last_seen=perception.timestamp
                    )
                    
                    self.active_objects[obj_id] = world_obj
    
    def _prediction_loop(self):
        """预测循环"""
        while self.is_running:
            try:
                # 预测未来状态
                future_states = self.world_predictor.predict_future_states()
                
                # 存储预测结果
                if future_states:
                    # 实际应用中这里应该保存预测结果用于评估
                    pass
                
                time.sleep(1.0)  # 1Hz预测频率
                
            except Exception as e:
                logger.error(f"预测循环错误: {e}")
                time.sleep(1)
    
    def _causal_inference_loop(self):
        """因果推理循环"""
        while self.is_running:
            try:
                # 定期进行因果推理
                if len(self.perception_buffer) > 50:
                    causal_relations = self.causal_engine.discover_causal_relationships(
                        list(self.perception_buffer), list(self.active_objects.values())
                    )
                    
                    # 更新因果图
                    for relation_key, causal_rel in causal_relations.items():
                        self.causal_engine.causal_graph.add_edge(
                            causal_rel.cause_object,
                            causal_rel.effect_object,
                            **asdict(causal_rel)
                        )
                
                time.sleep(2.0)  # 0.5Hz推理频率
                
            except Exception as e:
                logger.error(f"因果推理循环错误: {e}")
                time.sleep(1)
    
    def _environment_update_loop(self):
        """环境更新循环"""
        while self.is_running:
            try:
                # 更新环境状态
                environment_update = self.environment_modeler.update_environment_state(
                    list(self.perception_buffer), list(self.active_objects.values())
                )
                
                # 记录环境变化
                if environment_update['detected_changes']:
                    logger.info(f"检测到环境变化: {environment_update['num_changes']} 项")
                
                time.sleep(1.0 / self.environment_modeler.update_rate)
                
            except Exception as e:
                logger.error(f"环境更新循环错误: {e}")
                time.sleep(1)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'active_objects': len(self.active_objects),
            'perception_buffer_size': len(self.perception_buffer),
            'processed_perceptions': len(self.processed_perceptions),
            'performance_metrics': {
                'avg_processing_latency': np.mean(self.performance_metrics['processing_latency'][-10:]) if self.performance_metrics['processing_latency'] else 0.0,
                'avg_memory_usage': np.mean(self.performance_metrics['memory_usage'][-10:]) if self.performance_metrics['memory_usage'] else 0.0,
                'prediction_count': len(self.world_predictor.object_states),
                'causal_relations': len(self.causal_engine.causal_graph.edges())
            },
            'system_uptime': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def get_latest_perception(self) -> Optional[Dict[str, Any]]:
        """获取最新感知结果"""
        try:
            return self.processed_perceptions[-1]
        except IndexError:
            return None
    
    def export_model_state(self) -> Dict[str, Any]:
        """导出模型状态"""
        return {
            'world_model': {
                'object_states': self.world_predictor.object_states,
                'environmental_dynamics': self.world_predictor.environmental_dynamics
            },
            'spatial_model': {
                'occupancy_grid': self.spatial_navigator.occupancy_grid.tolist() if self.spatial_navigator.occupancy_grid is not None else None,
                'landmarks': self.spatial_navigator.landmarks,
                'navigation_waypoints': self.spatial_navigator.navigation_waypoints
            },
            'causal_model': {
                'causal_graph': dict(self.causal_engine.causal_graph.edges(data=True)),
                'temporal_sequences': dict(self.causal_engine.temporal_sequences)
            },
            'environment_model': {
                'environment_state': self.environment_modeler.environment_state,
                'stability_metrics': self.environment_modeler.stability_metrics
            },
            'export_timestamp': time.time()
        }


# =========================
# 使用示例和测试代码
# =========================

def demo_advanced_perception_system():
    """升级版感知系统演示"""
    print("=== 升级版多模态感知融合和世界模型构建系统演示 ===")
    
    # 配置系统
    config = {
        'visual_dim': 512,
        'audio_dim': 128,
        'spatial_dim': 256,
        'temporal_dim': 64,
        'fusion_dim': 1024
    }
    
    # 创建系统
    perception_system = AdvancedPerceptionModule(config)
    
    try:
        # 启动系统
        perception_system.start_perception_system()
        perception_system.start_time = time.time()
        
        print("系统运行中，模拟感知数据处理...")
        
        # 模拟感知数据输入
        for i in range(30):  # 演示30秒
            time.sleep(1)
            
            # 模拟多模态输入
            raw_inputs = {
                'visual': {
                    'frame': np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
                    'objects': [
                        {
                            'class': 'person',
                            'confidence': 0.9,
                            'position': [100 + i, 200, 3.0],
                            'bbox': [90, 190, 20, 30]
                        },
                        {
                            'class': 'chair',
                            'confidence': 0.8,
                            'position': [120 + i, 180, 0.5],
                            'bbox': [110, 170, 20, 20]
                        }
                    ]
                },
                'audio': {
                    'text': f"看到有人在移动 {i}",
                    'confidence': 0.7,
                    'language': 'zh'
                },
                'spatial': {
                    'point_cloud': np.random.randn(1000, 3),
                    'confidence': 0.9
                },
                'environment': {
                    'lighting': 'bright',
                    'weather': 'clear',
                    'time_of_day': 'morning'
                }
            }
            
            # 处理输入
            result = perception_system.process_multimodal_input(raw_inputs)
            
            print(f"时刻 {i+1}:")
            print(f"  处理时间: {result['processing_time']:.3f}秒")
            print(f"  置信度: {result['confidence_score']:.3f}")
            print(f"  活跃对象数: {len(result['world_state'])}")
            print(f"  空间特征数: {len(result.get('spatial_representation', {}).get('landmarks', {}))}")
            print(f"  因果关系数: {len(result.get('causal_relations', {}))}")
            
            # 每5秒显示详细状态
            if (i + 1) % 5 == 0:
                status = perception_system.get_system_status()
                print("系统状态:", json.dumps(status, indent=2, ensure_ascii=False))
                print()
        
        # 演示导航规划
        print("=== 演示空间导航功能 ===")
        start_position = np.array([0.0, 0.0, 0.0])
        goal_position = np.array([10.0, 0.0, 5.0])
        
        navigation_path = perception_system.spatial_navigator.plan_path(
            start_position, goal_position, {'avoid_occupied': True}
        )
        
        print(f"规划路径从 {start_position} 到 {goal_position}:")
        print(f"  路径点数: {len(navigation_path.waypoints)}")
        print(f"  总成本: {navigation_path.total_cost:.2f}")
        print(f"  预计时间: {navigation_path.estimated_time:.2f}秒")
        print(f"  替代路径数: {len(navigation_path.alternative_paths)}")
        
        # 演示因果推理
        print("\n=== 演示因果推理功能 ===")
        intervention = {
            'target_object': 'obj_person_0',
            'action': 'move_away'
        }
        
        predicted_outcomes = perception_system.causal_engine.predict_intervention_outcomes(intervention)
        print(f"预测干预结果: {predicted_outcomes}")
        
        # 导出模型状态
        print("\n=== 导出模型状态 ===")
        model_state = perception_system.export_model_state()
        print(f"导出时间戳: {model_state['export_timestamp']}")
        print(f"世界模型对象数: {len(model_state['world_model']['object_states'])}")
        print(f"空间地标数: {len(model_state['spatial_model']['landmarks'])}")
        print(f"因果关系数: {len(model_state['causal_model']['causal_graph'])}")
        
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止系统...")
    
    finally:
        # 停止系统
        perception_system.stop_perception_system()
        print("升级版感知系统已停止")


if __name__ == "__main__":
    # 运行演示
    demo_advanced_perception_system()