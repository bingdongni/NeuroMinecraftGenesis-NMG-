"""
程序化世界生成器

该模块实现动态的程序化世界生成功能，包括：
1. 基于复杂度的地形生成
2. 自适应资源分布
3. 动态生态系统生成
4. 气候和事件系统
5. 与难度调节器的深度集成
6. 实时世界更新和重构
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import random
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
try:
    import noise  # 如果没有这个库，将使用简化版本
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
    # logger将在logging.basicConfig后定义

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """地形类型枚举"""
    PLAINS = "plains"           # 平原
    HILLS = "hills"             # 丘陵
    MOUNTAINS = "mountains"     # 山地
    FOREST = "forest"           # 森林
    DESERT = "desert"           # 沙漠
    OCEAN = "ocean"             # 海洋
    CAVES = "caves"             # 洞穴
    RIVER = "river"             # 河流


class ResourceType(Enum):
    """资源类型枚举"""
    FOOD = "food"               # 食物
    WATER = "water"             # 水源
    WOOD = "wood"               # 木材
    STONE = "stone"             # 石头
    METAL = "metal"             # 金属
    TOOLS = "tools"             # 工具
    SHELTER = "shelter"         # 庇护所
    MEDICINE = "medicine"       # 药材
    FUEL = "fuel"               # 燃料


class EventType(Enum):
    """事件类型枚举"""
    WEATHER_CHANGE = "weather_change"     # 天气变化
    NATURAL_DISASTER = "natural_disaster" # 自然灾害
    HOSTILE_INVASION = "hostile_invasion" # 敌对入侵
    RESOURCE_DISCOVERY = "resource_discovery" # 资源发现
    TEMPLE_APPEARANCE = "temple_appearance"   # 神庙出现
    TIME_ACCELERATION = "time_acceleration"   # 时间加速
    ZOMBIE_SIEGE = "zombie_siege"             # 僵尸围城
    BLESSING_EVENT = "blessing_event"         # 祝福事件


@dataclass
class WorldConfig:
    """世界配置"""
    # 基础参数
    world_size: Tuple[int, int] = (256, 256)  # 世界尺寸 (width, height)
    max_height: float = 64.0                  # 最大高度
    base_seed: int = 12345                    # 基础种子
    complexity_target: float = 0.3            # 目标复杂度
    
    # 地形参数
    terrain_scale: float = 0.01               # 地形缩放
    terrain_octaves: int = 4                  # 地形 octave 数
    terrain_persistence: float = 0.5          # 地形 persistence
    cave_density: float = 0.3                 # 洞穴密度
    
    # 资源参数
    resource_density: float = 1.0             # 资源密度
    resource_types: int = 5                   # 资源种类数
    resource_quality_variance: float = 0.2    # 资源质量方差
    
    # 环境参数
    base_danger_level: float = 0.1            # 基础危险等级
    change_frequency: float = 0.1             # 变化频率
    event_probability: float = 0.05           # 事件概率
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'world_size': self.world_size,
            'max_height': self.max_height,
            'base_seed': self.base_seed,
            'complexity_target': self.complexity_target,
            'terrain_scale': self.terrain_scale,
            'terrain_octaves': self.terrain_octaves,
            'terrain_persistence': self.terrain_persistence,
            'cave_density': self.cave_density,
            'resource_density': self.resource_density,
            'resource_types': self.resource_types,
            'resource_quality_variance': self.resource_quality_variance,
            'base_danger_level': self.base_danger_level,
            'change_frequency': self.change_frequency,
            'event_probability': self.event_probability
        }


@dataclass
class TerrainCell:
    """地形单元"""
    x: int
    y: int
    height: float
    terrain_type: TerrainType
    material: str
    hardness: float  # 硬度 (0-1)
    fertility: float  # 肥沃度 (0-1)
    accessibility: float  # 可访问性 (0-1)
    resources: Dict[ResourceType, float] = field(default_factory=dict)
    dangers: Dict[str, float] = field(default_factory=dict)
    structures: List[str] = field(default_factory=list)


@dataclass
class ResourceNode:
    """资源节点"""
    x: int
    y: int
    resource_type: ResourceType
    quantity: float
    quality: float
    accessibility: float
    regrowth_rate: float
    depletion_level: float = 0.0


@dataclass
class EnvironmentEvent:
    """环境事件"""
    event_id: str
    event_type: EventType
    timestamp: float
    duration: float
    intensity: float  # 强度 (0-1)
    affected_area: Tuple[int, int, int, int]  # (x, y, width, height)
    effects: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class TerrainGenerator:
    """地形生成器"""
    
    def __init__(self, config: WorldConfig):
        """初始化地形生成器"""
        self.config = config
        self.width, self.height = config.world_size
        self.terrain_cache = {}
        self.noise_cache = {}
        
    def generate_terrain(self, complexity: float) -> List[List[TerrainCell]]:
        """
        生成地形
        
        Args:
            complexity: 当前复杂度 (0-1)
            
        Returns:
            List[List[TerrainCell]]: 地形网格
        """
        try:
            # 根据复杂度调整参数
            terrain_params = self._adjust_terrain_params(complexity)
            
            # 生成高度图
            height_map = self._generate_height_map(terrain_params)
            
            # 生成地形类型
            terrain_types = self._generate_terrain_types(height_map, terrain_params)
            
            # 生成洞穴系统
            cave_system = self._generate_cave_system(complexity, terrain_params)
            
            # 组合成地形网格
            terrain_grid = self._compose_terrain_grid(
                height_map, terrain_types, cave_system, terrain_params
            )
            
            logger.debug(f"地形生成完成 - 复杂度: {complexity:.3f}")
            return terrain_grid
            
        except Exception as e:
            logger.error(f"地形生成失败: {str(e)}")
            raise
    
    def _adjust_terrain_params(self, complexity: float) -> Dict[str, Any]:
        """根据复杂度调整地形参数"""
        # 复杂度从0.3逐步提升到0.8
        complexity_factor = (complexity - 0.3) / 0.5 if complexity > 0.3 else 0.0
        
        return {
            'scale': self.config.terrain_scale * (1 + complexity_factor * 0.5),
            'octaves': int(self.config.terrain_octaves + complexity_factor * 2),
            'persistence': min(0.8, self.config.terrain_persistence + complexity_factor * 0.3),
            'height_multiplier': 1.0 + complexity_factor * 0.5,
            'cave_density': min(0.8, self.config.cave_density + complexity_factor * 0.5)
        }
    
    def _generate_height_map(self, params: Dict[str, Any]) -> np.ndarray:
        """生成高度图"""
        width, height = self.width, self.height
        
        # 尝试使用 noise 库，如果不可用则使用简化的噪声生成
        if HAS_NOISE:
            height_map = np.zeros((width, height))
            for x in range(width):
                for y in range(height):
                    # 使用 Perlin 噪声生成高度
                    value = noise.pnoise2(
                        x * params['scale'],
                        y * params['scale'],
                        octaves=params['octaves'],
                        persistence=params['persistence'],
                        repeatx=1024,
                        repeaty=1024,
                        base=self.config.base_seed
                    )
                    # 归一化到 0-1
                    height_map[x, y] = (value + 1) / 2
                    
        else:
            # 简化版噪声生成
            logger.info("使用简化版噪声生成")
            height_map = self._simple_noise_generation(width, height, params)
        
        return height_map * params['height_multiplier']
    
    def _simple_noise_generation(self, width: int, height: int, params: Dict[str, Any]) -> np.ndarray:
        """简化版噪声生成"""
        scale = params['scale']
        
        # 生成随机种子网格
        seed_grid = np.random.random((int(width * scale * 10), int(height * scale * 10)))
        
        # 缩放到目标尺寸并应用平滑
        height_map = np.zeros((width, height))
        for x in range(width):
            for y in range(height):
                # 双线性插值
                ix = int(x * scale * 10)
                iy = int(y * scale * 10)
                
                # 边界检查
                ix = min(ix, seed_grid.shape[0] - 1)
                iy = min(iy, seed_grid.shape[1] - 1)
                
                height_map[x, y] = seed_grid[ix, iy]
        
        return height_map
    
    def _generate_terrain_types(self, 
                               height_map: np.ndarray, 
                               params: Dict[str, Any]) -> np.ndarray:
        """生成地形类型"""
        width, height = height_map.shape
        terrain_types = np.zeros((width, height), dtype=object)
        
        for x in range(width):
            for y in range(height):
                height_value = height_map[x, y]
                
                # 根据高度值确定地形类型
                if height_value < 0.2:
                    terrain_type = TerrainType.OCEAN
                elif height_value < 0.4:
                    terrain_type = TerrainType.PLAINS
                elif height_value < 0.6:
                    terrain_type = TerrainType.FOREST
                elif height_value < 0.8:
                    terrain_type = TerrainType.HILLS
                else:
                    terrain_type = TerrainType.MOUNTAINS
                
                terrain_types[x, y] = terrain_type
                
        return terrain_types
    
    def _generate_cave_system(self, 
                            complexity: float, 
                            params: Dict[str, Any]) -> np.ndarray:
        """生成洞穴系统"""
        width, height = self.width, self.height
        cave_density = params['cave_density']
        
        # 洞穴分布图
        cave_map = np.zeros((width, height))
        
        # 根据复杂度决定洞穴数量
        num_caves = int(cave_density * width * height * 0.01)
        
        for _ in range(num_caves):
            # 随机洞穴中心点
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            
            # 洞穴半径
            radius = random.uniform(2, 10) * complexity
            
            # 生成洞穴区域
            for x in range(max(0, int(center_x - radius)), 
                         min(width, int(center_x + radius + 1))):
                for y in range(max(0, int(center_y - radius)), 
                             min(height, int(center_y + radius + 1))):
                    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance <= radius:
                        cave_map[x, y] = 1.0
        
        return cave_map
    
    def _compose_terrain_grid(self, 
                            height_map: np.ndarray,
                            terrain_types: np.ndarray,
                            cave_map: np.ndarray,
                            params: Dict[str, Any]) -> List[List[TerrainCell]]:
        """组合地形网格"""
        width, height = height_map.shape
        terrain_grid = []
        
        for x in range(width):
            row = []
            for y in range(height):
                height_value = height_map[x, y]
                terrain_type = terrain_types[x, y]
                cave_present = cave_map[x, y] > 0.5
                
                # 确定材质
                material = self._get_material_for_terrain(terrain_type, cave_present)
                
                # 计算属性
                hardness = self._calculate_hardness(terrain_type, cave_present, height_value)
                fertility = self._calculate_fertility(terrain_type, height_value)
                accessibility = self._calculate_accessibility(terrain_type, cave_present, hardness)
                
                # 生成资源（暂时为空，后续由资源生成器填充）
                resources = {}
                
                # 生成危险
                dangers = self._generate_cell_dangers(terrain_type, cave_present)
                
                # 洞穴覆盖的地形类型特殊处理
                if cave_present:
                    terrain_type = TerrainType.CAVES
                
                cell = TerrainCell(
                    x=x,
                    y=y,
                    height=height_value,
                    terrain_type=terrain_type,
                    material=material,
                    hardness=hardness,
                    fertility=fertility,
                    accessibility=accessibility,
                    resources=resources,
                    dangers=dangers
                )
                
                row.append(cell)
            
            terrain_grid.append(row)
        
        return terrain_grid
    
    def _get_material_for_terrain(self, terrain_type: TerrainType, cave_present: bool) -> str:
        """获取地形材质"""
        material_map = {
            TerrainType.PLAINS: "grass",
            TerrainType.HILLS: "dirt",
            TerrainType.MOUNTAINS: "stone",
            TerrainType.FOREST: "wood",
            TerrainType.DESERT: "sand",
            TerrainType.OCEAN: "water",
            TerrainType.CAVES: "stone",
            TerrainType.RIVER: "water"
        }
        
        return material_map.get(terrain_type, "dirt")
    
    def _calculate_hardness(self, 
                          terrain_type: TerrainType, 
                          cave_present: bool, 
                          height_value: float) -> float:
        """计算地形硬度"""
        base_hardness = {
            TerrainType.PLAINS: 0.2,
            TerrainType.HILLS: 0.4,
            TerrainType.MOUNTAINS: 0.8,
            TerrainType.FOREST: 0.3,
            TerrainType.DESERT: 0.1,
            TerrainType.OCEAN: 0.0,
            TerrainType.CAVES: 0.9,
            TerrainType.RIVER: 0.0
        }
        
        hardness = base_hardness.get(terrain_type, 0.5)
        
        # 高度越高通常越硬
        if not cave_present:
            hardness += height_value * 0.2
        
        return min(1.0, hardness)
    
    def _calculate_fertility(self, terrain_type: TerrainType, height_value: float) -> float:
        """计算地形肥沃度"""
        base_fertility = {
            TerrainType.PLAINS: 0.8,
            TerrainType.HILLS: 0.6,
            TerrainType.MOUNTAINS: 0.2,
            TerrainType.FOREST: 0.7,
            TerrainType.DESERT: 0.1,
            TerrainType.OCEAN: 0.5,
            TerrainType.CAVES: 0.0,
            TerrainType.RIVER: 0.6
        }
        
        fertility = base_fertility.get(terrain_type, 0.5)
        
        # 适度的高度有利于肥沃度
        if 0.2 < height_value < 0.7:
            fertility += 0.1
        
        return max(0.0, min(1.0, fertility))
    
    def _calculate_accessibility(self, 
                               terrain_type: TerrainType, 
                               cave_present: bool, 
                               hardness: float) -> float:
        """计算可访问性"""
        base_accessibility = {
            TerrainType.PLAINS: 0.9,
            TerrainType.HILLS: 0.7,
            TerrainType.MOUNTAINS: 0.3,
            TerrainType.FOREST: 0.6,
            TerrainType.DESERT: 0.8,
            TerrainType.OCEAN: 0.2,
            TerrainType.CAVES: 0.4,
            TerrainType.RIVER: 0.5
        }
        
        accessibility = base_accessibility.get(terrain_type, 0.5)
        
        # 硬度影响可访问性
        accessibility *= (1.0 - hardness * 0.5)
        
        # 洞穴系统增加复杂性但可能降低可访问性
        if cave_present:
            accessibility *= 0.7
        
        return max(0.0, min(1.0, accessibility))
    
    def _generate_cell_dangers(self, terrain_type: TerrainType, cave_present: bool) -> Dict[str, float]:
        """生成单元格危险"""
        dangers = {}
        
        # 基础危险水平
        base_danger = {
            TerrainType.PLAINS: 0.1,
            TerrainType.HILLS: 0.2,
            TerrainType.MOUNTAINS: 0.4,
            TerrainType.FOREST: 0.3,
            TerrainType.DESERT: 0.2,
            TerrainType.OCEAN: 0.6,
            TerrainType.CAVES: 0.7,
            TerrainType.RIVER: 0.3
        }
        
        base_level = base_danger.get(terrain_type, 0.2)
        dangers['base_danger'] = base_level
        
        # 洞穴特殊危险
        if cave_present:
            dangers['cave_danger'] = 0.5
            dangers['collapse_risk'] = 0.3
        
        # 地形特定危险
        if terrain_type == TerrainType.MOUNTAINS:
            dangers['fall_risk'] = 0.4
        elif terrain_type == TerrainType.OCEAN:
            dangers['drowning_risk'] = 0.8
        
        return dangers


class ResourceGenerator:
    """资源生成器"""
    
    def __init__(self, config: WorldConfig):
        """初始化资源生成器"""
        self.config = config
        self.resource_nodes = []
        
    def generate_resources(self, 
                          terrain_grid: List[List[TerrainCell]], 
                          complexity: float) -> List[ResourceNode]:
        """
        生成资源
        
        Args:
            terrain_grid: 地形网格
            complexity: 当前复杂度
            
        Returns:
            List[ResourceNode]: 资源节点列表
        """
        try:
            # 根据复杂度调整资源参数
            resource_params = self._adjust_resource_params(complexity)
            
            # 生成资源分布
            resource_distribution = self._generate_resource_distribution(
                terrain_grid, resource_params
            )
            
            # 优化资源布局
            optimized_distribution = self._optimize_resource_layout(
                resource_distribution, resource_params
            )
            
            # 创建资源节点
            resource_nodes = self._create_resource_nodes(
                optimized_distribution, terrain_grid, resource_params
            )
            
            self.resource_nodes = resource_nodes
            logger.debug(f"资源生成完成 - 复杂度: {complexity:.3f}, 节点数: {len(resource_nodes)}")
            return resource_nodes
            
        except Exception as e:
            logger.error(f"资源生成失败: {str(e)}")
            raise
    
    def _adjust_resource_params(self, complexity: float) -> Dict[str, Any]:
        """根据复杂度调整资源参数"""
        # 资源稀缺度从1.0逐步降低到0.3
        scarcity_factor = 1.0 - (complexity - 0.3) * 0.7 / 0.5 if complexity > 0.3 else 1.0
        
        return {
            'scarcity_multiplier': max(0.3, scarcity_factor),
            'quality_variance': self.config.resource_quality_variance * (1 + complexity * 0.5),
            'accessibility_reduction': complexity * 0.3,  # 复杂度越高可访问性越低
            'regrowth_reduction': complexity * 0.2,       # 复杂度越高再生速度越慢
            'clustering_factor': complexity * 0.4         # 复杂度越高聚集度越高
        }
    
    def _generate_resource_distribution(self, 
                                      terrain_grid: List[List[TerrainCell]], 
                                      params: Dict[str, Any]) -> Dict[ResourceType, np.ndarray]:
        """生成资源分布"""
        width = len(terrain_grid)
        height = len(terrain_grid[0]) if terrain_grid else 0
        
        # 为每种资源类型生成分布图
        resource_distributions = {}
        
        for resource_type in ResourceType:
            distribution = np.zeros((width, height))
            
            # 根据地形类型偏好生成资源
            for x in range(width):
                for y in range(height):
                    cell = terrain_grid[x][y]
                    
                    # 计算在该地形生成此资源的概率
                    generation_prob = self._calculate_resource_probability(
                        resource_type, cell, params
                    )
                    
                    # 根据稀缺度调整
                    adjusted_prob = generation_prob * params['scarcity_multiplier']
                    
                    # 随机决定是否生成
                    if random.random() < adjusted_prob:
                        distribution[x, y] = random.uniform(0.3, 1.0)
            
            resource_distributions[resource_type] = distribution
        
        return resource_distributions
    
    def _calculate_resource_probability(self, 
                                      resource_type: ResourceType, 
                                      cell: TerrainCell, 
                                      params: Dict[str, Any]) -> float:
        """计算资源生成概率"""
        # 地形偏好
        terrain_preferences = {
            ResourceType.FOOD: {TerrainType.PLAINS: 0.8, TerrainType.FOREST: 0.9, TerrainType.HILLS: 0.6},
            ResourceType.WATER: {TerrainType.OCEAN: 1.0, TerrainType.RIVER: 0.9, TerrainType.PLAINS: 0.3},
            ResourceType.WOOD: {TerrainType.FOREST: 0.9, TerrainType.HILLS: 0.7, TerrainType.PLAINS: 0.4},
            ResourceType.STONE: {TerrainType.MOUNTAINS: 0.9, TerrainType.CAVES: 0.8, TerrainType.HILLS: 0.5},
            ResourceType.METAL: {TerrainType.CAVES: 0.8, TerrainType.MOUNTAINS: 0.6, TerrainType.HILLS: 0.3},
            ResourceType.TOOLS: {TerrainType.PLAINS: 0.5, TerrainType.FOREST: 0.6, TerrainType.HILLS: 0.4},
            ResourceType.SHELTER: {TerrainType.FOREST: 0.8, TerrainType.HILLS: 0.7, TerrainType.PLAINS: 0.3},
            ResourceType.MEDICINE: {TerrainType.FOREST: 0.8, TerrainType.PLAINS: 0.5, TerrainType.HILLS: 0.6},
            ResourceType.FUEL: {TerrainType.FOREST: 0.7, TerrainType.PLAINS: 0.6, TerrainType.DESERT: 0.4}
        }
        
        base_prob = terrain_preferences.get(resource_type, {}).get(cell.terrain_type, 0.2)
        
        # 肥沃度和可访问性影响
        fertility_factor = cell.fertility
        accessibility_factor = cell.accessibility
        
        # 复杂度和聚集度影响
        clustering_factor = 1.0 - params['clustering_factor'] * 0.5
        
        total_prob = base_prob * fertility_factor * accessibility_factor * clustering_factor
        
        return min(1.0, total_prob)
    
    def _optimize_resource_layout(self, 
                                 resource_distribution: Dict[ResourceType, np.ndarray],
                                 params: Dict[str, Any]) -> Dict[ResourceType, np.ndarray]:
        """优化资源布局"""
        optimized_distribution = {}
        
        for resource_type, distribution in resource_distribution.items():
            # 应用聚集算法
            clustered_distribution = self._apply_clustering(distribution, params)
            
            # 应用分布均衡算法
            balanced_distribution = self._balance_distribution(clustered_distribution)
            
            optimized_distribution[resource_type] = balanced_distribution
        
        return optimized_distribution
    
    def _apply_clustering(self, 
                         distribution: np.ndarray, 
                         params: Dict[str, Any]) -> np.ndarray:
        """应用聚集算法"""
        clustering_factor = params['clustering_factor']
        
        # 简化的聚集算法：增强已有的高值区域
        enhanced = distribution.copy()
        
        # 找到高值区域
        high_value_mask = distribution > 0.5
        
        # 增强高值区域的邻域
        for x in range(distribution.shape[0]):
            for y in range(distribution.shape[1]):
                if high_value_mask[x, y]:
                    # 增强周围的区域
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < distribution.shape[0] and 
                                0 <= ny < distribution.shape[1]):
                                enhanced[nx, ny] = max(enhanced[nx, ny], 
                                                     distribution[x, y] * 0.7)
        
        return enhanced * (1 + clustering_factor * 0.3)
    
    def _balance_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """平衡资源分布"""
        # 简化版本：确保没有过度聚集
        balanced = distribution.copy()
        
        # 设置最大值限制
        max_value = 1.0
        balanced = np.minimum(balanced, max_value)
        
        # 确保有足够的零值区域（避免过度集中）
        non_zero_ratio = np.count_nonzero(balanced) / balanced.size
        if non_zero_ratio > 0.3:  # 如果非零区域超过30%
            # 随机清除一些低值区域
            zero_indices = np.where(balanced < 0.3)
            if len(zero_indices[0]) > 0:
                num_to_zero = len(zero_indices[0]) // 4
                zero_indices = list(zip(*zero_indices))
                indices_to_zero = random.sample(zero_indices, min(num_to_zero, len(zero_indices)))
                for x, y in indices_to_zero:
                    balanced[x, y] = 0.0
        
        return balanced
    
    def _create_resource_nodes(self, 
                             optimized_distribution: Dict[ResourceType, np.ndarray],
                             terrain_grid: List[List[TerrainCell]], 
                             params: Dict[str, Any]) -> List[ResourceNode]:
        """创建资源节点"""
        resource_nodes = []
        
        for resource_type, distribution in optimized_distribution.items():
            width, height = distribution.shape
            
            for x in range(width):
                for y in range(height):
                    if distribution[x, y] > 0.1:  # 只为有足够资源的格子创建节点
                        cell = terrain_grid[x][y]
                        
                        # 资源数量
                        quantity = distribution[x, y] * random.uniform(50, 200)
                        
                        # 资源质量
                        quality = self._calculate_resource_quality(cell, params)
                        
                        # 可访问性
                        accessibility = cell.accessibility * (1.0 - params['accessibility_reduction'])
                        
                        # 再生速度
                        regrowth_rate = self._calculate_regrowth_rate(cell, params)
                        
                        node = ResourceNode(
                            x=x,
                            y=y,
                            resource_type=resource_type,
                            quantity=quantity,
                            quality=quality,
                            accessibility=max(0.1, accessibility),
                            regrowth_rate=max(0.01, regrowth_rate)
                        )
                        
                        resource_nodes.append(node)
        
        return resource_nodes
    
    def _calculate_resource_quality(self, 
                                  cell: TerrainCell, 
                                  params: Dict[str, Any]) -> float:
        """计算资源质量"""
        # 基础质量
        base_quality = cell.fertility * 0.7 + cell.accessibility * 0.3
        
        # 添加随机变化
        variance = params['quality_variance']
        quality_variation = random.uniform(-variance, variance)
        
        quality = base_quality + quality_variation
        
        return max(0.1, min(1.0, quality))
    
    def _calculate_regrowth_rate(self, 
                               cell: TerrainCell, 
                               params: Dict[str, Any]) -> float:
        """计算再生速度"""
        # 基础再生速度基于肥沃度
        base_rate = cell.fertility * 0.1
        
        # 复杂度影响
        reduction = params['regrowth_reduction']
        
        regrowth_rate = base_rate * (1.0 - reduction)
        
        return max(0.01, regrowth_rate)


class EventSystem:
    """事件系统"""
    
    def __init__(self, config: WorldConfig):
        """初始化事件系统"""
        self.config = config
        self.active_events = []
        self.event_history = deque(maxlen=100)
        self.last_event_time = 0
        
    def generate_events(self, 
                       current_time: float,
                       world_state: Dict[str, Any],
                       complexity: float) -> List[EnvironmentEvent]:
        """
        生成环境事件
        
        Args:
            current_time: 当前时间
            world_state: 世界状态
            complexity: 当前复杂度
            
        Returns:
            List[EnvironmentEvent]: 事件列表
        """
        try:
            events = []
            
            # 清理过期事件
            self._cleanup_expired_events(current_time)
            
            # 基于概率生成新事件
            event_probability = self.config.event_probability * (1 + complexity * 0.5)
            
            if random.random() < event_probability:
                new_event = self._create_random_event(current_time, world_state, complexity)
                if new_event:
                    events.append(new_event)
                    self.active_events.append(new_event)
            
            # 根据世界状态生成特定事件
            specific_events = self._generate_specific_events(current_time, world_state, complexity)
            events.extend(specific_events)
            
            # 更新事件状态
            for event in events:
                self._apply_event_effects(event, world_state)
            
            logger.debug(f"生成 {len(events)} 个事件 - 复杂度: {complexity:.3f}")
            return events
            
        except Exception as e:
            logger.error(f"事件生成失败: {str(e)}")
            return []
    
    def _create_random_event(self, 
                           current_time: float, 
                           world_state: Dict[str, Any], 
                           complexity: float) -> Optional[EnvironmentEvent]:
        """创建随机事件"""
        event_types = list(EventType)
        
        # 根据复杂度调整事件类型权重
        if complexity < 0.4:
            # 简单环境：主要是天气和资源事件
            available_types = [EventType.WEATHER_CHANGE, EventType.RESOURCE_DISCOVERY, 
                             EventType.BLESSING_EVENT]
        elif complexity < 0.7:
            # 中等环境：添加中等危险事件
            available_types = event_types[:-1]  # 除了极端事件
        else:
            # 复杂环境：包含所有事件
            available_types = event_types
        
        event_type = random.choice(available_types)
        
        # 创建事件
        event_id = f"event_{int(current_time)}_{random.randint(1000, 9999)}"
        
        # 影响区域
        world_width, world_height = world_state.get('size', (256, 256))
        area_x = random.randint(0, max(0, world_width // 4))
        area_y = random.randint(0, max(0, world_height // 4))
        area_width = random.randint(world_width // 4, world_width // 2)
        area_height = random.randint(world_height // 4, world_height // 2)
        
        affected_area = (area_x, area_y, area_width, area_height)
        
        # 事件参数
        duration = self._calculate_event_duration(event_type, complexity)
        intensity = random.uniform(0.3, 0.9) * (0.5 + complexity * 0.5)  # 复杂度影响强度
        
        event = EnvironmentEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=current_time,
            duration=duration,
            intensity=intensity,
            affected_area=affected_area
        )
        
        # 生成事件效果
        event.effects = self._generate_event_effects(event_type, intensity)
        
        return event
    
    def _calculate_event_duration(self, event_type: EventType, complexity: float) -> float:
        """计算事件持续时间"""
        base_durations = {
            EventType.WEATHER_CHANGE: 300.0,       # 5分钟
            EventType.NATURAL_DISASTER: 600.0,     # 10分钟
            EventType.HOSTILE_INVASION: 1200.0,    # 20分钟
            EventType.RESOURCE_DISCOVERY: 1800.0,  # 30分钟
            EventType.TEMPLE_APPEARANCE: 2400.0,   # 40分钟
            EventType.TIME_ACCELERATION: 180.0,    # 3分钟
            EventType.ZOMBIE_SIEGE: 1800.0,        # 30分钟
            EventType.BLESSING_EVENT: 900.0        # 15分钟
        }
        
        base_duration = base_durations.get(event_type, 600.0)
        
        # 复杂度影响持续时间
        if event_type in [EventType.ZOMBIE_SIEGE, EventType.HOSTILE_INVASION, EventType.NATURAL_DISASTER]:
            # 复杂事件持续时间随复杂度增加
            duration = base_duration * (0.8 + complexity * 0.4)
        else:
            # 其他事件受复杂度影响较小
            duration = base_duration * (0.9 + complexity * 0.2)
        
        return duration
    
    def _generate_event_effects(self, event_type: EventType, intensity: float) -> Dict[str, Any]:
        """生成事件效果"""
        effects = {}
        
        if event_type == EventType.WEATHER_CHANGE:
            effects = {
                'visibility_reduction': intensity * 0.5,
                'movement_slowdown': intensity * 0.3,
                'temperature_change': random.uniform(-10, 10) * intensity
            }
        
        elif event_type == EventType.NATURAL_DISASTER:
            disaster_type = random.choice(['earthquake', 'flood', 'storm', 'drought'])
            effects = {
                'disaster_type': disaster_type,
                'terrain_modification': intensity * 0.7,
                'resource_damage': intensity * 0.8,
                'danger_increase': intensity * 0.9
            }
        
        elif event_type == EventType.HOSTILE_INVASION:
            effects = {
                'hostile_density_increase': intensity * 1.5,
                'new_hostile_types': random.randint(1, 3),
                'territory_expansion': intensity * 0.6,
                'defensive_challenge': intensity * 0.8
            }
        
        elif event_type == EventType.RESOURCE_DISCOVERY:
            resource_type = random.choice(list(ResourceType))
            effects = {
                'new_resource_type': resource_type,
                'resource_abundance': intensity * 1.2,
                'quality_boost': intensity * 0.4,
                'accessibility_improvement': intensity * 0.3
            }
        
        elif event_type == EventType.TEMPLE_APPEARANCE:
            effects = {
                'special_structure': 'ancient_temple',
                'mysterious_power': intensity * 0.8,
                'knowledge_boost': intensity * 0.6,
                'challenge_enhancement': intensity * 0.5
            }
        
        elif event_type == EventType.TIME_ACCELERATION:
            effects = {
                'time_multiplier': 1.0 + intensity * 2.0,
                'resource_regrowth_boost': intensity * 2.0,
                'event_acceleration': intensity * 1.5,
                'learning_rate_boost': intensity * 0.4
            }
        
        elif event_type == EventType.ZOMBIE_SIEGE:
            effects = {
                'siege_duration': intensity * 1800.0,
                'siege_intensity': intensity,
                'fortification_requirement': intensity * 0.8,
                'survival_challenge': intensity * 0.9,
                'group_cooperation_needed': True
            }
        
        elif event_type == EventType.BLESSING_EVENT:
            effects = {
                'resource_abundance': intensity * 1.3,
                'danger_reduction': intensity * 0.4,
                'health_restoration': intensity * 0.6,
                'positive_mood_boost': intensity * 0.5
            }
        
        return effects
    
    def _generate_specific_events(self, 
                                current_time: float,
                                world_state: Dict[str, Any], 
                                complexity: float) -> List[EnvironmentEvent]:
        """生成特定事件"""
        events = []
        
        # 根据世界状态生成事件
        agent_count = len(world_state.get('agents', {}))
        resource_scarcity = world_state.get('resource_scarcity', 0.5)
        danger_level = world_state.get('danger_level', 0.3)
        
        # 资源极度稀缺时可能触发资源发现事件
        if resource_scarcity > 0.8 and random.random() < 0.3:
            event = self._create_specific_event(
                EventType.RESOURCE_DISCOVERY, current_time, 
                {'reason': 'extreme_scarcity'}, complexity
            )
            if event:
                events.append(event)
        
        # 危险等级过高时可能触发祝福事件
        if danger_level > 0.8 and random.random() < 0.2:
            event = self._create_specific_event(
                EventType.BLESSING_EVENT, current_time,
                {'reason': 'high_danger'}, complexity
            )
            if event:
                events.append(event)
        
        # 高复杂度环境定期触发僵尸围城
        if complexity > 0.7 and random.random() < complexity * 0.1:
            event = self._create_specific_event(
                EventType.ZOMBIE_SIEGE, current_time,
                {'reason': 'high_complexity'}, complexity
            )
            if event:
                events.append(event)
        
        return events
    
    def _create_specific_event(self, 
                             event_type: EventType,
                             current_time: float,
                             context: Dict[str, Any],
                             complexity: float) -> Optional[EnvironmentEvent]:
        """创建特定事件"""
        event_id = f"specific_{event_type.value}_{int(current_time)}_{random.randint(100, 999)}"
        
        # 使用上下文调整事件参数
        if event_type == EventType.RESOURCE_DISCOVERY:
            duration = 1800.0  # 30分钟
            intensity = 0.8
            affected_area = (0, 0, 256, 256)  # 全局
        elif event_type == EventType.BLESSING_EVENT:
            duration = 900.0   # 15分钟
            intensity = 0.6
            affected_area = (0, 0, 256, 256)  # 全局
        elif event_type == EventType.ZOMBIE_SIEGE:
            duration = 1800.0  # 30分钟
            intensity = 0.9
            affected_area = (50, 50, 156, 156)  # 中心区域
        else:
            return None
        
        event = EnvironmentEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=current_time,
            duration=duration,
            intensity=intensity,
            affected_area=affected_area,
            effects=self._generate_event_effects(event_type, intensity)
        )
        
        return event
    
    def _apply_event_effects(self, event: EnvironmentEvent, world_state: Dict[str, Any]):
        """应用事件效果到世界状态"""
        # 这里应该将事件效果实际应用到世界状态
        # 简化实现：记录效果
        if 'event_effects' not in world_state:
            world_state['event_effects'] = []
        
        world_state['event_effects'].append({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'effects': event.effects,
            'timestamp': event.timestamp
        })
    
    def _cleanup_expired_events(self, current_time: float):
        """清理过期事件"""
        expired_events = []
        
        for event in self.active_events:
            if current_time - event.timestamp > event.duration:
                expired_events.append(event)
                event.resolved = True
                self.event_history.append(event)
        
        # 移除过期事件
        for event in expired_events:
            if event in self.active_events:
                self.active_events.remove(event)


class DynamicWorldGenerator:
    """动态世界生成器"""
    
    def __init__(self, config: WorldConfig):
        """初始化动态世界生成器"""
        self.config = config
        self.terrain_generator = TerrainGenerator(config)
        self.resource_generator = ResourceGenerator(config)
        self.event_system = EventSystem(config)
        
        # 世界状态缓存
        self.current_world_state = {}
        self.world_history = deque(maxlen=50)
        
        # 性能统计
        self.generation_stats = {
            'total_generations': 0,
            'avg_generation_time': 0.0,
            'last_generation': None
        }
        
        logger.info("动态世界生成器初始化完成")
    
    def generate_world(self, 
                      complexity: float, 
                      target_metrics: Dict[str, float] = None,
                      preserve_progress: bool = True) -> Dict[str, Any]:
        """
        生成世界
        
        Args:
            complexity: 当前复杂度 (0-1)
            target_metrics: 目标指标
            preserve_progress: 是否保留现有进度
            
        Returns:
            Dict[str, Any]: 世界状态
        """
        start_time = time.time()
        
        try:
            # 保存当前世界状态用于渐进式更新
            previous_world = self.current_world_state.copy() if preserve_progress else {}
            
            logger.info(f"开始生成世界 - 复杂度: {complexity:.3f}")
            
            # 1. 生成地形
            logger.info("生成地形...")
            terrain_grid = self.terrain_generator.generate_terrain(complexity)
            
            # 2. 生成资源
            logger.info("生成资源...")
            resource_nodes = self.resource_generator.generate_resources(terrain_grid, complexity)
            
            # 3. 生成事件
            logger.info("生成环境事件...")
            current_time = time.time()
            events = self.event_system.generate_events(
                current_time, self.current_world_state, complexity
            )
            
            # 4. 组合世界状态
            world_state = self._compose_world_state(
                terrain_grid, resource_nodes, events, complexity
            )
            
            # 5. 应用渐进式更新（如果需要）
            if preserve_progress and previous_world:
                world_state = self._apply_progressive_updates(previous_world, world_state)
            
            # 6. 验证世界状态
            validation_result = self._validate_world_state(world_state)
            if not validation_result['valid']:
                logger.warning(f"世界状态验证失败: {validation_result['issues']}")
                world_state = self._fix_world_state_issues(world_state, validation_result)
            
            # 更新当前世界状态
            self.current_world_state = world_state
            
            # 记录生成历史
            generation_record = {
                'timestamp': start_time,
                'complexity': complexity,
                'generation_time': time.time() - start_time,
                'world_size': (len(world_state.get('terrain', [])), 
                             len(world_state.get('terrain', [[]])[0]) if world_state.get('terrain') else 0),
                'resource_count': len(world_state.get('resource_nodes', [])),
                'event_count': len(world_state.get('active_events', [])),
                'validation_result': validation_result
            }
            
            self.world_history.append(generation_record)
            
            # 更新统计信息
            self._update_generation_stats(time.time() - start_time)
            
            logger.info(f"世界生成完成 - 耗时: {generation_record['generation_time']:.2f}秒")
            return world_state
            
        except Exception as e:
            logger.error(f"世界生成失败: {str(e)}")
            raise
    
    def _compose_world_state(self, 
                           terrain_grid: List[List[TerrainCell]],
                           resource_nodes: List[ResourceNode],
                           events: List[EnvironmentEvent],
                           complexity: float) -> Dict[str, Any]:
        """组合世界状态"""
        world_state = {
            'terrain': terrain_grid,
            'resource_nodes': resource_nodes,
            'active_events': events,
            'complexity': complexity,
            'generation_timestamp': time.time(),
            'size': self.config.world_size,
            'max_height': self.config.max_height
        }
        
        # 添加统计信息
        world_state['statistics'] = self._calculate_world_statistics(terrain_grid, resource_nodes, events)
        
        # 添加环境指标
        world_state['metrics'] = self._calculate_environment_metrics(terrain_grid, resource_nodes, events)
        
        return world_state
    
    def _calculate_world_statistics(self, 
                                  terrain_grid: List[List[TerrainCell]],
                                  resource_nodes: List[ResourceNode],
                                  events: List[EnvironmentEvent]) -> Dict[str, Any]:
        """计算世界统计信息"""
        if not terrain_grid:
            return {}
        
        width, height = len(terrain_grid), len(terrain_grid[0])
        total_cells = width * height
        
        # 地形统计
        terrain_stats = defaultdict(int)
        avg_height = 0.0
        avg_hardness = 0.0
        avg_fertility = 0.0
        avg_accessibility = 0.0
        
        for row in terrain_grid:
            for cell in row:
                terrain_stats[cell.terrain_type.value] += 1
                avg_height += cell.height
                avg_hardness += cell.hardness
                avg_fertility += cell.fertility
                avg_accessibility += cell.accessibility
        
        avg_height /= total_cells
        avg_hardness /= total_cells
        avg_fertility /= total_cells
        avg_accessibility /= total_cells
        
        # 资源统计
        resource_stats = defaultdict(int)
        total_resource_value = 0.0
        
        for node in resource_nodes:
            resource_stats[node.resource_type.value] += 1
            total_resource_value += node.quantity * node.quality
        
        # 事件统计
        event_stats = defaultdict(int)
        for event in events:
            event_stats[event.event_type.value] += 1
        
        return {
            'terrain': {
                'total_cells': total_cells,
                'terrain_distribution': dict(terrain_stats),
                'average_height': avg_height,
                'average_hardness': avg_hardness,
                'average_fertility': avg_fertility,
                'average_accessibility': avg_accessibility
            },
            'resources': {
                'total_nodes': len(resource_nodes),
                'resource_distribution': dict(resource_stats),
                'total_resource_value': total_resource_value,
                'average_node_value': total_resource_value / len(resource_nodes) if resource_nodes else 0.0
            },
            'events': {
                'active_count': len(events),
                'event_distribution': dict(event_stats)
            }
        }
    
    def _calculate_environment_metrics(self, 
                                     terrain_grid: List[List[TerrainCell]],
                                     resource_nodes: List[ResourceNode],
                                     events: List[EnvironmentEvent]) -> Dict[str, float]:
        """计算环境指标"""
        if not terrain_grid:
            return {}
        
        # 地形复杂度
        terrain_complexity = self._calculate_terrain_complexity(terrain_grid)
        
        # 资源稀缺度
        resource_scarcity = self._calculate_resource_scarcity(terrain_grid, resource_nodes)
        
        # 危险系数
        danger_level = self._calculate_danger_level(terrain_grid, events)
        
        # 可访问性
        accessibility = self._calculate_average_accessibility(terrain_grid)
        
        # 稳定性
        temporal_stability = 1.0 - len(events) * 0.1  # 事件越多稳定性越低
        
        return {
            'terrain_complexity': terrain_complexity,
            'resource_scarcity': resource_scarcity,
            'danger_level': danger_level,
            'accessibility': accessibility,
            'temporal_stability': max(0.0, temporal_stability)
        }
    
    def _calculate_terrain_complexity(self, terrain_grid: List[List[TerrainCell]]) -> float:
        """计算地形复杂度"""
        if not terrain_grid:
            return 0.0
        
        width, height = len(terrain_grid), len(terrain_grid[0])
        
        # 高度变化
        height_variance = 0.0
        heights = [cell.height for row in terrain_grid for cell in row]
        height_variance = np.var(heights)
        
        # 地形类型多样性
        terrain_types = set(cell.terrain_type for row in terrain_grid for cell in row)
        type_diversity = len(terrain_types) / len(list(TerrainType))
        
        # 可访问性变化
        accessibility_values = [cell.accessibility for row in terrain_grid for cell in row]
        accessibility_variance = np.var(accessibility_values)
        
        complexity = (height_variance * 0.4 + 
                     type_diversity * 0.3 + 
                     accessibility_variance * 0.3)
        
        return min(1.0, complexity)
    
    def _calculate_resource_scarcity(self, 
                                   terrain_grid: List[List[TerrainCell]], 
                                   resource_nodes: List[ResourceNode]) -> float:
        """计算资源稀缺度"""
        if not terrain_grid:
            return 1.0
        
        width, height = len(terrain_grid), len(terrain_grid[0])
        total_cells = width * height
        
        # 基于资源节点密度的稀缺度
        resource_density = len(resource_nodes) / total_cells
        
        # 转换为稀缺度（密度越低，稀缺度越高）
        scarcity = 1.0 - min(1.0, resource_density * 10)
        
        return max(0.0, min(1.0, scarcity))
    
    def _calculate_danger_level(self, 
                              terrain_grid: List[List[TerrainCell]], 
                              events: List[EnvironmentEvent]) -> float:
        """计算危险系数"""
        if not terrain_grid:
            return 0.0
        
        # 地形基础危险
        total_danger = 0.0
        cell_count = 0
        
        for row in terrain_grid:
            for cell in row:
                if cell.dangers:
                    cell_danger = max(cell.dangers.values())
                    total_danger += cell_danger
                    cell_count += 1
        
        avg_terrain_danger = total_danger / cell_count if cell_count > 0 else 0.0
        
        # 事件增强的危险
        event_danger_boost = 0.0
        for event in events:
            if event.event_type in [EventType.HOSTILE_INVASION, EventType.ZOMBIE_SIEGE, 
                                  EventType.NATURAL_DISASTER]:
                event_danger_boost += event.intensity * 0.3
        
        total_danger = avg_terrain_danger + event_danger_boost
        
        return min(1.0, total_danger)
    
    def _calculate_average_accessibility(self, terrain_grid: List[List[TerrainCell]]) -> float:
        """计算平均可访问性"""
        if not terrain_grid:
            return 0.0
        
        total_accessibility = 0.0
        cell_count = 0
        
        for row in terrain_grid:
            for cell in row:
                total_accessibility += cell.accessibility
                cell_count += 1
        
        return total_accessibility / cell_count if cell_count > 0 else 0.0
    
    def _apply_progressive_updates(self, 
                                 previous_world: Dict[str, Any], 
                                 new_world: Dict[str, Any]) -> Dict[str, Any]:
        """应用渐进式更新"""
        # 简化实现：保留一些原有资源，添加新资源
        previous_resources = previous_world.get('resource_nodes', [])
        new_resources = new_world.get('resource_nodes', [])
        
        # 保留30%的旧资源
        preserved_count = int(len(previous_resources) * 0.3)
        if preserved_count > 0:
            preserved_resources = random.sample(previous_resources, 
                                              min(preserved_count, len(previous_resources)))
            new_resources.extend(preserved_resources)
        
        new_world['resource_nodes'] = new_resources
        
        return new_world
    
    def _validate_world_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """验证世界状态"""
        issues = []
        
        # 检查基本结构
        if 'terrain' not in world_state:
            issues.append('missing_terrain')
        
        if 'resource_nodes' not in world_state:
            issues.append('missing_resources')
        
        # 检查资源数量
        resource_count = len(world_state.get('resource_nodes', []))
        if resource_count == 0:
            issues.append('no_resources')
        elif resource_count > world_state.get('size', [256, 256])[0] * world_state.get('size', [256, 256])[1] * 0.5:
            issues.append('too_many_resources')
        
        # 检查地形合理性
        terrain = world_state.get('terrain', [])
        if terrain:
            avg_fertility = np.mean([cell.fertility for row in terrain for cell in row])
            if avg_fertility < 0.1:
                issues.append('low_fertility')
            
            avg_accessibility = np.mean([cell.accessibility for row in terrain for cell in row])
            if avg_accessibility < 0.2:
                issues.append('low_accessibility')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'severity': 'high' if any(issue in ['no_resources', 'missing_terrain'] for issue in issues) else 'medium'
        }
    
    def _fix_world_state_issues(self, 
                              world_state: Dict[str, Any], 
                              validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """修复世界状态问题"""
        issues = validation_result['issues']
        
        # 修复资源问题
        if 'no_resources' in issues:
            # 紧急添加一些基础资源
            terrain = world_state.get('terrain', [])
            if terrain:
                emergency_resources = []
                for i in range(min(20, len(terrain) * len(terrain[0]) // 10)):
                    x = random.randint(0, len(terrain) - 1)
                    y = random.randint(0, len(terrain[0]) - 1)
                    emergency_resources.append(ResourceNode(
                        x=x, y=y,
                        resource_type=ResourceType.FOOD,
                        quantity=100.0,
                        quality=0.5,
                        accessibility=0.5,
                        regrowth_rate=0.1
                    ))
                
                world_state['resource_nodes'].extend(emergency_resources)
        
        # 修复肥沃度问题
        if 'low_fertility' in issues:
            terrain = world_state.get('terrain', [])
            for row in terrain:
                for cell in row:
                    if cell.fertility < 0.1:
                        cell.fertility = 0.3
        
        # 修复可访问性问题
        if 'low_accessibility' in issues:
            terrain = world_state.get('terrain', [])
            for row in terrain:
                for cell in row:
                    if cell.accessibility < 0.2:
                        cell.accessibility = 0.4
        
        return world_state
    
    def _update_generation_stats(self, generation_time: float):
        """更新生成统计"""
        self.generation_stats['total_generations'] += 1
        current_avg = self.generation_stats['avg_generation_time']
        total_count = self.generation_stats['total_generations']
        
        self.generation_stats['avg_generation_time'] = \
            (current_avg * (total_count - 1) + generation_time) / total_count
        
        self.generation_stats['last_generation'] = time.time()
    
    def get_world_statistics(self) -> Dict[str, Any]:
        """获取世界统计信息"""
        return {
            'generation_stats': self.generation_stats,
            'history_size': len(self.world_history),
            'active_events': len(self.event_system.active_events),
            'total_resource_nodes': len(self.resource_generator.resource_nodes),
            'current_complexity': self.current_world_state.get('complexity', 0.0),
            'world_size': self.config.world_size
        }
    
    def export_world_data(self, filepath: str):
        """导出世界数据"""
        export_data = {
            'export_timestamp': time.time(),
            'config': self.config.to_dict(),
            'current_world_state': {
                'complexity': self.current_world_state.get('complexity'),
                'size': self.current_world_state.get('size'),
                'resource_count': len(self.current_world_state.get('resource_nodes', [])),
                'event_count': len(self.current_world_state.get('active_events', []))
            },
            'statistics': self.get_world_statistics(),
            'generation_history': list(self.world_history),
            'event_history': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'duration': event.duration,
                    'intensity': event.intensity,
                    'resolved': event.resolved
                }
                for event in self.event_system.event_history
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"世界数据已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出世界数据失败: {str(e)}")
            raise


# 工厂函数
def create_world_generator(config: Dict) -> DynamicWorldGenerator:
    """
    创建动态世界生成器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        DynamicWorldGenerator: 世界生成器实例
    """
    world_config = WorldConfig(
        world_size=tuple(config.get('world_size', [256, 256])),
        max_height=config.get('max_height', 64.0),
        base_seed=config.get('base_seed', 12345),
        complexity_target=config.get('complexity_target', 0.3),
        terrain_scale=config.get('terrain_scale', 0.01),
        terrain_octaves=config.get('terrain_octaves', 4),
        terrain_persistence=config.get('terrain_persistence', 0.5),
        cave_density=config.get('cave_density', 0.3),
        resource_density=config.get('resource_density', 1.0),
        resource_types=config.get('resource_types', 5),
        resource_quality_variance=config.get('resource_quality_variance', 0.2),
        base_danger_level=config.get('base_danger_level', 0.1),
        change_frequency=config.get('change_frequency', 0.1),
        event_probability=config.get('event_probability', 0.05)
    )
    
    return DynamicWorldGenerator(world_config)


if __name__ == "__main__":
    # 演示用法
    logger.info("动态世界生成器演示")
    
    # 创建配置
    config = {
        'world_size': [128, 128],
        'complexity_target': 0.5,
        'cave_density': 0.4,
        'resource_density': 0.8,
        'base_seed': 42
    }
    
    # 创建世界生成器
    generator = create_world_generator(config)
    
    # 生成不同复杂度的世界
    complexities = [0.3, 0.5, 0.7]
    
    for complexity in complexities:
        logger.info(f"生成复杂度为 {complexity} 的世界")
        world = generator.generate_world(complexity)
        
        # 显示世界统计信息
        stats = generator.get_world_statistics()
        print(f"复杂度 {complexity} 的世界统计:")
        print(f"  - 资源节点数: {stats['current_world_state']['resource_count']}")
        print(f"  - 活跃事件数: {stats['current_world_state']['event_count']}")
        print(f"  - 生成时间: {stats['generation_stats']['avg_generation_time']:.2f}秒")
        
        # 显示环境指标
        metrics = world.get('metrics', {})
        print(f"  - 地形复杂度: {metrics.get('terrain_complexity', 0):.3f}")
        print(f"  - 资源稀缺度: {metrics.get('resource_scarcity', 0):.3f}")
        print(f"  - 危险系数: {metrics.get('danger_level', 0):.3f}")
        print()
    
    # 获取总体统计
    total_stats = generator.get_world_statistics()
    print("总体统计:")
    print(json.dumps(total_stats, indent=2, ensure_ascii=False))