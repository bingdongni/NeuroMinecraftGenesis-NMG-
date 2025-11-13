# -*- coding: utf-8 -*-
"""
网络数据处理器
负责处理Nengo导出的网络数据，包括数据解析、格式转换和实时更新
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import gzip
import pickle
from pathlib import Path


@dataclass
class DataFormat:
    """数据格式配置"""
    position_format: str = "xyz"  # 位置坐标格式
    activity_range: Tuple[float, float] = (0.0, 1.0)  # 活动值范围
    connection_weight_range: Tuple[float, float] = (-1.0, 1.0)  # 连接权重范围
    time_precision: int = 6  # 时间精度（小数位数）
    compression_enabled: bool = True


@dataclass
class ProcessingConfig:
    """数据处理配置"""
    batch_size: int = 1000  # 批处理大小
    cache_size: int = 10000  # 缓存大小
    update_interval: float = 0.016  # 更新间隔（秒）
    max_history_length: int = 3600  # 最大历史记录长度（秒）
    spatial_layout_algorithm: str = "force_directed"  # 空间布局算法
    data_validation: bool = True  # 数据验证
    real_time_processing: bool = True  # 实时处理


class NetworkDataHandler:
    """
    网络数据处理器类
    
    负责处理Nengo导出的网络数据，包括解析、转换、验证和实时更新
    """
    
    def __init__(self, config: DataFormat = None, processing_config: ProcessingConfig = None):
        """
        初始化网络数据处理器
        
        Args:
            config: 数据格式配置
            processing_config: 处理配置
        """
        self.format_config = config or DataFormat()
        self.processing_config = processing_config or ProcessingConfig()
        
        # 数据存储
        self.neurons = {}  # neuron_id -> neuron_data
        self.connections = {}  # connection_id -> connection_data
        self.temporal_data = deque(maxlen=int(self.processing_config.max_history_length / self.processing_config.update_interval))
        self.activity_history = defaultdict(lambda: deque(maxlen=1000))
        
        # 数据转换和缓存
        self.position_cache = {}  # position_key -> position_data
        self.layout_cache = {}  # layout_key -> layout_data
        self.data_cache = {}  # data_key -> processed_data
        
        # 数据验证
        self.data_validators = {
            'position': self._validate_position,
            'activity': self._validate_activity,
            'connection': self._validate_connection,
            'spike': self._validate_spike
        }
        
        # 空间布局
        self.spatial_layouts = {
            'force_directed': self._force_directed_layout,
            'circular': self._circular_layout,
            'hierarchical': self._hierarchical_layout,
            'clustered': self._clustered_layout
        }
        
        # 统计信息
        self.data_stats = {
            'total_neurons': 0,
            'total_connections': 0,
            'data_points_processed': 0,
            'average_processing_time': 0.0,
            'last_update_time': time.time(),
            'cache_hit_rate': 0.0,
            'validation_errors': 0
        }
        
        self.logger = self._setup_logging()
        self.logger.info("网络数据处理器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('NetworkDataHandler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def initialize_network(self):
        """
        初始化网络数据结构
        
        Returns:
            Dict[str, Any]: 初始化数据
        """
        try:
            init_data = {
                'network_info': {
                    'format_version': '1.0',
                    'data_format': asdict(self.format_config),
                    'processing_config': asdict(self.processing_config),
                    'supported_layouts': list(self.spatial_layouts.keys())
                },
                'initial_state': {
                    'neurons': {},
                    'connections': {},
                    'temporal_data': []
                },
                'validation_rules': {
                    'position_range': [-100, 100],
                    'activity_range': self.format_config.activity_range,
                    'connection_weight_range': self.format_config.connection_weight_range,
                    'required_fields': ['id', 'position', 'activity']
                }
            }
            
            self.logger.info("网络数据处理器初始化完成")
            return init_data
            
        except Exception as e:
            self.logger.error(f"网络初始化失败: {e}")
            raise
    
    async def parse_nengo_data(self, nengo_data: Union[str, Dict, bytes]) -> Dict[str, Any]:
        """
        解析Nengo导出的网络数据
        
        Args:
            nengo_data: Nengo数据（JSON字符串、字典或压缩字节）
            
        Returns:
            Dict[str, Any]: 解析后的网络数据
        """
        try:
            start_time = time.time()
            
            # 解析输入数据
            if isinstance(nengo_data, bytes):
                # 解压缩数据
                if self.format_config.compression_enabled:
                    nengo_data = gzip.decompress(nengo_data).decode('utf-8')
                
                parsed_data = json.loads(nengo_data)
            elif isinstance(nengo_data, str):
                parsed_data = json.loads(nengo_data)
            elif isinstance(nengo_data, dict):
                parsed_data = nengo_data.copy()
            else:
                raise ValueError(f"不支持的数据类型: {type(nengo_data)}")
            
            # 数据验证
            if self.processing_config.data_validation:
                validation_result = await self._validate_network_data(parsed_data)
                if not validation_result['valid']:
                    self.logger.warning(f"数据验证失败: {validation_result['errors']}")
                    self.data_stats['validation_errors'] += 1
                    return self._handle_validation_errors(parsed_data, validation_result)
            
            # 转换数据格式
            converted_data = await self._convert_data_format(parsed_data)
            
            # 处理神经元数据
            neurons_data = await self._process_neurons(converted_data.get('neurons', []))
            
            # 处理连接数据
            connections_data = await self._process_connections(converted_data.get('connections', []))
            
            # 处理时间序列数据
            temporal_data = await self._process_temporal_data(converted_data.get('temporal', []))
            
            # 生成处理结果
            processed_data = {
                'neurons': neurons_data,
                'connections': connections_data,
                'temporal': temporal_data,
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'data_version': converted_data.get('version', 'unknown'),
                    'timestamp': time.time(),
                    'format': 'processed'
                }
            }
            
            # 缓存处理结果
            cache_key = f"network_data_{hash(str(converted_data))}"
            self.data_cache[cache_key] = processed_data
            
            # 更新统计信息
            self.data_stats['data_points_processed'] += 1
            self.data_stats['total_neurons'] = len(neurons_data)
            self.data_stats['total_connections'] = len(connections_data)
            
            self.logger.info(f"Nengo数据解析完成，处理了{len(neurons_data)}个神经元和{len(connections_data)}个连接")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"解析Nengo数据失败: {e}")
            raise
    
    async def _validate_network_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证网络数据"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # 验证基本结构
            if 'neurons' not in data:
                validation_result['errors'].append("缺少neurons字段")
                validation_result['valid'] = False
            
            if 'connections' not in data:
                validation_result['errors'].append("缺少connections字段")
                validation_result['valid'] = False
            
            # 验证神经元数据
            if 'neurons' in data:
                for i, neuron in enumerate(data['neurons']):
                    for field in self.data_stats.get('required_fields', []):
                        if field not in neuron:
                            validation_result['errors'].append(f"神经元{i}缺少{field}字段")
                            validation_result['valid'] = False
                    
                    # 验证位置数据
                    if 'position' in neuron:
                        pos = neuron['position']
                        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                            validation_result['errors'].append(f"神经元{i}位置数据格式错误")
                            validation_result['valid'] = False
                        elif not all(isinstance(coord, (int, float)) for coord in pos):
                            validation_result['errors'].append(f"神经元{i}位置坐标类型错误")
                            validation_result['valid'] = False
                    
                    # 验证活动值
                    if 'activity' in neuron:
                        activity = neuron['activity']
                        if not isinstance(activity, (int, float)):
                            validation_result['errors'].append(f"神经元{i}活动值类型错误")
                            validation_result['valid'] = False
                        elif not (self.format_config.activity_range[0] <= activity <= self.format_config.activity_range[1]):
                            validation_result['warnings'].append(f"神经元{i}活动值超出范围: {activity}")
            
            # 验证连接数据
            if 'connections' in data:
                connection_ids = set()
                for i, connection in enumerate(data['connections']):
                    if 'from' not in connection or 'to' not in connection:
                        validation_result['errors'].append(f"连接{i}缺少from或to字段")
                        validation_result['valid'] = False
                    
                    # 检查重复连接
                    connection_id = (connection.get('from'), connection.get('to'))
                    if connection_id in connection_ids:
                        validation_result['warnings'].append(f"发现重复连接: {connection_id}")
                    else:
                        connection_ids.add(connection_id)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return {
                'valid': False,
                'errors': [f"验证过程出错: {str(e)}"],
                'warnings': []
            }
    
    def _handle_validation_errors(self, data: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理验证错误"""
        try:
            # 尝试修复数据
            cleaned_data = data.copy()
            
            if 'neurons' in cleaned_data:
                cleaned_neurons = []
                for neuron in cleaned_data['neurons']:
                    # 确保必要字段存在
                    if 'id' not in neuron:
                        continue
                    if 'position' not in neuron:
                        neuron['position'] = [0, 0, 0]
                    if 'activity' not in neuron:
                        neuron['activity'] = 0.0
                    
                    cleaned_neurons.append(neuron)
                
                cleaned_data['neurons'] = cleaned_neurons
            
            if 'connections' in cleaned_data:
                cleaned_connections = []
                for connection in cleaned_data['connections']:
                    if 'from' in connection and 'to' in connection:
                        cleaned_connections.append(connection)
                
                cleaned_data['connections'] = cleaned_connections
            
            self.logger.info(f"数据验证后修复，保留{len(cleaned_data.get('neurons', []))}个神经元和{len(cleaned_data.get('connections', []))}个连接")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"处理验证错误失败: {e}")
            return data
    
    async def _convert_data_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换数据格式"""
        try:
            converted_data = {}
            
            # 转换神经元数据
            if 'neurons' in data:
                converted_neurons = []
                for neuron in data['neurons']:
                    converted_neuron = {
                        'id': neuron.get('id'),
                        'position': list(map(float, neuron.get('position', [0, 0, 0]))),
                        'activity': float(neuron.get('activity', 0.0)),
                        'type': neuron.get('type', 'standard'),
                        'layer': neuron.get('layer', 0)
                    }
                    converted_neurons.append(converted_neuron)
                
                converted_data['neurons'] = converted_neurons
            
            # 转换连接数据
            if 'connections' in data:
                converted_connections = []
                for connection in data['connections']:
                    converted_connection = {
                        'from': connection.get('from'),
                        'to': connection.get('to'),
                        'weight': float(connection.get('weight', 0.0)),
                        'delay': float(connection.get('delay', 0.0)),
                        'synapse': connection.get('synapse', 'alpha')
                    }
                    converted_connections.append(converted_connection)
                
                converted_data['connections'] = converted_connections
            
            # 转换时间序列数据
            if 'temporal' in data:
                converted_temporal = []
                for temporal in data['temporal']:
                    converted_temporal.append({
                        'timestamp': float(temporal.get('timestamp', time.time())),
                        'neuron_id': temporal.get('neuron_id'),
                        'spike': bool(temporal.get('spike', False)),
                        'activity': float(temporal.get('activity', 0.0))
                    })
                
                converted_data['temporal'] = converted_temporal
            
            # 添加版本信息
            converted_data['version'] = data.get('version', '1.0')
            converted_data['format'] = 'converted'
            
            return converted_data
            
        except Exception as e:
            self.logger.error(f"数据格式转换失败: {e}")
            raise
    
    async def _process_neurons(self, neurons_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理神经元数据"""
        try:
            processed_neurons = []
            
            for neuron in neurons_data:
                neuron_id = neuron.get('id')
                
                # 验证和清理数据
                processed_neuron = {
                    'id': neuron_id,
                    'position': await self._validate_and_clean_position(neuron.get('position')),
                    'activity': await self._validate_and_clean_activity(neuron.get('activity')),
                    'type': neuron.get('type', 'standard'),
                    'layer': neuron.get('layer', 0),
                    'neighbors': set(),  # 将在连接处理时填充
                    'last_updated': time.time()
                }
                
                # 存储到内部结构
                self.neurons[neuron_id] = processed_neuron
                processed_neurons.append(processed_neuron)
            
            return processed_neurons
            
        except Exception as e:
            self.logger.error(f"处理神经元数据失败: {e}")
            return []
    
    async def _process_connections(self, connections_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理连接数据"""
        try:
            processed_connections = []
            
            for connection in connections_data:
                from_id = connection.get('from')
                to_id = connection.get('to')
                
                if from_id not in self.neurons or to_id not in self.neurons:
                    self.logger.warning(f"跳过无效连接: {from_id} -> {to_id}")
                    continue
                
                # 验证和清理数据
                processed_connection = {
                    'id': f"{from_id}_{to_id}",
                    'from': from_id,
                    'to': to_id,
                    'weight': await self._validate_and_clean_weight(connection.get('weight')),
                    'delay': float(connection.get('delay', 0.0)),
                    'synapse': connection.get('synapse', 'alpha'),
                    'strength': abs(float(connection.get('weight', 0.0))),
                    'created_time': time.time()
                }
                
                # 更新神经元的邻居关系
                self.neurons[from_id]['neighbors'].add(to_id)
                self.neurons[to_id]['neighbors'].add(from_id)
                
                # 存储到内部结构
                connection_id = processed_connection['id']
                self.connections[connection_id] = processed_connection
                processed_connections.append(processed_connection)
            
            return processed_connections
            
        except Exception as e:
            self.logger.error(f"处理连接数据失败: {e}")
            return []
    
    async def _process_temporal_data(self, temporal_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理时间序列数据"""
        try:
            processed_temporal = []
            
            for temporal in temporal_data:
                neuron_id = temporal.get('neuron_id')
                
                if neuron_id not in self.neurons:
                    continue
                
                # 验证和清理数据
                processed_temporal_entry = {
                    'timestamp': float(temporal.get('timestamp', time.time())),
                    'neuron_id': neuron_id,
                    'spike': bool(temporal.get('spike', False)),
                    'activity': await self._validate_and_clean_activity(temporal.get('activity'))
                }
                
                # 更新神经元的活动历史
                self.activity_history[neuron_id].append(processed_temporal_entry)
                
                processed_temporal.append(processed_temporal_entry)
            
            # 按时间戳排序
            processed_temporal.sort(key=lambda x: x['timestamp'])
            
            # 存储到时间序列队列
            self.temporal_data.extend(processed_temporal)
            
            return processed_temporal
            
        except Exception as e:
            self.logger.error(f"处理时间序列数据失败: {e}")
            return []
    
    async def _validate_and_clean_position(self, position: List[float]) -> List[float]:
        """验证和清理位置数据"""
        try:
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                self.logger.warning(f"位置数据格式错误，使用默认位置: {position}")
                return [0.0, 0.0, 0.0]
            
            cleaned_position = []
            for coord in position:
                if isinstance(coord, (int, float)):
                    cleaned_position.append(float(coord))
                else:
                    self.logger.warning(f"坐标值类型错误: {coord}")
                    cleaned_position.append(0.0)
            
            return cleaned_position
            
        except Exception as e:
            self.logger.error(f"清理位置数据失败: {e}")
            return [0.0, 0.0, 0.0]
    
    async def _validate_and_clean_activity(self, activity: float) -> float:
        """验证和清理活动值"""
        try:
            if not isinstance(activity, (int, float)):
                self.logger.warning(f"活动值类型错误: {activity}")
                return 0.0
            
            cleaned_activity = float(activity)
            
            # 限制在有效范围内
            min_activity, max_activity = self.format_config.activity_range
            if cleaned_activity < min_activity:
                cleaned_activity = min_activity
            elif cleaned_activity > max_activity:
                cleaned_activity = max_activity
            
            return cleaned_activity
            
        except Exception as e:
            self.logger.error(f"清理活动值失败: {e}")
            return 0.0
    
    async def _validate_and_clean_weight(self, weight: float) -> float:
        """验证和清理连接权重"""
        try:
            if not isinstance(weight, (int, float)):
                self.logger.warning(f"连接权重类型错误: {weight}")
                return 0.0
            
            cleaned_weight = float(weight)
            
            # 限制在有效范围内
            min_weight, max_weight = self.format_config.connection_weight_range
            cleaned_weight = max(min_weight, min(max_weight, cleaned_weight))
            
            return cleaned_weight
            
        except Exception as e:
            self.logger.error(f"清理连接权重失败: {e}")
            return 0.0
    
    async def apply_spatial_layout(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        应用空间布局算法
        
        Args:
            connections_data: 连接数据
            
        Returns:
            Dict[str, Any]: 布局数据
        """
        try:
            algorithm = self.processing_config.spatial_layout_algorithm
            
            if algorithm not in self.spatial_layouts:
                self.logger.warning(f"未知的布局算法: {algorithm}，使用默认算法")
                algorithm = 'force_directed'
            
            # 执行布局算法
            layout_data = await self.spatial_layouts[algorithm](connections_data)
            
            # 缓存布局结果
            layout_key = f"layout_{algorithm}_{len(self.neurons)}"
            self.layout_cache[layout_key] = layout_data
            
            return layout_data
            
        except Exception as e:
            self.logger.error(f"应用空间布局失败: {e}")
            return {}
    
    async def _force_directed_layout(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """力导向布局算法"""
        try:
            # 简化的力导向布局实现
            positions = {}
            
            # 初始化位置（随机）
            for neuron_id in self.neurons:
                positions[neuron_id] = np.random.randn(3) * 10
            
            # 迭代优化
            iterations = 50
            k_repulsion = 1.0
            k_attraction = 0.1
            damping = 0.9
            
            for _ in range(iterations):
                forces = {neuron_id: np.zeros(3) for neuron_id in self.neurons}
                
                # 计算排斥力
                for neuron_id1, pos1 in positions.items():
                    for neuron_id2, pos2 in positions.items():
                        if neuron_id1 != neuron_id2:
                            distance = np.linalg.norm(pos2 - pos1)
                            if distance > 0:
                                force_direction = (pos1 - pos2) / distance
                                force_magnitude = k_repulsion / (distance ** 2)
                                forces[neuron_id1] += force_direction * force_magnitude
                
                # 计算吸引力
                for connection in connections_data:
                    from_id = connection.get('from')
                    to_id = connection.get('to')
                    
                    if from_id in positions and to_id in positions:
                        pos1 = positions[from_id]
                        pos2 = positions[to_id]
                        distance = np.linalg.norm(pos2 - pos1)
                        
                        if distance > 0:
                            force_direction = (pos2 - pos1) / distance
                            force_magnitude = k_attraction * distance
                            forces[from_id] += force_direction * force_magnitude
                            forces[to_id] -= force_direction * force_magnitude
                
                # 更新位置
                for neuron_id in positions:
                    positions[neuron_id] += forces[neuron_id] * damping
                    # 限制在合理范围内
                    positions[neuron_id] = np.clip(positions[neuron_id], -50, 50)
            
            # 转换格式
            layout_result = {
                'type': 'force_directed',
                'positions': {str(k): v.tolist() for k, v in positions.items()},
                'iterations': iterations,
                'parameters': {
                    'k_repulsion': k_repulsion,
                    'k_attraction': k_attraction,
                    'damping': damping
                }
            }
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"力导向布局失败: {e}")
            return {}
    
    async def _circular_layout(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """圆形布局算法"""
        try:
            positions = {}
            neuron_ids = list(self.neurons.keys())
            n_neurons = len(neuron_ids)
            
            if n_neurons == 0:
                return {}
            
            radius = 20.0
            center = np.array([0, 0, 0])
            
            for i, neuron_id in enumerate(neuron_ids):
                angle = 2 * np.pi * i / n_neurons
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = center[2]
                
                positions[neuron_id] = np.array([x, y, z])
            
            layout_result = {
                'type': 'circular',
                'positions': {str(k): v.tolist() for k, v in positions.items()},
                'radius': radius,
                'center': center.tolist()
            }
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"圆形布局失败: {e}")
            return {}
    
    async def _hierarchical_layout(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """层次布局算法"""
        try:
            # 计算神经元层级（基于连接关系）
            layer_assignments = await self._calculate_hierarchical_layers(connections_data)
            
            positions = {}
            neuron_ids = list(self.neurons.keys())
            layers = max(layer_assignments.values()) + 1 if layer_assignments else 1
            
            # 每层的垂直间距
            vertical_spacing = 10.0
            horizontal_spacing = 5.0
            
            # 按层组织神经元
            layer_neurons = defaultdict(list)
            for neuron_id, layer in layer_assignments.items():
                layer_neurons[layer].append(neuron_id)
            
            # 为每层计算位置
            for layer, layer_neuron_ids in layer_neurons.items():
                n_in_layer = len(layer_neuron_ids)
                y = layer * vertical_spacing
                
                for i, neuron_id in enumerate(layer_neuron_ids):
                    if n_in_layer == 1:
                        x = 0
                    else:
                        x = (i - (n_in_layer - 1) / 2) * horizontal_spacing
                    
                    positions[neuron_id] = np.array([x, y, 0])
            
            layout_result = {
                'type': 'hierarchical',
                'positions': {str(k): v.tolist() for k, v in positions.items()},
                'layers': layer_assignments,
                'spacing': {
                    'vertical': vertical_spacing,
                    'horizontal': horizontal_spacing
                }
            }
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"层次布局失败: {e}")
            return {}
    
    async def _clustered_layout(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚类布局算法"""
        try:
            # 简化的聚类布局
            positions = {}
            
            # 基于连接强度进行简单聚类
            clusters = await self._identify_clusters(connections_data)
            cluster_centers = [np.random.randn(3) * 30 for _ in range(len(clusters))]
            
            for cluster_id, cluster_neurons in enumerate(clusters):
                center = cluster_centers[cluster_id]
                
                for neuron_id in cluster_neurons:
                    # 在聚类中心周围添加随机偏移
                    offset = np.random.randn(3) * 5
                    positions[neuron_id] = center + offset
            
            layout_result = {
                'type': 'clustered',
                'positions': {str(k): v.tolist() for k, v in positions.items()},
                'clusters': clusters,
                'cluster_centers': [center.tolist() for center in cluster_centers]
            }
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"聚类布局失败: {e}")
            return {}
    
    async def _calculate_hierarchical_layers(self, connections_data: List[Dict[str, Any]]) -> Dict[int, int]:
        """计算层次布局的层级"""
        try:
            # 简化的层级计算算法
            layers = {}
            neuron_ids = set(self.neurons.keys())
            
            # 找到输入神经元（没有输入连接的神经元）
            input_neurons = set()
            all_inputs = set()
            all_outputs = set()
            
            for connection in connections_data:
                all_inputs.add(connection.get('to'))
                all_outputs.add(connection.get('from'))
            
            input_neurons = neuron_ids - all_inputs
            
            if not input_neurons:
                # 如果没有明确的输入层，随机分配
                for neuron_id in neuron_ids:
                    layers[neuron_id] = 0
                return layers
            
            # 分配层级
            current_layer = 0
            current_neurons = input_neurons
            
            while current_neurons:
                # 当前层神经元
                for neuron_id in current_neurons:
                    layers[neuron_id] = current_layer
                
                # 找到下一层神经元
                next_neurons = set()
                for connection in connections_data:
                    if connection.get('from') in current_neurons:
                        next_neurons.add(connection.get('to'))
                
                # 移除已分配的神经元
                next_neurons = next_neurons - set(layers.keys())
                current_neurons = next_neurons
                current_layer += 1
            
            return layers
            
        except Exception as e:
            self.logger.error(f"计算层次层级失败: {e}")
            return {}
    
    async def _identify_clusters(self, connections_data: List[Dict[str, Any]]) -> List[List[int]]:
        """识别神经元聚类"""
        try:
            # 简化的聚类算法：基于连接强度
            neuron_ids = list(self.neurons.keys())
            connection_strengths = {}
            
            # 计算神经元间的连接强度
            for connection in connections_data:
                from_id = connection.get('from')
                to_id = connection.get('to')
                weight = abs(connection.get('weight', 0.0))
                
                if from_id not in connection_strengths:
                    connection_strengths[from_id] = {}
                if to_id not in connection_strengths:
                    connection_strengths[to_id] = {}
                
                connection_strengths[from_id][to_id] = weight
                connection_strengths[to_id][from_id] = weight
            
            # 简单的聚类：将高连接强度的神经元分组
            clusters = []
            visited = set()
            
            for neuron_id in neuron_ids:
                if neuron_id in visited:
                    continue
                
                cluster = [neuron_id]
                visited.add(neuron_id)
                
                # 找到强连接的神经元
                to_check = [neuron_id]
                while to_check:
                    current_id = to_check.pop(0)
                    
                    if current_id in connection_strengths:
                        for connected_id, strength in connection_strengths[current_id].items():
                            if connected_id not in visited and strength > 0.5:
                                cluster.append(connected_id)
                                visited.add(connected_id)
                                to_check.append(connected_id)
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"识别聚类失败: {e}")
            return [list(self.neurons.keys())]
    
    async def process_connections(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理连接关系"""
        try:
            connections = network_data.get('connections', [])
            
            processed_connections = {
                'connection_matrix': [],
                'connection_stats': await self._calculate_connection_statistics(connections),
                'clustering_info': await self._analyze_clustering(connections),
                'path_analysis': await self._analyze_network_paths(connections)
            }
            
            return processed_connections
            
        except Exception as e:
            self.logger.error(f"处理连接失败: {e}")
            return {}
    
    async def _calculate_connection_statistics(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算连接统计信息"""
        try:
            if not connections:
                return {}
            
            weights = [abs(conn.get('weight', 0.0)) for conn in connections]
            delays = [conn.get('delay', 0.0) for conn in connections]
            
            stats = {
                'total_connections': len(connections),
                'weight_statistics': {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'median': np.median(weights)
                },
                'delay_statistics': {
                    'mean': np.mean(delays),
                    'std': np.std(delays),
                    'min': np.min(delays),
                    'max': np.max(delays)
                },
                'density': len(connections) / (len(self.neurons) * (len(self.neurons) - 1)),
                'average_degree': sum(len(self.neurons[conn.get('from', 0)].get('neighbors', set())) 
                                     for conn in connections) / len(self.neurons) if self.neurons else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算连接统计失败: {e}")
            return {}
    
    async def _analyze_clustering(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析聚类特性"""
        try:
            # 计算局部聚类系数
            local_clustering = {}
            
            for neuron_id, neuron_data in self.neurons.items():
                neighbors = neuron_data.get('neighbors', set())
                neighbor_count = len(neighbors)
                
                if neighbor_count < 2:
                    local_clustering[neuron_id] = 0.0
                    continue
                
                # 计算邻居间的连接数
                actual_connections = 0
                possible_connections = neighbor_count * (neighbor_count - 1) / 2
                
                for neighbor1 in neighbors:
                    for neighbor2 in neighbors:
                        if neighbor1 < neighbor2:  # 避免重复计算
                            # 检查是否有连接
                            has_connection = any(
                                (conn.get('from') == neighbor1 and conn.get('to') == neighbor2) or
                                (conn.get('from') == neighbor2 and conn.get('to') == neighbor1)
                                for conn in connections
                            )
                            if has_connection:
                                actual_connections += 1
                
                local_clustering[neuron_id] = actual_connections / possible_connections if possible_connections > 0 else 0
            
            avg_clustering = np.mean(list(local_clustering.values())) if local_clustering else 0
            
            return {
                'local_clustering': local_clustering,
                'average_clustering': avg_clustering,
                'clustering_coefficient': avg_clustering
            }
            
        except Exception as e:
            self.logger.error(f"分析聚类失败: {e}")
            return {}
    
    async def _analyze_network_paths(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析网络路径特性"""
        try:
            # 构建邻接矩阵
            neuron_ids = list(self.neurons.keys())
            n_neurons = len(neuron_ids)
            id_to_index = {neuron_id: i for i, neuron_id in enumerate(neuron_ids)}
            
            adjacency_matrix = np.zeros((n_neurons, n_neurons))
            
            for connection in connections:
                from_id = connection.get('from')
                to_id = connection.get('to')
                weight = abs(connection.get('weight', 0.0))
                
                if from_id in id_to_index and to_id in id_to_index:
                    from_idx = id_to_index[from_id]
                    to_idx = id_to_index[to_id]
                    adjacency_matrix[from_idx][to_idx] = weight
            
            # 计算最短路径（简化版本）
            path_lengths = []
            
            for i in range(n_neurons):
                for j in range(i + 1, n_neurons):
                    # 使用简化的距离计算
                    distance = await self._calculate_shortest_path(i, j, adjacency_matrix)
                    if distance < float('inf'):
                        path_lengths.append(distance)
            
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            
            return {
                'average_path_length': avg_path_length,
                'diameter': np.max(path_lengths) if path_lengths else 0,
                'path_length_distribution': {
                    'mean': avg_path_length,
                    'std': np.std(path_lengths) if path_lengths else 0,
                    'min': np.min(path_lengths) if path_lengths else 0,
                    'max': np.max(path_lengths) if path_lengths else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"分析网络路径失败: {e}")
            return {}
    
    async def _calculate_shortest_path(self, start_idx: int, end_idx: int, adjacency_matrix: np.ndarray) -> float:
        """计算两个神经元间的最短路径"""
        try:
            n = adjacency_matrix.shape[0]
            
            # 使用Dijkstra算法（简化版本）
            distances = np.full(n, float('inf'))
            distances[start_idx] = 0
            visited = np.zeros(n, dtype=bool)
            
            for _ in range(n):
                # 找到未访问的最近节点
                unvisited = ~visited
                if not unvisited.any():
                    break
                
                closest_idx = np.argmin(distances[unvisited])
                actual_idx = np.where(unvisited)[0][closest_idx]
                
                if distances[actual_idx] == float('inf'):
                    break
                
                visited[actual_idx] = True
                
                if actual_idx == end_idx:
                    break
                
                # 更新邻居距离
                for neighbor_idx in range(n):
                    if not visited[neighbor_idx] and adjacency_matrix[actual_idx][neighbor_idx] > 0:
                        new_distance = distances[actual_idx] + 1.0 / adjacency_matrix[actual_idx][neighbor_idx]
                        if new_distance < distances[neighbor_idx]:
                            distances[neighbor_idx] = new_distance
            
            return distances[end_idx]
            
        except Exception as e:
            self.logger.error(f"计算最短路径失败: {e}")
            return float('inf')
    
    def get_network_data(self) -> Dict[str, Any]:
        """获取当前网络数据"""
        try:
            return {
                'neurons': list(self.neurons.values()),
                'connections': list(self.connections.values()),
                'temporal_data': list(self.temporal_data),
                'statistics': self.data_stats,
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"获取网络数据失败: {e}")
            return {}
    
    def get_neuron_count(self) -> int:
        """获取神经元数量"""
        return len(self.neurons)
    
    def get_connection_count(self) -> int:
        """获取连接数量"""
        return len(self.connections)
    
    def should_recalculate_layout(self) -> bool:
        """判断是否需要重新计算布局"""
        try:
            # 如果神经元数量变化较大，需要重新计算布局
            last_layout_size = getattr(self, '_last_layout_size', 0)
            current_size = len(self.neurons)
            
            if abs(current_size - last_layout_size) / max(current_size, last_layout_size) > 0.1:
                self._last_layout_size = current_size
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"判断布局重计算失败: {e}")
            return False
    
    async def recalculate_layout(self) -> Dict[str, Any]:
        """重新计算布局"""
        try:
            connections_data = list(self.connections.values())
            layout_data = await self.apply_spatial_layout(connections_data)
            
            self.logger.info("重新计算网络布局完成")
            return layout_data
            
        except Exception as e:
            self.logger.error(f"重新计算布局失败: {e}")
            return {}
    
    async def update_neurons(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新神经元数据"""
        try:
            updates = {
                'updated_neurons': [],
                'new_neurons': [],
                'removed_neurons': []
            }
            
            new_neurons = new_data.get('neurons', [])
            
            for neuron in new_neurons:
                neuron_id = neuron.get('id')
                
                if neuron_id in self.neurons:
                    # 更新现有神经元
                    old_data = self.neurons[neuron_id].copy()
                    self.neurons[neuron_id].update({
                        'activity': neuron.get('activity', 0.0),
                        'position': neuron.get('position', [0, 0, 0]),
                        'last_updated': time.time()
                    })
                    updates['updated_neurons'].append({
                        'id': neuron_id,
                        'old_data': old_data,
                        'new_data': self.neurons[neuron_id]
                    })
                else:
                    # 添加新神经元
                    self.neurons[neuron_id] = {
                        'id': neuron_id,
                        'position': neuron.get('position', [0, 0, 0]),
                        'activity': neuron.get('activity', 0.0),
                        'type': neuron.get('type', 'standard'),
                        'layer': neuron.get('layer', 0),
                        'neighbors': set(),
                        'last_updated': time.time()
                    }
                    updates['new_neurons'].append(neuron)
            
            # 检查删除的神经元
            current_ids = set(neuron.get('id') for neuron in new_neurons)
            existing_ids = set(self.neurons.keys())
            
            removed_ids = existing_ids - current_ids
            for neuron_id in removed_ids:
                updates['removed_neurons'].append(self.neurons[neuron_id])
                del self.neurons[neuron_id]
            
            self.logger.info(f"神经元数据更新完成：{len(updates['updated_neurons'])}个更新，{len(updates['new_neurons'])}个新增，{len(updates['removed_neurons'])}个删除")
            
            return updates
            
        except Exception as e:
            self.logger.error(f"更新神经元失败: {e}")
            return {}
    
    async def update_connections(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新连接数据"""
        try:
            updates = {
                'updated_connections': [],
                'new_connections': [],
                'removed_connections': []
            }
            
            new_connections = new_data.get('connections', [])
            
            # 重建连接映射
            new_connection_map = {}
            for connection in new_connections:
                connection_id = f"{connection.get('from')}_{connection.get('to')}"
                new_connection_map[connection_id] = connection
            
            # 处理现有连接
            existing_connection_ids = set(self.connections.keys())
            new_connection_ids = set(new_connection_map.keys())
            
            # 新增或更新的连接
            for connection_id, connection_data in new_connection_map.items():
                if connection_id in self.connections:
                    # 更新现有连接
                    old_data = self.connections[connection_id].copy()
                    self.connections[connection_id].update({
                        'weight': connection_data.get('weight', 0.0),
                        'delay': connection_data.get('delay', 0.0),
                        'strength': abs(connection_data.get('weight', 0.0))
                    })
                    updates['updated_connections'].append({
                        'id': connection_id,
                        'old_data': old_data,
                        'new_data': self.connections[connection_id]
                    })
                else:
                    # 新增连接
                    self.connections[connection_id] = {
                        'id': connection_id,
                        'from': connection_data.get('from'),
                        'to': connection_data.get('to'),
                        'weight': connection_data.get('weight', 0.0),
                        'delay': connection_data.get('delay', 0.0),
                        'synapse': connection_data.get('synapse', 'alpha'),
                        'strength': abs(connection_data.get('weight', 0.0)),
                        'created_time': time.time()
                    }
                    updates['new_connections'].append(connection_data)
            
            # 删除的连接
            removed_ids = existing_connection_ids - new_connection_ids
            for connection_id in removed_ids:
                updates['removed_connections'].append(self.connections[connection_id])
                del self.connections[connection_id]
            
            # 更新神经元的邻居关系
            await self._update_neuron_neighbors()
            
            self.logger.info(f"连接数据更新完成：{len(updates['updated_connections'])}个更新，{len(updates['new_connections'])}个新增，{len(updates['removed_connections'])}个删除")
            
            return updates
            
        except Exception as e:
            self.logger.error(f"更新连接失败: {e}")
            return {}
    
    async def _update_neuron_neighbors(self):
        """更新神经元的邻居关系"""
        try:
            # 清空所有邻居关系
            for neuron_id in self.neurons:
                self.neurons[neuron_id]['neighbors'].clear()
            
            # 重新构建邻居关系
            for connection in self.connections.values():
                from_id = connection.get('from')
                to_id = connection.get('to')
                
                if from_id in self.neurons and to_id in self.neurons:
                    self.neurons[from_id]['neighbors'].add(to_id)
                    self.neurons[to_id]['neighbors'].add(from_id)
            
        except Exception as e:
            self.logger.error(f"更新神经元邻居失败: {e}")
    
    async def get_latest_updates(self) -> Dict[str, Any]:
        """获取最新的网络更新"""
        try:
            # 模拟获取最新的网络活动数据
            updates = {
                'neurons': [],
                'connections': [],
                'temporal': list(self.temporal_data)[-100:],  # 最近100个时间点
                'timestamp': time.time(),
                'frame': int(time.time() * 60)  # 假设60FPS
            }
            
            # 添加一些模拟的活动数据
            for neuron_id, neuron_data in list(self.neurons.items())[:10]:  # 只更新部分神经元以节省性能
                # 模拟活动值变化
                base_activity = neuron_data.get('activity', 0.0)
                variation = np.random.normal(0, 0.1)
                new_activity = max(0.0, min(1.0, base_activity + variation))
                
                updates['neurons'].append({
                    'id': neuron_id,
                    'activity': new_activity,
                    'last_updated': time.time()
                })
            
            return updates
            
        except Exception as e:
            self.logger.error(f"获取最新更新失败: {e}")
            return {}
    
    def export_data(self, format: str = 'json') -> Union[str, bytes]:
        """导出网络数据"""
        try:
            data = self.get_network_data()
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format.lower() == 'pickle':
                return pickle.dumps(data)
            elif format.lower() == 'compressed':
                json_str = json.dumps(data, default=str)
                return gzip.compress(json_str.encode('utf-8'))
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
        except Exception as e:
            self.logger.error(f"导出数据失败: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.neurons.clear()
            self.connections.clear()
            self.temporal_data.clear()
            self.activity_history.clear()
            self.position_cache.clear()
            self.layout_cache.clear()
            self.data_cache.clear()
            
            self.logger.info("网络数据处理器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"数据处理器清理失败: {e}")
    
    # 数据验证方法
    async def _validate_position(self, position: List[float]) -> bool:
        """验证位置数据"""
        try:
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                return False
            
            return all(isinstance(coord, (int, float)) and -1000 <= coord <= 1000 for coord in position)
            
        except Exception:
            return False
    
    async def _validate_activity(self, activity: float) -> bool:
        """验证活动值"""
        try:
            if not isinstance(activity, (int, float)):
                return False
            
            min_val, max_val = self.format_config.activity_range
            return min_val <= activity <= max_val
            
        except Exception:
            return False
    
    async def _validate_connection(self, connection: Dict[str, Any]) -> bool:
        """验证连接数据"""
        try:
            required_fields = ['from', 'to', 'weight']
            return all(field in connection for field in required_fields)
            
        except Exception:
            return False
    
    async def _validate_spike(self, spike: Dict[str, Any]) -> bool:
        """验证脉冲数据"""
        try:
            required_fields = ['neuron_id', 'timestamp']
            return all(field in spike for field in required_fields)
            
        except Exception:
            return False
    
    def reset(self):
        """重置处理器"""
        try:
            self.neurons.clear()
            self.connections.clear()
            self.temporal_data.clear()
            self.activity_history.clear()
            self.position_cache.clear()
            self.layout_cache.clear()
            self.data_cache.clear()
            
            # 重置统计信息
            self.data_stats.update({
                'total_neurons': 0,
                'total_connections': 0,
                'data_points_processed': 0,
                'validation_errors': 0
            })
            
            self.logger.info("网络数据处理器已重置")
            
        except Exception as e:
            self.logger.error(f"重置处理器失败: {e}")


if __name__ == "__main__":
    # 示例用法
    async def test_network_data_handler():
        config = DataFormat()
        processing_config = ProcessingConfig()
        
        handler = NetworkDataHandler(config, processing_config)
        
        # 初始化
        init_data = await handler.initialize_network()
        print(f"初始化数据: {init_data}")
        
        # 模拟Nengo数据
        nengo_data = {
            'neurons': [
                {'id': 1, 'position': [0, 0, 0], 'activity': 0.5},
                {'id': 2, 'position': [1, 0, 0], 'activity': 0.8},
                {'id': 3, 'position': [0, 1, 0], 'activity': 1.0}
            ],
            'connections': [
                {'from': 1, 'to': 2, 'weight': 0.5},
                {'from': 2, 'to': 3, 'weight': -0.3}
            ],
            'temporal': [
                {'neuron_id': 1, 'timestamp': time.time(), 'spike': True, 'activity': 1.0}
            ]
        }
        
        # 解析数据
        parsed_data = await handler.parse_nengo_data(nengo_data)
        print(f"解析数据: {parsed_data}")
        
        # 应用布局
        layout_data = await handler.apply_spatial_layout(parsed_data['connections'])
        print(f"布局数据: {layout_data}")
        
        # 处理连接
        connection_data = await handler.process_connections(parsed_data)
        print(f"连接数据: {connection_data}")
        
        # 获取网络数据
        network_data = handler.get_network_data()
        print(f"网络数据: {network_data}")
        
        # 获取最新更新
        updates = await handler.get_latest_updates()
        print(f"最新更新: {updates}")
        
        # 清理资源
        await handler.cleanup()
    
    # 运行测试
    asyncio.run(test_network_data_handler())