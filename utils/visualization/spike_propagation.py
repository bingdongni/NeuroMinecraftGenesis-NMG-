# -*- coding: utf-8 -*-
"""
脉冲传播类
负责计算和动画化神经网络中脉冲信号的传播过程
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from dataclasses import dataclass
import heapq
from collections import defaultdict, deque
import math
import time


@dataclass
class PropagationConfig:
    """脉冲传播配置"""
    propagation_speed: float = 2.0  # 脉冲传播速度 (units/s)
    max_propagation_distance: float = 20.0  # 最大传播距离
    spike_threshold: float = 0.8  # 脉冲阈值
    refractory_period: float = 0.01  # 绝对不应期 (s)
    relative_refractory_period: float = 0.05  # 相对不应期 (s)
    decay_factor: float = 0.9  # 脉冲衰减因子
    propagation_delay: float = 0.001  # 突触延迟 (s)
    
    # 动画参数
    animation_fps: int = 60
    wave_visualization: bool = True
    particle_effects: bool = True
    
    # 网络参数
    connection_probability: float = 0.1
    connection_strength_range: Tuple[float, float] = (0.1, 1.0)
    synaptic_delay_range: Tuple[float, float] = (0.001, 0.01)
    
    # 统计参数
    max_history_length: int = 1000
    sampling_interval: float = 0.1


class SpikePropagation:
    """
    脉冲传播计算和动画类
    
    负责计算脉冲在神经网络中的传播路径、时间序列和视觉效果
    """
    
    def __init__(self, config: PropagationConfig):
        """
        初始化脉冲传播系统
        
        Args:
            config: 传播配置参数
        """
        self.config = config
        
        # 网络状态
        self.neurons = {}  # neuron_id -> neuron_data
        self.connections = {}  # connection_id -> connection_data
        self.synaptic_weights = {}  # (from_id, to_id) -> weight
        
        # 脉冲事件
        self.spike_events = []  # 活跃的脉冲事件
        self.spike_queue = []  # 脉冲事件优先队列 (按时间排序)
        self.completed_spikes = deque(maxlen=self.config.max_history_length)  # 已完成的脉冲
        
        # 传播状态
        self.active_propagations = {}  # propagation_id -> propagation_data
        self.wave_fronts = []  # 波前数据
        self.network_activity = []  # 网络活动历史
        
        # 性能统计
        self.stats = {
            'total_spikes': 0,
            'active_spikes': 0,
            'average_propagation_time': 0.0,
            'network_firing_rate': 0.0,
            'active_connections': 0,
            'last_update_time': time.time()
        }
        
        # 缓存
        self.distance_cache = {}  # 距离计算缓存
        self.path_cache = {}  # 传播路径缓存
        
        self.logger = self._setup_logging()
        self.logger.info("脉冲传播系统初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('SpikePropagation')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def initialize_propagation(self):
        """
        初始化脉冲传播系统
        
        Returns:
            Dict[str, Any]: 初始化数据
        """
        try:
            init_data = {
                'config': {
                    'propagation_speed': self.config.propagation_speed,
                    'max_distance': self.config.max_propagation_distance,
                    'threshold': self.config.spike_threshold,
                    'refractory_period': self.config.refractory_period
                },
                'network_topology': await self._build_network_topology(),
                'initial_spikes': [],
                'wave_visualization': self.config.wave_visualization,
                'particle_effects': self.config.particle_effects
            }
            
            self.logger.info("脉冲传播系统初始化完成")
            return init_data
            
        except Exception as e:
            self.logger.error(f"脉冲传播初始化失败: {e}")
            raise
    
    async def _build_network_topology(self) -> Dict[str, Any]:
        """构建网络拓扑"""
        try:
            topology = {
                'nodes': [],
                'edges': [],
                'clustering': await self._calculate_clustering_coefficient(),
                'path_length': await self._calculate_average_path_length(),
                'small_world_properties': await self._analyze_small_world_properties()
            }
            
            return topology
            
        except Exception as e:
            self.logger.error(f"构建网络拓扑失败: {e}")
            return {}
    
    async def _calculate_clustering_coefficient(self) -> float:
        """计算聚类系数"""
        try:
            # 简化的聚类系数计算
            total_clustering = 0.0
            node_count = 0
            
            for neuron_id, neuron_data in self.neurons.items():
                neighbors = neuron_data.get('neighbors', set())
                neighbor_count = len(neighbors)
                
                if neighbor_count < 2:
                    continue
                
                # 计算邻居间的连接数
                actual_connections = 0
                possible_connections = neighbor_count * (neighbor_count - 1) / 2
                
                for neighbor1 in neighbors:
                    for neighbor2 in neighbors:
                        if neighbor1 < neighbor2:  # 避免重复计算
                            if (neighbor1, neighbor2) in self.synaptic_weights or \
                               (neighbor2, neighbor1) in self.synaptic_weights:
                                actual_connections += 1
                
                local_clustering = actual_connections / possible_connections if possible_connections > 0 else 0
                total_clustering += local_clustering
                node_count += 1
            
            return total_clustering / node_count if node_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"计算聚类系数失败: {e}")
            return 0.0
    
    async def _calculate_average_path_length(self) -> float:
        """计算平均路径长度"""
        try:
            # 使用Dijkstra算法计算最短路径
            total_path_length = 0.0
            path_count = 0
            
            neuron_ids = list(self.neurons.keys())
            
            for i, start_id in enumerate(neuron_ids):
                for j, end_id in enumerate(neuron_ids[i+1:], i+1):
                    path_length = await self._find_shortest_path(start_id, end_id)
                    if path_length > 0:
                        total_path_length += path_length
                        path_count += 1
            
            return total_path_length / path_count if path_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"计算平均路径长度失败: {e}")
            return 0.0
    
    async def _find_shortest_path(self, start_id: int, end_id: int) -> float:
        """使用Dijkstra算法找到最短路径"""
        try:
            # 缓存检查
            cache_key = (start_id, end_id)
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]
            
            # Dijkstra算法
            distances = {start_id: 0.0}
            visited = set()
            heap = [(0.0, start_id)]
            
            while heap:
                current_dist, current_id = heapq.heappop(heap)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                
                if current_id == end_id:
                    # 缓存结果
                    self.path_cache[cache_key] = current_dist
                    return current_dist
                
                # 检查邻居
                for neighbor_id in self.neurons.get(current_id, {}).get('neighbors', set()):
                    if neighbor_id not in visited:
                        distance = await self._get_connection_distance(current_id, neighbor_id)
                        new_dist = current_dist + distance
                        
                        if neighbor_id not in distances or new_dist < distances[neighbor_id]:
                            distances[neighbor_id] = new_dist
                            heapq.heappush(heap, (new_dist, neighbor_id))
            
            return float('inf')  # 无路径
            
        except Exception as e:
            self.logger.error(f"查找最短路径失败: {e}")
            return float('inf')
    
    async def _get_connection_distance(self, from_id: int, to_id: int) -> float:
        """获取连接距离"""
        try:
            # 缓存检查
            cache_key = (from_id, to_id)
            if cache_key in self.distance_cache:
                return self.distance_cache[cache_key]
            
            from_pos = self.neurons.get(from_id, {}).get('position', [0, 0, 0])
            to_pos = self.neurons.get(to_id, {}).get('position', [0, 0, 0])
            
            distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
            
            # 缓存结果
            self.distance_cache[cache_key] = distance
            
            return distance
            
        except Exception as e:
            self.logger.error(f"获取连接距离失败: {e}")
            return 0.0
    
    async def _analyze_small_world_properties(self) -> Dict[str, float]:
        """分析小世界网络属性"""
        try:
            # 计算小世界商 (σ)
            clustering_coeff = await self._calculate_clustering_coefficient()
            avg_path_length = await self._calculate_average_path_length()
            
            # 假设随机网络的聚类系数和路径长度（简化计算）
            random_clustering = self.config.connection_probability
            node_count = len(self.neurons)
            random_path_length = math.log(node_count) / math.log(node_count * self.config.connection_probariance) if node_count > 1 else 1.0
            
            clustering_ratio = clustering_coeff / random_clustering if random_clustering > 0 else 1.0
            path_ratio = avg_path_length / random_path_length if random_path_length > 0 else 1.0
            
            small_world_quotient = clustering_ratio / path_ratio
            
            return {
                'clustering_coefficient': clustering_coeff,
                'average_path_length': avg_path_length,
                'clustering_ratio': clustering_ratio,
                'path_ratio': path_ratio,
                'small_world_quotient': small_world_quotient
            }
            
        except Exception as e:
            self.logger.error(f"分析小世界属性失败: {e}")
            return {}
    
    async def calculate_paths(self, spike_data: Dict[str, Any]):
        """
        计算脉冲传播路径
        
        Args:
            spike_data: 脉冲数据
            
        Returns:
            Dict[str, Any]: 传播路径数据
        """
        try:
            source_neurons = spike_data.get('source_neurons', [])
            spike_times = spike_data.get('spike_times', [])
            propagation_data = []
            
            for i, source_id in enumerate(source_neurons):
                if i < len(spike_times):
                    spike_time = spike_times[i]
                    
                    # 计算从源神经元开始的传播路径
                    paths = await self._calculate_propagation_from_source(
                        source_id, spike_time
                    )
                    
                    propagation_data.append({
                        'source_id': source_id,
                        'spike_time': spike_time,
                        'paths': paths
                    })
            
            return {
                'propagation_data': propagation_data,
                'total_paths': len(propagation_data),
                'calculation_time': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"计算传播路径失败: {e}")
            return {}
    
    async def _calculate_propagation_from_source(self, source_id: int, spike_time: float) -> List[Dict[str, Any]]:
        """从源神经元计算传播路径"""
        try:
            paths = []
            visited = set()
            propagation_queue = [(0.0, source_id, [source_id])]  # (distance, neuron_id, path)
            
            while propagation_queue:
                current_distance, current_id, current_path = heapq.heappop(propagation_queue)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                
                # 检查是否达到最大传播距离
                if current_distance > self.config.max_propagation_distance:
                    continue
                
                # 记录路径
                path_data = {
                    'path': current_path,
                    'total_distance': current_distance,
                    'arrival_time': spike_time + current_distance / self.config.propagation_speed,
                    'path_length': len(current_path)
                }
                paths.append(path_data)
                
                # 继续传播到邻居
                neighbors = self.neurons.get(current_id, {}).get('neighbors', set())
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        connection_distance = await self._get_connection_distance(current_id, neighbor_id)
                        new_distance = current_distance + connection_distance
                        new_path = current_path + [neighbor_id]
                        
                        if new_distance <= self.config.max_propagation_distance:
                            heapq.heappush(propagation_queue, (new_distance, neighbor_id, new_path))
            
            return paths
            
        except Exception as e:
            self.logger.error(f"计算传播路径失败: {e}")
            return []
    
    async def animate_propagation(self, propagation_paths: Dict[str, Any]):
        """
        动画化脉冲传播
        
        Args:
            propagation_paths: 传播路径数据
            
        Returns:
            Dict[str, Any]: 动画数据
        """
        try:
            animation_data = {
                'wave_animations': [],
                'particle_animations': [],
                'neuron_activations': [],
                'propagation_effects': [],
                'timestamp': time.time()
            }
            
            current_time = time.time()
            
            # 创建波前动画
            for prop_data in propagation_paths.get('propagation_data', []):
                source_id = prop_data.get('source_id')
                spike_time = prop_data.get('spike_time')
                paths = prop_data.get('paths', [])
                
                # 创建径向波纹
                if self.config.wave_visualization:
                    wave_animation = await self._create_radial_wave(source_id, spike_time, current_time)
                    animation_data['wave_animations'].append(wave_animation)
                
                # 创建粒子效果
                if self.config.particle_effects:
                    particle_animations = await self._create_particle_animations(source_id, paths, current_time)
                    animation_data['particle_animations'].extend(particle_animations)
                
                # 神经元激活动画
                neuron_activations = await self._create_neuron_activations(paths, spike_time, current_time)
                animation_data['neuron_activations'].extend(neuron_activations)
                
                # 传播效果
                propagation_effects = await self._create_propagation_effects(paths, current_time)
                animation_data['propagation_effects'].extend(propagation_effects)
            
            return animation_data
            
        except Exception as e:
            self.logger.error(f"动画化传播失败: {e}")
            return {}
    
    async def _create_radial_wave(self, source_id: int, spike_time: float, current_time: float) -> Dict[str, Any]:
        """创建径向波纹"""
        try:
            source_pos = self.neurons.get(source_id, {}).get('position', [0, 0, 0])
            
            wave_animation = {
                'type': 'radial_wave',
                'center': source_pos,
                'start_time': spike_time,
                'current_time': current_time,
                'wave_radius': self.config.propagation_speed * (current_time - spike_time),
                'max_radius': self.config.max_propagation_distance,
                'intensity': 1.0 - (current_time - spike_time) / 5.0,  # 衰减
                'wave_speed': self.config.propagation_speed
            }
            
            return wave_animation
            
        except Exception as e:
            self.logger.error(f"创建径向波纹失败: {e}")
            return {}
    
    async def _create_particle_animations(self, source_id: int, paths: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
        """创建粒子动画"""
        try:
            particle_animations = []
            
            for path_data in paths:
                path = path_data.get('path', [])
                arrival_time = path_data.get('arrival_time', current_time)
                
                # 为每个传播路径创建粒子
                for i, neuron_id in enumerate(path[:-1]):
                    particle_data = {
                        'type': 'propagation_particle',
                        'start_neuron': neuron_id,
                        'end_neuron': path[i + 1],
                        'start_time': current_time + i * 0.01,  # 轻微延迟
                        'duration': 1.0 / self.config.propagation_speed,
                        'particle_count': 5,
                        'color': [1.0, 1.0, 0.0],  # 黄色
                        'size': 0.02
                    }
                    particle_animations.append(particle_data)
            
            return particle_animations
            
        except Exception as e:
            self.logger.error(f"创建粒子动画失败: {e}")
            return []
    
    async def _create_neuron_activations(self, paths: List[Dict[str, Any]], spike_time: float, current_time: float) -> List[Dict[str, Any]]:
        """创建神经元激活动画"""
        try:
            neuron_activations = []
            activated_neurons = set()
            
            for path_data in paths:
                path = path_data.get('path', [])
                arrival_time = path_data.get('arrival_time', current_time)
                
                for neuron_id in path:
                    if neuron_id not in activated_neurons:
                        activation_data = {
                            'type': 'neuron_activation',
                            'neuron_id': neuron_id,
                            'activation_time': arrival_time,
                            'current_time': current_time,
                            'intensity': 1.0,
                            'duration': self.config.refractory_period,
                            'color': [1.0, 1.0, 0.0]  # 黄色激活
                        }
                        neuron_activations.append(activation_data)
                        activated_neurons.add(neuron_id)
            
            return neuron_activations
            
        except Exception as e:
            self.logger.error(f"创建神经元激活动画失败: {e}")
            return []
    
    async def _create_propagation_effects(self, paths: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
        """创建传播效果"""
        try:
            propagation_effects = []
            
            for path_data in paths:
                path = path_data.get('path', [])
                
                # 创建连接高亮效果
                for i in range(len(path) - 1):
                    from_neuron = path[i]
                    to_neuron = path[i + 1]
                    
                    effect_data = {
                        'type': 'connection_highlight',
                        'from_neuron': from_neuron,
                        'to_neuron': to_neuron,
                        'start_time': current_time,
                        'duration': 0.5,
                        'intensity': 1.0,
                        'color': [0.0, 1.0, 1.0]  # 青色高亮
                    }
                    propagation_effects.append(effect_data)
            
            return propagation_effects
            
        except Exception as e:
            self.logger.error(f"创建传播效果失败: {e}")
            return []
    
    async def update_spikes(self):
        """
        更新脉冲状态
        
        Returns:
            Dict[str, Any]: 更新结果
        """
        try:
            current_time = time.time()
            updates = {
                'new_spikes': [],
                'completed_propagations': [],
                'active_wave_fronts': [],
                'stats_update': {}
            }
            
            # 处理新的脉冲事件
            while self.spike_queue and self.spike_queue[0][0] <= current_time:
                spike_time, spike_event = heapq.heappop(self.spike_queue)
                await self._process_spike_event(spike_event, spike_time)
                updates['new_spikes'].append(spike_event)
            
            # 更新活跃的传播
            completed_propagations = []
            for propagation_id, prop_data in list(self.active_propagations.items()):
                if current_time >= prop_data.get('end_time', current_time):
                    completed_propagations.append(prop_data)
                    del self.active_propagations[propagation_id]
            
            updates['completed_propagations'] = completed_propagations
            
            # 更新波前
            active_wave_fronts = await self._update_wave_fronts(current_time)
            updates['active_wave_fronts'] = active_wave_fronts
            
            # 更新统计信息
            await self._update_statistics()
            updates['stats_update'] = self.stats.copy()
            
            return updates
            
        except Exception as e:
            self.logger.error(f"更新脉冲失败: {e}")
            return {}
    
    async def _process_spike_event(self, spike_event: Dict[str, Any], spike_time: float):
        """处理脉冲事件"""
        try:
            source_id = spike_event.get('source_id')
            
            # 标记源神经元发放脉冲
            if source_id in self.neurons:
                self.neurons[source_id]['last_spike_time'] = spike_time
                self.neurons[source_id]['spike_count'] = self.neurons[source_id].get('spike_count', 0) + 1
            
            # 创建传播事件
            propagation_id = f"prop_{source_id}_{spike_time}"
            propagation_data = {
                'id': propagation_id,
                'source_id': source_id,
                'start_time': spike_time,
                'end_time': spike_time + self.config.max_propagation_distance / self.config.propagation_speed,
                'status': 'active'
            }
            
            self.active_propagations[propagation_id] = propagation_data
            
            # 触发突触后脉冲
            neighbors = self.neurons.get(source_id, {}).get('neighbors', set())
            for neighbor_id in neighbors:
                await self._trigger_synaptic_event(source_id, neighbor_id, spike_time)
            
            self.stats['total_spikes'] += 1
            
        except Exception as e:
            self.logger.error(f"处理脉冲事件失败: {e}")
    
    async def _trigger_synaptic_event(self, from_id: int, to_id: int, spike_time: float):
        """触发突触事件"""
        try:
            # 计算突触延迟
            connection_distance = await self._get_connection_distance(from_id, to_id)
            synaptic_delay = self.config.propagation_delay + connection_distance / self.config.propagation_speed
            
            # 计算突触后电位
            weight = self.synaptic_weights.get((from_id, to_id), 0.0)
            psp = weight * math.exp(-synaptic_delay / 0.01)  # 突触后电位衰减
            
            # 如果超过阈值，触发新的脉冲
            if psp >= self.config.spike_threshold:
                new_spike_time = spike_time + synaptic_delay
                spike_event = {
                    'source_id': to_id,
                    'psp': psp,
                    'from_neuron': from_id
                }
                
                # 检查不应期
                if await self._check_refractory_period(to_id, new_spike_time):
                    heapq.heappush(self.spike_queue, (new_spike_time, spike_event))
            
        except Exception as e:
            self.logger.error(f"触发突触事件失败: {e}")
    
    async def _check_refractory_period(self, neuron_id: int, spike_time: float) -> bool:
        """检查神经元是否在不应期内"""
        try:
            neuron_data = self.neurons.get(neuron_id, {})
            last_spike_time = neuron_data.get('last_spike_time', -1)
            
            if last_spike_time < 0:
                return True
            
            time_since_last_spike = spike_time - last_spike_time
            return time_since_last_spike >= self.config.refractory_period
            
        except Exception as e:
            self.logger.error(f"检查不应期失败: {e}")
            return True
    
    async def _update_wave_fronts(self, current_time: float) -> List[Dict[str, Any]]:
        """更新波前"""
        try:
            active_waves = []
            
            # 更新径向波前
            for prop_id, prop_data in self.active_propagations.items():
                source_id = prop_data.get('source_id')
                start_time = prop_data.get('start_time', current_time)
                
                wave_radius = self.config.propagation_speed * (current_time - start_time)
                if wave_radius <= self.config.max_propagation_distance:
                    wave_data = {
                        'propagation_id': prop_id,
                        'source_id': source_id,
                        'radius': wave_radius,
                        'intensity': 1.0 - (wave_radius / self.config.max_propagation_distance),
                        'age': current_time - start_time
                    }
                    active_waves.append(wave_data)
            
            self.wave_fronts = active_waves
            return active_waves
            
        except Exception as e:
            self.logger.error(f"更新波前失败: {e}")
            return []
    
    async def _update_statistics(self):
        """更新统计信息"""
        try:
            current_time = time.time()
            
            # 更新活跃脉冲数
            self.stats['active_spikes'] = len(self.spike_queue) + len(self.active_propagations)
            
            # 计算网络发放率
            recent_spikes = [spike for spike in self.completed_spikes 
                           if current_time - spike.get('timestamp', current_time) <= 1.0]
            
            self.stats['network_firing_rate'] = len(recent_spikes)
            
            # 更新活跃连接数
            active_connections = 0
            for prop_data in self.active_propagations.values():
                source_id = prop_data.get('source_id')
                if source_id in self.neurons:
                    neighbors = self.neurons[source_id].get('neighbors', set())
                    active_connections += len(neighbors)
            
            self.stats['active_connections'] = active_connections
            self.stats['last_update_time'] = current_time
            
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    def add_neuron(self, neuron_id: int, position: List[float], neighbors: Set[int] = None):
        """添加神经元"""
        try:
            self.neurons[neuron_id] = {
                'id': neuron_id,
                'position': position,
                'neighbors': neighbors or set(),
                'last_spike_time': -1,
                'spike_count': 0,
                'activity_level': 0.0
            }
            
            # 添加到邻居的邻居列表
            if neighbors:
                for neighbor_id in neighbors:
                    if neighbor_id in self.neurons:
                        self.neurons[neighbor_id]['neighbors'].add(neuron_id)
            
        except Exception as e:
            self.logger.error(f"添加神经元失败: {e}")
    
    def add_connection(self, from_id: int, to_id: int, weight: float):
        """添加连接"""
        try:
            self.synaptic_weights[(from_id, to_id)] = weight
            
        except Exception as e:
            self.logger.error(f"添加连接失败: {e}")
    
    def trigger_spike(self, neuron_id: int, spike_time: float = None):
        """手动触发脉冲"""
        try:
            if spike_time is None:
                spike_time = time.time()
            
            spike_event = {
                'source_id': neuron_id,
                'manual_trigger': True
            }
            
            heapq.heappush(self.spike_queue, (spike_time, spike_event))
            
        except Exception as e:
            self.logger.error(f"触发脉冲失败: {e}")
    
    def get_active_spike_count(self) -> int:
        """获取活跃脉冲数量"""
        return len(self.spike_queue) + len(self.active_propagations)
    
    def get_network_activity(self) -> List[Dict[str, Any]]:
        """获取网络活动历史"""
        return list(self.network_activity)
    
    async def reset(self):
        """重置脉冲传播系统"""
        try:
            self.spike_events.clear()
            self.spike_queue.clear()
            self.completed_spikes.clear()
            self.active_propagations.clear()
            self.wave_fronts.clear()
            self.network_activity.clear()
            self.distance_cache.clear()
            self.path_cache.clear()
            
            # 重置统计信息
            self.stats.update({
                'total_spikes': 0,
                'active_spikes': 0,
                'average_propagation_time': 0.0,
                'network_firing_rate': 0.0,
                'active_connections': 0,
                'last_update_time': time.time()
            })
            
            self.logger.info("脉冲传播系统已重置")
            
        except Exception as e:
            self.logger.error(f"重置脉冲传播系统失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.reset()
            self.neurons.clear()
            self.connections.clear()
            self.synaptic_weights.clear()
            
            self.logger.info("脉冲传播系统资源清理完成")
            
        except Exception as e:
            self.logger.error(f"脉冲传播系统清理失败: {e}")


if __name__ == "__main__":
    # 示例用法
    async def test_spike_propagation():
        config = PropagationConfig()
        propagation = SpikePropagation(config)
        
        # 初始化
        init_data = await propagation.initialize_propagation()
        print(f"初始化数据: {init_data}")
        
        # 添加神经元
        propagation.add_neuron(1, [0, 0, 0], {2, 3})
        propagation.add_neuron(2, [1, 0, 0], {1, 4})
        propagation.add_neuron(3, [0, 1, 0], {1, 4})
        propagation.add_neuron(4, [1, 1, 0], {2, 3})
        
        # 添加连接
        propagation.add_connection(1, 2, 0.8)
        propagation.add_connection(1, 3, 0.6)
        propagation.add_connection(2, 4, 0.9)
        propagation.add_connection(3, 4, 0.7)
        
        # 触发脉冲
        propagation.trigger_spike(1)
        
        # 计算传播路径
        spike_data = {
            'source_neurons': [1],
            'spike_times': [time.time()]
        }
        
        paths = await propagation.calculate_paths(spike_data)
        print(f"传播路径: {paths}")
        
        # 动画化传播
        animations = await propagation.animate_propagation(paths)
        print(f"动画数据: {animations}")
        
        # 更新脉冲状态
        updates = await propagation.update_spikes()
        print(f"更新结果: {updates}")
        
        # 清理资源
        await propagation.cleanup()
    
    # 运行测试
    asyncio.run(test_spike_propagation())