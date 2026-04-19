# -*- coding: utf-8 -*-
"""
神经元渲染器
负责3D场景中神经元的渲染、颜色更新和视觉效果
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
import json
import math
import colorsys


@dataclass
class NeuronRenderConfig:
    """神经元渲染配置"""
    neuron_radius: float = 0.1
    base_color: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # 蓝色
    spike_color: Tuple[float, float, float] = (1.0, 1.0, 0.0)  # 黄色
    connection_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # 灰色
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.1)  # 深蓝色背景
    
    # 动画参数
    spike_animation_duration: float = 0.5
    fade_in_duration: float = 0.3
    fade_out_duration: float = 0.3
    
    # 粒子效果
    particle_enabled: bool = True
    particle_count_per_spike: int = 10
    particle_lifetime: float = 1.0
    
    # LOD (Level of Detail) 参数
    lod_distance_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.lod_distance_thresholds is None:
            self.lod_distance_thresholds = [5.0, 10.0, 20.0, 50.0]


class NeuronRenderer:
    """
    神经元渲染器类
    
    负责处理神经元的3D渲染、颜色动画、粒子效果等视觉表现
    """
    
    def __init__(self, config: NeuronRenderConfig):
        """
        初始化神经元渲染器
        
        Args:
            config: 渲染配置参数
        """
        self.config = config
        
        # 渲染状态
        self.neurons = {}  # neuron_id -> neuron_data
        self.connections = {}  # connection_id -> connection_data
        self.materials = {}  # material_id -> material_data
        
        # 动画状态
        self.active_animations = {}
        self.spike_timeline = {}
        self.color_transitions = {}
        
        # 粒子系统
        self.particles = []
        self.particle_system_active = self.config.particle_enabled
        
        # LOD系统
        self.lod_levels = [0, 1, 2, 3]  # LOD级别
        self.lod_distances = self.config.lod_distance_thresholds
        
        # 性能优化
        self.render_distance = 100.0
        self.frustum_culling = True
        self.batch_update_size = 100
        
        # 统计信息
        self.render_stats = {
            'visible_neurons': 0,
            'total_neurons': 0,
            'visible_connections': 0,
            'total_connections': 0,
            'active_particles': 0
        }
        
        self.logger = self._setup_logging()
        self.logger.info("神经元渲染器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('NeuronRenderer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def initialize_scene(self):
        """
        初始化3D场景
        
        Returns:
            Dict[str, Any]: 场景初始化数据
        """
        try:
            scene_data = {
                'background_color': self.config.background_color,
                'lighting': await self._setup_lighting(),
                'materials': await self._setup_materials(),
                'geometry': await self._setup_base_geometry(),
                'particles': await self._initialize_particle_system()
            }
            
            self.logger.info("3D场景初始化完成")
            return scene_data
            
        except Exception as e:
            self.logger.error(f"场景初始化失败: {e}")
            raise
    
    async def _setup_lighting(self) -> Dict[str, Any]:
        """设置灯光配置"""
        try:
            lighting_config = {
                'ambient_light': {
                    'color': [0.3, 0.3, 0.4],
                    'intensity': 0.6
                },
                'directional_light': {
                    'color': [1.0, 0.95, 0.9],
                    'intensity': 0.8,
                    'position': [10, 10, 5]
                },
                'point_lights': [
                    {
                        'color': [0.7, 0.8, 1.0],
                        'intensity': 1.0,
                        'position': [0, 5, 0]
                    }
                ]
            }
            
            return lighting_config
            
        except Exception as e:
            self.logger.error(f"灯光设置失败: {e}")
            return {}
    
    async def _setup_materials(self) -> Dict[str, Any]:
        """设置材质配置"""
        try:
            base_color = list(self.config.base_color)
            spike_color = list(self.config.spike_color)
            connection_color = list(self.config.connection_color)
            
            materials = {
                'neuron_base': {
                    'type': 'MeshPhongMaterial',
                    'color': base_color,
                    'emissive': [0.0, 0.0, 0.2],
                    'shininess': 100
                },
                'neuron_spike': {
                    'type': 'MeshPhongMaterial',
                    'color': spike_color,
                    'emissive': [0.3, 0.3, 0.0],
                    'shininess': 150
                },
                'neuron_fade': {
                    'type': 'MeshPhongMaterial',
                    'color': base_color,
                    'transparent': True,
                    'opacity': 0.5,
                    'emissive': [0.1, 0.1, 0.3]
                },
                'connection': {
                    'type': 'LineBasicMaterial',
                    'color': connection_color,
                    'transparent': True,
                    'opacity': 0.3
                },
                'particle': {
                    'type': 'PointsMaterial',
                    'color': spike_color,
                    'size': 0.02,
                    'transparent': True,
                    'opacity': 0.8
                }
            }
            
            # 保存材质到内部状态
            self.materials = materials
            
            return materials
            
        except Exception as e:
            self.logger.error(f"材质设置失败: {e}")
            return {}
    
    async def _setup_base_geometry(self) -> Dict[str, Any]:
        """设置基础几何体"""
        try:
            geometry = {
                'neuron_sphere': {
                    'type': 'SphereGeometry',
                    'radius': self.config.neuron_radius,
                    'width_segments': 16,
                    'height_segments': 12
                },
                'connection_cylinder': {
                    'type': 'CylinderGeometry',
                    'radius_top': 0.005,
                    'radius_bottom': 0.005,
                    'height': 1.0,
                    'radial_segments': 8
                }
            }
            
            return geometry
            
        except Exception as e:
            self.logger.error(f"几何体设置失败: {e}")
            return {}
    
    async def _initialize_particle_system(self) -> Dict[str, Any]:
        """初始化粒子系统"""
        try:
            if not self.config.particle_enabled:
                return {}
            
            particle_config = {
                'enabled': self.config.particle_enabled,
                'max_particles': self.config.particle_count_per_spike * 100,
                'particle_lifetime': self.config.particle_lifetime,
                'emission_rate': 50,  # particles per second
                'particle_lifecycle': {
                    'birth': {
                        'size_range': [0.005, 0.02],
                        'color_range': [
                            self.config.spike_color,
                            self.config.base_color
                        ]
                    },
                    'life': {
                        'fade_in': self.config.fade_in_duration,
                        'fade_out': self.config.fade_out_duration
                    },
                    'death': {
                        'size_shrink': 0.5
                    }
                }
            }
            
            return particle_config
            
        except Exception as e:
            self.logger.error(f"粒子系统初始化失败: {e}")
            return {}
    
    async def render_network(self, network_data: Dict[str, Any]):
        """
        渲染脑网络结构
        
        Args:
            network_data: 网络数据
            
        Returns:
            Dict[str, Any]: 渲染数据
        """
        try:
            neurons_data = network_data.get('neurons', [])
            connections_data = network_data.get('connections', [])
            
            # 渲染神经元
            neuron_render_data = await self._render_neurons(neurons_data)
            
            # 渲染连接
            connection_render_data = await self._render_connections(connections_data)
            
            # 应用LOD优化
            lod_data = await self._apply_lod_optimization()
            
            # 更新统计信息
            self._update_render_stats(neuron_render_data, connection_render_data)
            
            result = {
                'neurons': neuron_render_data,
                'connections': connection_render_data,
                'lod': lod_data,
                'statistics': self.render_stats
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"网络渲染失败: {e}")
            raise
    
    async def _render_neurons(self, neurons_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """渲染神经元"""
        try:
            render_data = {
                'instances': [],
                'colors': [],
                'positions': [],
                'animations': [],
                'lod_levels': []
            }
            
            for neuron_data in neurons_data:
                neuron_id = neuron_data.get('id')
                position = neuron_data.get('position', [0, 0, 0])
                activity = neuron_data.get('activity', 0.0)
                
                # 根据活动值确定颜色和动画
                color_data = await self._calculate_neuron_color(neuron_id, activity)
                
                # 添加到渲染数据
                render_data['instances'].append({
                    'id': neuron_id,
                    'position': position,
                    'material_id': 'neuron_base' if activity < 0.5 else 'neuron_spike'
                })
                
                render_data['colors'].append(color_data['color'])
                render_data['positions'].append(position)
                
                # 设置LOD级别
                lod_level = await self._calculate_lod_level(position)
                render_data['lod_levels'].append(lod_level)
                
                # 如果有活动，设置动画
                if activity > 0.8:
                    animation_data = await self._create_spike_animation(neuron_id, activity)
                    render_data['animations'].append(animation_data)
            
            return render_data
            
        except Exception as e:
            self.logger.error(f"神经元渲染失败: {e}")
            return {}
    
    async def _render_connections(self, connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """渲染连接"""
        try:
            render_data = {
                'lines': [],
                'weight_colors': [],
                'opacity_values': []
            }
            
            for connection_data in connections_data:
                from_id = connection_data.get('from')
                to_id = connection_data.get('to')
                weight = connection_data.get('weight', 0.0)
                
                # 获取神经元位置
                from_pos = self._get_neuron_position(from_id)
                to_pos = self._get_neuron_position(to_id)
                
                if from_pos and to_pos:
                    # 根据权重计算颜色和透明度
                    color_data = await self._calculate_connection_color(weight)
                    opacity = max(0.1, abs(weight))
                    
                    render_data['lines'].append({
                        'from': from_pos,
                        'to': to_pos,
                        'weight': weight,
                        'material_id': 'connection'
                    })
                    
                    render_data['weight_colors'].append(color_data)
                    render_data['opacity_values'].append(opacity)
            
            return render_data
            
        except Exception as e:
            self.logger.error(f"连接渲染失败: {e}")
            return {}
    
    async def _calculate_neuron_color(self, neuron_id: int, activity: float) -> Dict[str, Any]:
        """
        计算神经元颜色
        
        Args:
            neuron_id: 神经元ID
            activity: 活动值 [0, 1]
            
        Returns:
            Dict[str, Any]: 颜色数据
        """
        try:
            # 基于活动值在基础颜色和脉冲颜色之间插值
            base_color = np.array(self.config.base_color)
            spike_color = np.array(self.config.spike_color)
            
            # 活动的缓动函数，让颜色变化更自然
            eased_activity = self._ease_out_quart(min(1.0, activity))
            
            # 颜色插值
            mixed_color = base_color * (1 - eased_activity) + spike_color * eased_activity
            
            # 添加发光效果
            emissive_intensity = eased_activity * 0.3
            emissive_color = spike_color * emissive_intensity
            
            return {
                'color': mixed_color.tolist(),
                'emissive': emissive_color.tolist(),
                'activity': activity,
                'pulse_intensity': eased_activity
            }
            
        except Exception as e:
            self.logger.error(f"计算神经元颜色失败: {e}")
            return {'color': self.config.base_color, 'emissive': [0, 0, 0], 'activity': 0}
    
    def _ease_out_quart(self, t: float) -> float:
        """四次缓动函数"""
        return 1 - pow(1 - t, 4)
    
    async def _calculate_connection_color(self, weight: float) -> List[float]:
        """
        计算连接颜色
        
        Args:
            weight: 连接权重
            
        Returns:
            List[float]: RGB颜色值
        """
        try:
            # 权重归一化到 [0, 1]
            normalized_weight = (weight + 1.0) / 2.0
            
            # 根据权重映射到颜色
            if weight > 0:
                # 正权重：蓝色到绿色
                return [
                    0.0,
                    normalized_weight,
                    1.0 - normalized_weight * 0.5
                ]
            else:
                # 负权重：红色到灰色
                intensity = abs(normalized_weight)
                return [
                    intensity,
                    0.2 * (1 - intensity),
                    0.2 * (1 - intensity)
                ]
                
        except Exception as e:
            self.logger.error(f"计算连接颜色失败: {e}")
            return list(self.config.connection_color)
    
    async def _calculate_lod_level(self, position: List[float]) -> int:
        """
        计算LOD级别
        
        Args:
            position: 位置坐标
            
        Returns:
            int: LOD级别 (0=最高质量, 3=最低质量)
        """
        try:
            # 计算到相机的距离（简化版本）
            distance = np.linalg.norm(np.array(position))
            
            # 根据距离确定LOD级别
            for i, threshold in enumerate(self.lod_distances):
                if distance <= threshold:
                    return i
            
            return len(self.lod_distances)  # 最远距离
            
        except Exception as e:
            self.logger.error(f"计算LOD级别失败: {e}")
            return 0
    
    async def _apply_lod_optimization(self) -> Dict[str, Any]:
        """应用LOD优化"""
        try:
            lod_config = {
                'enabled': True,
                'distance_thresholds': self.lod_distances,
                'lod_rules': {
                    0: {'detail_level': 'high', 'particles': True, 'glow': True},
                    1: {'detail_level': 'medium', 'particles': True, 'glow': False},
                    2: {'detail_level': 'low', 'particles': False, 'glow': False},
                    3: {'detail_level': 'minimal', 'particles': False, 'glow': False}
                }
            }
            
            return lod_config
            
        except Exception as e:
            self.logger.error(f"LOD优化失败: {e}")
            return {}
    
    async def _create_spike_animation(self, neuron_id: int, activity: float) -> Dict[str, Any]:
        """
        创建脉冲动画
        
        Args:
            neuron_id: 神经元ID
            activity: 活动值
            
        Returns:
            Dict[str, Any]: 动画数据
        """
        try:
            animation_id = f"spike_{neuron_id}_{asyncio.get_event_loop().time()}"
            
            animation_data = {
                'id': animation_id,
                'type': 'spike',
                'neuron_id': neuron_id,
                'duration': self.config.spike_animation_duration,
                'activity': activity,
                'keyframes': [
                    {
                        'time': 0.0,
                        'scale': [1.0, 1.0, 1.0],
                        'color': self.config.base_color,
                        'intensity': 0.0
                    },
                    {
                        'time': 0.1,
                        'scale': [1.5, 1.5, 1.5],
                        'color': self.config.spike_color,
                        'intensity': 1.0
                    },
                    {
                        'time': 0.5,
                        'scale': [1.2, 1.2, 1.2],
                        'color': self.config.spike_color,
                        'intensity': 0.7
                    },
                    {
                        'time': 1.0,
                        'scale': [1.0, 1.0, 1.0],
                        'color': self.config.base_color,
                        'intensity': 0.0
                    }
                ]
            }
            
            # 添加到活跃动画列表
            self.active_animations[animation_id] = animation_data
            
            return animation_data
            
        except Exception as e:
            self.logger.error(f"创建脉冲动画失败: {e}")
            return {}
    
    def _get_neuron_position(self, neuron_id: int) -> Optional[List[float]]:
        """获取神经元位置"""
        neuron_data = self.neurons.get(neuron_id)
        return neuron_data.get('position') if neuron_data else None
    
    def _update_render_stats(self, neuron_data: Dict[str, Any], connection_data: Dict[str, Any]):
        """更新渲染统计信息"""
        try:
            self.render_stats.update({
                'total_neurons': len(neuron_data.get('instances', [])),
                'visible_neurons': len(neuron_data.get('instances', [])),
                'total_connections': len(connection_data.get('lines', [])),
                'visible_connections': len(connection_data.get('lines', [])),
                'active_particles': len(self.particles)
            })
            
        except Exception as e:
            self.logger.error(f"更新渲染统计失败: {e}")
    
    async def update_neurons(self, spike_data: Dict[str, Any], animation_data: Dict[str, Any]):
        """
        更新神经元状态和动画
        
        Args:
            spike_data: 脉冲数据
            animation_data: 动画数据
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        try:
            updates = {
                'color_updates': [],
                'position_updates': [],
                'animation_updates': [],
                'particle_updates': []
            }
            
            # 更新脉冲触发的神经元
            spiking_neurons = spike_data.get('spiking_neurons', [])
            for neuron_id in spiking_neurons:
                color_update = await self._create_color_transition(neuron_id, 'spike')
                updates['color_updates'].append(color_update)
            
            # 更新动画
            active_animations = animation_data.get('animations', [])
            for anim_data in active_animations:
                animation_update = await self._update_animation(anim_data)
                updates['animation_updates'].append(animation_update)
            
            # 清理已完成的动画
            await self._cleanup_completed_animations()
            
            # 更新粒子系统
            if self.particle_system_active:
                particle_updates = await self._update_particle_system()
                updates['particle_updates'] = particle_updates
            
            return updates
            
        except Exception as e:
            self.logger.error(f"更新神经元失败: {e}")
            return {}
    
    async def _create_color_transition(self, neuron_id: int, transition_type: str) -> Dict[str, Any]:
        """创建颜色过渡动画"""
        try:
            transition_id = f"color_{neuron_id}_{transition_type}"
            
            if transition_type == 'spike':
                # 脉冲颜色过渡
                transition_data = {
                    'id': transition_id,
                    'neuron_id': neuron_id,
                    'from_color': self.config.base_color,
                    'to_color': self.config.spike_color,
                    'duration': self.config.spike_animation_duration,
                    'start_time': asyncio.get_event_loop().time(),
                    'easing': 'ease_out_quart'
                }
                
                self.color_transitions[transition_id] = transition_data
                return transition_data
            
            return {}
            
        except Exception as e:
            self.logger.error(f"创建颜色过渡失败: {e}")
            return {}
    
    async def _update_animation(self, anim_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新动画状态"""
        try:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - anim_data.get('start_time', current_time)
            duration = anim_data.get('duration', 1.0)
            
            progress = min(1.0, elapsed / duration)
            
            # 更新动画状态
            return {
                'animation_id': anim_data.get('id'),
                'progress': progress,
                'current_frame': int(progress * len(anim_data.get('keyframes', []))),
                'status': 'completed' if progress >= 1.0 else 'running'
            }
            
        except Exception as e:
            self.logger.error(f"更新动画失败: {e}")
            return {}
    
    async def _cleanup_completed_animations(self):
        """清理已完成的动画"""
        try:
            completed_animations = []
            current_time = asyncio.get_event_loop().time()
            
            for anim_id, anim_data in self.active_animations.items():
                start_time = anim_data.get('start_time', current_time)
                duration = anim_data.get('duration', 1.0)
                
                if current_time - start_time >= duration:
                    completed_animations.append(anim_id)
            
            # 移除已完成的动画
            for anim_id in completed_animations:
                self.active_animations.pop(anim_id, None)
                self.color_transitions.pop(anim_id, None)
                
        except Exception as e:
            self.logger.error(f"清理动画失败: {e}")
    
    async def _update_particle_system(self) -> List[Dict[str, Any]]:
        """更新粒子系统"""
        try:
            if not self.particle_system_active:
                return []
            
            particle_updates = []
            
            # 更新现有粒子
            for particle in self.particles:
                particle['age'] += 0.016  # 假设60FPS
                
                if particle['age'] >= particle['lifetime']:
                    # 粒子死亡
                    particle_updates.append({
                        'action': 'remove',
                        'particle_id': particle['id']
                    })
                else:
                    # 更新粒子位置和属性
                    progress = particle['age'] / particle['lifetime']
                    particle_updates.append({
                        'action': 'update',
                        'particle_id': particle['id'],
                        'position': particle['position'],
                        'progress': progress,
                        'opacity': 1.0 - progress
                    })
            
            # 清理过期粒子
            self.particles = [p for p in self.particles if p['age'] < p['lifetime']]
            
            return particle_updates
            
        except Exception as e:
            self.logger.error(f"更新粒子系统失败: {e}")
            return []
    
    async def create_wave_effects(self) -> Dict[str, Any]:
        """创建波纹效果"""
        try:
            wave_effects = {
                'radial_waves': [],
                'connection_waves': [],
                'global_oscillation': {
                    'enabled': True,
                    'frequency': 0.5,
                    'amplitude': 0.1,
                    'phase': 0.0
                }
            }
            
            # 基于当前活跃的脉冲创建径向波纹
            current_time = asyncio.get_event_loop().time()
            for anim_id, anim_data in self.active_animations.items():
                if anim_data.get('type') == 'spike':
                    neuron_id = anim_data.get('neuron_id')
                    position = self._get_neuron_position(neuron_id)
                    
                    if position:
                        wave_data = {
                            'center': position,
                            'start_time': current_time,
                            'speed': 2.0,
                            'max_radius': 10.0,
                            'decay': 0.5
                        }
                        wave_effects['radial_waves'].append(wave_data)
            
            return wave_effects
            
        except Exception as e:
            self.logger.error(f"创建波纹效果失败: {e}")
            return {}
    
    async def cleanup(self):
        """清理渲染器资源"""
        try:
            self.neurons.clear()
            self.connections.clear()
            self.materials.clear()
            self.active_animations.clear()
            self.spike_timeline.clear()
            self.color_transitions.clear()
            self.particles.clear()
            
            self.logger.info("神经元渲染器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"渲染器清理失败: {e}")


if __name__ == "__main__":
    # 示例用法
    async def test_neuron_renderer():
        config = NeuronRenderConfig()
        renderer = NeuronRenderer(config)
        
        # 初始化场景
        scene_data = await renderer.initialize_scene()
        print(f"场景数据: {json.dumps(scene_data, indent=2, default=str)}")
        
        # 测试网络数据
        test_network_data = {
            'neurons': [
                {'id': 1, 'position': [0, 0, 0], 'activity': 0.5},
                {'id': 2, 'position': [1, 0, 0], 'activity': 0.8},
                {'id': 3, 'position': [0, 1, 0], 'activity': 1.0}
            ],
            'connections': [
                {'from': 1, 'to': 2, 'weight': 0.5},
                {'from': 2, 'to': 3, 'weight': -0.3}
            ]
        }
        
        # 渲染网络
        render_data = await renderer.render_network(test_network_data)
        print(f"渲染数据: {json.dumps(render_data, indent=2, default=str)}")
        
        # 测试脉冲数据更新
        spike_data = {
            'spiking_neurons': [2, 3],
            'timestamp': asyncio.get_event_loop().time()
        }
        
        animation_data = {'animations': []}
        update_result = await renderer.update_neurons(spike_data, animation_data)
        print(f"更新结果: {json.dumps(update_result, indent=2, default=str)}")
        
        # 创建波纹效果
        wave_effects = await renderer.create_wave_effects()
        print(f"波纹效果: {json.dumps(wave_effects, indent=2, default=str)}")
        
        # 清理资源
        await renderer.cleanup()
    
    # 运行测试
    asyncio.run(test_neuron_renderer())