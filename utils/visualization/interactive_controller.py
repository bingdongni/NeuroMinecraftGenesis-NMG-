# -*- coding: utf-8 -*-
"""
交互控制器
处理用户与3D脑网络的交互操作
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class InteractionConfig:
    """交互配置参数"""
    rotation_sensitivity: float = 1.0
    zoom_sensitivity: float = 0.1
    selection_radius: float = 1.0
    camera_distance: float = 10.0
    min_camera_distance: float = 2.0
    max_camera_distance: float = 100.0


class InteractiveController:
    """
    交互控制器类
    
    处理用户的视角控制、对象选择、数据过滤等交互操作
    """
    
    def __init__(self, config: InteractionConfig = None):
        """
        初始化交互控制器
        
        Args:
            config: 交互配置参数
        """
        self.config = config or InteractionConfig()
        self.logger = self._setup_logging()
        
        # 相机状态
        self.camera_position = np.array([0, 0, self.config.camera_distance])
        self.camera_rotation = np.array([0, 0, 0])  # pitch, yaw, roll
        self.target_position = np.array([0, 0, 0])
        
        # 交互状态
        self.selected_neurons = set()
        self.active_filters = {}
        self.interaction_history = []
        
        # 动画状态
        self.is_animating = False
        self.animation_start = None
        self.animation_duration = 0.5
        self.animation_from = None
        self.animation_to = None
        self.animation_type = None
        
        self.logger.info("交互控制器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('InteractiveController')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def initialize_controls(self):
        """
        初始化交互控制
        
        Returns:
            Dict[str, Any]: 初始控制配置
        """
        try:
            control_config = {
                'camera': {
                    'position': self.camera_position.tolist(),
                    'rotation': self.camera_rotation.tolist(),
                    'target': self.target_position.tolist(),
                    'distance': self.config.camera_distance
                },
                'interaction': {
                    'rotation_sensitivity': self.config.rotation_sensitivity,
                    'zoom_sensitivity': self.config.zoom_sensitivity,
                    'selection_radius': self.config.selection_radius
                },
                'filters': self.active_filters,
                'selected_neurons': list(self.selected_neurons)
            }
            
            self.logger.info("交互控制初始化完成")
            return control_config
            
        except Exception as e:
            self.logger.error(f"交互控制初始化失败: {e}")
            raise
    
    async def handle_rotation(self, rotation_data: Dict[str, Any]):
        """
        处理视角旋转
        
        Args:
            rotation_data: 旋转数据 {delta_x, delta_y, sensitivity}
            
        Returns:
            Dict[str, Any]: 旋转结果数据
        """
        try:
            delta_x = rotation_data.get('delta_x', 0)
            delta_y = rotation_data.get('delta_y', 0)
            sensitivity = rotation_data.get('sensitivity', self.config.rotation_sensitivity)
            
            # 应用旋转敏感性
            delta_x *= sensitivity
            delta_y *= sensitivity
            
            # 更新相机旋转 (pitch, yaw, roll)
            self.camera_rotation[0] += delta_y * 0.01  # pitch
            self.camera_rotation[1] += delta_x * 0.01  # yaw
            
            # 限制pitch角度在 [-π/2, π/2] 范围内
            self.camera_rotation[0] = max(-np.pi/2, min(np.pi/2, self.camera_rotation[0]))
            
            # 更新相机位置
            await self._update_camera_position()
            
            # 记录交互历史
            self._record_interaction('rotation', rotation_data)
            
            result = {
                'camera_position': self.camera_position.tolist(),
                'camera_rotation': self.camera_rotation.tolist(),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理旋转失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_zoom(self, zoom_data: Dict[str, Any]):
        """
        处理缩放操作
        
        Args:
            zoom_data: 缩放数据 {delta, sensitivity}
            
        Returns:
            Dict[str, Any]: 缩放结果数据
        """
        try:
            delta = zoom_data.get('delta', 0)
            sensitivity = zoom_data.get('sensitivity', self.config.zoom_sensitivity)
            
            # 计算新的相机距离
            distance_change = delta * sensitivity * self.config.camera_distance
            
            # 限制相机距离在有效范围内
            new_distance = np.linalg.norm(self.camera_position - self.target_position) + distance_change
            new_distance = max(self.config.min_camera_distance, 
                             min(self.config.max_camera_distance, new_distance))
            
            # 计算相机位置的方向向量
            direction = self.camera_position - self.target_position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                # 更新相机位置
                self.camera_position = self.target_position + direction * new_distance
            
            # 记录交互历史
            self._record_interaction('zoom', zoom_data)
            
            result = {
                'camera_position': self.camera_position.tolist(),
                'distance': np.linalg.norm(self.camera_position - self.target_position),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理缩放失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_neuron_selection(self, selection_data: Dict[str, Any]):
        """
        处理神经元选择
        
        Args:
            selection_data: 选择数据 {neuron_id, action, position}
            
        Returns:
            Dict[str, Any]: 选择结果数据
        """
        try:
            neuron_id = selection_data.get('neuron_id')
            action = selection_data.get('action', 'toggle')  # 'select', 'deselect', 'toggle'
            position = selection_data.get('position')
            
            if neuron_id is None:
                return {'success': False, 'error': '缺少神经元ID'}
            
            # 根据动作类型处理选择
            if action == 'select':
                self.selected_neurons.add(neuron_id)
            elif action == 'deselect':
                self.selected_neurons.discard(neuron_id)
            elif action == 'toggle':
                if neuron_id in self.selected_neurons:
                    self.selected_neurons.discard(neuron_id)
                else:
                    self.selected_neurons.add(neuron_id)
            elif action == 'clear':
                self.selected_neurons.clear()
            elif action == 'add_region':
                # 根据位置添加区域内的神经元
                if position:
                    nearby_neurons = await self._find_neurons_in_radius(position, self.config.selection_radius)
                    self.selected_neurons.update(nearby_neurons)
            
            # 记录交互历史
            self._record_interaction('neuron_selection', selection_data)
            
            result = {
                'selected_neurons': list(self.selected_neurons),
                'selection_count': len(self.selected_neurons),
                'action': action,
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理神经元选择失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_filtering(self, filter_data: Dict[str, Any]):
        """
        处理数据过滤
        
        Args:
            filter_data: 过滤数据 {filter_type, parameters, action}
            
        Returns:
            Dict[str, Any]: 过滤结果数据
        """
        try:
            filter_type = filter_data.get('filter_type')
            parameters = filter_data.get('parameters', {})
            action = filter_data.get('action', 'set')  # 'set', 'remove', 'clear'
            
            if filter_type is None:
                return {'success': False, 'error': '缺少过滤器类型'}
            
            # 根据动作处理过滤器
            if action == 'set':
                self.active_filters[filter_type] = parameters
            elif action == 'remove':
                self.active_filters.pop(filter_type, None)
            elif action == 'clear':
                self.active_filters.clear()
            
            # 记录交互历史
            self._record_interaction('filtering', filter_data)
            
            result = {
                'active_filters': self.active_filters,
                'filter_count': len(self.active_filters),
                'action': action,
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理过滤失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def focus_on_neuron(self, neuron_id: int, duration: float = 1.0):
        """
        聚焦到指定神经元
        
        Args:
            neuron_id: 神经元ID
            duration: 动画持续时间
            
        Returns:
            Dict[str, Any]: 聚焦结果数据
        """
        try:
            # 获取神经元位置（这里需要从网络数据中获取）
            neuron_position = await self._get_neuron_position(neuron_id)
            
            if neuron_position is None:
                return {'success': False, 'error': f'找不到神经元 {neuron_id}'}
            
            # 设置动画参数
            self.animation_from = self.camera_position.copy()
            self.animation_to = neuron_position + np.array([0, 0, self.config.camera_distance])
            self.animation_duration = duration
            self.animation_start = asyncio.get_event_loop().time()
            self.animation_type = 'focus'
            self.is_animating = True
            
            # 更新目标位置
            self.target_position = neuron_position
            
            result = {
                'neuron_id': neuron_id,
                'target_position': neuron_position.tolist(),
                'duration': duration,
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"聚焦到神经元失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def reset_camera(self, duration: float = 1.0):
        """
        重置相机到默认位置
        
        Args:
            duration: 动画持续时间
            
        Returns:
            Dict[str, Any]: 重置结果数据
        """
        try:
            # 设置动画参数
            self.animation_from = self.camera_position.copy()
            self.animation_to = np.array([0, 0, self.config.camera_distance])
            self.animation_duration = duration
            self.animation_start = asyncio.get_event_loop().time()
            self.animation_type = 'reset'
            self.is_animating = True
            
            # 重置目标位置和旋转
            self.target_position = np.array([0, 0, 0])
            self.camera_rotation = np.array([0, 0, 0])
            
            result = {
                'target_position': self.target_position.tolist(),
                'duration': duration,
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"重置相机失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def update_animation(self):
        """更新动画状态"""
        if not self.is_animating:
            return
        
        try:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self.animation_start
            progress = elapsed / self.animation_duration
            
            if progress >= 1.0:
                # 动画完成
                self.camera_position = self.animation_to.copy()
                self.is_animating = False
            else:
                # 应用缓动函数
                eased_progress = self._ease_in_out_cubic(progress)
                
                # 插值计算当前位置
                self.camera_position = (self.animation_from * (1 - eased_progress) + 
                                      self.animation_to * eased_progress)
            
        except Exception as e:
            self.logger.error(f"更新动画失败: {e}")
            self.is_animating = False
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """三次缓动函数"""
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
    
    async def _update_camera_position(self):
        """更新相机位置"""
        try:
            # 基于球坐标系计算相机位置
            pitch, yaw = self.camera_rotation[0], self.camera_rotation[1]
            distance = self.config.camera_distance
            
            x = distance * math.cos(pitch) * math.sin(yaw)
            y = distance * math.sin(pitch)
            z = distance * math.cos(pitch) * math.cos(yaw)
            
            self.camera_position = self.target_position + np.array([x, y, z])
            
        except Exception as e:
            self.logger.error(f"更新相机位置失败: {e}")
    
    async def _find_neurons_in_radius(self, center_position: List[float], radius: float) -> List[int]:
        """
        在指定半径内查找神经元
        
        Args:
            center_position: 中心位置
            radius: 搜索半径
            
        Returns:
            List[int]: 找到的神经元ID列表
        """
        try:
            # 这里需要根据实际的神经元位置数据来实现
            # 暂时返回空列表作为占位符
            return []
            
        except Exception as e:
            self.logger.error(f"查找神经元失败: {e}")
            return []
    
    async def _get_neuron_position(self, neuron_id: int) -> Optional[np.ndarray]:
        """
        获取神经元位置
        
        Args:
            neuron_id: 神经元ID
            
        Returns:
            Optional[np.ndarray]: 神经元位置，如果未找到则返回None
        """
        try:
            # 这里需要根据实际的神经元位置数据来实现
            # 暂时返回随机位置作为占位符
            return np.random.randn(3)
            
        except Exception as e:
            self.logger.error(f"获取神经元位置失败: {e}")
            return None
    
    def _record_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """记录交互历史"""
        try:
            interaction_record = {
                'type': interaction_type,
                'data': data,
                'timestamp': asyncio.get_event_loop().time(),
                'camera_position': self.camera_position.tolist(),
                'camera_rotation': self.camera_rotation.tolist()
            }
            
            self.interaction_history.append(interaction_record)
            
            # 限制历史记录数量
            if len(self.interaction_history) > 1000:
                self.interaction_history = self.interaction_history[-500:]
                
        except Exception as e:
            self.logger.error(f"记录交互历史失败: {e}")
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """获取交互历史"""
        return self.interaction_history.copy()
    
    def get_current_camera_state(self) -> Dict[str, Any]:
        """获取当前相机状态"""
        return {
            'position': self.camera_position.tolist(),
            'rotation': self.camera_rotation.tolist(),
            'target': self.target_position.tolist(),
            'distance': np.linalg.norm(self.camera_position - self.target_position),
            'is_animating': self.is_animating,
            'selected_neurons': list(self.selected_neurons),
            'active_filters': self.active_filters.copy()
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.selected_neurons.clear()
            self.active_filters.clear()
            self.interaction_history.clear()
            self.is_animating = False
            self.logger.info("交互控制器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"交互控制器清理失败: {e}")


if __name__ == "__main__":
    # 示例用法
    async def test_interactive_controller():
        controller = InteractiveController()
        
        # 测试旋转
        rotation_result = await controller.handle_rotation({
            'delta_x': 0.1,
            'delta_y': 0.1
        })
        print(f"旋转结果: {rotation_result}")
        
        # 测试缩放
        zoom_result = await controller.handle_zoom({
            'delta': 1.0
        })
        print(f"缩放结果: {zoom_result}")
        
        # 测试神经元选择
        selection_result = await controller.handle_neuron_selection({
            'neuron_id': 1,
            'action': 'select'
        })
        print(f"选择结果: {selection_result}")
        
        # 获取相机状态
        camera_state = controller.get_current_camera_state()
        print(f"相机状态: {camera_state}")
        
        # 清理资源
        await controller.cleanup()
    
    # 运行测试
    asyncio.run(test_interactive_controller())