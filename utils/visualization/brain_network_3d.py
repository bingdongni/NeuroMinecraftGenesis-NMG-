# -*- coding: utf-8 -*-
"""
Three.js 3D脑网络可视化主类
实现动态神经元放电动画和网络结构渲染
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import websockets
from flask import Flask, render_template, jsonify
import threading
import time

from .neuron_renderer import NeuronRenderer
from .spike_propagation import SpikePropagation
from .network_data_handler import NetworkDataHandler
from .interactive_controller import InteractiveController


@dataclass
class NetworkConfig:
    """脑网络配置参数"""
    neuron_count: int = 1000
    connection_density: float = 0.01
    neuron_radius: float = 0.1
    max_spike_rate: float = 100.0  # Hz
    propagation_speed: float = 2.0  # units/s
    animation_fps: int = 60
    websocket_port: int = 8765
    flask_port: int = 5000
    
    # 视觉效果参数
    particle_enabled: bool = True
    wave_visualization: bool = True
    glow_effects: bool = True
    lod_distance_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.lod_distance_thresholds is None:
            self.lod_distance_thresholds = [5.0, 10.0, 20.0, 50.0]


class BrainNetwork3D:
    """
    3D脑网络可视化主类
    
    负责协调所有组件，管理网络数据和实时渲染
    """
    
    def __init__(self, config: NetworkConfig = None):
        """
        初始化3D脑网络可视化系统
        
        Args:
            config: 网络配置参数
        """
        self.config = config or NetworkConfig()
        
        # 初始化组件
        # 为每个组件创建适当的配置对象
        from .neuron_renderer import NeuronRenderConfig
        from .spike_propagation import PropagationConfig
        from .network_data_handler import DataFormat, ProcessingConfig
        
        neuron_render_config = NeuronRenderConfig(
            neuron_radius=self.config.neuron_radius,
            particle_enabled=self.config.particle_enabled,
            lod_distance_thresholds=self.config.lod_distance_thresholds
        )
        
        propagation_config = PropagationConfig(
            propagation_speed=self.config.propagation_speed,
            max_propagation_distance=20.0,
            animation_fps=self.config.animation_fps,
            wave_visualization=self.config.wave_visualization,
            particle_effects=self.config.particle_enabled
        )
        
        data_format = DataFormat()
        processing_config = ProcessingConfig()
        
        # 初始化组件
        self.neuron_renderer = NeuronRenderer(neuron_render_config)
        self.spike_propagation = SpikePropagation(propagation_config)
        self.network_data_handler = NetworkDataHandler(data_format, processing_config)
        self.interactive_controller = InteractiveController()
        
        # 状态管理
        self.is_running = False
        self.is_paused = False
        self.current_frame = 0
        
        # Web服务
        self.flask_app = Flask(__name__)
        self.ws_server = None
        self.clients = set()
        
        # 性能统计
        self.frame_count = 0
        self.fps_counter = {'start_time': time.time(), 'frames': 0}
        
        # 设置日志
        self.logger = self._setup_logging()
        
        # 绑定路由
        self._setup_routes()
        
        self.logger.info("3D脑网络可视化系统初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('BrainNetwork3D')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.flask_app.route('/')
        def index():
            """主页，返回3D脑网络可视化页面"""
            return render_template('brain_network.html')
        
        @self.flask_app.route('/api/network-data')
        def get_network_data():
            """获取网络数据API"""
            try:
                data = self.network_data_handler.get_network_data()
                return jsonify({
                    'status': 'success',
                    'data': data
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.flask_app.route('/api/network-stats')
        def get_network_stats():
            """获取网络统计信息API"""
            try:
                stats = self.get_network_statistics()
                return jsonify({
                    'status': 'success',
                    'data': stats
                })
            except Exception as e:
                return jsonify({
                    'status': 'error', 
                    'message': str(e)
                }), 500
        
        @self.flask_app.route('/api/config', methods=['GET', 'POST'])
        def handle_config():
            """配置管理API"""
            if self.flask_app.request.method == 'GET':
                return jsonify({
                    'status': 'success',
                    'data': {
                        'neuron_count': self.config.neuron_count,
                        'connection_density': self.config.connection_density,
                        'neuron_radius': self.config.neuron_radius,
                        'max_spike_rate': self.config.max_spike_rate,
                        'propagation_speed': self.config.propagation_speed
                    }
                })
            elif self.flask_app.request.method == 'POST':
                try:
                    data = self.flask_app.request.get_json()
                    self._update_config(data)
                    return jsonify({'status': 'success'})
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    }), 400
    
    def _update_config(self, new_config: Dict[str, Any]):
        """更新网络配置"""
        if 'neuron_count' in new_config:
            self.config.neuron_count = new_config['neuron_count']
        if 'connection_density' in new_config:
            self.config.connection_density = new_config['connection_density']
        if 'neuron_radius' in new_config:
            self.config.neuron_radius = new_config['neuron_radius']
        if 'max_spike_rate' in new_config:
            self.config.max_spike_rate = new_config['max_spike_rate']
        if 'propagation_speed' in new_config:
            self.config.propagation_speed = new_config['propagation_speed']
        
        # 重新初始化相关组件
        self.neuron_renderer = NeuronRenderer(self.config)
        self.spike_propagation = SpikePropagation(self.config)
        self.network_data_handler = NetworkDataHandler(self.config)
        
        self.logger.info(f"配置已更新: {new_config}")
    
    async def initialize_scene(self):
        """
        初始化Three.js场景
        
        Returns:
            Dict[str, Any]: 场景初始化数据
        """
        try:
            self.logger.info("开始初始化3D场景...")
            
            # 初始化神经元渲染器
            scene_data = await self.neuron_renderer.initialize_scene()
            
            # 初始化脉冲传播系统
            propagation_data = await self.spike_propagation.initialize_propagation()
            
            # 初始化网络数据处理器
            network_data = await self.network_data_handler.initialize_network()
            
            # 初始化交互控制器
            interaction_data = await self.interactive_controller.initialize_controls()
            
            scene_config = {
                'scene': scene_data,
                'propagation': propagation_data,
                'network': network_data,
                'interaction': interaction_data,
                'config': {
                    'neuron_count': self.config.neuron_count,
                    'neuron_radius': self.config.neuron_radius,
                    'max_spike_rate': self.config.max_spike_rate,
                    'propagation_speed': self.config.propagation_speed,
                    'animation_fps': self.config.animation_fps
                }
            }
            
            self.logger.info("3D场景初始化完成")
            return scene_config
            
        except Exception as e:
            self.logger.error(f"场景初始化失败: {e}")
            raise
    
    async def render_network(self, network_data: Dict[str, Any]):
        """
        渲染脑网络结构
        
        Args:
            network_data: 网络数据
        """
        try:
            # 渲染神经元位置和连接
            render_data = await self.neuron_renderer.render_network(network_data)
            
            # 处理连接关系
            connections = await self.network_data_handler.process_connections(network_data)
            
            # 应用空间布局算法
            layout_data = await self.network_data_handler.apply_spatial_layout(connections)
            
            return {
                'neurons': render_data,
                'connections': connections,
                'layout': layout_data,
                'frame': self.current_frame
            }
            
        except Exception as e:
            self.logger.error(f"网络渲染失败: {e}")
            raise
    
    async def animate_spikes(self, spike_data: Dict[str, Any]):
        """
        动画脉冲传播效果
        
        Args:
            spike_data: 脉冲数据
        """
        try:
            # 计算脉冲传播路径
            propagation_paths = await self.spike_propagation.calculate_paths(spike_data)
            
            # 动画脉冲在网络中的传播
            animation_data = await self.spike_propagation.animate_propagation(
                propagation_paths
            )
            
            # 更新神经元颜色和状态
            neuron_updates = await self.neuron_renderer.update_neurons(
                spike_data, animation_data
            )
            
            return {
                'spikes': animation_data,
                'neuron_updates': neuron_updates,
                'wave_effects': await self.spike_propagation.create_wave_effects()
            }
            
        except Exception as e:
            self.logger.error(f"脉冲动画失败: {e}")
            raise
    
    async def update_network_data(self, new_data: Dict[str, Any]):
        """
        更新网络数据
        
        Args:
            new_data: 新的网络数据
        """
        try:
            # 更新神经元数据
            neuron_data = await self.network_data_handler.update_neurons(new_data)
            
            # 更新连接权重
            connection_updates = await self.network_data_handler.update_connections(new_data)
            
            # 重新计算布局（如果需要）
            if self.network_data_handler.should_recalculate_layout():
                layout_data = await self.network_data_handler.recalculate_layout()
            else:
                layout_data = None
            
            # 生成更新数据包
            update_packet = {
                'neurons': neuron_data,
                'connections': connection_updates,
                'layout': layout_data,
                'timestamp': time.time(),
                'frame': self.current_frame
            }
            
            # 向所有客户端广播更新
            await self.broadcast_update(update_packet)
            
            return update_packet
            
        except Exception as e:
            self.logger.error(f"网络数据更新失败: {e}")
            raise
    
    async def handle_interaction(self, interaction_data: Dict[str, Any]):
        """
        处理用户交互
        
        Args:
            interaction_data: 交互数据
        """
        try:
            interaction_type = interaction_data.get('type')
            
            if interaction_type == 'rotate':
                # 处理视角旋转
                rotation_data = await self.interactive_controller.handle_rotation(
                    interaction_data
                )
                return rotation_data
            
            elif interaction_type == 'zoom':
                # 处理缩放
                zoom_data = await self.interactive_controller.handle_zoom(
                    interaction_data
                )
                return zoom_data
            
            elif interaction_type == 'select_neuron':
                # 处理神经元选择
                selection_data = await self.interactive_controller.handle_neuron_selection(
                    interaction_data
                )
                return selection_data
            
            elif interaction_type == 'filter':
                # 处理数据过滤
                filter_data = await self.interactive_controller.handle_filtering(
                    interaction_data
                )
                return filter_data
            
            else:
                self.logger.warning(f"未知的交互类型: {interaction_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"交互处理失败: {e}")
            raise
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """向所有WebSocket客户端广播更新"""
        if self.clients:
            message = json.dumps({
                'type': 'network_update',
                'data': data,
                'timestamp': time.time()
            })
            
            # 批量发送以提高性能
            disconnected_clients = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    self.logger.warning(f"向客户端发送消息失败: {e}")
                    disconnected_clients.add(client)
            
            # 清理断开的连接
            self.clients.difference_update(disconnected_clients)
    
    async def handle_websocket(self, websocket, path):
        """处理WebSocket连接"""
        self.clients.add(websocket)
        self.logger.info(f"新客户端连接，当前连接数: {len(self.clients)}")
        
        try:
            # 发送初始网络数据
            initial_data = await self.network_data_handler.get_network_data()
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': initial_data
            }))
            
            # 持续处理客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    self.logger.warning(f"收到无效JSON消息: {message}")
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息失败: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.logger.info(f"客户端断开连接，当前连接数: {len(self.clients)}")
    
    async def _process_websocket_message(self, websocket, data: Dict[str, Any]):
        """处理WebSocket消息"""
        message_type = data.get('type')
        
        if message_type == 'get_network_data':
            network_data = await self.network_data_handler.get_network_data()
            await websocket.send(json.dumps({
                'type': 'network_data',
                'data': network_data
            }))
        
        elif message_type == 'interaction':
            interaction_result = await self.handle_interaction(data.get('data', {}))
            if interaction_result:
                await websocket.send(json.dumps({
                    'type': 'interaction_result',
                    'data': interaction_result
                }))
        
        elif message_type == 'update_config':
            self._update_config(data.get('config', {}))
            await websocket.send(json.dumps({
                'type': 'config_updated',
                'success': True
            }))
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        try:
            stats = {
                'neurons': self.network_data_handler.get_neuron_count(),
                'connections': self.network_data_handler.get_connection_count(),
                'active_spikes': self.spike_propagation.get_active_spike_count(),
                'frame_rate': self._calculate_fps(),
                'connected_clients': len(self.clients),
                'uptime': time.time() - self.fps_counter['start_time'],
                'current_frame': self.current_frame,
                'system_resources': self._get_system_resources()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取网络统计失败: {e}")
            return {}
    
    def _calculate_fps(self) -> float:
        """计算当前帧率"""
        self.frame_count += 1
        elapsed_time = time.time() - self.fps_counter['start_time']
        
        if elapsed_time >= 1.0:  # 每秒更新一次FPS
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_counter['start_time'] = time.time()
            return fps
        
        return 0.0
    
    def _get_system_resources(self) -> Dict[str, float]:
        """获取系统资源使用情况"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3)
            }
        except ImportError:
            return {}
    
    async def start(self):
        """启动3D脑网络可视化系统"""
        try:
            if self.is_running:
                self.logger.warning("系统已在运行中")
                return
            
            self.is_running = True
            self.logger.info("启动3D脑网络可视化系统...")
            
            # 启动WebSocket服务器
            ws_server = await websockets.serve(
                self.handle_websocket,
                "localhost",
                self.config.websocket_port
            )
            
            # 启动Flask应用（在后台线程中）
            flask_thread = threading.Thread(
                target=self._run_flask_app,
                daemon=True
            )
            flask_thread.start()
            
            # 启动主循环
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"启动系统失败: {e}")
            self.is_running = False
            raise
    
    def _run_flask_app(self):
        """运行Flask应用"""
        try:
            self.flask_app.run(
                host='0.0.0.0',
                port=self.config.flask_port,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"Flask应用运行失败: {e}")
    
    async def _main_loop(self):
        """主渲染循环"""
        try:
            frame_interval = 1.0 / self.config.animation_fps
            
            while self.is_running and not self.is_paused:
                start_time = time.time()
                
                # 更新当前帧
                self.current_frame += 1
                
                # 更新网络数据
                network_updates = await self.network_data_handler.get_latest_updates()
                if network_updates:
                    await self.broadcast_update(network_updates)
                
                # 更新脉冲传播
                spike_updates = await self.spike_propagation.update_spikes()
                if spike_updates:
                    await self.broadcast_update({
                        'spikes': spike_updates,
                        'frame': self.current_frame
                    })
                
                # 计算睡眠时间以维持目标帧率
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # 如果渲染时间超过帧间隔，记录警告
                    self.logger.debug(f"渲染时间超过帧间隔: {elapsed_time:.3f}s")
                    
        except asyncio.CancelledError:
            self.logger.info("主循环被取消")
        except Exception as e:
            self.logger.error(f"主循环错误: {e}")
        finally:
            self.is_running = False
    
    async def stop(self):
        """停止3D脑网络可视化系统"""
        try:
            self.logger.info("停止3D脑网络可视化系统...")
            self.is_running = False
            self.is_paused = False
            
            if self.ws_server:
                self.ws_server.close()
                await self.ws_server.wait_closed()
            
            # 清理资源
            await self._cleanup_resources()
            
            self.logger.info("系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止系统失败: {e}")
    
    async def _cleanup_resources(self):
        """清理资源"""
        try:
            # 清理WebSocket客户端
            self.clients.clear()
            
            # 清理组件资源
            await self.neuron_renderer.cleanup()
            await self.spike_propagation.cleanup()
            await self.network_data_handler.cleanup()
            await self.interactive_controller.cleanup()
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
    
    def pause(self):
        """暂停系统"""
        self.is_paused = True
        self.logger.info("系统已暂停")
    
    def resume(self):
        """恢复系统"""
        self.is_paused = False
        self.logger.info("系统已恢复")
    
    def reset(self):
        """重置系统"""
        self.current_frame = 0
        self.network_data_handler.reset()
        self.spike_propagation.reset()
        self.logger.info("系统已重置")


# 工厂函数
def create_brain_network(config: NetworkConfig = None) -> BrainNetwork3D:
    """
    创建3D脑网络可视化实例
    
    Args:
        config: 网络配置参数
        
    Returns:
        BrainNetwork3D: 3D脑网络可视化实例
    """
    return BrainNetwork3D(config)


# 便捷配置
DEFAULT_CONFIGS = {
    'small': NetworkConfig(
        neuron_count=100,
        connection_density=0.05,
        animation_fps=60
    ),
    'medium': NetworkConfig(
        neuron_count=1000,
        connection_density=0.01,
        animation_fps=60
    ),
    'large': NetworkConfig(
        neuron_count=5000,
        connection_density=0.005,
        animation_fps=30
    )
}


def create_brain_network_with_preset(preset: str = 'medium') -> BrainNetwork3D:
    """
    使用预设配置创建3D脑网络
    
    Args:
        preset: 预设名称 ('small', 'medium', 'large')
        
    Returns:
        BrainNetwork3D: 配置好的3D脑网络可视化实例
    """
    config = DEFAULT_CONFIGS.get(preset, DEFAULT_CONFIGS['medium'])
    return create_brain_network(config)


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建中等规模的网络
        brain_network = create_brain_network_with_preset('medium')
        
        try:
            # 启动系统
            await brain_network.start()
            
            # 保持运行
            while brain_network.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("收到中断信号，正在停止系统...")
        finally:
            await brain_network.stop()
    
    # 运行示例
    asyncio.run(main())