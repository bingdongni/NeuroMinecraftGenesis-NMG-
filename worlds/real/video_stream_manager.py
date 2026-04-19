"""
USB摄像头场景捕获系统 - 视频流管理模块

该模块实现视频流的传输、管理和优化，包括流媒体服务器、
网络传输协议、缓冲管理和性能监控等功能。

作者: AI助手
创建时间: 2025-11-13
版本: 1.0
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
import socket
import json
import zlib
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor


class StreamQuality(Enum):
    """流质量枚举"""
    LOW = "低质量"    # 320x240, 15fps
    MEDIUM = "中等质量"  # 640x480, 20fps
    HIGH = "高质量"    # 1280x720, 30fps
    ULTRA = "超高质量"  # 1920x1080, 30fps


class StreamProtocol(Enum):
    """传输协议枚举"""
    TCP = "TCP"
    UDP = "UDP"
    WEBSOCKET = "WebSocket"
    HTTP = "HTTP"


@dataclass
class StreamConfig:
    """流配置参数"""
    quality: StreamQuality = StreamQuality.MEDIUM
    protocol: StreamProtocol = StreamProtocol.TCP
    port: int = 8080
    max_clients: int = 10
    buffer_size: int = 30
    compression: bool = True
    compression_level: int = 6
    fps_limit: int = 30
    bitrate_limit: int = 2000000  # 2Mbps
    timeout: int = 30


@dataclass
class StreamMetrics:
    """流指标数据结构"""
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    frame_count: int = 0
    dropped_frames: int = 0
    avg_frame_time: float = 0.0
    network_latency: float = 0.0
    client_count: int = 0
    bitrate: float = 0.0


class VideoStreamManager:
    """
    视频流管理器
    
    该类负责：
    1. 视频流的编码和压缩
    2. 多客户端连接管理
    3. 网络传输协议实现
    4. 流质量控制和优化
    5. 性能监控和统计
    6. 错误处理和恢复
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """初始化视频流管理器"""
        self.config = config or StreamConfig()
        self.logger = logging.getLogger(__name__)
        
        # 流状态
        self.is_streaming = False
        self.server_socket = None
        self.clients = {}
        self.client_lock = threading.Lock()
        
        # 缓冲队列
        self.frame_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.client_queues = {}
        
        # 线程管理
        self.stream_threads = []
        self.processing_threads = []
        
        # 指标统计
        self.metrics = StreamMetrics()
        self.metrics_history = []
        self.max_history = 100
        
        # 编解码器
        self.jpeg_quality = 85
        self.encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
        ]
        
        # 性能监控
        self.last_frame_time = time.time()
        self.frame_times = []
        
        # 回调函数
        self.frame_callbacks = []
        self.client_callbacks = {}
        
        # 压缩器
        self.compressor = None
        if self.config.compression:
            self.compressor = zlib.compressobj(self.config.compression_level)
    
    def stream_video(self, frame: np.ndarray) -> bool:
        """
        流式传输视频帧
        
        Args:
            frame: 要传输的视频帧
            
        Returns:
            bool: 传输是否成功
        """
        if not self.is_streaming:
            return False
        
        start_time = time.time()
        
        try:
            # 帧预处理
            processed_frame = self._preprocess_frame(frame)
            
            # 编码帧
            encoded_frame = self._encode_frame(processed_frame)
            if encoded_frame is None:
                return False
            
            # 压缩帧
            if self.compressor:
                encoded_frame = self.compressor.compress(encoded_frame)
                encoded_frame += self.compressor.flush(zlib.Z_SYNC_FLUSH)
            
            # 计算流质量
            current_quality = self._calculate_current_quality(encoded_frame)
            
            # 分发到客户端
            success = self._distribute_to_clients(encoded_frame, current_quality)
            
            # 更新指标
            self._update_metrics(encoded_frame, time.time() - start_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"视频流传输失败: {e}")
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理视频帧"""
        if frame is None:
            return None
        
        # 根据质量设置调整分辨率
        target_width, target_height = self._get_target_resolution()
        current_height, current_width = frame.shape[:2]
        
        if current_width != target_width or current_height != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # 帧率控制
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        target_frame_time = 1.0 / self.config.fps_limit
        
        if time_since_last < target_frame_time:
            # 跳过这一帧以保持目标帧率
            return None
        
        self.last_frame_time = current_time
        
        return frame
    
    def _get_target_resolution(self) -> tuple:
        """获取目标分辨率"""
        quality_map = {
            StreamQuality.LOW: (320, 240),
            StreamQuality.MEDIUM: (640, 480),
            StreamQuality.HIGH: (1280, 720),
            StreamQuality.ULTRA: (1920, 1080)
        }
        return quality_map.get(self.config.quality, (640, 480))
    
    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """编码视频帧"""
        try:
            # 动态调整JPEG质量
            target_size = self.config.bitrate_limit // (self.config.fps_limit * 8)
            current_size = len(cv2.imencode('.jpg', frame, self.encode_params)[1].tobytes())
            
            if current_size > target_size:
                # 降低质量
                self.jpeg_quality = max(30, self.jpeg_quality - 5)
            elif current_size < target_size * 0.8:
                # 提高质量
                self.jpeg_quality = min(95, self.jpeg_quality + 1)
            
            # 更新编码参数
            self.encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
            
            # 编码
            ret, encoded = cv2.imencode('.jpg', frame, self.encode_params)
            return encoded.tobytes() if ret else None
            
        except Exception as e:
            self.logger.error(f"帧编码失败: {e}")
            return None
    
    def _calculate_current_quality(self, encoded_frame: bytes) -> float:
        """计算当前流质量"""
        frame_size = len(encoded_frame)
        target_bitrate = self.config.bitrate_limit / 8  # bytes per second
        quality_score = min(1.0, target_bitrate / frame_size)
        
        return quality_score
    
    def _distribute_to_clients(self, encoded_frame: bytes, quality: float) -> bool:
        """分发帧到客户端"""
        success_count = 0
        
        with self.client_lock:
            client_ids = list(self.clients.keys())
        
        for client_id in client_ids:
            try:
                success = self._send_to_client(client_id, encoded_frame, quality)
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"发送到客户端 {client_id} 失败: {e}")
                self._remove_client(client_id)
        
        return success_count > 0
    
    def _send_to_client(self, client_id: str, encoded_frame: bytes, quality: float) -> bool:
        """发送帧到指定客户端"""
        client_info = self.clients.get(client_id)
        if not client_info:
            return False
        
        try:
            # 创建帧数据包
            frame_data = {
                'type': 'frame',
                'timestamp': time.time(),
                'size': len(encoded_frame),
                'quality': quality,
                'data': base64.b64encode(encoded_frame).decode('utf-8') if self.config.compression else encoded_frame.hex()
            }
            
            serialized_data = json.dumps(frame_data).encode('utf-8')
            
            # 根据协议发送
            if self.config.protocol == StreamProtocol.TCP:
                return self._send_tcp(client_info, serialized_data)
            elif self.config.protocol == StreamProtocol.UDP:
                return self._send_udp(client_info, serialized_data)
            elif self.config.protocol == StreamProtocol.WEBSOCKET:
                return self._send_websocket(client_info, serialized_data)
            
        except Exception as e:
            self.logger.error(f"发送帧到客户端 {client_id} 失败: {e}")
            return False
        
        return True
    
    def _send_tcp(self, client_info: dict, data: bytes) -> bool:
        """通过TCP发送数据"""
        try:
            client_socket = client_info.get('socket')
            if not client_socket:
                return False
            
            # 添加长度前缀
            data_length = len(data).to_bytes(4, 'big')
            client_socket.sendall(data_length + data)
            return True
            
        except Exception as e:
            self.logger.debug(f"TCP发送失败: {e}")
            return False
    
    def _send_udp(self, client_info: dict, data: bytes) -> bool:
        """通过UDP发送数据"""
        try:
            client_socket = client_info.get('socket')
            client_address = client_info.get('address')
            if not client_socket or not client_address:
                return False
            
            # UDP数据包大小限制
            max_packet_size = 65507
            if len(data) <= max_packet_size:
                client_socket.sendto(data, client_address)
                return True
            else:
                # 分片发送
                return self._send_udp_fragments(client_socket, client_address, data)
                
        except Exception as e:
            self.logger.debug(f"UDP发送失败: {e}")
            return False
    
    def _send_udp_fragments(self, socket, address: tuple, data: bytes) -> bool:
        """分片UDP发送"""
        try:
            max_payload = 1000  # 为头部预留空间
            fragment_id = int(time.time() * 1000)  # 时间戳作为片段ID
            total_fragments = (len(data) + max_payload - 1) // max_payload
            
            for i in range(total_fragments):
                start = i * max_payload
                end = min(start + max_payload, len(data))
                fragment_data = data[start:end]
                
                # 添加分片头部
                header = {
                    'fragment_id': fragment_id,
                    'fragment_index': i,
                    'total_fragments': total_fragments,
                    'payload': fragment_data.hex()
                }
                fragment_packet = json.dumps(header).encode('utf-8')
                
                socket.sendto(fragment_packet, address)
                
                # 短暂延迟避免网络拥塞
                time.sleep(0.001)
            
            return True
            
        except Exception as e:
            self.logger.debug(f"UDP分片发送失败: {e}")
            return False
    
    def _send_websocket(self, client_info: dict, data: bytes) -> bool:
        """通过WebSocket发送数据"""
        # WebSocket发送实现
        try:
            websocket = client_info.get('websocket')
            if websocket:
                asyncio.run_coroutine_threadsafe(
                    websocket.send(data),
                    asyncio.get_event_loop()
                )
                return True
        except Exception as e:
            self.logger.debug(f"WebSocket发送失败: {e}")
        
        return False
    
    def start_server(self) -> bool:
        """启动流媒体服务器"""
        try:
            self.is_streaming = True
            
            # 根据协议启动服务器
            if self.config.protocol == StreamProtocol.TCP:
                return self._start_tcp_server()
            elif self.config.protocol == StreamProtocol.UDP:
                return self._start_udp_server()
            elif self.config.protocol == StreamProtocol.WEBSOCKET:
                return self._start_websocket_server()
            else:
                self.logger.error(f"不支持的协议: {self.config.protocol}")
                return False
                
        except Exception as e:
            self.logger.error(f"启动服务器失败: {e}")
            self.is_streaming = False
            return False
    
    def _start_tcp_server(self) -> bool:
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.config.port))
            self.server_socket.listen(self.config.max_clients)
            self.server_socket.settimeout(1.0)
            
            # 启动客户端接受线程
            accept_thread = threading.Thread(target=self._accept_tcp_clients)
            accept_thread.daemon = True
            accept_thread.start()
            
            self.logger.info(f"TCP服务器启动在端口 {self.config.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"TCP服务器启动失败: {e}")
            return False
    
    def _start_udp_server(self) -> bool:
        """启动UDP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind(('0.0.0.0', self.config.port))
            
            # 启动UDP客户端监听线程
            listen_thread = threading.Thread(target=self._listen_udp_clients)
            listen_thread.daemon = True
            listen_thread.start()
            
            self.logger.info(f"UDP服务器启动在端口 {self.config.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"UDP服务器启动失败: {e}")
            return False
    
    def _start_websocket_server(self) -> bool:
        """启动WebSocket服务器"""
        # 简化的WebSocket服务器实现
        try:
            # 这里可以集成websockets库或aiohttp
            self.logger.info("WebSocket服务器功能需要额外的库支持")
            return False
        except Exception as e:
            self.logger.error(f"WebSocket服务器启动失败: {e}")
            return False
    
    def _accept_tcp_clients(self):
        """接受TCP客户端连接"""
        while self.is_streaming:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_id = f"{client_address[0]}:{client_address[1]}"
                
                with self.client_lock:
                    self.clients[client_id] = {
                        'socket': client_socket,
                        'address': client_address,
                        'connected_time': time.time(),
                        'bytes_sent': 0
                    }
                
                self.logger.info(f"新TCP客户端连接: {client_id}")
                
                # 启动客户端处理线程
                client_thread = threading.Thread(
                    target=self._handle_tcp_client,
                    args=(client_id,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_streaming:
                    self.logger.error(f"接受TCP客户端失败: {e}")
    
    def _listen_udp_clients(self):
        """监听UDP客户端"""
        while self.is_streaming:
            try:
                data, client_address = self.server_socket.recvfrom(4096)
                client_id = f"{client_address[0]}:{client_address[1]}"
                
                # 注册或更新UDP客户端
                with self.client_lock:
                    if client_id not in self.clients:
                        self.clients[client_id] = {
                            'address': client_address,
                            'connected_time': time.time(),
                            'last_seen': time.time()
                        }
                        self.logger.info(f"新UDP客户端: {client_id}")
                    else:
                        self.clients[client_id]['last_seen'] = time.time()
                
                # 处理客户端消息（如果需要）
                self._handle_udp_message(client_id, data)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_streaming:
                    self.logger.error(f"UDP监听失败: {e}")
    
    def _handle_tcp_client(self, client_id: str):
        """处理TCP客户端"""
        try:
            client_info = self.clients.get(client_id)
            if not client_info:
                return
            
            client_socket = client_info['socket']
            
            while self.is_streaming:
                try:
                    # 接收客户端消息
                    length_data = client_socket.recv(4)
                    if not length_data:
                        break
                    
                    message_length = int.from_bytes(length_data, 'big')
                    message_data = b''
                    
                    while len(message_data) < message_length:
                        chunk = client_socket.recv(message_length - len(message_data))
                        if not chunk:
                            break
                        message_data += chunk
                    
                    if len(message_data) == message_length:
                        self._handle_client_message(client_id, message_data)
                
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.debug(f"TCP客户端 {client_id} 通信错误: {e}")
                    break
            
        except Exception as e:
            self.logger.error(f"处理TCP客户端 {client_id} 失败: {e}")
        finally:
            self._remove_client(client_id)
    
    def _handle_udp_message(self, client_id: str, data: bytes):
        """处理UDP客户端消息"""
        try:
            # 解压缩数据
            if self.config.compression:
                try:
                    data = zlib.decompress(data)
                except:
                    pass
            
            # 解析JSON消息
            message = json.loads(data.decode('utf-8'))
            self._handle_client_message(client_id, json.dumps(message).encode('utf-8'))
            
        except Exception as e:
            self.logger.debug(f"UDP消息处理失败: {e}")
    
    def _handle_client_message(self, client_id: str, message_data: bytes):
        """处理客户端消息"""
        try:
            message = json.loads(message_data.decode('utf-8'))
            message_type = message.get('type')
            
            if message_type == 'config_request':
                # 客户端请求配置信息
                config_response = {
                    'type': 'config_response',
                    'quality': self.config.quality.value,
                    'protocol': self.config.protocol.value,
                    'fps_limit': self.config.fps_limit,
                    'bitrate_limit': self.config.bitrate_limit
                }
                self._send_to_client(client_id, json.dumps(config_response).encode('utf-8'), 1.0)
            
            elif message_type == 'quality_change':
                # 客户端请求改变质量
                new_quality = message.get('quality')
                self._change_stream_quality(new_quality)
            
        except Exception as e:
            self.logger.debug(f"客户端消息处理失败: {e}")
    
    def _remove_client(self, client_id: str):
        """移除客户端"""
        with self.client_lock:
            if client_id in self.clients:
                client_info = self.clients[client_id]
                
                # 关闭连接
                if 'socket' in client_info:
                    try:
                        client_info['socket'].close()
                    except:
                        pass
                
                del self.clients[client_id]
                self.logger.info(f"客户端 {client_id} 已断开连接")
    
    def _change_stream_quality(self, new_quality: str):
        """改变流质量"""
        try:
            quality_map = {
                "低质量": StreamQuality.LOW,
                "中等质量": StreamQuality.MEDIUM,
                "高质量": StreamQuality.HIGH,
                "超高质量": StreamQuality.ULTRA
            }
            
            if new_quality in quality_map:
                self.config.quality = quality_map[new_quality]
                self.logger.info(f"流质量已更改为: {new_quality}")
                
                # 广播质量变更到所有客户端
                with self.client_lock:
                    for client_id in self.clients:
                        self._send_to_client(client_id, 
                                           json.dumps({'type': 'quality_changed', 'quality': new_quality}).encode('utf-8'),
                                           1.0)
        except Exception as e:
            self.logger.error(f"改变流质量失败: {e}")
    
    def _update_metrics(self, encoded_frame: bytes, processing_time: float):
        """更新流指标"""
        self.metrics.frame_count += 1
        self.metrics.bytes_sent += len(encoded_frame)
        self.metrics.packets_sent += 1
        
        # 计算平均帧处理时间
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        self.metrics.avg_frame_time = np.mean(self.frame_times)
        
        # 计算实际比特率
        if self.metrics.frame_count % 30 == 0:  # 每30帧计算一次
            current_time = time.time()
            if hasattr(self, '_last_metrics_time'):
                time_diff = current_time - self._last_metrics_time
                bytes_diff = self.metrics.bytes_sent - getattr(self, '_last_bytes_sent', 0)
                self.metrics.bitrate = (bytes_diff * 8) / time_diff / 1000  # kbps
            
            self._last_metrics_time = current_time
            self._last_bytes_sent = self.metrics.bytes_sent
        
        # 更新客户端数量
        self.metrics.client_count = len(self.clients)
    
    def stop_server(self):
        """停止流媒体服务器"""
        self.logger.info("正在停止流媒体服务器...")
        
        self.is_streaming = False
        
        # 关闭服务器socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # 关闭所有客户端连接
        with self.client_lock:
            for client_id in list(self.clients.keys()):
                self._remove_client(client_id)
        
        # 等待线程结束
        for thread in self.stream_threads + self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("流媒体服务器已停止")
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], bool]):
        """添加帧处理回调函数"""
        self.frame_callbacks.append(callback)
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """获取流统计信息"""
        return {
            'is_streaming': self.is_streaming,
            'protocol': self.config.protocol.value,
            'quality': self.config.quality.value,
            'port': self.config.port,
            'metrics': {
                'frame_count': self.metrics.frame_count,
                'bytes_sent': self.metrics.bytes_sent,
                'packets_sent': self.metrics.packets_sent,
                'avg_frame_time': self.metrics.avg_frame_time,
                'client_count': self.metrics.client_count,
                'bitrate_kbps': self.metrics.bitrate / 1000,
                'jpeg_quality': self.jpeg_quality
            },
            'clients': list(self.clients.keys()),
            'configuration': {
                'fps_limit': self.config.fps_limit,
                'bitrate_limit': self.config.bitrate_limit,
                'buffer_size': self.config.buffer_size,
                'compression': self.config.compression
            }
        }


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='视频流管理器')
    parser.add_argument('--protocol', type=str, default='tcp',
                       choices=['tcp', 'udp', 'websocket'],
                       help='传输协议')
    parser.add_argument('--port', type=int, default=8080, help='监听端口')
    parser.add_argument('--quality', type=str, default='medium',
                       choices=['low', 'medium', 'high', 'ultra'],
                       help='流质量')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID')
    
    args = parser.parse_args()
    
    # 解析参数
    protocol_map = {
        'tcp': StreamProtocol.TCP,
        'udp': StreamProtocol.UDP,
        'websocket': StreamProtocol.WEBSOCKET
    }
    
    quality_map = {
        'low': StreamQuality.LOW,
        'medium': StreamQuality.MEDIUM,
        'high': StreamQuality.HIGH,
        'ultra': StreamQuality.ULTRA
    }
    
    config = StreamConfig(
        protocol=protocol_map[args.protocol],
        port=args.port,
        quality=quality_map[args.quality]
    )
    
    # 创建流管理器
    stream_manager = VideoStreamManager(config)
    
    # 启动服务器
    if stream_manager.start_server():
        print(f"流媒体服务器启动成功，协议: {args.protocol}, 端口: {args.port}")
        
        # 打开摄像头
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("无法打开摄像头")
            stream_manager.stop_server()
            exit(1)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 流式传输
                stream_manager.stream_video(frame)
                
                # 显示本地预览
                cv2.imshow('流媒体服务器预览', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断程序")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            stream_manager.stop_server()
            print("程序结束")
            
            # 显示统计信息
            stats = stream_manager.get_stream_statistics()
            print(f"流统计信息: {stats}")
    else:
        print("流媒体服务器启动失败")