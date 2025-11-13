"""
USB摄像头场景捕获系统 - 实时捕获主系统

该模块是整个USB摄像头场景捕获系统的核心控制器，
整合摄像头捕获、场景处理、帧分析和视频流管理功能。

作者: AI助手
创建时间: 2025-11-13
版本: 1.0
"""

import cv2
import numpy as np
import time
import threading
import logging
import argparse
import json
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# 导入自定义模块
from camera_capture import CameraCapture, CameraConfig, CameraStatus
from scene_processor import SceneProcessor, SceneInfo, SceneType
from frame_analyzer import FrameAnalyzer, FrameAnalysisResult, AnalysisType
from video_stream_manager import VideoStreamManager, StreamConfig, StreamQuality, StreamProtocol


class SystemMode(Enum):
    """系统运行模式枚举"""
    LOCAL_DISPLAY = "本地显示"
    STREAM_SERVER = "流媒体服务器"
    ANALYSIS_ONLY = "仅分析"
    RECORDING = "录像模式"
    BATCH_PROCESSING = "批量处理"


@dataclass
class SystemConfig:
    """系统配置参数"""
    # 摄像头配置
    camera_config: CameraConfig = None
    
    # 分析配置
    analysis_type: AnalysisType = AnalysisType.REAL_TIME
    enable_face_detection: bool = True
    enable_object_detection: bool = True
    enable_text_detection: bool = False
    
    # 流媒体配置
    stream_config: StreamConfig = None
    enable_streaming: bool = False
    
    # 录像配置
    record_video: bool = False
    record_path: str = "./recordings/"
    record_format: str = "mp4"
    
    # 性能配置
    max_fps: int = 30
    buffer_size: int = 10
    enable_profiling: bool = False
    
    # 界面配置
    show_preview: bool = True
    show_annotations: bool = True
    show_statistics: bool = True
    window_title: str = "USB摄像头场景捕获系统"
    
    def __post_init__(self):
        if self.camera_config is None:
            self.camera_config = CameraConfig()
        if self.stream_config is None:
            self.stream_config = StreamConfig()


class RealTimeCapture:
    """
    实时捕获系统主控制器
    
    该类负责：
    1. 整合所有子系统组件
    2. 协调摄像头捕获、场景处理、帧分析和流媒体
    3. 系统状态管理和错误处理
    4. 性能监控和优化
    5. 用户界面和交互
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """初始化实时捕获系统"""
        self.config = config or SystemConfig()
        
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 系统组件
        self.camera = None
        self.scene_processor = None
        self.frame_analyzer = None
        self.stream_manager = None
        
        # 系统状态
        self.is_running = False
        self.system_mode = SystemMode.LOCAL_DISPLAY
        self.frame_count = 0
        self.start_time = time.time()
        
        # 线程管理
        self.main_thread = None
        self.processing_threads = []
        
        # 数据存储
        self.current_scene_info = None
        self.current_frame_analysis = None
        self.performance_stats = {}
        
        # 回调函数
        self.frame_callbacks = []
        self.error_callbacks = []
        
        # 录像相关
        self.video_writer = None
        self.record_start_time = None
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 初始化摄像头
            self.camera = CameraCapture(self.config.camera_config)
            
            # 初始化场景处理器
            self.scene_processor = SceneProcessor()
            
            # 初始化帧分析器
            self.frame_analyzer = FrameAnalyzer(self.config.analysis_type)
            
            # 初始化流管理器
            if self.config.enable_streaming:
                self.stream_manager = VideoStreamManager(self.config.stream_config)
            
            # 创建录像目录
            if self.config.record_video:
                os.makedirs(self.config.record_path, exist_ok=True)
            
            self.logger.info("系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def initialize_camera(self) -> bool:
        """
        初始化摄像头
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("正在初始化摄像头...")
            
            # 扫描可用摄像头
            if not self.camera.available_cameras:
                self.camera._scan_available_cameras()
            
            if not self.camera.available_cameras:
                self.logger.error("未检测到任何可用摄像头")
                return False
            
            # 尝试初始化第一个可用摄像头
            success = self.camera.initialize_camera()
            
            if success:
                self.logger.info("摄像头初始化成功")
                return True
            else:
                self.logger.error("摄像头初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"摄像头初始化异常: {e}")
            return False
    
    def start_capture(self) -> bool:
        """
        开始捕获
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                self.logger.warning("系统已在运行中")
                return True
            
            # 初始化摄像头
            if not self.initialize_camera():
                return False
            
            # 启动流媒体服务器
            if self.config.enable_streaming and self.stream_manager:
                if not self.stream_manager.start_server():
                    self.logger.warning("流媒体服务器启动失败，继续本地捕获")
            
            # 启动录像
            if self.config.record_video:
                self._start_recording()
            
            self.is_running = True
            self.start_time = time.time()
            
            # 启动主处理线程
            self.main_thread = threading.Thread(target=self._main_capture_loop)
            self.main_thread.daemon = True
            self.main_thread.start()
            
            self.logger.info("系统捕获已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动捕获失败: {e}")
            return False
    
    def stop_capture(self):
        """停止捕获"""
        self.logger.info("正在停止系统...")
        
        self.is_running = False
        
        # 等待主线程结束
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
        
        # 停止录像
        if self.video_writer:
            self._stop_recording()
        
        # 停止流媒体服务器
        if self.stream_manager:
            self.stream_manager.stop_server()
        
        # 释放摄像头
        if self.camera:
            self.camera.release_camera()
        
        self.logger.info("系统已停止")
    
    def _main_capture_loop(self):
        """主捕获循环"""
        try:
            while self.is_running:
                start_time = time.time()
                
                # 捕获场景
                frame = self.camera.capture_scene()
                if frame is None:
                    self.logger.warning("捕获帧失败，尝试重新初始化摄像头")
                    time.sleep(1)
                    if not self.initialize_camera():
                        break
                    continue
                
                # 处理帧
                processed_frame = self._process_frame(frame)
                if processed_frame is None:
                    continue
                
                # 显示预览
                if self.config.show_preview:
                    self._display_frame(processed_frame)
                
                # 录像
                if self.config.record_video and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # 性能统计
                if self.config.enable_profiling:
                    self._update_performance_stats(time.time() - start_time)
                
                # 控制帧率
                self._control_frame_rate(start_time)
                
        except Exception as e:
            self.logger.error(f"主循环异常: {e}")
            self._handle_error(e)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """处理单帧"""
        try:
            # 场景处理
            self.current_scene_info = self.scene_processor.analyze_environment(frame)
            processed_frame = self.scene_processor.process_frame(frame)
            
            # 帧分析
            self.current_frame_analysis = self.frame_analyzer.analyze_frame(processed_frame)
            
            # 流媒体传输
            if self.config.enable_streaming and self.stream_manager:
                self.stream_manager.stream_video(processed_frame)
            
            # 调用回调函数
            for callback in self.frame_callbacks:
                try:
                    callback(processed_frame, self.current_scene_info, self.current_frame_analysis)
                except Exception as e:
                    self.logger.warning(f"帧回调函数执行失败: {e}")
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return frame
    
    def _display_frame(self, frame: np.ndarray):
        """显示帧"""
        try:
            display_frame = frame.copy()
            
            if self.config.show_annotations:
                display_frame = self._add_annotations(display_frame)
            
            # 显示帧
            cv2.imshow(self.config.window_title, display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            self._handle_keypress(key)
            
        except Exception as e:
            self.logger.error(f"显示帧失败: {e}")
    
    def _add_annotations(self, frame: np.ndarray) -> np.ndarray:
        """添加注释到帧"""
        try:
            # 场景信息
            if self.current_scene_info:
                scene_text = f"场景: {self.current_scene_info.scene_type.value} ({self.current_scene_info.confidence:.2f})"
                cv2.putText(frame, scene_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                quality_text = f"质量: {self.current_scene_info.quality_score:.2f}"
                cv2.putText(frame, quality_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 帧分析结果
            if self.current_frame_analysis:
                # 检测到的人脸
                for (x, y, w, h) in self.current_frame_analysis.faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 检测到的对象
                for obj in self.current_frame_analysis.objects:
                    x, y, w, h = obj.bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{obj.class_name} ({obj.confidence:.2f})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 文本区域
                for (x, y, w, h) in self.current_frame_analysis.text_regions:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                
                # 运动区域
                for (x, y, w, h) in self.current_frame_analysis.motion_vectors:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
                
                # 场景变化标记
                if self.current_frame_analysis.scene_change:
                    cv2.putText(frame, "场景变化!", (10, frame.shape[0]-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 性能统计
            if self.config.show_statistics:
                self._add_performance_info(frame)
            
            # 流媒体信息
            if self.config.enable_streaming and self.stream_manager:
                stream_stats = self.stream_manager.get_stream_statistics()
                clients_text = f"客户端: {stream_stats['metrics']['client_count']}"
                cv2.putText(frame, clients_text, (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"添加注释失败: {e}")
            return frame
    
    def _add_performance_info(self, frame: np.ndarray):
        """添加性能信息"""
        try:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 计算当前FPS
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 显示FPS
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示帧数
            frame_text = f"帧数: {self.frame_count}"
            cv2.putText(frame, frame_text, (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示运行时间
            runtime_text = f"运行时间: {elapsed_time:.1f}s"
            cv2.putText(frame, runtime_text, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 摄像头状态
            if self.camera:
                camera_status = self.camera.status.value
                status_text = f"摄像头: {camera_status}"
                cv2.putText(frame, status_text, (frame.shape[1]-200, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception as e:
            self.logger.debug(f"性能信息显示失败: {e}")
    
    def _handle_keypress(self, key: int):
        """处理按键"""
        if key == ord('q'):
            self.logger.info("用户按Q键退出")
            self.stop_capture()
        elif key == ord('s'):
            self._save_screenshot()
        elif key == ord('r'):
            self._toggle_recording()
        elif key == ord('p'):
            self.config.show_statistics = not self.config.show_statistics
            self.logger.info(f"性能统计显示: {self.config.show_statistics}")
        elif key == ord('a'):
            self.config.show_annotations = not self.config.show_annotations
            self.logger.info(f"注释显示: {self.config.show_annotations}")
        elif key == ord('c'):
            self._capture_scene_info()
    
    def _save_screenshot(self):
        """保存截图"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join(self.config.record_path, filename)
            
            cv2.imwrite(filepath, self.camera.camera.read()[1])
            self.logger.info(f"截图已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存截图失败: {e}")
    
    def _toggle_recording(self):
        """切换录像状态"""
        if self.video_writer:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """开始录像"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.{self.config.record_format}"
            filepath = os.path.join(self.config.record_path, filename)
            
            # 获取摄像头信息
            camera_info = self.camera.get_camera_info()
            width = camera_info.get('width', 640)
            height = camera_info.get('height', 480)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v' if self.config.record_format == 'mp4' else 'avi')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
            self.record_start_time = time.time()
            
            self.logger.info(f"开始录像: {filepath}")
            
        except Exception as e:
            self.logger.error(f"开始录像失败: {e}")
    
    def _stop_recording(self):
        """停止录像"""
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
                if self.record_start_time:
                    duration = time.time() - self.record_start_time
                    self.logger.info(f"录像已停止，时长: {duration:.1f}秒")
                
        except Exception as e:
            self.logger.error(f"停止录像失败: {e}")
    
    def _capture_scene_info(self):
        """捕获场景信息"""
        try:
            if self.current_scene_info:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"scene_info_{timestamp}.json"
                filepath = os.path.join(self.config.record_path, filename)
                
                # 准备场景信息数据
                scene_data = {
                    'timestamp': timestamp,
                    'scene_type': self.current_scene_info.scene_type.value,
                    'confidence': self.current_scene_info.confidence,
                    'quality_score': self.current_scene_info.quality_score,
                    'features': {
                        'brightness': self.current_scene_info.features.brightness,
                        'contrast': self.current_scene_info.features.contrast,
                        'saturation': self.current_scene_info.features.saturation,
                        'edge_density': self.current_scene_info.features.edge_density,
                        'object_count': self.current_scene_info.features.object_count,
                        'motion_level': self.current_scene_info.features.motion_level
                    }
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(scene_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"场景信息已保存: {filepath}")
                
        except Exception as e:
            self.logger.error(f"保存场景信息失败: {e}")
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self.frame_count += 1
        
        if 'processing_times' not in self.performance_stats:
            self.performance_stats['processing_times'] = []
        
        self.performance_stats['processing_times'].append(processing_time)
        
        # 保持最近100次记录
        if len(self.performance_stats['processing_times']) > 100:
            self.performance_stats['processing_times'].pop(0)
    
    def _control_frame_rate(self, frame_start_time: float):
        """控制帧率"""
        target_frame_time = 1.0 / self.config.max_fps
        elapsed_time = time.time() - frame_start_time
        
        if elapsed_time < target_frame_time:
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _handle_error(self, error: Exception):
        """处理错误"""
        self.logger.error(f"系统错误: {error}")
        
        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.warning(f"错误回调失败: {e}")
    
    def add_frame_callback(self, callback):
        """添加帧处理回调"""
        self.frame_callbacks.append(callback)
    
    def add_error_callback(self, callback):
        """添加错误处理回调"""
        self.error_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'is_running': self.is_running,
            'system_mode': self.system_mode.value,
            'frame_count': self.frame_count,
            'runtime': time.time() - self.start_time if self.start_time else 0,
            'camera': self.camera.get_camera_info() if self.camera else {},
            'scene_processor': self.scene_processor.get_scene_statistics() if self.scene_processor else {},
            'frame_analyzer': self.frame_analyzer.get_analysis_statistics() if self.frame_analyzer else {},
            'stream_manager': self.stream_manager.get_stream_statistics() if self.stream_manager else {},
            'configuration': {
                'max_fps': self.config.max_fps,
                'analysis_type': self.config.analysis_type.value,
                'enable_streaming': self.config.enable_streaming,
                'record_video': self.config.record_video,
                'show_preview': self.config.show_preview
            }
        }
        
        if self.config.enable_profiling and 'processing_times' in self.performance_stats:
            processing_times = self.performance_stats['processing_times']
            status['performance'] = {
                'avg_processing_time': np.mean(processing_times),
                'min_processing_time': np.min(processing_times),
                'max_processing_time': np.max(processing_times),
                'current_fps': 1.0 / np.mean(processing_times) if processing_times else 0
            }
        
        return status
    
    def export_system_report(self, filepath: str):
        """导出系统报告"""
        try:
            report = {
                'system_info': {
                    'version': '1.0',
                    'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_runtime': time.time() - self.start_time
                },
                'system_status': self.get_system_status(),
                'performance_stats': self.performance_stats,
                'configuration': {
                    'camera_config': self.config.camera_config.__dict__,
                    'analysis_type': self.config.analysis_type.value,
                    'stream_config': self.config.stream_config.__dict__,
                    'record_settings': {
                        'enabled': self.config.record_video,
                        'path': self.config.record_path,
                        'format': self.config.record_format
                    }
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"系统报告已导出: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出系统报告失败: {e}")


def create_default_config() -> SystemConfig:
    """创建默认配置"""
    camera_config = CameraConfig(
        width=1280,
        height=720,
        fps=30,
        brightness=50,
        contrast=50
    )
    
    stream_config = StreamConfig(
        quality=StreamQuality.MEDIUM,
        protocol=StreamProtocol.TCP,
        port=8080,
        max_clients=10,
        compression=True
    )
    
    return SystemConfig(
        camera_config=camera_config,
        analysis_type=AnalysisType.REAL_TIME,
        stream_config=stream_config,
        enable_streaming=False,
        record_video=False,
        show_preview=True,
        show_annotations=True,
        show_statistics=True,
        max_fps=30
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='USB摄像头场景捕获系统')
    
    # 系统配置
    parser.add_argument('--mode', type=str, default='local',
                       choices=['local', 'stream', 'analysis', 'record'],
                       help='系统运行模式')
    
    # 摄像头配置
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--width', type=int, default=1280, help='图像宽度')
    parser.add_argument('--height', type=int, default=720, help='图像高度')
    parser.add_argument('--fps', type=int, default=30, help='目标帧率')
    
    # 分析配置
    parser.add_argument('--analysis', type=str, default='realtime',
                       choices=['basic', 'detailed', 'realtime', 'batch'],
                       help='分析类型')
    parser.add_argument('--no-face', action='store_true', help='禁用人脸检测')
    parser.add_argument('--no-object', action='store_true', help='禁用目标检测')
    
    # 流媒体配置
    parser.add_argument('--stream', action='store_true', help='启用流媒体')
    parser.add_argument('--protocol', type=str, default='tcp',
                       choices=['tcp', 'udp'], help='流媒体协议')
    parser.add_argument('--port', type=int, default=8080, help='流媒体端口')
    parser.add_argument('--quality', type=str, default='medium',
                       choices=['low', 'medium', 'high', 'ultra'], help='流质量')
    
    # 录像配置
    parser.add_argument('--record', action='store_true', help='启用录像')
    parser.add_argument('--record-path', type=str, default='./recordings/', help='录像保存路径')
    
    # 其他配置
    parser.add_argument('--no-preview', action='store_true', help='禁用预览窗口')
    parser.add_argument('--no-annotations', action='store_true', help='禁用注释')
    parser.add_argument('--no-stats', action='store_true', help='禁用统计显示')
    parser.add_argument('--max-fps', type=int, default=30, help='最大帧率')
    parser.add_argument('--report', type=str, help='导出系统报告文件路径')
    
    args = parser.parse_args()
    
    # 创建配置
    config = create_default_config()
    
    # 摄像头配置
    config.camera_config.device_id = args.camera
    config.camera_config.width = args.width
    config.camera_config.height = args.height
    config.camera_config.fps = args.fps
    
    # 分析配置
    analysis_map = {
        'basic': AnalysisType.BASIC,
        'detailed': AnalysisType.DETAILED,
        'realtime': AnalysisType.REAL_TIME,
        'batch': AnalysisType.BATCH
    }
    config.analysis_type = analysis_map[args.analysis]
    config.enable_face_detection = not args.no_face
    config.enable_object_detection = not args.no_object
    
    # 系统模式配置
    mode_map = {
        'local': SystemMode.LOCAL_DISPLAY,
        'stream': SystemMode.STREAM_SERVER,
        'analysis': SystemMode.ANALYSIS_ONLY,
        'record': SystemMode.RECORDING
    }
    config.system_mode = mode_map[args.mode]
    
    # 流媒体配置
    config.enable_streaming = args.stream
    if args.stream:
        protocol_map = {'tcp': StreamProtocol.TCP, 'udp': StreamProtocol.UDP}
        quality_map = {
            'low': StreamQuality.LOW,
            'medium': StreamQuality.MEDIUM,
            'high': StreamQuality.HIGH,
            'ultra': StreamQuality.ULTRA
        }
        
        config.stream_config.protocol = protocol_map[args.protocol]
        config.stream_config.port = args.port
        config.stream_config.quality = quality_map[args.quality]
    
    # 录像配置
    config.record_video = args.record
    config.record_path = args.record_path
    
    # 界面配置
    config.show_preview = not args.no_preview
    config.show_annotations = not args.no_annotations
    config.show_statistics = not args.no_stats
    config.max_fps = args.max_fps
    
    # 创建实时捕获系统
    capture_system = RealTimeCapture(config)
    
    try:
        # 启动系统
        if capture_system.start_capture():
            print("USB摄像头场景捕获系统已启动")
            print("按键说明:")
            print("  q - 退出程序")
            print("  s - 保存截图")
            print("  r - 切换录像")
            print("  p - 切换性能统计显示")
            print("  a - 切换注释显示")
            print("  c - 捕获场景信息")
            
            # 等待用户中断
            while capture_system.is_running:
                time.sleep(0.1)
        else:
            print("系统启动失败")
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        # 停止系统
        capture_system.stop_capture()
        
        # 导出报告
        if args.report:
            capture_system.export_system_report(args.report)
            print(f"系统报告已导出: {args.report}")
        
        # 显示最终统计
        status = capture_system.get_system_status()
        print("\n=== 系统统计 ===")
        print(f"运行时间: {status['runtime']:.1f}秒")
        print(f"处理帧数: {status['frame_count']}")
        print(f"平均FPS: {status['frame_count'] / status['runtime']:.1f}")
        
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == "__main__":
    main()