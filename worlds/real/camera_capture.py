"""
USB摄像头场景捕获系统 - 摄像头捕获核心模块

该模块实现USB摄像头的检测、初始化、配置和捕获功能，
支持多摄像头输入和实时视频流处理。

作者: AI助手
创建时间: 2025-11-13
版本: 1.0
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
import os


class CameraStatus(Enum):
    """摄像头状态枚举"""
    IDLE = "空闲"
    CAPTURING = "捕获中"
    ERROR = "错误"
    DISCONNECTED = "已断开"


@dataclass
class CameraConfig:
    """摄像头配置参数"""
    device_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    brightness: int = 50
    contrast: int = 50
    saturation: int = 50
    exposure: int = -6
    white_balance: int = 4000
    auto_focus: bool = True
    auto_exposure: bool = True
    auto_white_balance: bool = True


class CameraCapture:
    """
    摄像头捕获主类
    
    该类负责：
    1. USB摄像头的自动检测和初始化
    2. 摄像头参数的配置和优化
    3. 实时视频帧的捕获和处理
    4. 多摄像头切换和管理
    5. 性能监控和错误处理
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        初始化摄像头捕获系统
        
        Args:
            config: 摄像头配置参数，如果为None则使用默认配置
        """
        self.config = config or CameraConfig()
        self.camera = None
        self.status = CameraStatus.IDLE
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.available_cameras = []
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 线程锁
        self.capture_lock = threading.Lock()
        
        # 性能统计
        self.capture_times = []
        self.max_capture_time_history = 100
        
    def initialize_camera(self, device_id: Optional[int] = None) -> bool:
        """
        初始化USB摄像头设备
        
        Args:
            device_id: 摄像头设备ID，None为自动检测
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 扫描可用摄像头
            if not self.available_cameras:
                self._scan_available_cameras()
            
            # 选择设备
            if device_id is None:
                device_id = self._select_best_camera()
                self.logger.info(f"自动选择摄像头设备: {device_id}")
            else:
                if device_id not in self.available_cameras:
                    self.logger.error(f"设备ID {device_id} 不可用")
                    return False
            
            # 创建摄像头实例
            self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"无法打开摄像头设备: {device_id}")
                return False
            
            # 配置摄像头参数
            success = self._configure_camera()
            if not success:
                self.logger.error("摄像头参数配置失败")
                return False
            
            # 验证摄像头功能
            success = self._test_camera_functionality()
            if not success:
                self.logger.error("摄像头功能测试失败")
                return False
            
            self.status = CameraStatus.IDLE
            self.config.device_id = device_id
            self.logger.info(f"摄像头初始化成功，设备ID: {device_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头初始化失败: {e}")
            self.status = CameraStatus.ERROR
            return False
    
    def _scan_available_cameras(self):
        """扫描系统中可用的摄像头设备"""
        self.available_cameras = []
        
        # 扫描前10个可能的设备ID
        for device_id in range(10):
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    # 测试是否能读取帧
                    ret, frame = cap.read()
                    if ret:
                        self.available_cameras.append(device_id)
                        self.logger.info(f"检测到可用摄像头设备: {device_id}")
                    cap.release()
            except Exception as e:
                self.logger.debug(f"设备 {device_id} 检测失败: {e}")
                continue
        
        self.logger.info(f"总共检测到 {len(self.available_cameras)} 个可用摄像头")
    
    def _select_best_camera(self) -> int:
        """选择最佳摄像头设备"""
        if not self.available_cameras:
            self._scan_available_cameras()
        
        if not self.available_cameras:
            self.logger.error("未检测到任何可用摄像头")
            return 0
        
        # 优先选择内置摄像头（通常设备ID较小）
        return self.available_cameras[0]
    
    def _configure_camera(self) -> bool:
        """配置摄像头参数"""
        if not self.camera:
            return False
        
        try:
            # 设置分辨率
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 设置图像参数
            if hasattr(cv2, 'CAP_PROP_BRIGHTNESS'):
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            if hasattr(cv2, 'CAP_PROP_CONTRAST'):
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            if hasattr(cv2, 'CAP_PROP_SATURATION'):
                self.camera.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            
            # 设置自动对焦和曝光
            if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.auto_focus else 0)
            if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if self.config.auto_exposure else 0)
            if hasattr(cv2, 'CAP_PROP_AUTO_WB'):
                self.camera.set(cv2.CAP_PROP_AUTO_WB, 1 if self.config.auto_white_balance else 0)
            
            # 验证配置结果
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"摄像头配置验证 - 分辨率: {actual_width}x{actual_height}, 帧率: {actual_fps}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头配置失败: {e}")
            return False
    
    def _test_camera_functionality(self) -> bool:
        """测试摄像头功能"""
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.logger.error("无法从摄像头读取帧")
                return False
            
            # 测试多次读取确保稳定性
            for _ in range(5):
                ret, test_frame = self.camera.read()
                if not ret:
                    self.logger.error("摄像头读取不稳定")
                    return False
                time.sleep(0.01)
            
            self.logger.info("摄像头功能测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头功能测试失败: {e}")
            return False
    
    def capture_scene(self) -> Optional[np.ndarray]:
        """
        捕获场景图像
        
        Returns:
            Optional[np.ndarray]: 捕获的图像帧，失败时返回None
        """
        if not self.camera or not self.camera.isOpened():
            self.logger.error("摄像头未正确初始化")
            return None
        
        start_time = time.time()
        
        try:
            with self.capture_lock:
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    self.logger.warning("摄像头读取失败")
                    self.status = CameraStatus.ERROR
                    return None
                
                # 更新帧统计
                self.frame_count += 1
                self.fps_counter += 1
                
                # 计算实际帧率
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # 记录捕获时间
                capture_time = time.time() - start_time
                self.capture_times.append(capture_time)
                if len(self.capture_times) > self.max_capture_time_history:
                    self.capture_times.pop(0)
                
                self.status = CameraStatus.CAPTURING
                return frame
                
        except Exception as e:
            self.logger.error(f"场景捕获失败: {e}")
            self.status = CameraStatus.ERROR
            return None
    
    def switch_camera(self, device_id: int) -> bool:
        """
        切换到指定摄像头
        
        Args:
            device_id: 目标摄像头设备ID
            
        Returns:
            bool: 切换是否成功
        """
        if device_id == self.config.device_id:
            return True
        
        self.logger.info(f"切换摄像头从 {self.config.device_id} 到 {device_id}")
        
        # 释放当前摄像头
        self.release_camera()
        
        # 初始化新摄像头
        return self.initialize_camera(device_id)
    
    def release_camera(self):
        """释放摄像头资源"""
        if self.camera:
            try:
                self.camera.release()
                self.camera = None
                self.status = CameraStatus.IDLE
                self.logger.info("摄像头资源已释放")
            except Exception as e:
                self.logger.error(f"释放摄像头资源失败: {e}")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        获取摄像头信息
        
        Returns:
            Dict[str, Any]: 摄像头信息字典
        """
        if not self.camera or not self.camera.isOpened():
            return {}
        
        info = {
            "device_id": self.config.device_id,
            "status": self.status.value,
            "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.camera.get(cv2.CAP_PROP_FPS),
            "frame_count": self.frame_count,
            "current_fps": self.current_fps,
            "available_cameras": self.available_cameras,
            "capture_time_avg": np.mean(self.capture_times) if self.capture_times else 0,
        }
        
        return info
    
    def set_brightness(self, value: int):
        """设置亮度"""
        if self.camera and hasattr(cv2, 'CAP_PROP_BRIGHTNESS'):
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, value)
            self.config.brightness = value
    
    def set_contrast(self, value: int):
        """设置对比度"""
        if self.camera and hasattr(cv2, 'CAP_PROP_CONTRAST'):
            self.camera.set(cv2.CAP_PROP_CONTRAST, value)
            self.config.contrast = value
    
    def set_exposure(self, value: int):
        """设置曝光"""
        if self.camera and hasattr(cv2, 'CAP_PROP_EXPOSURE'):
            self.camera.set(cv2.CAP_PROP_EXPOSURE, value)
            self.config.exposure = value
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.release_camera()


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='USB摄像头捕获系统')
    parser.add_argument('--device', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度')
    parser.add_argument('--height', type=int, default=1080, help='图像高度')
    parser.add_argument('--fps', type=int, default=30, help='目标帧率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = CameraConfig(
        device_id=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # 创建摄像头捕获实例
    camera = CameraCapture(config)
    
    try:
        # 初始化摄像头
        if camera.initialize_camera():
            print("摄像头初始化成功！")
            print(f"摄像头信息: {camera.get_camera_info()}")
            
            # 实时显示捕获的视频流
            while True:
                frame = camera.capture_scene()
                if frame is not None:
                    cv2.imshow('USB摄像头场景捕获', frame)
                    
                    # 显示摄像头信息
                    info = camera.get_camera_info()
                    info_text = f"FPS: {info.get('current_fps', 0):.1f} | 分辨率: {info.get('width', 0)}x{info.get('height', 0)}"
                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("捕获失败，尝试重新初始化...")
                    time.sleep(1)
        else:
            print("摄像头初始化失败！")
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        camera.release_camera()
        cv2.destroyAllWindows()
        print("程序结束")