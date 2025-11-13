"""
OpenCV真实物体识别系统 - 模型加载器
ModelLoader类：负责加载和管理各种深度学习模型
"""

import cv2
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Tuple
import requests
import urllib.request
from pathlib import Path
import zipfile
import tarfile


class ModelLoader:
    """
    模型加载器类
    功能：
    - 加载YOLO、SSD、RCNN等深度学习模型
    - 管理模型文件和缓存
    - 预处理配置管理
    - 模型性能优化
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化模型加载器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 支持的模型类型
        self.supported_models = {
            'yolo': {
                'names': ['yolov3', 'yolov4', 'yolov5', 'yolov7', 'yolov8'],
                'config_files': ['.cfg', '.onnx'],
                'weight_files': ['.weights', '.pth', '.onnx']
            },
            'ssd': {
                'names': ['ssd_mobilenet', 'ssd_resnet'],
                'config_files': ['.prototxt'],
                'weight_files': ['.caffemodel', '.pb']
            },
            'rcnn': {
                'names': ['faster_rcnn', 'mask_rcnn'],
                'config_files': ['.config'],
                'weight_files': ['.pth', '.pb']
            }
        }
        
        # 物体类别定义
        self.object_classes = {
            'bottle': '水瓶',
            'book': '书本', 
            'pen': '笔',
            'laptop': '笔记本电脑',
            'phone': '手机',
            'cup': '杯子',
            'chair': '椅子',
            'table': '桌子',
            'bag': '包',
            'wallet': '钱包'
        }
        
        # 预训练模型配置
        self.model_configs = {
            'yolov8n': {
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                'size': (640, 640),
                'input_shape': (3, 640, 640)
            },
            'yolov8s': {
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt', 
                'size': (640, 640),
                'input_shape': (3, 640, 640)
            },
            'ssd_mobilenet': {
                'config_url': 'https://github.com/tensorflow/models/raw/master/research/object_detection/ssd_mobilenet_v2_coco.config',
                'model_url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
                'size': (300, 300),
                'input_shape': (3, 300, 300)
            }
        }
        
        # 已加载的模型缓存
        self.loaded_models = {}
        self.model_configs_loaded = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("模型加载器初始化完成")
    
    def load_yolo_model(self, model_name: str = 'yolov8n', device: str = 'cpu') -> Any:
        """
        加载YOLO模型
        
        Args:
            model_name: 模型名称
            device: 设备类型 ('cpu' 或 'cuda')
            
        Returns:
            加载的模型对象
        """
        try:
            self.logger.info(f"正在加载YOLO模型: {model_name}")
            
            # 尝试导入ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                self.logger.warning("未安装ultralytics，使用OpenCV DNN模块")
                return self._load_yolo_opencv(model_name, device)
            
            # 下载或加载模型
            model_path = self._get_model_path(model_name, '.pt')
            
            if not model_path.exists():
                self.logger.info(f"模型文件不存在，正在下载: {model_path}")
                self._download_model(model_name)
            
            # 加载模型
            model = YOLO(str(model_path))
            
            if device == 'cuda':
                model.to('cuda')
            
            self.loaded_models[f'yolo_{model_name}'] = model
            self.logger.info(f"YOLO模型 {model_name} 加载成功")
            
            return model
            
        except Exception as e:
            self.logger.error(f"加载YOLO模型失败: {e}")
            # 降级到OpenCV DNN
            return self._load_yolo_opencv(model_name, device)
    
    def _load_yolo_opencv(self, model_name: str, device: str = 'cpu') -> Any:
        """
        使用OpenCV DNN加载YOLO模型
        
        Args:
            model_name: 模型名称
            device: 设备类型
            
        Returns:
            OpenCV DNN网络对象
        """
        try:
            # 下载YOLOv3-tiny配置文件
            config_path = self._get_yolo_config_path()
            model_path = self._get_yolo_weights_path()
            
            if not config_path.exists() or not model_path.exists():
                self._download_yolo_opencv_model()
            
            # 读取网络
            net = cv2.dnn.readNet(str(model_path), str(config_path))
            
            # 设置计算后端
            if device == 'cuda':
                try:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.logger.info("已启用CUDA加速")
                except:
                    self.logger.warning("CUDA不可用，使用CPU")
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.loaded_models[f'yolo_opencv'] = net
            self.logger.info("OpenCV YOLO模型加载成功")
            
            return net
            
        except Exception as e:
            self.logger.error(f"OpenCV YOLO模型加载失败: {e}")
            return None
    
    def load_ssd_model(self, model_name: str = 'ssd_mobilenet') -> Any:
        """
        加载SSD模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            加载的模型对象
        """
        try:
            self.logger.info(f"正在加载SSD模型: {model_name}")
            
            config_path = self._get_ssd_config_path()
            model_path = self._get_ssd_weights_path()
            
            if not config_path.exists() or not model_path.exists():
                self._download_ssd_model()
            
            # 加载网络
            net = cv2.dnn.readNet(str(model_path), str(config_path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.loaded_models[f'ssd_{model_name}'] = net
            self.logger.info(f"SSD模型 {model_name} 加载成功")
            
            return net
            
        except Exception as e:
            self.logger.error(f"加载SSD模型失败: {e}")
            return None
    
    def _download_model(self, model_name: str) -> None:
        """
        下载预训练模型
        
        Args:
            model_name: 模型名称
        """
        if model_name not in self.model_configs:
            raise ValueError(f"不支持的模型: {model_name}")
        
        config = self.model_configs[model_name]
        model_path = self._get_model_path(model_name, '.pt')
        
        try:
            if 'url' in config:
                self.logger.info(f"下载模型: {config['url']}")
                
                # 创建临时下载路径
                temp_path = self.models_dir / f"{model_name}.temp"
                
                # 下载文件
                urllib.request.urlretrieve(config['url'], str(temp_path))
                
                # 移动到最终位置
                temp_path.rename(model_path)
                
                self.logger.info(f"模型下载完成: {model_path}")
            
        except Exception as e:
            self.logger.error(f"下载模型失败: {e}")
            raise
    
    def _get_model_path(self, model_name: str, extension: str) -> Path:
        """获取模型文件路径"""
        return self.models_dir / f"{model_name}{extension}"
    
    def _get_yolo_config_path(self) -> Path:
        """获取YOLO配置文件路径"""
        return self.models_dir / "yolov3-tiny.cfg"
    
    def _get_yolo_weights_path(self) -> Path:
        """获取YOLO权重文件路径"""
        return self.models_dir / "yolov3-tiny.weights"
    
    def _get_ssd_config_path(self) -> Path:
        """获取SSD配置文件路径"""
        return self.models_dir / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    
    def _get_ssd_weights_path(self) -> Path:
        """获取SSD权重文件路径"""
        return self.models_dir / "ssd_mobilenet_v2_coco_2018_03_29.pb"
    
    def _download_yolo_opencv_model(self) -> None:
        """下载OpenCV YOLO模型文件"""
        try:
            # 下载配置文件
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
            config_path = self._get_yolo_config_path()
            
            urllib.request.urlretrieve(config_url, str(config_path))
            
            # 下载权重文件
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            weights_path = self._get_yolo_weights_path()
            
            urllib.request.urlretrieve(weights_url, str(weights_path))
            
            self.logger.info("OpenCV YOLO模型文件下载完成")
            
        except Exception as e:
            self.logger.error(f"下载OpenCV YOLO模型失败: {e}")
            # 创建备用配置文件
            self._create_backup_yolo_config()
    
    def _download_ssd_model(self) -> None:
        """下载SSD模型文件"""
        try:
            # 下载TensorFlow对象检测模型
            model_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
            temp_path = self.models_dir / "ssd_mobilenet.tar.gz"
            
            urllib.request.urlretrieve(model_url, str(temp_path))
            
            # 解压文件
            with tarfile.open(str(temp_path)) as tar:
                tar.extractall(str(self.models_dir))
            
            # 清理临时文件
            temp_path.unlink()
            
            # 重命名提取的文件
            extracted_dir = self.models_dir / "ssd_mobilenet_v2_coco_2018_03_29"
            if extracted_dir.exists():
                (extracted_dir / "frozen_inference_graph.pb").rename(self._get_ssd_weights_path())
                (extracted_dir / "pipeline.config").rename(self._get_ssd_config_path())
                
                # 删除解压目录
                import shutil
                shutil.rmtree(str(extracted_dir))
            
            self.logger.info("SSD模型文件下载完成")
            
        except Exception as e:
            self.logger.error(f"下载SSD模型失败: {e}")
            # 创建备用配置
            self._create_backup_ssd_config()
    
    def _create_backup_yolo_config(self) -> None:
        """创建备用YOLO配置文件"""
        config_content = """[net]
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=16
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=16
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

##############

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
size=1
stride=1
pad=1
filters=18
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=18
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
size=1
stride=1
pad=1
filters=256
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61

[convolutional]
batch_normalize=1
size=1
stride=1
pad=1
filters=128
activation=leaky

[convolutional]
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=128
activation=leaky

[convolutional]
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=128
activation=leaky

[convolutional]
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=18
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=18
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1"""
        
        with open(self._get_yolo_config_path(), 'w') as f:
            f.write(config_content)
        
        self.logger.info("已创建备用YOLO配置文件")
    
    def _create_backup_ssd_config(self) -> None:
        """创建备用SSD配置文件"""
        config_content = """tensorflow_serving {
  config_key: "model_server_config"
}

model_config_list: [
  {
    name: "ssd_mobilenet_v2",
    base_path: "/models/ssd_mobilenet_v2",
    model_platform: "tensorflow_serving",
    model_version_policy: { latest: { num_versions: 2 }},
    logging_config {
      level: 3
    }
  }
]"""
        
        with open(self._get_ssd_config_path(), 'w') as f:
            f.write(config_content)
        
        self.logger.info("已创建备用SSD配置文件")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        if model_name in self.model_configs:
            return self.model_configs[model_name].copy()
        else:
            return {
                'size': (416, 416),
                'input_shape': (3, 416, 416),
                'classes': len(self.object_classes)
            }
    
    def list_loaded_models(self) -> Dict[str, str]:
        """
        列出已加载的模型
        
        Returns:
            已加载模型字典
        """
        return {name: type(model).__name__ for name, model in self.loaded_models.items()}
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            卸载是否成功
        """
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                self.logger.info(f"模型 {model_name} 已卸载")
                return True
            else:
                self.logger.warning(f"模型 {model_name} 未找到")
                return False
        except Exception as e:
            self.logger.error(f"卸载模型失败: {e}")
            return False
    
    def get_object_classes(self) -> Dict[str, str]:
        """获取支持的物体类别"""
        return self.object_classes.copy()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        检查模型是否已加载
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型是否已加载
        """
        return model_name in self.loaded_models
    
    def get_available_models(self) -> Dict[str, list]:
        """
        获取可用的模型列表
        
        Returns:
            可用模型字典
        """
        return {
            model_type: info['names'] 
            for model_type, info in self.supported_models.items()
        }