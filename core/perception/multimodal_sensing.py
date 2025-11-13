#!/usr/bin/env python3
"""
多模态世界模型感知系统

该模块实现了完整的感知系统，整合视觉、音频和空间感知能力，
构建动态世界模型并进行多模态特征融合。

功能模块：
1. 视觉感知：USB摄像头 + OpenCV物体识别
2. 音频感知：Whisper语音识别  
3. 空间感知：激光雷达点云处理
4. 世界模型动态构建
5. 多模态融合和特征提取

作者: NeuroMinecraftGenesis
创建时间: 2025-11-13
"""

import cv2
import numpy as np
import open3d as o3d
import pyaudio
import json
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import whisper
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import torch
import librosa
import soundfile as sf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerceptionData:
    """感知数据结构"""
    timestamp: float
    modality: str
    data: Dict[str, Any]
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass 
class WorldObject:
    """世界对象定义"""
    id: str
    position: np.ndarray
    attributes: Dict[str, Any]
    confidence: float
    last_seen: float
    modality_sources: List[str]


@dataclass
class SpatialFeature:
    """空间特征"""
    centroid: np.ndarray
    bounds: np.ndarray
    surface_area: float
    volume: float
    orientation: np.ndarray


class CameraPerception:
    """视觉感知模块"""
    
    def __init__(self, camera_id: int = 0, enable_object_detection: bool = True):
        self.camera_id = camera_id
        self.enable_object_detection = enable_object_detection
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 加载预训练模型（如果启用）
        if enable_object_detection:
            try:
                self.net = cv2.dnn.readNetFromDarknet(
                    "yolo_weights/yolov4.cfg", 
                    "yolo_weights/yolov4.weights"
                )
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                logger.info("YOLO物体检测模型加载成功")
            except:
                logger.warning("YOLO模型加载失败，将使用基础的OpenCV检测")
                self.net = None
                self.output_layers = None
        
        # COCO数据集类别
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        
        self.frame_queue = Queue(maxsize=10)
        self.is_running = False
    
    def start_capture(self):
        """启动摄像头捕获"""
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()
        logger.info("摄像头捕获已启动")
    
    def stop_capture(self):
        """停止摄像头捕获"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()
        logger.info("摄像头捕获已停止")
    
    def _capture_loop(self):
        """摄像头捕获循环"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("摄像头读取失败")
                continue
            
            # 预处理帧
            processed_frame = self._preprocess_frame(frame)
            
            # 物体检测
            objects = self._detect_objects(processed_frame)
            
            # 添加到队列
            try:
                self.frame_queue.put_nowait({
                    'timestamp': time.time(),
                    'frame': processed_frame,
                    'objects': objects
                })
            except:
                pass  # 队列满，跳过
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        # 调整大小
        frame = cv2.resize(frame, (416, 416))
        
        # 归一化
        frame = frame / 255.0
        
        return frame
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """物体检测"""
        objects = []
        
        if self.net is not None and self.output_layers is not None:
            # 使用YOLO检测
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # 获取边界框
                        center_x = int(detection[0] * 416)
                        center_y = int(detection[1] * 416)
                        w = int(detection[2] * 416)
                        h = int(detection[3] * 416)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        objects.append({
                            'class': self.classes[class_id],
                            'confidence': float(confidence),
                            'bbox': [x, y, w, h]
                        })
        else:
            # 基础检测：边缘检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 最小面积阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        'class': 'detected_object',
                        'confidence': 0.7,
                        'bbox': [x, y, w, h]
                    })
        
        return objects
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None


class AudioPerception:
    """音频感知模块"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # 初始化Whisper模型
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper语音识别模型加载成功")
        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            self.whisper_model = None
        
        # PyAudio设置
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        
        # 音频缓冲区
        self.audio_queue = Queue(maxsize=30)
        self.is_recording = False
        
        # 音频分析参数
        self.silence_threshold = 0.01
        self.silence_duration = 2.0  # 2秒静音检测
    
    def start_recording(self):
        """启动音频录制"""
        self.is_recording = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()
        logger.info("音频录制已启动")
    
    def stop_recording(self):
        """停止音频录制"""
        self.is_recording = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logger.info("音频录制已停止")
    
    def _record_loop(self):
        """音频录制循环"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            audio_buffer = []
            silence_start = None
            
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # 检查静音
                if np.max(np.abs(audio_chunk)) < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duration:
                        # 处理音频片段
                        if len(audio_buffer) > 0:
                            self._process_audio_segment(audio_buffer)
                            audio_buffer = []
                        silence_start = None
                else:
                    silence_start = None
                
                audio_buffer.append(audio_chunk)
                
                # 控制缓冲区大小
                if len(audio_buffer) > 100:  # 约6秒音频
                    self._process_audio_segment(audio_buffer[:50])  # 处理前3秒
                    audio_buffer = audio_buffer[50:]
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def _process_audio_segment(self, audio_buffer: List[np.ndarray]):
        """处理音频片段"""
        audio_data = np.concatenate(audio_buffer)
        
        # 添加到队列
        try:
            self.audio_queue.put_nowait({
                'timestamp': time.time(),
                'audio': audio_data,
                'sample_rate': self.sample_rate
            })
        except:
            pass
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """语音转文本"""
        if self.whisper_model is None:
            return {'text': '', 'confidence': 0.0, 'language': 'unknown'}
        
        try:
            # 使用Whisper进行转录
            result = self.whisper_model.transcribe(
                audio_data,
                language='zh',  # 中文优先
                task='transcribe'
            )
            
            # 计算置信度（简化）
            confidence = min(result.get('confidence', 0.7), 1.0)
            
            return {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'confidence': confidence,
                'segments': result.get('segments', [])
            }
        
        except Exception as e:
            logger.error(f"语音识别错误: {e}")
            return {'text': '', 'confidence': 0.0, 'language': 'unknown'}
    
    def get_latest_audio(self) -> Optional[Dict[str, Any]]:
        """获取最新音频"""
        try:
            return self.audio_queue.get_nowait()
        except Empty:
            return None


class SpatialPerception:
    """空间感知模块"""
    
    def __init__(self, num_points: int = 20000):
        self.num_points = num_points
        self.point_cloud_queue = Queue(maxsize=10)
        
        # 点云处理参数
        self.downsampling_voxel_size = 0.05
        self.outlier_nb_neighbors = 20
        self.outlier_std_ratio = 2.0
        self.cluster_eps = 0.1
    
    def start_point_cloud_capture(self):
        """启动点云捕获"""
        self.thread = threading.Thread(target=self._capture_point_cloud)
        self.thread.start()
        logger.info("点云捕获已启动")
    
    def stop_point_cloud_capture(self):
        """停止点云捕获"""
        if hasattr(self, 'thread'):
            self.thread.join()
        logger.info("点云捕获已停止")
    
    def _capture_point_cloud(self):
        """点云捕获循环"""
        while True:
            try:
                # 模拟激光雷达数据（在实际应用中替换为真实传感器数据）
                points = self._simulate_lidar_data()
                
                # 点云处理
                processed_cloud = self._process_point_cloud(points)
                
                # 添加到队列
                try:
                    self.point_cloud_queue.put_nowait({
                        'timestamp': time.time(),
                        'point_cloud': processed_cloud,
                        'num_points': len(processed_cloud.points)
                    })
                except:
                    pass
                
                time.sleep(0.1)  # 10Hz更新率
            
            except Exception as e:
                logger.error(f"点云捕获错误: {e}")
                time.sleep(1)
    
    def _simulate_lidar_data(self) -> np.ndarray:
        """模拟激光雷达数据"""
        # 在实际应用中，这将是真实的传感器数据
        # 这里模拟一个房间环境
        points = []
        
        # 地面
        for x in np.linspace(-5, 5, 50):
            for z in np.linspace(-5, 5, 50):
                points.append([x, 0, z])
        
        # 墙壁
        for y in np.linspace(0, 3, 30):
            for z in np.linspace(-5, 5, 50):
                points.append([-5, y, z])
                points.append([5, y, z])
        
        for x in np.linspace(-5, 5, 50):
            for y in np.linspace(0, 3, 30):
                points.append([x, y, -5])
                points.append([x, y, 5])
        
        # 添加一些随机物体
        for _ in range(100):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(0.5, 2.5)
            z = np.random.uniform(-4, 4)
            points.append([x, y, z])
        
        return np.array(points)
    
    def _process_point_cloud(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """处理点云"""
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 降采样
        pcd = pcd.voxel_down_sample(voxel_size=self.downsampling_voxel_size)
        
        # 去除离群点
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )
        
        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        return pcd
    
    def extract_spatial_features(self, pcd: o3d.geometry.PointCloud) -> List[SpatialFeature]:
        """提取空间特征"""
        features = []
        
        # 聚类分析
        points = np.asarray(pcd.points)
        
        if len(points) > 0:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=10).fit(points)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # 噪声点
                    continue
                
                cluster_points = points[labels == label]
                
                if len(cluster_points) > 10:  # 最小聚类大小
                    feature = self._compute_spatial_feature(cluster_points)
                    features.append(feature)
        
        return features
    
    def _compute_spatial_feature(self, cluster_points: np.ndarray) -> SpatialFeature:
        """计算单个空间特征"""
        # 计算质心
        centroid = np.mean(cluster_points, axis=0)
        
        # 计算边界框
        min_coords = np.min(cluster_points, axis=0)
        max_coords = np.max(cluster_points, axis=0)
        bounds = np.array([min_coords, max_coords])
        
        # 计算表面积和体积
        dims = max_coords - min_coords
        volume = np.prod(dims)
        surface_area = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])
        
        # 计算主方向（简化版）
        if len(cluster_points) > 3:
            # PCA降维
            pca = PCA(n_components=3)
            pca.fit(cluster_points - centroid)
            orientation = pca.components_[0]  # 第一主成分
        else:
            orientation = np.array([1, 0, 0])
        
        return SpatialFeature(
            centroid=centroid,
            bounds=bounds,
            surface_area=surface_area,
            volume=volume,
            orientation=orientation
        )
    
    def get_latest_point_cloud(self) -> Optional[Dict[str, Any]]:
        """获取最新点云"""
        try:
            return self.point_cloud_queue.get_nowait()
        except Empty:
            return None


class WorldModel:
    """世界模型动态构建"""
    
    def __init__(self):
        self.objects: Dict[str, WorldObject] = {}
        self.spatial_relationships: Dict[str, Dict[str, float]] = {}
        self.temporal_evolution: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}
        
        # 更新参数
        self.object_max_age = 60.0  # 对象最大存活时间（秒）
        self.confidence_decay = 0.95  # 置信度衰减因子
        self.position_threshold = 0.5  # 位置匹配阈值
    
    def update_world_state(self, perception_data: List[PerceptionData]):
        """更新世界状态"""
        current_time = time.time()
        
        # 清理过期对象
        self._cleanup_expired_objects(current_time)
        
        # 处理各种感知数据
        for data in perception_data:
            if data.modality == 'visual':
                self._process_visual_data(data, current_time)
            elif data.modality == 'spatial':
                self._process_spatial_data(data, current_time)
            elif data.modality == 'audio':
                self._process_audio_data(data, current_time)
        
        # 更新空间关系
        self._update_spatial_relationships()
        
        # 记录时间序列
        self.temporal_evolution.append({
            'timestamp': current_time,
            'objects': list(self.objects.keys()),
            'num_objects': len(self.objects)
        })
        
        # 限制时间序列长度
        if len(self.temporal_evolution) > 1000:
            self.temporal_evolution = self.temporal_evolution[-500:]
    
    def _process_visual_data(self, data: PerceptionData, current_time: float):
        """处理视觉数据"""
        for obj in data.data.get('objects', []):
            obj_id = f"obj_{obj['class']}_{len(self.objects)}"
            
            # 计算位置（从边界框推断深度）
            bbox = obj['bbox']
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            # 简化深度估算
            estimated_depth = 3.0  # 默认深度
            position = np.array([center_x, center_y, estimated_depth])
            
            # 更新或创建对象
            if obj_id in self.objects:
                existing_obj = self.objects[obj_id]
                # 位置融合
                new_position = 0.7 * existing_obj.position + 0.3 * position
                existing_obj.position = new_position
                existing_obj.confidence = min(existing_obj.confidence * self.confidence_decay + 0.3, 1.0)
                existing_obj.last_seen = current_time
            else:
                self.objects[obj_id] = WorldObject(
                    id=obj_id,
                    position=position,
                    attributes=obj,
                    confidence=obj['confidence'],
                    last_seen=current_time,
                    modality_sources=['visual']
                )
    
    def _process_spatial_data(self, data: PerceptionData, current_time: float):
        """处理空间数据"""
        for feature in data.data.get('features', []):
            feature_id = f"spatial_{len(self.objects)}"
            
            # 将空间特征转换为对象
            position = feature.centroid
            
            self.objects[feature_id] = WorldObject(
                id=feature_id,
                position=position,
                attributes={
                    'type': 'spatial_feature',
                    'volume': feature.volume,
                    'surface_area': feature.surface_area,
                    'bounds': feature.bounds.tolist()
                },
                confidence=0.8,
                last_seen=current_time,
                modality_sources=['spatial']
            )
    
    def _process_audio_data(self, data: PerceptionData, current_time: float):
        """处理音频数据"""
        # 音频数据主要是语义信息，添加到环境状态
        if 'text' in data.data:
            self.current_state['last_audio'] = {
                'text': data.data['text'],
                'confidence': data.data.get('confidence', 0.0),
                'timestamp': current_time
            }
    
    def _cleanup_expired_objects(self, current_time: float):
        """清理过期对象"""
        expired_ids = []
        for obj_id, obj in self.objects.items():
            if current_time - obj.last_seen > self.object_max_age:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            del self.objects[obj_id]
            if obj_id in self.spatial_relationships:
                del self.spatial_relationships[obj_id]
    
    def _update_spatial_relationships(self):
        """更新空间关系"""
        obj_ids = list(self.objects.keys())
        n_objects = len(obj_ids)
        
        # 重置关系矩阵
        self.spatial_relationships = {obj_id: {} for obj_id in obj_ids}
        
        # 计算对象间距离
        if n_objects > 1:
            positions = np.array([self.objects[obj_id].position for obj_id in obj_ids])
            
            for i, obj_id1 in enumerate(obj_ids):
                for j, obj_id2 in enumerate(obj_ids):
                    if i != j:
                        distance = np.linalg.norm(positions[i] - positions[j])
                        self.spatial_relationships[obj_id1][obj_id2] = distance
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前世界状态"""
        return {
            'objects': {obj_id: asdict(obj) for obj_id, obj in self.objects.items()},
            'spatial_relationships': self.spatial_relationships,
            'current_state': self.current_state,
            'num_objects': len(self.objects),
            'timestamp': time.time()
        }


class MultimodalFusion:
    """多模态融合和特征提取"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.fusion_weights = {
            'visual': 0.4,
            'spatial': 0.3,
            'audio': 0.3
        }
        
        # 特征提取器
        self.visual_encoder = self._create_visual_encoder()
        self.spatial_encoder = self._create_spatial_encoder()
        self.audio_encoder = self._create_audio_encoder()
    
    def _create_visual_encoder(self) -> Dict[str, Any]:
        """创建视觉编码器"""
        return {
            'type': 'cnn',
            'input_shape': (224, 224, 3),
            'output_dim': self.feature_dim // 3
        }
    
    def _create_spatial_encoder(self) -> Dict[str, Any]:
        """创建空间编码器"""
        return {
            'type': 'point_net',
            'input_dim': 3,
            'output_dim': self.feature_dim // 3
        }
    
    def _create_audio_encoder(self) -> Dict[str, Any]:
        """创建音频编码器"""
        return {
            'type': 'spectral',
            'n_mfcc': 13,
            'output_dim': self.feature_dim // 3
        }
    
    def extract_fused_features(self, perception_data: List[PerceptionData]) -> np.ndarray:
        """提取融合特征"""
        features = []
        
        for data in perception_data:
            if data.modality == 'visual':
                feature = self._extract_visual_features(data)
            elif data.modality == 'spatial':
                feature = self._extract_spatial_features(data)
            elif data.modality == 'audio':
                feature = self._extract_audio_features(data)
            else:
                continue
            
            if feature is not None:
                features.append(feature * data.confidence)  # 按置信度加权
        
        # 特征融合
        if len(features) == 0:
            return np.zeros(self.feature_dim)
        
        # 对不同模态进行加权融合
        fused_features = np.zeros(self.feature_dim)
        
        for i, feature in enumerate(features):
            modality = perception_data[i].modality
            weight = self.fusion_weights.get(modality, 1.0)
            
            if len(feature) + len(fused_features) > self.feature_dim:
                # 截断特征
                feature = feature[:self.feature_dim - len(fused_features)]
            
            fused_features[:len(feature)] += feature * weight
        
        # 归一化
        if np.linalg.norm(fused_features) > 0:
            fused_features = fused_features / np.linalg.norm(fused_features)
        
        return fused_features
    
    def _extract_visual_features(self, data: PerceptionData) -> Optional[np.ndarray]:
        """提取视觉特征"""
        try:
            frame = data.data.get('frame')
            if frame is None:
                return None
            
            # 调整大小
            if len(frame.shape) == 3:
                frame = cv2.resize(frame, (224, 224))
                # 全局平均池化特征（简化版）
                features = np.mean(frame, axis=(0, 1, 2))
                # 重复到目标维度
                repeated = np.tile(features, self.visual_encoder['output_dim'] // 3 + 1)
                return repeated[:self.visual_encoder['output_dim']]
            return None
        except:
            return None
    
    def _extract_spatial_features(self, data: PerceptionData) -> Optional[np.ndarray]:
        """提取空间特征"""
        try:
            features = data.data.get('features', [])
            if not features:
                return None
            
            # 提取空间特征的统计信息
            if len(features) > 0:
                feature = features[0]  # 使用第一个特征
                
                # 组合特征
                spatial_stats = np.array([
                    feature.volume,
                    feature.surface_area,
                    np.linalg.norm(feature.orientation)
                ])
                
                # 重复到目标维度
                repeated = np.tile(spatial_stats, self.spatial_encoder['output_dim'] // 3 + 1)
                return repeated[:self.spatial_encoder['output_dim']]
            return None
        except:
            return None
    
    def _extract_audio_features(self, data: PerceptionData) -> Optional[np.ndarray]:
        """提取音频特征"""
        try:
            audio = data.data.get('audio')
            sample_rate = data.data.get('sample_rate', 16000)
            
            if audio is None or len(audio) == 0:
                return None
            
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=sample_rate, 
                n_mfcc=self.audio_encoder['n_mfcc']
            )
            
            # 计算统计特征
            feature_vector = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1)
            ])
            
            # 重复到目标维度
            repeated = np.tile(feature_vector, self.audio_encoder['output_dim'] // len(feature_vector) + 1)
            return repeated[:self.audio_encoder['output_dim']]
        
        except Exception as e:
            logger.error(f"音频特征提取错误: {e}")
            return None
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似度"""
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))


class MultimodalSensingSystem:
    """多模态感知系统主控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个感知模块
        self.camera_perception = CameraPerception(
            camera_id=self.config.get('camera_id', 0),
            enable_object_detection=self.config.get('enable_object_detection', True)
        )
        
        self.audio_perception = AudioPerception(
            sample_rate=self.config.get('audio_sample_rate', 16000),
            chunk_size=self.config.get('audio_chunk_size', 1024)
        )
        
        self.spatial_perception = SpatialPerception(
            num_points=self.config.get('num_points', 20000)
        )
        
        self.world_model = WorldModel()
        self.fusion_engine = MultimodalFusion(
            feature_dim=self.config.get('feature_dim', 512)
        )
        
        # 数据队列
        self.perception_queue = Queue(maxsize=50)
        self.processed_queue = Queue(maxsize=20)
        
        # 系统状态
        self.is_running = False
        self.perception_threads = []
        self.fusion_thread = None
        
        # 性能统计
        self.stats = {
            'frame_count': 0,
            'audio_count': 0,
            'point_cloud_count': 0,
            'fusion_count': 0
        }
    
    def start_system(self):
        """启动多模态感知系统"""
        logger.info("启动多模态感知系统...")
        
        self.is_running = True
        
        # 启动各感知模块
        self.camera_perception.start_capture()
        self.audio_perception.start_recording()
        self.spatial_perception.start_point_cloud_capture()
        
        # 启动融合处理线程
        self.fusion_thread = threading.Thread(target=self._fusion_loop)
        self.fusion_thread.start()
        
        # 启动感知数据聚合线程
        perception_thread = threading.Thread(target=self._perception_loop)
        perception_thread.start()
        self.perception_threads.append(perception_thread)
        
        logger.info("多模态感知系统启动完成")
    
    def stop_system(self):
        """停止多模态感知系统"""
        logger.info("停止多模态感知系统...")
        
        self.is_running = False
        
        # 停止各感知模块
        self.camera_perception.stop_capture()
        self.audio_perception.stop_recording()
        self.spatial_perception.stop_point_cloud_capture()
        
        # 等待线程结束
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join()
        
        for thread in self.perception_threads:
            if thread.is_alive():
                thread.join()
        
        logger.info("多模态感知系统已停止")
    
    def _perception_loop(self):
        """感知数据聚合循环"""
        while self.is_running:
            try:
                current_data = []
                
                # 获取视觉数据
                visual_data = self.camera_perception.get_latest_frame()
                if visual_data:
                    perception_obj = PerceptionData(
                        timestamp=visual_data['timestamp'],
                        modality='visual',
                        data=visual_data,
                        confidence=0.8
                    )
                    current_data.append(perception_obj)
                    self.stats['frame_count'] += 1
                
                # 获取音频数据
                audio_data = self.audio_perception.get_latest_audio()
                if audio_data:
                    # 语音转文本
                    transcription = self.audio_perception.transcribe_audio(
                        audio_data['audio'], 
                        audio_data['sample_rate']
                    )
                    audio_data.update(transcription)
                    
                    perception_obj = PerceptionData(
                        timestamp=audio_data['timestamp'],
                        modality='audio',
                        data=audio_data,
                        confidence=transcription.get('confidence', 0.5)
                    )
                    current_data.append(perception_obj)
                    self.stats['audio_count'] += 1
                
                # 获取空间数据
                spatial_data = self.spatial_perception.get_latest_point_cloud()
                if spatial_data:
                    # 提取空间特征
                    features = self.spatial_perception.extract_spatial_features(
                        spatial_data['point_cloud']
                    )
                    spatial_data['features'] = features
                    
                    perception_obj = PerceptionData(
                        timestamp=spatial_data['timestamp'],
                        modality='spatial',
                        data=spatial_data,
                        confidence=0.7
                    )
                    current_data.append(perception_obj)
                    self.stats['point_cloud_count'] += 1
                
                # 添加到处理队列
                if current_data:
                    try:
                        self.perception_queue.put_nowait(current_data)
                    except:
                        pass  # 队列满，跳过
                
                time.sleep(0.1)  # 10Hz更新率
            
            except Exception as e:
                logger.error(f"感知循环错误: {e}")
                time.sleep(1)
    
    def _fusion_loop(self):
        """融合处理循环"""
        while self.is_running:
            try:
                # 获取感知数据
                try:
                    perception_data = self.perception_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 更新世界模型
                self.world_model.update_world_state(perception_data)
                
                # 特征融合
                fused_features = self.fusion_engine.extract_fused_features(perception_data)
                
                # 构建输出数据
                output = {
                    'timestamp': time.time(),
                    'world_state': self.world_model.get_current_state(),
                    'fused_features': fused_features.tolist(),
                    'raw_perception': [asdict(data) for data in perception_data],
                    'stats': self.stats.copy()
                }
                
                # 添加到输出队列
                try:
                    self.processed_queue.put_nowait(output)
                except:
                    pass  # 队列满，丢弃旧数据
                
                self.stats['fusion_count'] += 1
            
            except Exception as e:
                logger.error(f"融合处理错误: {e}")
                time.sleep(1)
    
    def get_latest_perception(self) -> Optional[Dict[str, Any]]:
        """获取最新感知结果"""
        try:
            return self.processed_queue.get_nowait()
        except Empty:
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'objects_count': len(self.world_model.objects),
            'queue_sizes': {
                'perception': self.perception_queue.qsize(),
                'processed': self.processed_queue.qsize()
            },
            'uptime': time.time() - (getattr(self, 'start_time', time.time()))
        }


# 使用示例和测试代码
def demo_multimodal_sensing():
    """多模态感知系统演示"""
    print("=== 多模态世界模型感知系统演示 ===")
    
    # 配置系统
    config = {
        'camera_id': 0,
        'enable_object_detection': True,
        'audio_sample_rate': 16000,
        'num_points': 10000,
        'feature_dim': 512
    }
    
    # 创建系统
    sensing_system = MultimodalSensingSystem(config)
    
    try:
        # 启动系统
        sensing_system.start_system()
        sensing_system.start_time = time.time()
        
        # 运行演示
        print("系统运行中，收集感知数据...")
        
        for i in range(20):  # 演示20秒
            time.sleep(1)
            
            # 获取最新感知结果
            perception = sensing_system.get_latest_perception()
            if perception:
                world_state = perception['world_state']
                stats = perception['stats']
                
                print(f"时刻 {i+1}:")
                print(f"  检测到的对象数: {world_state['num_objects']}")
                print(f"  帧数: {stats['frame_count']}")
                print(f"  音频段数: {stats['audio_count']}")
                print(f"  点云数: {stats['point_cloud_count']}")
                print(f"  融合次数: {stats['fusion_count']}")
                
                if world_state['current_state'].get('last_audio'):
                    audio_text = world_state['current_state']['last_audio']['text']
                    if audio_text:
                        print(f"  最新语音: {audio_text}")
                
                print()
            
            # 每5秒显示系统状态
            if (i + 1) % 5 == 0:
                status = sensing_system.get_system_status()
                print("系统状态:", json.dumps(status, indent=2, ensure_ascii=False))
                print()
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止系统...")
    
    finally:
        # 停止系统
        sensing_system.stop_system()
        print("系统已停止")


if __name__ == "__main__":
    # 运行演示
    demo_multimodal_sensing()