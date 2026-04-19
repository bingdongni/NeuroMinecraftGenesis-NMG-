"""
OpenCV真实物体识别系统 - 主检测器
ObjectDetector类：集成所有组件的主要物体检测器
"""

import cv2
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

# 导入本地模块
from .model_loader import ModelLoader
from .detection_engine import DetectionEngine, DetectionResult
from .classification_system import ClassificationSystem, ClassificationResult
from .object_tracker import ObjectTracker, Track


class ObjectDetector:
    """
    物体检测器主类
    功能：
    - 集成模型加载、检测、分类和跟踪功能
    - 提供实时物体识别API
    - 支持水瓶、书本、笔等日常物体检测
    - 高性能实时处理（<100ms检测时间）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化物体检测器
        
        Args:
            config: 配置字典
        """
        # 默认配置
        self.config = {
            'model_type': 'yolo',
            'model_name': 'yolov8n',
            'device': 'cpu',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'max_disappeared': 30,
            'max_distance': 100,
            'enable_tracking': True,
            'enable_classification': True,
            'enable_visualization': True,
            'max_detection_time': 100.0,  # ms
            'min_detection_area': 100,
            'max_detection_area': 50000,
            'thread_pool_size': 4
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.model_loader = ModelLoader()
        self.detection_engine = DetectionEngine(
            confidence_threshold=self.config['confidence_threshold'],
            nms_threshold=self.config['nms_threshold']
        )
        self.classification_system = ClassificationSystem()
        self.object_tracker = ObjectTracker(
            max_disappeared=self.config['max_disappeared'],
            max_distance=self.config['max_distance']
        )
        
        # 状态管理
        self.model = None
        self.is_initialized = False
        self.detection_history = []
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_detection_time': 0.0,
            'accuracy_rate': 0.0,
            'last_update': 0.0
        }
        
        # 线程池用于并发处理
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['thread_pool_size'])
        
        # 缓存
        self.roi_cache = {}
        self.classification_cache = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("物体检测器初始化开始")
        
        # 初始化完成
        self.is_initialized = True
        self.logger.info("物体检测器初始化完成")
    
    def load_detection_model(self, model_type: Optional[str] = None, 
                           model_name: Optional[str] = None, 
                           device: Optional[str] = None) -> bool:
        """
        加载检测模型
        
        Args:
            model_type: 模型类型 ('yolo', 'ssd', 'rcnn')
            model_name: 模型名称
            device: 设备类型 ('cpu', 'cuda')
            
        Returns:
            加载是否成功
        """
        try:
            # 使用配置或参数
            model_type = model_type or self.config['model_type']
            model_name = model_name or self.config['model_name']
            device = device or self.config['device']
            
            self.logger.info(f"正在加载检测模型: {model_type}/{model_name} on {device}")
            
            start_time = time.time()
            
            # 加载模型
            if model_type.lower() == 'yolo':
                self.model = self.model_loader.load_yolo_model(model_name, device)
            elif model_type.lower() == 'ssd':
                self.model = self.model_loader.load_ssd_model(model_name)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            if self.model is None:
                raise ValueError(f"模型加载失败: {model_type}/{model_name}")
            
            load_time = time.time() - start_time
            self.logger.info(f"模型加载成功，耗时: {load_time:.2f}秒")
            
            # 更新配置
            self.config.update({
                'model_type': model_type,
                'model_name': model_name,
                'device': device
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载检测模型失败: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray, return_tracks: bool = True, 
                      return_classifications: bool = True) -> Dict[str, Any]:
        """
        检测图像中的物体
        
        Args:
            image: 输入图像
            return_tracks: 是否返回轨迹信息
            return_classifications: 是否返回分类信息
            
        Returns:
            检测结果字典
        """
        if not self.is_initialized or self.model is None:
            raise RuntimeError("检测器未初始化或模型未加载")
        
        start_time = time.time()
        
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 执行检测
            detections = self._perform_detection(processed_image)
            
            # 过滤检测结果
            detections = self._filter_detections(detections)
            
            # 轨迹跟踪
            tracks = {}
            if self.config['enable_tracking'] and return_tracks:
                tracks = self.object_tracker.track_objects(
                    detections, time.time(), processed_image
                )
            
            # 物体分类
            classifications = {}
            if self.config['enable_classification'] and return_classifications:
                classifications = self._perform_classification(
                    processed_image, detections
                )
            
            # 后处理和结果整理
            result = self._postprocess_results(
                image, detections, tracks, classifications
            )
            
            # 更新统计信息
            self._update_performance_stats(start_time, len(detections))
            
            self.logger.debug(f"物体检测完成: {len(detections)}个物体, "
                            f"{len(tracks)}个轨迹, {len(classifications)}个分类, "
                            f"用时{time.time() - start_time:.3f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"物体检测失败: {e}")
            return self._create_error_result(str(e))
    
    def classify_object(self, roi: np.ndarray, class_name: str) -> ClassificationResult:
        """
        对特定区域进行分类
        
        Args:
            roi: 物体区域图像
            class_name: 基础类别名称
            
        Returns:
            分类结果
        """
        if not self.config['enable_classification']:
            return self._create_empty_classification_result(class_name)
        
        try:
            # 创建虚拟检测结果
            detection_result = type('DetectionResult', (), {
                'bbox': (0, 0, roi.shape[1], roi.shape[0]),
                'confidence': 1.0,
                'class_name': class_name,
                'center': (roi.shape[1] // 2, roi.shape[0] // 2),
                'area': roi.shape[0] * roi.shape[1],
                'aspect_ratio': roi.shape[1] / roi.shape[0] if roi.shape[0] > 0 else 0
            })()
            
            return self.classification_system.classify_object(roi, detection_result)
            
        except Exception as e:
            self.logger.error(f"物体分类失败: {e}")
            return self._create_empty_classification_result(class_name)
    
    def track_objects(self, detections: List[DetectionResult], 
                     timestamp: float, image: np.ndarray) -> Dict[int, Track]:
        """
        跟踪检测到的物体
        
        Args:
            detections: 检测结果列表
            timestamp: 时间戳
            image: 当前图像
            
        Returns:
            轨迹字典
        """
        return self.object_tracker.track_objects(detections, timestamp, image)
    
    def get_object_properties(self, detection: DetectionResult, 
                            classification: Optional[ClassificationResult] = None) -> Dict[str, Any]:
        """
        获取物体属性
        
        Args:
            detection: 检测结果
            classification: 分类结果（可选）
            
        Returns:
            物体属性字典
        """
        properties = {
            'basic_info': {
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'center': detection.center,
                'area': detection.area,
                'aspect_ratio': detection.aspect_ratio
            },
            'geometric_properties': self._calculate_geometric_properties(detection),
            'visual_properties': {}
        }
        
        # 添加分类属性
        if classification:
            properties['classification'] = {
                'primary_class': classification.primary_class,
                'primary_confidence': classification.primary_confidence,
                'sub_classes': classification.sub_classes,
                'size_category': classification.size_category,
                'material_estimate': classification.material_estimate,
                'color_info': classification.color_info,
                'shape_info': classification.shape_info,
                'attributes': classification.attributes
            }
        
        return properties
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 确保图像格式正确
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 图像归一化
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 调整图像尺寸（如果过大）
            max_size = 1920  # 最大宽度
            if image.shape[1] > max_size:
                scale = max_size / image.shape[1]
                new_width = max_size
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return image
    
    def _perform_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """
        执行物体检测
        
        Args:
            image: 预处理后的图像
            
        Returns:
            检测结果列表
        """
        try:
            if self.config['model_type'] == 'yolo':
                return self.detection_engine.detect_with_yolo(image, self.model)
            elif self.config['model_type'] == 'ssd':
                return self.detection_engine.detect_with_ssd(image, self.model)
            else:
                raise ValueError(f"不支持的模型类型: {self.config['model_type']}")
                
        except Exception as e:
            self.logger.error(f"检测执行失败: {e}")
            return []
    
    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        过滤检测结果
        
        Args:
            detections: 检测结果列表
            
        Returns:
            过滤后的检测结果
        """
        # 按置信度和面积过滤
        filtered = self.detection_engine.filter_detections(
            detections, 
            min_area=self.config['min_detection_area'],
            max_area=self.config['max_detection_area']
        )
        
        # 按置信度排序
        filtered = self.detection_engine.sort_detections_by_confidence(filtered, ascending=False)
        
        return filtered
    
    def _perform_classification(self, image: np.ndarray, 
                              detections: List[DetectionResult]) -> Dict[int, ClassificationResult]:
        """
        执行物体分类
        
        Args:
            image: 图像
            detections: 检测结果列表
            
        Returns:
            分类结果字典
        """
        classifications = {}
        
        # 使用线程池并发分类
        futures = {}
        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            roi = image[y:y+h, x:x+w]
            
            if roi.size > 0:
                # 检查缓存
                cache_key = f"{detection.class_name}_{w}_{h}"
                if cache_key in self.classification_cache:
                    classifications[i] = self.classification_cache[cache_key]
                else:
                    # 提交到线程池
                    future = self.thread_pool.submit(
                        self.classification_system.classify_object, roi, detection
                    )
                    futures[i] = future
        
        # 获取分类结果
        for i, future in futures.items():
            try:
                classification_result = future.result(timeout=1.0)  # 1秒超时
                classifications[i] = classification_result
                
                # 缓存结果
                cache_key = f"{detections[i].class_name}_{detections[i].bbox[2]}_{detections[i].bbox[3]}"
                self.classification_cache[cache_key] = classification_result
                
                # 清理缓存（防止无限增长）
                if len(self.classification_cache) > 1000:
                    self.classification_cache = dict(list(self.classification_cache.items())[-500:])
                    
            except Exception as e:
                self.logger.error(f"分类失败: {e}")
                classifications[i] = self._create_empty_classification_result(
                    detections[i].class_name
                )
        
        return classifications
    
    def _calculate_geometric_properties(self, detection: DetectionResult) -> Dict[str, Any]:
        """
        计算几何属性
        
        Args:
            detection: 检测结果
            
        Returns:
            几何属性字典
        """
        x, y, w, h = detection.bbox
        
        return {
            'width': w,
            'height': h,
            'perimeter': 2 * (w + h),
            'diagonal_length': np.sqrt(w**2 + h**2),
            'circularity': 4 * np.pi * detection.area / (2 * (w + h))**2 if (w + h) > 0 else 0,
            'solidity': 1.0,  # 简化为1.0，实际需要轮廓分析
            'extent': detection.area / (w * h) if (w * h) > 0 else 0,
            'equi_diameter': 2 * np.sqrt(detection.area / np.pi),
            'center_coordinates': detection.center,
            'bounding_box_aspect_ratio': detection.aspect_ratio
        }
    
    def _postprocess_results(self, original_image: np.ndarray, 
                           detections: List[DetectionResult],
                           tracks: Dict[int, Track],
                           classifications: Dict[int, ClassificationResult]) -> Dict[str, Any]:
        """
        后处理结果
        
        Args:
            original_image: 原始图像
            detections: 检测结果列表
            tracks: 轨迹字典
            classifications: 分类结果字典
            
        Returns:
            处理后的结果字典
        """
        # 构建物体列表
        objects = []
        
        for i, detection in enumerate(detections):
            # 获取对应的分类结果
            classification = classifications.get(i)
            
            # 获取轨迹信息
            track_info = None
            for track_id, track in tracks.items():
                if self._is_same_detection(track, detection):
                    track_info = {
                        'track_id': track_id,
                        'age': track.age,
                        'velocity': track.velocity,
                        'total_visible_count': track.total_visible_count
                    }
                    break
            
            # 获取物体属性
            properties = self.get_object_properties(detection, classification)
            
            # 创建物体对象
            obj = {
                'detection': {
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'class_name': detection.class_name,
                    'class_name_cn': detection.class_name_cn,
                    'center': detection.center,
                    'area': detection.area,
                    'aspect_ratio': detection.aspect_ratio
                },
                'properties': properties,
                'track_info': track_info
            }
            
            # 添加分类信息
            if classification:
                obj['classification'] = {
                    'primary_class': classification.primary_class,
                    'primary_confidence': classification.primary_confidence,
                    'sub_classes': classification.sub_classes,
                    'size_category': classification.size_category,
                    'material_estimate': classification.material_estimate,
                    'color_info': classification.color_info,
                    'shape_info': classification.shape_info,
                    'attributes': classification.attributes
                }
            
            objects.append(obj)
        
        # 添加可视化（如果启用）
        if self.config['enable_visualization']:
            visualization = self._create_visualization(original_image, detections, tracks)
        else:
            visualization = None
        
        # 构建最终结果
        result = {
            'success': True,
            'timestamp': time.time(),
            'image_shape': original_image.shape,
            'object_count': len(detections),
            'objects': objects,
            'detection_engine_stats': self.detection_engine.get_performance_stats(),
            'tracking_stats': self.object_tracker.get_tracking_stats(),
            'visualization': visualization
        }
        
        return result
    
    def _is_same_detection(self, track: Track, detection: DetectionResult) -> bool:
        """
        检查轨迹和检测是否为同一物体
        
        Args:
            track: 轨迹对象
            detection: 检测结果
            
        Returns:
            是否为同一物体
        """
        # 简单的中心点距离检查
        track_center = (track.bbox[0] + track.bbox[2] // 2, 
                       track.bbox[1] + track.bbox[3] // 2)
        detection_center = detection.center
        
        distance = np.sqrt((track_center[0] - detection_center[0])**2 + 
                          (track_center[1] - detection_center[1])**2)
        
        return distance < 50  # 50像素阈值
    
    def _create_visualization(self, image: np.ndarray, detections: List[DetectionResult],
                            tracks: Dict[int, Track]) -> Dict[str, Any]:
        """
        创建可视化结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            tracks: 轨迹字典
            
        Returns:
            可视化结果
        """
        try:
            vis_image = image.copy()
            
            # 绘制检测框
            for detection in detections:
                x, y, w, h = detection.bbox
                
                # 选择颜色
                color = self._get_class_color(detection.class_name)
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # 绘制标签
                label = f"{detection.class_name_cn}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(vis_image, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制轨迹
            for track_id, track in tracks.items():
                if len(track.history) > 1:
                    # 绘制轨迹线
                    for i in range(1, len(track.history)):
                        _, prev_bbox = track.history[i-1]
                        _, curr_bbox = track.history[i]
                        
                        prev_center = (prev_bbox[0] + prev_bbox[2] // 2, 
                                     prev_bbox[1] + prev_bbox[3] // 2)
                        curr_center = (curr_bbox[0] + curr_bbox[2] // 2, 
                                     curr_bbox[1] + curr_bbox[3] // 2)
                        
                        cv2.line(vis_image, prev_center, curr_center, 
                               (0, 255, 0), 2)
                    
                    # 绘制轨迹ID
                    x, y, w, h = track.bbox
                    cv2.putText(vis_image, f"ID:{track_id}", (x, y + h + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 转换为base64
            import base64
            import io
            img_buffer = io.BytesIO()
            cv2.imwrite(img_buffer, vis_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return {
                'image': img_str,
                'detection_count': len(detections),
                'track_count': len(tracks)
            }
            
        except Exception as e:
            self.logger.error(f"创建可视化失败: {e}")
            return None
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        获取类别颜色
        
        Args:
            class_name: 类别名称
            
        Returns:
            BGR颜色元组
        """
        color_map = {
            'bottle': (0, 0, 255),      # 红色
            'book': (0, 165, 255),      # 橙色
            'pen': (0, 255, 0),         # 绿色
            'laptop': (255, 0, 0),      # 蓝色
            'phone': (128, 0, 128),     # 紫色
            'cup': (0, 255, 255),       # 青色
            'chair': (255, 255, 0),     # 黄色
            'table': (255, 0, 255),     # 品红
            'bag': (128, 128, 128),     # 灰色
            'wallet': (0, 128, 128)     # 深青色
        }
        
        return color_map.get(class_name, (255, 255, 255))
    
    def _create_empty_classification_result(self, class_name: str) -> Any:
        """创建空的分类结果"""
        return type('ClassificationResult', (), {
            'primary_class': class_name,
            'primary_confidence': 0.0,
            'sub_classes': {},
            'attributes': {},
            'color_info': {},
            'shape_info': {},
            'size_category': 'unknown',
            'material_estimate': 'unknown'
        })()
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'error': error_msg,
            'timestamp': time.time(),
            'object_count': 0,
            'objects': []
        }
    
    def _update_performance_stats(self, start_time: float, detection_count: int):
        """更新性能统计"""
        detection_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 更新统计
        self.performance_stats['total_detections'] += 1
        self.performance_stats['last_update'] = time.time()
        
        if detection_count > 0:
            self.performance_stats['successful_detections'] += 1
        
        # 计算平均检测时间
        total_time = (self.performance_stats['average_detection_time'] * 
                     (self.performance_stats['total_detections'] - 1) + detection_time)
        self.performance_stats['average_detection_time'] = (
            total_time / self.performance_stats['total_detections']
        )
        
        # 计算准确率
        if self.performance_stats['total_detections'] > 0:
            self.performance_stats['accuracy_rate'] = (
                self.performance_stats['successful_detections'] / 
                self.performance_stats['total_detections']
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        stats = self.performance_stats.copy()
        stats.update({
            'detection_engine_stats': self.detection_engine.get_performance_stats(),
            'tracking_stats': self.object_tracker.get_tracking_stats(),
            'system_info': {
                'model_type': self.config['model_type'],
                'model_name': self.config['model_name'],
                'device': self.config['device'],
                'thread_pool_size': self.config['thread_pool_size']
            }
        })
        
        return stats
    
    def save_detection_report(self, results: List[Dict[str, Any]], 
                            output_path: Union[str, Path]) -> None:
        """
        保存检测报告
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
        """
        try:
            report = {
                'report_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'detector_config': self.config,
                'performance_stats': self.get_performance_stats(),
                'total_detections': len(results),
                'detection_results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"检测报告已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存检测报告失败: {e}")
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_detection_time': 0.0,
            'accuracy_rate': 0.0,
            'last_update': 0.0
        }
        
        self.detection_engine.reset_performance_stats()
        self.object_tracker.reset_tracker()
        
        self.logger.info("性能统计已重置")
    
    def set_config(self, **kwargs):
        """
        设置配置参数
        
        Args:
            **kwargs: 配置参数
        """
        self.config.update(kwargs)
        
        # 更新组件配置
        if 'confidence_threshold' in kwargs:
            self.detection_engine.set_thresholds(confidence=kwargs['confidence_threshold'])
        
        if 'nms_threshold' in kwargs:
            self.detection_engine.set_thresholds(nms=kwargs['nms_threshold'])
        
        if 'max_disappeared' in kwargs or 'max_distance' in kwargs:
            max_disappeared = kwargs.get('max_disappeared', self.config['max_disappeared'])
            max_distance = kwargs.get('max_distance', self.config['max_distance'])
            self.object_tracker.set_tracking_params(max_disappeared, max_distance)
        
        self.logger.info(f"配置已更新: {kwargs}")
    
    def export_model_info(self) -> Dict[str, Any]:
        """
        导出模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_loader_info': {
                'loaded_models': self.model_loader.list_loaded_models(),
                'available_models': self.model_loader.get_available_models(),
                'object_classes': self.model_loader.get_object_classes()
            },
            'detection_engine_info': {
                'confidence_threshold': self.detection_engine.confidence_threshold,
                'nms_threshold': self.detection_engine.nms_threshold
            },
            'object_tracker_info': {
                'max_disappeared': self.object_tracker.max_disappeared,
                'max_distance': self.object_tracker.max_distance
            }
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)