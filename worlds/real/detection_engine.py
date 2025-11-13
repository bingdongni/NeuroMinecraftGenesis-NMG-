"""
OpenCV真实物体识别系统 - 检测引擎
DetectionEngine类：负责执行物体检测的核心算法
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    class_id: int
    class_name: str
    class_name_cn: str
    center: Tuple[int, int]
    area: int
    aspect_ratio: float


class DetectionEngine:
    """
    物体检测引擎类
    功能：
    - 执行YOLO、SSD、RCNN等算法
    - 处理图像预处理和后处理
    - 提供实时检测性能
    - 支持多物体检测
    """
    
    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        初始化检测引擎
        
        Args:
            confidence_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # 检测性能统计
        self.detection_times = []
        self.average_inference_time = 0.0
        
        # 物体类别映射
        self.object_classes = {
            0: 'bottle',
            1: 'book', 
            2: 'pen',
            3: 'laptop',
            4: 'phone',
            5: 'cup',
            6: 'chair',
            7: 'table',
            8: 'bag',
            9: 'wallet'
        }
        
        self.object_classes_cn = {
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
        
        # YOLO标签文件
        self.yolo_labels_path = self._create_yolo_labels_file()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("检测引擎初始化完成")
    
    def detect_with_yolo(self, image: np.ndarray, model: Any) -> List[DetectionResult]:
        """
        使用YOLO模型进行检测
        
        Args:
            image: 输入图像
            model: YOLO模型对象
            
        Returns:
            检测结果列表
        """
        start_time = time.time()
        
        try:
            # 检查模型类型
            if hasattr(model, 'predict'):
                # Ultralytics YOLO模型
                results = model.predict(image, conf=self.confidence_threshold, verbose=False)
                
                detections = []
                if results and len(results) > 0:
                    result = results[0]  # 取第一个结果
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i in range(len(boxes.xyxy)):
                            # 获取边界框
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            
                            # 获取置信度和类别
                            conf = float(boxes.conf[i])
                            class_id = int(boxes.cls[i])
                            
                            # 过滤低置信度检测
                            if conf >= self.confidence_threshold:
                                class_name = self.object_classes.get(class_id, f'class_{class_id}')
                                class_name_cn = self.object_classes_cn.get(class_name, class_name)
                                
                                # 计算中心点和面积
                                center = (x + w // 2, y + h // 2)
                                area = w * h
                                aspect_ratio = w / h if h > 0 else 0
                                
                                detection = DetectionResult(
                                    bbox=(x, y, w, h),
                                    confidence=conf,
                                    class_id=class_id,
                                    class_name=class_name,
                                    class_name_cn=class_name_cn,
                                    center=center,
                                    area=area,
                                    aspect_ratio=aspect_ratio
                                )
                                detections.append(detection)
                
            else:
                # OpenCV DNN YOLO模型
                detections = self._detect_with_opencv_yolo(image, model)
            
            # 应用NMS
            detections = self._apply_nms(detections)
            
            # 记录性能
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            self.logger.debug(f"YOLO检测完成: {len(detections)}个物体, 用时{inference_time:.3f}秒")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO检测失败: {e}")
            return []
    
    def _detect_with_opencv_yolo(self, image: np.ndarray, net: Any) -> List[DetectionResult]:
        """
        使用OpenCV DNN进行YOLO检测
        
        Args:
            image: 输入图像
            net: OpenCV DNN网络对象
            
        Returns:
            检测结果列表
        """
        # 图像预处理
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # 设置输入
        net.setInput(blob)
        
        # 前向传播
        outs = net.forward()
        
        # 后处理
        detections = []
        
        # 获取图像尺寸
        (h, w) = image.shape[:2]
        
        # 解析输出
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # 获取边界框
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # 应用NMS
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # 生成检测结果
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                
                class_name = self.object_classes.get(class_id, f'class_{class_id}')
                class_name_cn = self.object_classes_cn.get(class_name, class_name)
                
                center = (x + w // 2, y + h // 2)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                detection = DetectionResult(
                    bbox=(x, y, w, h),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    class_name_cn=class_name_cn,
                    center=center,
                    area=area,
                    aspect_ratio=aspect_ratio
                )
                detections.append(detection)
        
        return detections
    
    def detect_with_ssd(self, image: np.ndarray, model: Any) -> List[DetectionResult]:
        """
        使用SSD模型进行检测
        
        Args:
            image: 输入图像
            model: SSD模型对象
            
        Returns:
            检测结果列表
        """
        start_time = time.time()
        
        try:
            # 图像预处理
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5))
            
            # 设置输入
            model.setInput(blob)
            
            # 前向传播
            detections = model.forward()
            
            # 解析检测结果
            results = []
            (h, w) = image.shape[:2]
            
            # 遍历检测结果
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # 过滤低置信度检测
                if confidence > self.confidence_threshold:
                    # 获取边界框
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # 确保边界框在图像范围内
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # 计算实际尺寸
                    x = startX
                    y = startY
                    width = endX - startX
                    height = endY - startY
                    
                    # 获取类别ID
                    class_id = int(detections[0, 0, i, 1])
                    
                    # 映射到自定义类别
                    mapped_class_id = self._map_ssd_class(class_id)
                    class_name = self.object_classes.get(mapped_class_id, f'class_{mapped_class_id}')
                    class_name_cn = self.object_classes_cn.get(class_name, class_name)
                    
                    # 计算属性
                    center = (x + width // 2, y + height // 2)
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    detection = DetectionResult(
                        bbox=(x, y, width, height),
                        confidence=confidence,
                        class_id=mapped_class_id,
                        class_name=class_name,
                        class_name_cn=class_name_cn,
                        center=center,
                        area=area,
                        aspect_ratio=aspect_ratio
                    )
                    results.append(detection)
            
            # 记录性能
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            self.logger.debug(f"SSD检测完成: {len(results)}个物体, 用时{inference_time:.3f}秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"SSD检测失败: {e}")
            return []
    
    def _map_ssd_class(self, ssd_class_id: int) -> int:
        """
        映射SSD类别到自定义类别
        
        Args:
            ssd_class_id: SSD模型输出的类别ID
            
        Returns:
            映射后的类别ID
        """
        # COCO数据集类别映射 (简化版)
        coco_mapping = {
            1: 9,   # person -> wallet (近似)
            2: 3,   # bicycle -> laptop (近似)
            3: 3,   # car -> laptop (近似)
            4: 3,   # motorbike -> laptop (近似)
            5: 4,   # airplane -> phone (近似)
            6: 6,   # bus -> cup (近似)
            7: 3,   # train -> laptop (近似)
            8: 4,   # truck -> phone (近似)
            9: 5,   # boat -> cup (近似)
            10: 6,  # traffic light -> cup (近似)
            11: 8,  # fire hydrant -> bag (近似)
            12: 6,  # stop sign -> cup (近似)
            13: 7,  # parking meter -> chair (近似)
            14: 7,  # bench -> chair (近似)
            15: 2,  # bird -> pen (近似)
            16: 0,  # cat -> bottle (近似)
            17: 1,  # dog -> book (近似)
            18: 1,  # horse -> book (近似)
            19: 1,  # sheep -> book (近似)
            20: 1,  # cow -> book (近似)
            21: 1,  # elephant -> book (近似)
            22: 1,  # bear -> book (近似)
            23: 1,  # zebra -> book (近似)
            24: 1,  # giraffe -> book (近似)
            25: 2,  # backpack -> pen (近似)
            26: 2,  # umbrella -> pen (近似)
            27: 1,  # handbag -> book (近似)
            28: 2,  # tie -> pen (近似)
            29: 0,  # suitcase -> bottle (近似)
            30: 0,  # frisbee -> bottle (近似)
            31: 1,  # skis -> book (近似)
            32: 1,  # snowboard -> book (近似)
            33: 0,  # sports ball -> bottle (近似)
            34: 0,  # kite -> bottle (近似)
            35: 1,  # baseball bat -> book (近似)
            36: 1,  # baseball glove -> book (近似)
            37: 0,  # skateboard -> bottle (近似)
            38: 0,  # surfboard -> bottle (近似)
            39: 0,  # tennis racket -> bottle (近似)
            40: 0,  # bottle -> bottle
            41: 0,  # wine glass -> bottle (近似)
            42: 0,  # cup -> bottle (近似)
            43: 0,  # fork -> bottle (近似)
            44: 0,  # knife -> bottle (近似)
            45: 0,  # spoon -> bottle (近似)
            46: 1,  # bowl -> book (近似)
            47: 1,  # banana -> book (近似)
            48: 1,  # apple -> book (近似)
            49: 1,  # sandwich -> book (近似)
            50: 1,  # orange -> book (近似)
            51: 1,  # broccoli -> book (近似)
            52: 1,  # carrot -> book (近似)
            53: 1,  # hot dog -> book (近似)
            54: 1,  # pizza -> book (近似)
            55: 1,  # donut -> book (近似)
            56: 1,  # cake -> book (近似)
            57: 2,  # chair -> pen (近似)
            58: 7,  # couch -> table (近似)
            59: 6,  # potted plant -> chair (近似)
            60: 7,  # bed -> table (近似)
            61: 7,  # dining table -> table
            62: 4,  # toilet -> phone (近似)
            63: 3,  # tv -> laptop (近似)
            64: 3,  # laptop -> laptop
            65: 0,  # mouse -> bottle (近似)
            66: 3,  # remote -> laptop (近似)
            67: 3,  # keyboard -> laptop (近似)
            68: 3,  # cell phone -> phone
            69: 1,  # microwave -> book (近似)
            70: 3,  # oven -> laptop (近似)
            71: 3,  # toaster -> laptop (近似)
            72: 3,  # sink -> laptop (近似)
            73: 3,  # refrigerator -> laptop (近似)
            74: 1,  # book -> book
            75: 2,  # clock -> pen (近似)
            76: 2,  # vase -> pen (近似)
            77: 2,  # scissors -> pen (近似)
            78: 2,  # teddy bear -> pen (近似)
            79: 2,  # hair drier -> pen (近似)
            80: 0,  # toothbrush -> bottle (近似)
        }
        
        return coco_mapping.get(ssd_class_id, 0)
    
    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            抑制后的检测结果
        """
        if len(detections) <= 1:
            return detections
        
        # 提取边界框、置信度和类别
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x, y, w, h = det.bbox
            boxes.append([x, y, x + w, y + h])
            confidences.append(det.confidence)
            class_ids.append(det.class_id)
        
        # 转换为numpy数组
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)
        
        # 按类别分组进行NMS
        keep_indices = []
        
        unique_classes = np.unique(class_ids)
        for class_id in unique_classes:
            # 获取该类别的索引
            class_indices = np.where(class_ids == class_id)[0]
            
            if len(class_indices) == 1:
                keep_indices.append(class_indices[0])
                continue
            
            # 获取该类别的边界框和置信度
            class_boxes = boxes[class_indices]
            class_confidences = confidences[class_indices]
            
            # 计算面积
            class_areas = (class_boxes[:, 2] - class_boxes[:, 0]) * (class_boxes[:, 3] - class_boxes[:, 1])
            
            # 按置信度排序
            sorted_indices = np.argsort(class_confidences)[::-1]
            
            while len(sorted_indices) > 0:
                # 选择置信度最高的检测
                current = sorted_indices[0]
                keep_indices.append(class_indices[current])
                
                if len(sorted_indices) == 1:
                    break
                
                # 计算与剩余检测的IoU
                remaining_indices = sorted_indices[1:]
                
                # 当前边界框
                current_box = class_boxes[current]
                current_area = class_areas[current]
                
                # 其他边界框
                other_boxes = class_boxes[remaining_indices]
                other_areas = class_areas[remaining_indices]
                
                # 计算IoU
                xx1 = np.maximum(current_box[0], other_boxes[:, 0])
                yy1 = np.maximum(current_box[1], other_boxes[:, 1])
                xx2 = np.minimum(current_box[2], other_boxes[:, 2])
                yy2 = np.minimum(current_box[3], other_boxes[:, 3])
                
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                
                intersection = w * h
                union = current_area + other_areas - intersection
                iou = intersection / (union + 1e-6)
                
                # 保留IoU小于阈值的检测
                remaining_indices = remaining_indices[iou <= self.nms_threshold]
                sorted_indices = remaining_indices
        
        # 返回保留下来的检测结果
        keep_indices = sorted(keep_indices)
        return [detections[i] for i in keep_indices]
    
    def _update_performance_stats(self, inference_time: float) -> None:
        """
        更新性能统计信息
        
        Args:
            inference_time: 推理时间
        """
        self.detection_times.append(inference_time)
        
        # 保持最近100次的统计
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-100:]
        
        # 计算平均时间
        self.average_inference_time = np.mean(self.detection_times)
    
    def _create_yolo_labels_file(self) -> Path:
        """
        创建YOLO标签文件
        
        Returns:
            标签文件路径
        """
        labels_content = '\n'.join([
            'bottle',
            'book',
            'pen',
            'laptop',
            'phone',
            'cup',
            'chair',
            'table',
            'bag',
            'wallet'
        ])
        
        labels_path = Path('yolo_labels.txt')
        with open(labels_path, 'w') as f:
            f.write(labels_content)
        
        return labels_path
    
    def filter_detections(self, detections: List[DetectionResult], 
                         min_area: int = 100, max_area: int = 50000) -> List[DetectionResult]:
        """
        过滤检测结果
        
        Args:
            detections: 检测结果列表
            min_area: 最小面积
            max_area: 最大面积
            
        Returns:
            过滤后的检测结果
        """
        filtered = []
        
        for det in detections:
            if min_area <= det.area <= max_area:
                filtered.append(det)
        
        return filtered
    
    def sort_detections_by_confidence(self, detections: List[DetectionResult], 
                                    ascending: bool = False) -> List[DetectionResult]:
        """
        按置信度排序检测结果
        
        Args:
            detections: 检测结果列表
            ascending: 是否升序
            
        Returns:
            排序后的检测结果
        """
        return sorted(detections, key=lambda x: x.confidence, reverse=not ascending)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if not self.detection_times:
            return {
                'average_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0,
                'total_detections': 0,
                'fps_estimate': 0.0
            }
        
        return {
            'average_inference_time': self.average_inference_time,
            'min_inference_time': min(self.detection_times),
            'max_inference_time': max(self.detection_times),
            'total_detections': len(self.detection_times),
            'fps_estimate': 1.0 / self.average_inference_time if self.average_inference_time > 0 else 0.0
        }
    
    def reset_performance_stats(self) -> None:
        """重置性能统计"""
        self.detection_times.clear()
        self.average_inference_time = 0.0
    
    def set_thresholds(self, confidence: float = None, nms: float = None) -> None:
        """
        设置检测阈值
        
        Args:
            confidence: 置信度阈值
            nms: NMS阈值
        """
        if confidence is not None:
            self.confidence_threshold = confidence
        
        if nms is not None:
            self.nms_threshold = nms
        
        self.logger.info(f"阈值设置: 置信度={self.confidence_threshold}, NMS={self.nms_threshold}")