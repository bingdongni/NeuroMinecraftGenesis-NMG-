"""
OpenCV真实物体识别系统 - 物体跟踪器
ObjectTracker类：负责实时跟踪检测到的物体
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import time
from scipy.optimize import linear_sum_assignment
import math
from pathlib import Path


@dataclass
class Track:
    """物体轨迹数据类"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    timestamp: float
    history: List[Tuple[float, Tuple[int, int, int, int]]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    total_visible_count: int = 0
    consecutive_invisible_count: int = 0
    last_seen: float = 0.0
    color_histogram: np.ndarray = field(default_factory=lambda: np.zeros(256))
    features: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.history:
            self.history.append((self.timestamp, self.bbox))
        if self.last_seen == 0.0:
            self.last_seen = self.timestamp
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, timestamp: float, 
               color_hist: Optional[np.ndarray] = None, features: Optional[List[np.ndarray]] = None):
        """更新轨迹"""
        self.bbox = bbox
        self.confidence = confidence
        self.timestamp = timestamp
        
        # 更新历史记录
        if len(self.history) >= 10:  # 保持最近10个位置
            self.history.pop(0)
        self.history.append((timestamp, bbox))
        
        # 计算速度和加速度
        if len(self.history) >= 2:
            prev_time, prev_bbox = self.history[-2]
            dt = timestamp - prev_time
            
            if dt > 0:
                center_curr = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                center_prev = (prev_bbox[0] + prev_bbox[2] // 2, prev_bbox[1] + prev_bbox[3] // 2)
                
                velocity_x = (center_curr[0] - center_prev[0]) / dt
                velocity_y = (center_curr[1] - center_prev[1]) / dt
                
                # 计算加速度
                if len(self.history) >= 3:
                    _, prev_bbox2 = self.history[-3]
                    prev_time2 = self.history[-2][0]
                    
                    center_prev2 = (prev_bbox2[0] + prev_bbox2[2] // 2, prev_bbox2[1] + prev_bbox2[3] // 2)
                    
                    prev_velocity_x = (center_prev[0] - center_prev2[0]) / (prev_time2 - prev_time)
                    prev_velocity_y = (center_prev[1] - center_prev2[1]) / (prev_time2 - prev_time)
                    
                    acceleration_x = (velocity_x - prev_velocity_x) / dt
                    acceleration_y = (velocity_y - prev_velocity_y) / dt
                    
                    self.acceleration = (acceleration_x, acceleration_y)
                
                self.velocity = (velocity_x, velocity_y)
        
        # 更新其他属性
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        self.last_seen = timestamp
        self.age += 1
        
        # 更新特征
        if color_hist is not None:
            self.color_histogram = color_hist
        if features is not None:
            self.features.extend(features)
            # 只保留最近的特征
            if len(self.features) > 5:
                self.features = self.features[-5:]
    
    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """预测下一个位置"""
        if len(self.history) < 2:
            return self.bbox
        
        # 简单的线性预测
        timestamp, bbox = self.history[-1]
        
        # 使用最近的几个点进行线性回归
        recent_points = self.history[-5:] if len(self.history) >= 5 else self.history
        
        if len(recent_points) < 2:
            return bbox
        
        # 提取时间和中心点
        centers = []
        times = []
        
        for t, (x, y, w, h) in recent_points:
            centers.append((x + w // 2, y + h // 2))
            times.append(t)
        
        # 简单线性预测
        dt = times[-1] - times[0] if len(times) > 1 else 1
        
        if dt > 0:
            # 计算速度
            center_curr = centers[-1]
            center_start = centers[0]
            
            velocity_x = (center_curr[0] - center_start[0]) / dt
            velocity_y = (center_curr[1] - center_start[1]) / dt
            
            # 预测下一个中心点
            next_center_x = int(center_curr[0] + velocity_x * dt)
            next_center_y = int(center_curr[1] + velocity_y * dt)
            
            # 使用当前尺寸
            x, y, w, h = bbox
            
            return (next_center_x - w // 2, next_center_y - h // 2, w, h)
        
        return bbox
    
    def mark_invisible(self):
        """标记为不可见"""
        self.consecutive_invisible_count += 1
    
    def is_valid(self, max_invisible_age: int = 10) -> bool:
        """检查轨迹是否有效"""
        return self.consecutive_invisible_count <= max_invisible_age


class ObjectTracker:
    """
    物体跟踪器类
    功能：
    - 实时跟踪多个物体
    - 使用多种算法（Kalman滤波、匈牙利算法等）
    - 处理物体进入/离开视野
    - 提供运动预测和轨迹分析
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 100, 
                 detection_threshold: float = 0.5, min_feature_similarity: float = 0.3):
        """
        初始化物体跟踪器
        
        Args:
            max_disappeared: 最大消失帧数
            max_distance: 最大匹配距离
            detection_threshold: 检测阈值
            min_feature_similarity: 最小特征相似度
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.detection_threshold = detection_threshold
        self.min_feature_similarity = min_feature_similarity
        
        # 轨迹管理
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.track_counter = 0
        
        # 性能统计
        self.tracking_times = []
        self.average_tracking_time = 0.0
        
        # 颜色直方图bins
        self.color_bins = 16
        
        # 特征检测器
        self.feature_detector = cv2.ORB_create(nfeatures=50)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("物体跟踪器初始化完成")
    
    def track_objects(self, detections: List[Any], timestamp: float, 
                     image: Optional[np.ndarray] = None) -> Dict[int, Track]:
        """
        跟踪物体
        
        Args:
            detections: 检测结果列表
            timestamp: 时间戳
            image: 当前图像（可选，用于特征提取）
            
        Returns:
            当前活跃的轨迹
        """
        start_time = time.time()
        
        try:
            # 过滤低置信度检测
            valid_detections = [det for det in detections if det.confidence >= self.detection_threshold]
            
            # 更新现有轨迹
            self._update_existing_tracks(valid_detections, timestamp, image)
            
            # 清理过期轨迹
            self._cleanup_expired_tracks(timestamp)
            
            # 添加新轨迹
            self._add_new_tracks(valid_detections, timestamp, image)
            
            # 记录性能
            tracking_time = time.time() - start_time
            self._update_performance_stats(tracking_time)
            
            # 返回有效轨迹
            active_tracks = {track_id: track for track_id, track in self.tracks.items() 
                           if track.is_valid(self.max_disappeared)}
            
            self.logger.debug(f"跟踪完成: {len(active_tracks)}个活跃轨迹, 用时{tracking_time:.3f}秒")
            
            return active_tracks
            
        except Exception as e:
            self.logger.error(f"物体跟踪失败: {e}")
            return {}
    
    def _update_existing_tracks(self, detections: List[Any], timestamp: float, 
                              image: Optional[np.ndarray] = None):
        """更新现有轨迹"""
        if not self.tracks or not detections:
            return
        
        # 准备距离矩阵
        track_ids = list(self.tracks.keys())
        detection_centers = []
        detection_features = []
        
        for det in detections:
            center = det.center
            detection_centers.append(center)
            
            # 提取特征（如果有图像）
            features = []
            color_hist = None
            
            if image is not None:
                # 提取颜色直方图
                color_hist = self._extract_color_histogram(image, det.bbox)
                
                # 提取ORB特征
                features = self._extract_features(image, det.bbox)
            
            detection_features.append((color_hist, features))
        
        # 计算成本矩阵
        cost_matrix = self._calculate_cost_matrix(
            track_ids, detection_centers, detection_features, timestamp
        )
        
        # 使用匈牙利算法进行最优匹配
        if cost_matrix.size > 0:
            # 转换为最小化问题
            cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1e6)
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # 获取已匹配的轨迹和检测
            used_track_indices = set()
            used_detection_indices = set()
            
            for track_idx, det_idx in zip(track_indices, detection_indices):
                track_id = track_ids[track_idx]
                detection = detections[det_idx]
                
                cost = cost_matrix[track_idx, det_idx]
                
                # 检查匹配是否有效
                if cost < self.max_distance and np.isfinite(cost):
                    # 更新轨迹
                    color_hist, features = detection_features[det_idx]
                    self.tracks[track_id].update(
                        detection.bbox, detection.confidence, timestamp, color_hist, features
                    )
                    
                    used_track_indices.add(track_idx)
                    used_detection_indices.add(det_idx)
            
            # 标记未匹配的检测
            unmatched_detections = [i for i in range(len(detections)) 
                                  if i not in used_detection_indices]
            
            # 标记未匹配的轨迹为不可见
            for i, track_id in enumerate(track_ids):
                if i not in used_track_indices:
                    self.tracks[track_id].mark_invisible()
    
    def _calculate_cost_matrix(self, track_ids: List[int], detection_centers: List[Tuple[int, int]],
                              detection_features: List[Tuple[Optional[np.ndarray], List[np.ndarray]]],
                              timestamp: float) -> np.ndarray:
        """
        计算成本矩阵
        
        Args:
            track_ids: 轨迹ID列表
            detection_centers: 检测中心点列表
            detection_features: 检测特征列表
            timestamp: 时间戳
            
        Returns:
            成本矩阵
        """
        num_tracks = len(track_ids)
        num_detections = len(detection_centers)
        
        if num_tracks == 0 or num_detections == 0:
            return np.array([])
        
        cost_matrix = np.zeros((num_tracks, num_detections))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_center = (track.bbox[0] + track.bbox[2] // 2, 
                           track.bbox[1] + track.bbox[3] // 2)
            
            for j, (det_center, (color_hist, features)) in enumerate(zip(detection_centers, detection_features)):
                # 位置距离成本
                position_cost = np.sqrt((track_center[0] - det_center[0])**2 + 
                                      (track_center[1] - det_center[1])**2)
                
                # 类别匹配成本
                class_cost = 0.0 if track.class_name == self._get_detection_class(j) else 50.0
                
                # 外观匹配成本
                appearance_cost = 0.0
                if color_hist is not None and np.sum(track.color_histogram) > 0:
                    # 颜色直方图相似度
                    color_sim = cv2.compareHist(track.color_histogram, color_hist, cv2.HISTCMP_CORREL)
                    appearance_cost += 20.0 * (1 - color_sim)  # 转换为成本
                
                # 特征匹配成本
                if features and track.features:
                    try:
                        # 匹配特征点
                        matches = self.feature_matcher.match(features, track.features[-1])
                        if matches:
                            distances = [m.distance for m in matches]
                            avg_distance = np.mean(distances)
                            appearance_cost += avg_distance * 2.0  # 转换为成本
                        else:
                            appearance_cost += 30.0
                    except:
                        appearance_cost += 30.0
                
                # 时间一致性成本
                time_cost = 0.0
                if track.last_seen > 0:
                    time_diff = timestamp - track.last_seen
                    time_cost = min(time_diff * 5.0, 100.0)  # 时间越久成本越高
                
                # 合并成本
                total_cost = position_cost + class_cost + appearance_cost + time_cost
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def _get_detection_class(self, detection_idx: int) -> str:
        """获取检测类别（这个方法需要在实际使用时根据检测结果调整）"""
        # 这里需要从外部传入检测结果的类别信息
        # 在实际使用中，这个方法应该被整合到主逻辑中
        return "unknown"
    
    def _cleanup_expired_tracks(self, timestamp: float):
        """清理过期轨迹"""
        expired_tracks = []
        
        for track_id, track in self.tracks.items():
            # 检查是否超时
            if track.consecutive_invisible_count > self.max_disappeared:
                expired_tracks.append(track_id)
            elif timestamp - track.last_seen > 30.0:  # 30秒未更新
                expired_tracks.append(track_id)
        
        # 删除过期轨迹
        for track_id in expired_tracks:
            del self.tracks[track_id]
            self.logger.debug(f"轨迹 {track_id} 已过期，清理")
    
    def _add_new_tracks(self, detections: List[Any], timestamp: float, 
                       image: Optional[np.ndarray] = None):
        """添加新轨迹"""
        for det in detections:
            # 检查是否已经匹配到现有轨迹
            already_tracked = False
            for track in self.tracks.values():
                if self._is_same_object(track, det):
                    already_tracked = True
                    break
            
            if not already_tracked:
                # 创建新轨迹
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # 提取特征
                color_hist = None
                features = []
                
                if image is not None:
                    color_hist = self._extract_color_histogram(image, det.bbox)
                    features = self._extract_features(image, det.bbox)
                
                new_track = Track(
                    track_id=track_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    timestamp=timestamp
                )
                
                # 更新特征
                if color_hist is not None:
                    new_track.color_histogram = color_hist
                if features:
                    new_track.features = features
                
                self.tracks[track_id] = new_track
                
                self.logger.debug(f"创建新轨迹: ID={track_id}, 类别={det.class_name}")
    
    def _is_same_object(self, track: Track, detection: Any) -> bool:
        """
        检查轨迹和检测是否为同一物体
        
        Args:
            track: 轨迹对象
            detection: 检测结果
            
        Returns:
            是否为同一物体
        """
        # 检查类别
        if track.class_name != detection.class_name:
            return False
        
        # 检查中心点距离
        track_center = (track.bbox[0] + track.bbox[2] // 2, 
                       track.bbox[1] + track.bbox[3] // 2)
        detection_center = detection.center
        
        distance = np.sqrt((track_center[0] - detection_center[0])**2 + 
                          (track_center[1] - detection_center[1])**2)
        
        # 检查置信度
        if abs(track.confidence - detection.confidence) > 0.3:
            return False
        
        # 计算IoU
        track_bbox = track.bbox
        detection_bbox = detection.bbox
        
        iou = self._calculate_iou(track_bbox, detection_bbox)
        
        return iou > 0.3  # IoU阈值
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 第一个边界框 (x, y, w, h)
            bbox2: 第二个边界框 (x, y, w, h)
            
        Returns:
            IoU值
        """
        # 转换为左上右下格式
        x1_min, y1_min, w1, h1 = bbox1
        x1_max, y1_max = x1_min + w1, y1_min + h1
        
        x2_min, y2_min, w2, h2 = bbox2
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # 计算交集
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_min >= xi_max or yi_min >= yi_max:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # 计算并集
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_color_histogram(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        提取颜色直方图
        
        Args:
            image: 图像
            bbox: 边界框
            
        Returns:
            颜色直方图
        """
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return np.zeros(256)
            
            # 转换到HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 计算H通道的直方图
            hist = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
            
            # 归一化
            cv2.normalize(hist, hist)
            
            return hist.flatten()
            
        except Exception as e:
            self.logger.error(f"提取颜色直方图失败: {e}")
            return np.zeros(self.color_bins)
    
    def _extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        提取图像特征
        
        Args:
            image: 图像
            bbox: 边界框
            
        Returns:
            特征列表
        """
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0 or w < 10 or h < 10:
                return []
            
            # 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 检测ORB特征
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            # 只保留关键点
            keypoints = [kp.pt for kp in keypoints]
            
            # 转换为numpy数组
            if descriptors is not None:
                return [descriptors]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"提取特征失败: {e}")
            return []
    
    def _update_performance_stats(self, tracking_time: float):
        """更新性能统计"""
        self.tracking_times.append(tracking_time)
        
        # 保持最近100次的统计
        if len(self.tracking_times) > 100:
            self.tracking_times = self.tracking_times[-100:]
        
        # 计算平均时间
        self.average_tracking_time = np.mean(self.tracking_times)
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """
        获取跟踪统计信息
        
        Returns:
            跟踪统计字典
        """
        active_tracks = [track for track in self.tracks.values() 
                        if track.is_valid(self.max_disappeared)]
        
        if not self.tracking_times:
            return {
                'total_tracks': len(self.tracks),
                'active_tracks': len(active_tracks),
                'average_tracking_time': 0.0,
                'fps_estimate': 0.0,
                'track_success_rate': 0.0
            }
        
        # 计算成功率
        total_tracks = len(self.tracks)
        successful_tracks = len([t for t in self.tracks.values() 
                               if t.total_visible_count >= 5])
        success_rate = successful_tracks / total_tracks if total_tracks > 0 else 0.0
        
        return {
            'total_tracks': total_tracks,
            'active_tracks': len(active_tracks),
            'average_tracking_time': self.average_tracking_time,
            'fps_estimate': 1.0 / self.average_tracking_time if self.average_tracking_time > 0 else 0.0,
            'track_success_rate': success_rate
        }
    
    def reset_tracker(self):
        """重置跟踪器"""
        self.tracks.clear()
        self.next_track_id = 1
        self.track_counter = 0
        self.tracking_times.clear()
        self.average_tracking_time = 0.0
        self.logger.info("物体跟踪器已重置")
    
    def export_track_data(self, output_path: Union[str, Path]) -> None:
        """
        导出轨迹数据
        
        Args:
            output_path: 输出文件路径
        """
        try:
            track_data = {
                'export_time': time.time(),
                'total_tracks': len(self.tracks),
                'tracks': {}
            }
            
            for track_id, track in self.tracks.items():
                track_info = {
                    'track_id': track.track_id,
                    'class_name': track.class_name,
                    'age': track.age,
                    'total_visible_count': track.total_visible_count,
                    'consecutive_invisible_count': track.consecutive_invisible_count,
                    'current_bbox': track.bbox,
                    'velocity': track.velocity,
                    'acceleration': track.acceleration,
                    'history': track.history
                }
                
                track_data['tracks'][str(track_id)] = track_info
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(track_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"轨迹数据已导出: {output_path}")
            
        except Exception as e:
            self.logger.error(f"导出轨迹数据失败: {e}")
    
    def set_tracking_params(self, max_disappeared: int = None, max_distance: int = None):
        """
        设置跟踪参数
        
        Args:
            max_disappeared: 最大消失帧数
            max_distance: 最大匹配距离
        """
        if max_disappeared is not None:
            self.max_disappeared = max_disappeared
        
        if max_distance is not None:
            self.max_distance = max_distance
        
        self.logger.info(f"跟踪参数更新: max_disappeared={self.max_disappeared}, max_distance={self.max_distance}")
    
    def get_track_predictions(self) -> Dict[int, Tuple[int, int, int, int]]:
        """
        获取所有轨迹的预测位置
        
        Returns:
            轨迹ID到预测位置的映射
        """
        predictions = {}
        for track_id, track in self.tracks.items():
            if track.is_valid(self.max_disappeared):
                predictions[track_id] = track.predict_next_position()
        
        return predictions