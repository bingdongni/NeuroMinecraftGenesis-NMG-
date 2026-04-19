"""
USB摄像头场景捕获系统 - 帧分析模块

该模块实现对视频帧的深度分析，包括目标检测、特征提取、
图像质量评估和内容理解等功能。

作者: AI助手
创建时间: 2025-11-13
版本: 1.0
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, List, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from io import BytesIO


class AnalysisType(Enum):
    """分析类型枚举"""
    BASIC = "基础分析"
    DETAILED = "详细分析"
    REAL_TIME = "实时分析"
    BATCH = "批量分析"


@dataclass
class ObjectDetection:
    """目标检测结果"""
    class_name: str = ""
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    mask: Optional[np.ndarray] = None


@dataclass
class FrameMetrics:
    """帧指标数据结构"""
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    saturation: float = 0.0
    exposure: float = 0.0
    white_balance: float = 0.0
    dynamic_range: float = 0.0


@dataclass
class FrameAnalysisResult:
    """帧分析结果"""
    frame_id: int = 0
    timestamp: float = 0.0
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    objects: List[ObjectDetection] = field(default_factory=list)
    faces: List[Tuple[int, int, int, int]] = field(default_factory=list)
    text_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    motion_vectors: List[Tuple[int, int, int, int]] = field(default_factory=list)
    scene_change: bool = False
    quality_score: float = 0.0
    analysis_time: float = 0.0


class FrameAnalyzer:
    """
    帧分析器
    
    该类负责：
    1. 图像质量评估和指标计算
    2. 目标检测和识别
    3. 人脸检测和表情分析
    4. 文本区域检测
    5. 颜色分析和主色调提取
    6. 运动向量分析
    7. 场景变化检测
    """
    
    def __init__(self, analysis_type: AnalysisType = AnalysisType.REAL_TIME):
        """初始化帧分析器"""
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(__name__)
        
        # 分析状态
        self.frame_count = 0
        self.previous_frame = None
        self.baseline_metrics = None
        
        # 性能统计
        self.analysis_times = []
        self.max_history = 100
        
        # 初始化检测器
        self._initialize_detectors()
        
        # 颜色分析相关
        self.color_categories = {
            'red': (0, 50, 180, 10, 255, 255),
            'orange': (10, 50, 180, 25, 255, 255),
            'yellow': (25, 50, 180, 35, 255, 255),
            'green': (35, 50, 180, 85, 255, 255),
            'cyan': (85, 50, 180, 100, 255, 255),
            'blue': (100, 50, 180, 130, 255, 255),
            'purple': (130, 50, 180, 160, 255, 255),
            'pink': (160, 50, 180, 180, 255, 255)
        }
    
    def _initialize_detectors(self):
        """初始化各种检测器"""
        try:
            # 初始化人脸检测器
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 初始化眼睛检测器
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # 初始化行人检测器（HOG）
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # 初始化文字检测器（EAST文本检测器）
            self.text_detector = self._load_text_detector()
            
            self.logger.info("检测器初始化完成")
            
        except Exception as e:
            self.logger.warning(f"部分检测器初始化失败: {e}")
    
    def _load_text_detector(self) -> Optional[Any]:
        """加载文本检测器（简化版本）"""
        # 这里使用简化的文本区域检测方法
        # 实际项目中可以集成EAST、CRNN等深度学习文本检测器
        return None
    
    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysisResult:
        """
        分析单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            FrameAnalysisResult: 帧分析结果
        """
        start_time = time.time()
        
        if frame is None:
            return FrameAnalysisResult()
        
        try:
            result = FrameAnalysisResult()
            result.frame_id = self.frame_count
            result.timestamp = time.time()
            
            # 基础指标分析
            result.metrics = self._calculate_frame_metrics(frame)
            
            # 根据分析类型执行不同深度的分析
            if self.analysis_type == AnalysisType.BASIC:
                self._basic_analysis(frame, result)
            elif self.analysis_type == AnalysisType.DETAILED:
                self._detailed_analysis(frame, result)
            elif self.analysis_type == AnalysisType.REAL_TIME:
                self._realtime_analysis(frame, result)
            elif self.analysis_type == AnalysisType.BATCH:
                self._batch_analysis(frame, result)
            
            # 场景变化检测
            result.scene_change = self._detect_scene_change(frame)
            
            # 计算质量评分
            result.quality_score = self._calculate_quality_score(result.metrics)
            
            # 记录分析时间
            result.analysis_time = time.time() - start_time
            self.analysis_times.append(result.analysis_time)
            if len(self.analysis_times) > self.max_history:
                self.analysis_times.pop(0)
            
            self.frame_count += 1
            return result
            
        except Exception as e:
            self.logger.error(f"帧分析失败: {e}")
            return FrameAnalysisResult()
    
    def _calculate_frame_metrics(self, frame: np.ndarray) -> FrameMetrics:
        """计算帧指标"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            gray = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2HSV)
        
        metrics = FrameMetrics()
        
        # 亮度
        metrics.brightness = np.mean(gray)
        
        # 对比度
        metrics.contrast = np.std(gray)
        
        # 饱和度
        metrics.saturation = np.mean(hsv[:,:,1])
        
        # 曝光度（基于高亮像素比例）
        highlight_threshold = 250
        metrics.exposure = np.sum(gray > highlight_threshold) / (gray.shape[0] * gray.shape[1])
        
        # 锐度（拉普拉斯方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics.sharpness = np.var(laplacian)
        
        # 噪声水平
        metrics.noise_level = self._estimate_noise_level(gray)
        
        # 白平衡指标（基于色温估算）
        metrics.white_balance = self._estimate_white_balance(hsv)
        
        # 动态范围
        metrics.dynamic_range = np.max(gray) - np.min(gray)
        
        return metrics
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """估算噪声水平"""
        # 使用高斯滤波和原图的差值来估算噪声
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = cv2.absdiff(gray, blurred)
        return np.mean(noise)
    
    def _estimate_white_balance(self, hsv: np.ndarray) -> float:
        """估算白平衡"""
        # 基于色相分布估算白平衡偏离度
        hue = hsv[:,:,0]
        # 统计每个色相的像素数
        hist = np.histogram(hue, bins=36, range=(0, 180))[0]
        # 计算色相分布的方差
        hue_variance = np.var(hist)
        # 白平衡良好时色相分布相对均匀
        return 1.0 / (1.0 + hue_variance / 1000.0)
    
    def _basic_analysis(self, frame: np.ndarray, result: FrameAnalysisResult):
        """基础分析"""
        # 人脸检测
        result.faces = self._detect_faces(frame)
        
        # 颜色分析
        result.dominant_colors = self._extract_dominant_colors(frame)
        
        # 运动分析
        if self.previous_frame is not None:
            result.motion_vectors = self._analyze_motion(self.previous_frame, frame)
        
        self.previous_frame = frame.copy()
    
    def _detailed_analysis(self, frame: np.ndarray, result: FrameAnalysisResult):
        """详细分析"""
        # 基础分析
        self._basic_analysis(frame, result)
        
        # 目标检测
        result.objects = self._detect_objects(frame)
        
        # 文本区域检测
        result.text_regions = self._detect_text_regions(frame)
        
        # 行人检测
        people = self._detect_people(frame)
        for person in people:
            obj = ObjectDetection("person", 0.9, person)
            result.objects.append(obj)
    
    def _realtime_analysis(self, frame: np.ndarray, result: FrameAnalysisResult):
        """实时分析（优化版本）"""
        # 基础分析
        self._basic_analysis(frame, result)
        
        # 简化的目标检测
        objects = self._fast_object_detection(frame)
        result.objects.extend(objects)
        
        # 文本检测（只在每几帧执行一次）
        if self.frame_count % 5 == 0:
            result.text_regions = self._fast_text_detection(frame)
    
    def _batch_analysis(self, frame: np.ndarray, result: FrameAnalysisResult):
        """批量分析（最全面）"""
        # 执行所有分析
        self._detailed_analysis(frame, result)
        
        # 添加更多高级分析
        result.motion_vectors = self._analyze_motion(self.previous_frame, frame) if self.previous_frame is not None else []
        self.previous_frame = frame.copy()
        
        # 更详细的颜色分析
        result.dominant_colors = self._extract_dominant_colors_detailed(frame)
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测人脸"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return [(x, y, w, h) for (x, y, w, h) in faces]
        except Exception as e:
            self.logger.debug(f"人脸检测失败: {e}")
            return []
    
    def _detect_objects(self, frame: np.ndarray) -> List[ObjectDetection]:
        """检测目标对象（简化版本）"""
        objects = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 轮廓检测
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 过滤小的噪声
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 简单分类逻辑
                    class_name = self._classify_object(contour, frame[y:y+h, x:x+w])
                    confidence = min(area / 10000, 1.0)  # 基于面积的置信度
                    
                    objects.append(ObjectDetection(class_name, confidence, (x, y, w, h)))
            
        except Exception as e:
            self.logger.debug(f"目标检测失败: {e}")
        
        return objects
    
    def _classify_object(self, contour: np.ndarray, roi: np.ndarray) -> str:
        """简单对象分类"""
        area = cv2.contourArea(contour)
        if area < 1000:
            return "small_object"
        elif area > 50000:
            return "large_object"
        else:
            # 基于长宽比的简单分类
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1
            
            if aspect_ratio > 2:
                return "horizontal_object"
            elif aspect_ratio < 0.5:
                return "vertical_object"
            else:
                return "regular_object"
    
    def _detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测行人"""
        try:
            height, width = frame.shape[:2]
            # 调整大小以提高检测速度
            if width > 640:
                scale = 640 / width
                resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
            else:
                resized = frame
                scale = 1.0
            
            rects, weights = self.hog.detectMultiScale(resized, winStride=(4, 4), padding=(8, 8), scale=1.05)
            
            # 调整边界框到原始尺寸
            people = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in rects]
            return people
            
        except Exception as e:
            self.logger.debug(f"行人检测失败: {e}")
            return []
    
    def _detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测文本区域（简化版本）"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100 and area < 10000:  # 文本区域大小过滤
                    x, y, w, h = cv2.boundingRect(contour)
                    # 文本区域通常是细长的矩形
                    if 0.2 < w/h < 5.0:
                        text_regions.append((x, y, w, h))
            
            return text_regions
            
        except Exception as e:
            self.logger.debug(f"文本检测失败: {e}")
            return []
    
    def _fast_object_detection(self, frame: np.ndarray) -> List[ObjectDetection]:
        """快速目标检测（实时优化）"""
        objects = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours[:10]:  # 限制检测数量以提高速度
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(area / 5000, 1.0)
                    objects.append(ObjectDetection("object", confidence, (x, y, w, h)))
            
        except Exception as e:
            self.logger.debug(f"快速目标检测失败: {e}")
        
        return objects
    
    def _fast_text_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """快速文本检测（实时优化）"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 简化的文本检测
            text_regions = []
            for y in range(0, gray.shape[0], 20):
                for x in range(0, gray.shape[1], 20):
                    roi = gray[y:y+20, x:x+20]
                    if np.std(roi) < 5:  # 低方差区域可能是文本
                        text_regions.append((x, y, 20, 20))
            
            return text_regions[:5]  # 限制数量
            
        except Exception as e:
            self.logger.debug(f"快速文本检测失败: {e}")
            return []
    
    def _analyze_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """分析运动向量"""
        if prev_frame.shape != curr_frame.shape:
            return []
        
        try:
            # 计算帧差
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
                curr_gray = curr_frame
            
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            
            # 阈值化
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # 查找运动区域
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_regions.append((x, y, w, h))
            
            return motion_regions
            
        except Exception as e:
            self.logger.debug(f"运动分析失败: {e}")
            return []
    
    def _extract_dominant_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """提取主色调"""
        try:
            # 使用K-means聚类提取主色调
            data = frame.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = 5
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 计算每个颜色的像素数
            color_counts = np.bincount(labels.flatten())
            dominant_colors = []
            
            # 排序并选择前5个主色调
            for i in range(min(5, len(color_counts))):
                dominant_color = centers[i].astype(int)
                dominant_colors.append((int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])))  # BGR格式
            
            return dominant_colors
            
        except Exception as e:
            self.logger.debug(f"颜色提取失败: {e}")
            return []
    
    def _extract_dominant_colors_detailed(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """详细颜色分析"""
        # 简化为基本的颜色提取
        return self._extract_dominant_colors(frame)
    
    def _detect_scene_change(self, frame: np.ndarray) -> bool:
        """检测场景变化"""
        if self.previous_frame is None:
            return False
        
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                prev_gray = self.previous_frame
            
            # 计算直方图比较
            hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # 使用相关性比较
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 如果相关性低于阈值，认为发生场景变化
            return correlation < 0.7
            
        except Exception as e:
            self.logger.debug(f"场景变化检测失败: {e}")
            return False
    
    def _calculate_quality_score(self, metrics: FrameMetrics) -> float:
        """计算图像质量评分"""
        scores = []
        
        # 亮度评分
        if 80 <= metrics.brightness <= 180:
            scores.append(1.0)
        elif 50 <= metrics.brightness <= 220:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # 对比度评分
        if metrics.contrast > 30:
            scores.append(1.0)
        elif metrics.contrast > 15:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        # 锐度评分
        if metrics.sharpness > 100:
            scores.append(1.0)
        elif metrics.sharpness > 50:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        # 噪声评分
        if metrics.noise_level < 2:
            scores.append(1.0)
        elif metrics.noise_level < 5:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        # 曝光评分
        if metrics.exposure < 0.1:
            scores.append(1.0)
        elif metrics.exposure < 0.2:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def export_analysis_results(self, results: List[FrameAnalysisResult], 
                              output_file: str, format: str = "json"):
        """
        导出分析结果
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
            format: 导出格式 ("json" 或 "csv")
        """
        try:
            if format.lower() == "json":
                self._export_json(results, output_file)
            elif format.lower() == "csv":
                self._export_csv(results, output_file)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"分析结果已导出到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"导出分析结果失败: {e}")
    
    def _export_json(self, results: List[FrameAnalysisResult], output_file: str):
        """导出为JSON格式"""
        export_data = []
        
        for result in results:
            data = {
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "metrics": {
                    "sharpness": result.metrics.sharpness,
                    "brightness": result.metrics.brightness,
                    "contrast": result.metrics.contrast,
                    "noise_level": result.metrics.noise_level,
                    "saturation": result.metrics.saturation,
                    "exposure": result.metrics.exposure,
                    "white_balance": result.metrics.white_balance,
                    "dynamic_range": result.metrics.dynamic_range
                },
                "objects": [{"class_name": obj.class_name, "confidence": obj.confidence, 
                           "bbox": obj.bbox} for obj in result.objects],
                "faces": result.faces,
                "text_regions": result.text_regions,
                "dominant_colors": result.dominant_colors,
                "scene_change": result.scene_change,
                "quality_score": result.quality_score,
                "analysis_time": result.analysis_time
            }
            export_data.append(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, results: List[FrameAnalysisResult], output_file: str):
        """导出为CSV格式"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                "frame_id", "timestamp", "brightness", "contrast", "sharpness",
                "noise_level", "saturation", "exposure", "quality_score",
                "object_count", "face_count", "text_region_count", "scene_change"
            ])
            
            # 写入数据
            for result in results:
                writer.writerow([
                    result.frame_id,
                    result.timestamp,
                    result.metrics.brightness,
                    result.metrics.contrast,
                    result.metrics.sharpness,
                    result.metrics.noise_level,
                    result.metrics.saturation,
                    result.metrics.exposure,
                    result.quality_score,
                    len(result.objects),
                    len(result.faces),
                    len(result.text_regions),
                    result.scene_change
                ])
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        avg_analysis_time = np.mean(self.analysis_times) if self.analysis_times else 0
        
        return {
            "total_frames_analyzed": self.frame_count,
            "analysis_type": self.analysis_type.value,
            "average_analysis_time": avg_analysis_time,
            "analysis_fps": 1.0 / avg_analysis_time if avg_analysis_time > 0 else 0,
            "available_detectors": {
                "face_detector": self.face_cascade is not None,
                "hog_detector": self.hog is not None,
                "text_detector": self.text_detector is not None
            }
        }


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='帧分析系统')
    parser.add_argument('--input', type=str, help='输入视频文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--output', type=str, help='分析结果输出文件')
    parser.add_argument('--type', type=str, default='realtime', 
                       choices=['basic', 'detailed', 'realtime', 'batch'],
                       help='分析类型')
    
    args = parser.parse_args()
    
    # 选择分析类型
    analysis_type = AnalysisType(args.type)
    analyzer = FrameAnalyzer(analysis_type)
    
    # 选择输入源
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("无法打开输入源")
        exit(1)
    
    # 分析结果存储
    analysis_results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 分析帧
            result = analyzer.analyze_frame(frame)
            analysis_results.append(result)
            
            # 实时显示分析结果
            display_frame = frame.copy()
            
            # 绘制检测结果
            # 人脸
            for (x, y, w, h) in result.faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 目标
            for obj in result.objects:
                x, y, w, h = obj.bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(display_frame, f"{obj.class_name} ({obj.confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 文本区域
            for (x, y, w, h) in result.text_regions:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
            # 运动区域
            for (x, y, w, h) in result.motion_vectors:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
            
            # 信息显示
            info_text = f"质量: {result.quality_score:.2f} | 分析: {result.analysis_time:.3f}s"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if result.scene_change:
                cv2.putText(display_frame, "场景变化!", (10, display_frame.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('帧分析系统', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("程序结束")
        
        # 导出分析结果
        if args.output:
            analyzer.export_analysis_results(analysis_results, args.output)
            print(f"分析结果已保存到: {args.output}")
        
        # 显示统计信息
        stats = analyzer.get_analysis_statistics()
        print(f"分析统计: {stats}")
        print(f"处理的帧数: {len(analysis_results)}")