"""
OpenCV真实物体识别系统 - 分类系统
ClassificationSystem类：负责物体的细粒度分类和属性提取
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass
class ClassificationResult:
    """分类结果数据类"""
    primary_class: str
    primary_confidence: float
    sub_classes: Dict[str, float]
    attributes: Dict[str, Any]
    color_info: Dict[str, Union[str, float, Tuple]]
    shape_info: Dict[str, Any]
    size_category: str
    material_estimate: str


@dataclass
class ObjectAttributes:
    """物体属性数据类"""
    bbox: Tuple[int, int, int, int]
    area: int
    aspect_ratio: float
    center: Tuple[int, int]
    contour_area: float
    perimeter: float
    circularity: float
    solidity: float


class ClassificationSystem:
    """
    物体分类系统类
    功能：
    - 对检测到的物体进行细粒度分类
    - 提取物体属性（颜色、形状、材质等）
    - 识别水瓶、书本、笔等特定物体特征
    - 提供物体质量评估
    """
    
    def __init__(self):
        """初始化分类系统"""
        # 物体主类别定义
        self.object_categories = {
            'bottle': {
                'sub_classes': ['water_bottle', 'plastic_bottle', 'glass_bottle', 'thermos'],
                'attributes': ['transparency', 'color', 'height_ratio', 'cap_type'],
                'material_estimate': ['plastic', 'glass', 'metal'],
                'size_range': [(50, 200), (200, 500)]  # 面积范围
            },
            'book': {
                'sub_classes': ['textbook', 'notebook', 'magazine', 'dictionary'],
                'attributes': ['page_count', 'thickness', 'color_cover', 'spine_type'],
                'material_estimate': ['paper', 'leather', 'plastic'],
                'size_range': [(200, 1000), (1000, 3000)]
            },
            'pen': {
                'sub_classes': ['ballpoint_pen', 'fountain_pen', 'pencil', 'highlighter'],
                'attributes': ['length', 'tip_type', 'grip_style', 'ink_color'],
                'material_estimate': ['plastic', 'metal', 'wood'],
                'size_range': [(10, 50), (50, 200)]
            },
            'laptop': {
                'sub_classes': ['gaming_laptop', 'ultrabook', 'chromebook', 'workstation'],
                'attributes': ['screen_size', 'keyboard_type', 'brand_indicators'],
                'material_estimate': ['metal', 'plastic', 'carbon_fiber'],
                'size_range': [(5000, 15000), (15000, 30000)]
            },
            'phone': {
                'sub_classes': ['smartphone', 'feature_phone', 'tablet_phone'],
                'attributes': ['screen_ratio', 'camera_bump', 'notch_type'],
                'material_estimate': ['glass', 'metal', 'plastic'],
                'size_range': [(300, 800), (800, 2000)]
            },
            'cup': {
                'sub_classes': ['coffee_cup', 'tea_cup', 'water_cup', 'mug'],
                'attributes': ['handle_type', 'material', 'fill_level'],
                'material_estimate': ['ceramic', 'glass', 'metal', 'plastic'],
                'size_range': [(100, 400), (400, 1000)]
            },
            'chair': {
                'sub_classes': ['office_chair', 'dining_chair', 'gaming_chair'],
                'attributes': ['back_height', 'armrest', 'material'],
                'material_estimate': ['wood', 'metal', 'plastic', 'fabric'],
                'size_range': [(3000, 8000), (8000, 20000)]
            },
            'table': {
                'sub_classes': ['dining_table', 'coffee_table', 'desk', 'work_table'],
                'attributes': ['shape', 'leg_count', 'surface_type'],
                'material_estimate': ['wood', 'metal', 'glass'],
                'size_range': [(10000, 30000), (30000, 100000)]
            },
            'bag': {
                'sub_classes': ['backpack', 'handbag', 'briefcase', 'tote_bag'],
                'attributes': ['strap_type', 'compartment_count', 'closure_type'],
                'material_estimate': ['fabric', 'leather', 'plastic'],
                'size_range': [(1000, 5000), (5000, 15000)]
            },
            'wallet': {
                'sub_classes': ['bifold_wallet', 'trifold_wallet', 'card_holder'],
                'attributes': ['card_slots', 'coin_pocket', 'material'],
                'material_estimate': ['leather', 'fabric', 'plastic'],
                'size_range': [(50, 200), (200, 500)]
            }
        }
        
        # 颜色映射
        self.color_ranges = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([50, 50, 50], [70, 255, 255]),
            'yellow': ([15, 100, 100], [35, 255, 255]),
            'orange': ([10, 100, 100], [25, 255, 255]),
            'purple': ([120, 50, 50], [150, 255, 255]),
            'pink': ([140, 50, 50], [170, 255, 255]),
            'brown': ([10, 100, 100], [20, 255, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'gray': ([0, 0, 50], [180, 255, 200])
        }
        
        # 形状特征定义
        self.shape_features = {
            'circular': {'min_circularity': 0.7, 'aspect_ratio_range': (0.8, 1.2)},
            'rectangular': {'min_circularity': 0.3, 'aspect_ratio_range': (1.5, 4.0)},
            'elongated': {'min_circularity': 0.1, 'aspect_ratio_range': (3.0, 10.0)}
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("分类系统初始化完成")
    
    def classify_object(self, image: np.ndarray, detection_result: Any) -> ClassificationResult:
        """
        对检测到的物体进行分类
        
        Args:
            image: 输入图像
            detection_result: 检测结果
            
        Returns:
            分类结果
        """
        try:
            # 提取ROI
            x, y, w, h = detection_result.bbox
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return self._create_empty_result(detection_result.class_name)
            
            # 获取物体属性
            attributes = self._extract_object_attributes(roi, detection_result.bbox)
            
            # 颜色分析
            color_info = self._analyze_color(roi)
            
            # 形状分析
            shape_info = self._analyze_shape(roi, attributes)
            
            # 细粒度分类
            primary_class, primary_confidence = self._classify_primary(detection_result, attributes, color_info)
            
            # 子类别分类
            sub_classes = self._classify_sub_classes(primary_class, roi, attributes, color_info)
            
            # 属性提取
            extracted_attributes = self._extract_detailed_attributes(
                primary_class, roi, attributes, color_info, shape_info
            )
            
            # 材质估算
            material_estimate = self._estimate_material(primary_class, color_info, shape_info)
            
            # 尺寸分类
            size_category = self._classify_size(detection_result.area, primary_class)
            
            result = ClassificationResult(
                primary_class=primary_class,
                primary_confidence=primary_confidence,
                sub_classes=sub_classes,
                attributes=extracted_attributes,
                color_info=color_info,
                shape_info=shape_info,
                size_category=size_category,
                material_estimate=material_estimate
            )
            
            self.logger.debug(f"物体分类完成: {primary_class}, 置信度: {primary_confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"物体分类失败: {e}")
            return self._create_empty_result(detection_result.class_name)
    
    def _extract_object_attributes(self, roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> ObjectAttributes:
        """
        提取物体基本属性
        
        Args:
            roi: 物体区域图像
            bbox: 边界框
            
        Returns:
            物体属性
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # 计算圆形度
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # 计算凸包面积和实心度
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
            else:
                contour_area = 0
                perimeter = 0
                circularity = 0
                solidity = 0
            
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            center = (w // 2, h // 2)
            
            return ObjectAttributes(
                bbox=bbox,
                area=area,
                aspect_ratio=aspect_ratio,
                center=center,
                contour_area=contour_area,
                perimeter=perimeter,
                circularity=circularity,
                solidity=solidity
            )
            
        except Exception as e:
            self.logger.error(f"提取物体属性失败: {e}")
            # 返回默认属性
            x, y, w, h = bbox
            return ObjectAttributes(
                bbox=bbox,
                area=w * h,
                aspect_ratio=w / h if h > 0 else 0,
                center=(w // 2, h // 2),
                contour_area=0,
                perimeter=0,
                circularity=0,
                solidity=0
            )
    
    def _analyze_color(self, roi: np.ndarray) -> Dict[str, Any]:
        """
        分析物体颜色
        
        Args:
            roi: 物体区域图像
            
        Returns:
            颜色分析结果
        """
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 统计各种颜色的像素数
            color_stats = {}
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for color_name, (lower, upper) in self.color_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                # 创建掩码
                mask = cv2.inRange(hsv, lower, upper)
                
                # 计算该颜色的像素数
                color_pixels = cv2.countNonZero(mask)
                color_percentage = color_pixels / total_pixels
                
                color_stats[color_name] = {
                    'count': color_pixels,
                    'percentage': color_percentage
                }
            
            # 找到主要颜色
            dominant_color = max(color_stats.keys(), key=lambda k: color_stats[k]['percentage'])
            
            # 计算平均颜色
            mean_color = cv2.mean(roi)[:3]  # BGR格式
            mean_color_bgr = tuple(int(x) for x in mean_color)
            mean_color_rgb = (mean_color_bgr[2], mean_color_bgr[1], mean_color_bgr[0])
            
            return {
                'dominant_color': dominant_color,
                'dominant_percentage': color_stats[dominant_color]['percentage'],
                'color_distribution': color_stats,
                'mean_color_bgr': mean_color_bgr,
                'mean_color_rgb': mean_color_rgb,
                'color_variance': np.var(roi.reshape(-1, 3), axis=0).mean()
            }
            
        except Exception as e:
            self.logger.error(f"颜色分析失败: {e}")
            return {
                'dominant_color': 'unknown',
                'dominant_percentage': 0.0,
                'color_distribution': {},
                'mean_color_bgr': (128, 128, 128),
                'mean_color_rgb': (128, 128, 128),
                'color_variance': 0.0
            }
    
    def _analyze_shape(self, roi: np.ndarray, attributes: ObjectAttributes) -> Dict[str, Any]:
        """
        分析物体形状
        
        Args:
            roi: 物体区域图像
            attributes: 物体属性
            
        Returns:
            形状分析结果
        """
        try:
            # 转换到灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 应用阈值处理
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 使用最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 近似多边形
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 计算顶点数
                vertex_count = len(approx)
                
                # 判断形状类型
                shape_type = self._determine_shape_type(attributes, vertex_count)
                
                # 计算边界矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                return {
                    'shape_type': shape_type,
                    'vertex_count': vertex_count,
                    'is_regular': self._is_regular_polygon(vertex_count, attributes.circularity),
                    'bounding_rect': (x, y, w, h),
                    'aspect_ratio': attributes.aspect_ratio,
                    'circularity': attributes.circularity,
                    'solidity': attributes.solidity,
                    'contour_area': attributes.contour_area
                }
            else:
                return {
                    'shape_type': 'unknown',
                    'vertex_count': 0,
                    'is_regular': False,
                    'bounding_rect': (0, 0, roi.shape[1], roi.shape[0]),
                    'aspect_ratio': attributes.aspect_ratio,
                    'circularity': attributes.circularity,
                    'solidity': attributes.solidity,
                    'contour_area': attributes.contour_area
                }
                
        except Exception as e:
            self.logger.error(f"形状分析失败: {e}")
            return {
                'shape_type': 'unknown',
                'vertex_count': 0,
                'is_regular': False,
                'bounding_rect': (0, 0, roi.shape[1], roi.shape[0]),
                'aspect_ratio': attributes.aspect_ratio,
                'circularity': attributes.circularity,
                'solidity': attributes.solidity,
                'contour_area': attributes.contour_area
            }
    
    def _determine_shape_type(self, attributes: ObjectAttributes, vertex_count: int) -> str:
        """
        确定物体形状类型
        
        Args:
            attributes: 物体属性
            vertex_count: 顶点数
            
        Returns:
            形状类型
        """
        if attributes.circularity > 0.7:
            return 'circular'
        elif attributes.aspect_ratio > 2.5:
            return 'elongated'
        elif vertex_count >= 4 and vertex_count <= 8:
            return 'polygonal'
        else:
            return 'irregular'
    
    def _is_regular_polygon(self, vertex_count: int, circularity: float) -> bool:
        """
        判断是否为规则多边形
        
        Args:
            vertex_count: 顶点数
            circularity: 圆形度
            
        Returns:
            是否为规则多边形
        """
        # 规则多边形应该有适当的顶点数和较高的圆形度
        regular_counts = [3, 4, 5, 6, 7, 8]
        return vertex_count in regular_counts and circularity > 0.5
    
    def _classify_primary(self, detection_result: Any, attributes: ObjectAttributes, 
                         color_info: Dict[str, Any]) -> Tuple[str, float]:
        """
        主类别分类
        
        Args:
            detection_result: 检测结果
            attributes: 物体属性
            color_info: 颜色信息
            
        Returns:
            (主类别, 置信度)
        """
        # 基于检测结果的类别
        base_class = detection_result.class_name
        
        # 根据形状和尺寸特征调整置信度
        confidence = detection_result.confidence
        
        # 特殊规则调整
        if base_class == 'bottle':
            if attributes.aspect_ratio > 1.5:  # 高瓶子形状
                confidence *= 1.1
            elif color_info['dominant_color'] in ['blue', 'transparent']:
                confidence *= 1.05
                
        elif base_class == 'book':
            if attributes.aspect_ratio > 1.2 and attributes.aspect_ratio < 2.5:  # 书本比例
                confidence *= 1.1
            elif color_info['dominant_color'] in ['brown', 'black', 'blue']:
                confidence *= 1.05
                
        elif base_class == 'pen':
            if attributes.aspect_ratio > 4.0:  # 细长形状
                confidence *= 1.1
            elif attributes.aspect_ratio > 2.0:  # 笔状
                confidence *= 1.05
        
        # 确保置信度不超过1.0
        confidence = min(confidence, 1.0)
        
        return base_class, confidence
    
    def _classify_sub_classes(self, primary_class: str, roi: np.ndarray, 
                            attributes: ObjectAttributes, color_info: Dict[str, Any]) -> Dict[str, float]:
        """
        子类别分类
        
        Args:
            primary_class: 主类别
            roi: 物体区域图像
            attributes: 物体属性
            color_info: 颜色信息
            
        Returns:
            子类别及其置信度
        """
        sub_classes = {}
        
        if primary_class in self.object_categories:
            for sub_class in self.object_categories[primary_class]['sub_classes']:
                confidence = 0.5  # 默认置信度
                
                # 根据特征调整置信度
                if primary_class == 'bottle':
                    if sub_class == 'water_bottle' and color_info['dominant_color'] == 'transparent':
                        confidence = 0.8
                    elif sub_class == 'thermos' and attributes.aspect_ratio > 2.0:
                        confidence = 0.7
                    elif sub_class == 'plastic_bottle' and color_info['dominant_color'] in ['blue', 'green']:
                        confidence = 0.7
                        
                elif primary_class == 'book':
                    if sub_class == 'notebook' and attributes.aspect_ratio < 1.5:
                        confidence = 0.8
                    elif sub_class == 'textbook' and attributes.area > 500:
                        confidence = 0.7
                    elif sub_class == 'magazine' and color_info['dominant_color'] == 'white':
                        confidence = 0.6
                        
                elif primary_class == 'pen':
                    if sub_class == 'ballpoint_pen' and attributes.aspect_ratio > 5.0:
                        confidence = 0.8
                    elif sub_class == 'fountain_pen' and attributes.aspect_ratio > 6.0:
                        confidence = 0.7
                    elif sub_class == 'pencil' and color_info['dominant_color'] in ['yellow', 'orange']:
                        confidence = 0.8
                
                sub_classes[sub_class] = confidence
        
        return sub_classes
    
    def _extract_detailed_attributes(self, primary_class: str, roi: np.ndarray,
                                   attributes: ObjectAttributes, color_info: Dict[str, Any],
                                   shape_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取详细属性
        
        Args:
            primary_class: 主类别
            roi: 物体区域图像
            attributes: 物体属性
            color_info: 颜色信息
            shape_info: 形状信息
            
        Returns:
            详细属性字典
        """
        detailed_attrs = {}
        
        if primary_class == 'bottle':
            detailed_attrs.update({
                'height_ratio': attributes.aspect_ratio,
                'transparency': 'high' if color_info['dominant_color'] == 'transparent' else 'low',
                'cap_present': self._detect_bottle_cap(roi),
                'label_present': self._detect_bottle_label(roi)
            })
            
        elif primary_class == 'book':
            detailed_attrs.update({
                'thickness_estimate': attributes.aspect_ratio,
                'pages_visible': self._detect_book_pages(roi),
                'spine_type': 'visible' if attributes.aspect_ratio > 1.5 else 'hidden',
                'cover_material': color_info['dominant_color']
            })
            
        elif primary_class == 'pen':
            detailed_attrs.update({
                'length_category': 'long' if attributes.aspect_ratio > 6.0 else 'short',
                'tip_type': self._detect_pen_tip(roi),
                'grip_present': self._detect_pen_grip(roi),
                'ink_indicator': color_info['dominant_color']
            })
            
        elif primary_class == 'phone':
            detailed_attrs.update({
                'screen_ratio': 'fullscreen' if attributes.aspect_ratio > 1.8 else 'standard',
                'camera_bump': self._detect_camera_bump(roi),
                'notch_present': self._detect_screen_notch(roi)
            })
        
        return detailed_attrs
    
    def _detect_bottle_cap(self, roi: np.ndarray) -> bool:
        """检测瓶子是否有盖子"""
        # 简化的盖子检测逻辑
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 检查顶部区域是否有明显的对比
        top_region = gray[:height//4, :]
        bottom_region = gray[height//4:height//2, :]
        
        top_mean = np.mean(top_region)
        bottom_mean = np.mean(bottom_region)
        
        return abs(top_mean - bottom_mean) > 30
    
    def _detect_bottle_label(self, roi: np.ndarray) -> bool:
        """检测瓶子是否有标签"""
        # 简化标签检测逻辑
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 查找水平的边缘
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # 近似水平线
                    horizontal_lines += 1
        
        return horizontal_lines >= 2
    
    def _detect_book_pages(self, roi: np.ndarray) -> bool:
        """检测书是否显示页面"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 检查是否有规则的线条（页面边缘）
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if abs(theta - np.pi/2) < 0.3:  # 垂直线
                    return True
        
        return False
    
    def _detect_pen_tip(self, roi: np.ndarray) -> str:
        """检测笔尖类型"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 检查底部尖部
        tip_region = gray[-height//4:, :]
        tip_mean = np.mean(tip_region)
        
        if tip_mean > 200:
            return 'metal'
        else:
            return 'plastic'
    
    def _detect_pen_grip(self, roi: np.ndarray) -> bool:
        """检测笔是否有握把"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 检查中间区域是否有纹理
        height, width = gray.shape
        middle_region = gray[height//3:2*height//3, :]
        
        # 计算梯度变化
        grad_x = cv2.Sobel(middle_region, cv2.CV_64F, 1, 0, ksize=3)
        gradient_variance = np.var(grad_x)
        
        return gradient_variance > 100
    
    def _detect_camera_bump(self, roi: np.ndarray) -> bool:
        """检测手机摄像头凸起"""
        # 在手机背面区域查找圆形结构
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 检测圆形
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=5, maxRadius=30)
        
        return circles is not None and len(circles[0]) > 0
    
    def _detect_screen_notch(self, roi: np.ndarray) -> bool:
        """检测屏幕缺口"""
        # 检查顶部是否有缺口特征
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        top_region = gray[:height//8, :]
        edges = cv2.Canny(top_region, 50, 150)
        
        # 查找U形或V形特征
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  # 适中的缺口大小
                return True
        
        return False
    
    def _estimate_material(self, primary_class: str, color_info: Dict[str, Any],
                          shape_info: Dict[str, Any]) -> str:
        """
        估算物体材质
        
        Args:
            primary_class: 主类别
            color_info: 颜色信息
            shape_info: 形状信息
            
        Returns:
            材质估算结果
        """
        if primary_class in self.object_categories:
            possible_materials = self.object_categories[primary_class]['material_estimate']
            
            # 基于颜色和形状特征估算
            if primary_class == 'bottle':
                if color_info['dominant_color'] == 'transparent':
                    return 'glass'
                elif color_info['dominant_color'] in ['blue', 'green']:
                    return 'plastic'
                else:
                    return possible_materials[0]
                    
            elif primary_class == 'book':
                if color_info['color_variance'] > 1000:
                    return 'paper'
                else:
                    return 'leather'
                    
            elif primary_class == 'cup':
                if color_info['dominant_color'] == 'white':
                    return 'ceramic'
                elif color_info['dominant_color'] == 'metal':
                    return 'metal'
                else:
                    return 'plastic'
            
            return possible_materials[0] if possible_materials else 'unknown'
        
        return 'unknown'
    
    def _classify_size(self, area: int, primary_class: str) -> str:
        """
        根据面积分类尺寸
        
        Args:
            area: 物体面积
            primary_class: 主类别
            
        Returns:
            尺寸类别
        """
        if primary_class in self.object_categories:
            size_ranges = self.object_categories[primary_class]['size_range']
            
            for i, (min_area, max_area) in enumerate(size_ranges):
                if min_area <= area <= max_area:
                    size_names = ['small', 'medium', 'large']
                    return size_names[i] if i < len(size_names) else 'large'
        
        # 默认分类
        if area < 500:
            return 'small'
        elif area < 2000:
            return 'medium'
        else:
            return 'large'
    
    def _create_empty_result(self, fallback_class: str) -> ClassificationResult:
        """
        创建空分类结果
        
        Args:
            fallback_class: 备用类别
            
        Returns:
            空分类结果
        """
        return ClassificationResult(
            primary_class=fallback_class,
            primary_confidence=0.0,
            sub_classes={},
            attributes={},
            color_info={
                'dominant_color': 'unknown',
                'dominant_percentage': 0.0,
                'color_distribution': {},
                'mean_color_bgr': (128, 128, 128),
                'mean_color_rgb': (128, 128, 128),
                'color_variance': 0.0
            },
            shape_info={
                'shape_type': 'unknown',
                'vertex_count': 0,
                'is_regular': False,
                'bounding_rect': (0, 0, 0, 0),
                'aspect_ratio': 0.0,
                'circularity': 0.0,
                'solidity': 0.0,
                'contour_area': 0.0
            },
            size_category='unknown',
            material_estimate='unknown'
        )
    
    def get_object_info(self, primary_class: str) -> Dict[str, Any]:
        """
        获取物体类别信息
        
        Args:
            primary_class: 主类别
            
        Returns:
            物体类别信息
        """
        return self.object_categories.get(primary_class, {})
    
    def save_classification_report(self, results: List[ClassificationResult], 
                                  output_path: Union[str, Path]) -> None:
        """
        保存分类报告
        
        Args:
            results: 分类结果列表
            output_path: 输出文件路径
        """
        try:
            report = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_objects': len(results),
                'results': []
            }
            
            for result in results:
                result_dict = {
                    'primary_class': result.primary_class,
                    'primary_confidence': result.primary_confidence,
                    'sub_classes': result.sub_classes,
                    'attributes': result.attributes,
                    'color_info': result.color_info,
                    'shape_info': result.shape_info,
                    'size_category': result.size_category,
                    'material_estimate': result.material_estimate
                }
                report['results'].append(result_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分类报告已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存分类报告失败: {e}")