"""
USB摄像头场景捕获系统 - 场景处理模块

该模块实现对捕获图像的场景分析、处理和增强功能，
包括图像预处理、特征提取、场景理解等。

作者: AI助手
创建时间: 2025-11-13
版本: 1.0
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, List, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import colorsys


class SceneType(Enum):
    """场景类型枚举"""
    UNKNOWN = "未知"
    OFFICE = "办公室"
    BEDROOM = "卧室"
    LIVING_ROOM = "客厅"
    KITCHEN = "厨房"
    LABORATORY = "实验室"
    OUTDOOR = "户外"


@dataclass
class SceneFeatures:
    """场景特征数据结构"""
    brightness: float = 0.0
    contrast: float = 0.0
    color_temperature: float = 0.0
    saturation: float = 0.0
    texture_variance: float = 0.0
    edge_density: float = 0.0
    object_count: int = 0
    depth_hint: float = 0.0
    motion_level: float = 0.0


@dataclass
class SceneInfo:
    """场景信息数据结构"""
    scene_type: SceneType = SceneType.UNKNOWN
    confidence: float = 0.0
    features: SceneFeatures = None
    timestamp: float = 0.0
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.features is None:
            self.features = SceneFeatures()
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SceneProcessor:
    """
    场景处理器
    
    该类负责：
    1. 图像预处理和增强
    2. 场景特征提取和分析
    3. 场景类型识别和分类
    4. 图像质量评估和优化
    5. 实时场景变化检测
    """
    
    def __init__(self):
        """初始化场景处理器"""
        self.logger = logging.getLogger(__name__)
        
        # 上一帧用于运动检测
        self.previous_frame = None
        self.frame_count = 0
        
        # 统计信息
        self.processing_times = []
        self.max_history = 100
        
        # 场景变化阈值
        self.motion_threshold = 5000.0  # 像素变化阈值
        self.brightness_change_threshold = 20.0
        
        # 初始化OpenCV的HOG行人检测器（可选）
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.people_detector_enabled = True
        except Exception as e:
            self.logger.warning(f"无法初始化HOG检测器: {e}")
            self.people_detector_enabled = False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            np.ndarray: 处理后的图像帧
        """
        if frame is None:
            return None
        
        start_time = time.time()
        
        try:
            # 图像预处理
            processed_frame = self._preprocess_image(frame)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_history:
                self.processing_times.pop(0)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return frame
    
    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """图像预处理和增强"""
        # 确保输入是BGR格式
        if len(frame.shape) == 3:
            bgr_frame = frame.copy()
        else:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 降噪处理
        denoised = cv2.fastNlMeansDenoisingColored(bgr_frame, None, 10, 10, 7, 21)
        
        # 对比度和亮度调整
        alpha = 1.1  # 对比度控制
        beta = 10    # 亮度控制
        enhanced = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        
        # 锐化处理
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 直方图均衡化（CLAHE）
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def analyze_environment(self, frame: np.ndarray) -> SceneInfo:
        """
        分析环境特征
        
        Args:
            frame: 输入图像帧
            
        Returns:
            SceneInfo: 场景分析结果
        """
        if frame is None:
            return SceneInfo()
        
        try:
            # 提取场景特征
            features = self._extract_scene_features(frame)
            
            # 场景类型识别
            scene_type = self._classify_scene_type(features)
            
            # 计算置信度
            confidence = self._calculate_confidence(features, scene_type)
            
            # 质量评估
            quality_score = self._assess_image_quality(frame, features)
            
            scene_info = SceneInfo(
                scene_type=scene_type,
                confidence=confidence,
                features=features,
                quality_score=quality_score
            )
            
            self.frame_count += 1
            return scene_info
            
        except Exception as e:
            self.logger.error(f"环境分析失败: {e}")
            return SceneInfo()
    
    def _extract_scene_features(self, frame: np.ndarray) -> SceneFeatures:
        """提取场景特征"""
        features = SceneFeatures()
        
        try:
            # 转换到不同色彩空间进行分析
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # 亮度和对比度
            features.brightness = np.mean(gray)
            features.contrast = np.std(gray)
            
            # 颜色温度估算（基于色相）
            hue = hsv[:,:,0]
            features.color_temperature = np.mean(hue) * 2  # 转换为近似温度
            
            # 饱和度统计
            features.saturation = np.mean(hsv[:,:,1])
            
            # 纹理方差（高频信息）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features.texture_variance = np.var(laplacian)
            
            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            features.edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 运动检测
            motion_level = self._detect_motion(gray)
            features.motion_level = motion_level
            
            # 对象数量估算（简单的轮廓检测）
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 过滤小的噪声轮廓
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            features.object_count = len(valid_contours)
            
            # 深度提示（基于焦点模糊和对比度）
            features.depth_hint = self._estimate_depth_hint(gray, features.contrast)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return features
    
    def _detect_motion(self, gray_frame: np.ndarray) -> float:
        """检测运动水平"""
        if self.previous_frame is None:
            self.previous_frame = gray_frame.copy()
            return 0.0
        
        # 计算帧差
        frame_diff = cv2.absdiff(gray_frame, self.previous_frame)
        motion_pixels = np.sum(frame_diff > 25)  # 阈值化
        
        # 运动水平归一化
        total_pixels = gray_frame.shape[0] * gray_frame.shape[1]
        motion_level = motion_pixels / total_pixels
        
        # 更新前一帧
        self.previous_frame = gray_frame.copy()
        
        return motion_level
    
    def _estimate_depth_hint(self, gray_frame: np.ndarray, contrast: float) -> float:
        """估算深度提示"""
        # 基于局部对比度和锐度估算深度
        kernel_sizes = [3, 5, 9, 15]
        sharpness_scores = []
        
        for ksize in kernel_sizes:
            blurred = cv2.GaussianBlur(gray_frame, (ksize, ksize), 0)
            diff = cv2.absdiff(gray_frame, blurred)
            sharpness = np.mean(diff)
            sharpness_scores.append(sharpness)
        
        # 深度估算（高锐度表示前景，低锐度表示背景）
        depth_hint = np.mean(sharpness_scores) / 255.0
        return np.clip(depth_hint, 0.0, 1.0)
    
    def _classify_scene_type(self, features: SceneFeatures) -> SceneType:
        """场景类型分类"""
        # 基于特征的场景分类逻辑
        
        # 实验室场景：高亮度、均匀照明、高对比度
        if (features.brightness > 150 and 
            features.contrast > 50 and 
            features.color_temperature < 80):
            return SceneType.LABORATORY
        
        # 办公室场景：中等亮度、蓝色色调、有规律的纹理
        elif (features.brightness > 100 and 
              features.brightness < 200 and
              features.color_temperature > 80 and 
              features.color_temperature < 140):
            return SceneType.OFFICE
        
        # 卧室场景：温暖色调、低到中等亮度
        elif (features.color_temperature > 140 and 
              features.brightness < 150):
            return SceneType.BEDROOM
        
        # 客厅场景：较高亮度、丰富颜色
        elif (features.brightness > 120 and 
              features.saturation > 50):
            return SceneType.LIVING_ROOM
        
        # 厨房场景：明亮、白色/银色调
        elif (features.brightness > 140 and 
              features.color_temperature < 120 and
              features.saturation < 40):
            return SceneType.KITCHEN
        
        # 户外场景：高对比度、变化的光照
        elif (features.contrast > 60 or 
              features.motion_level > 0.1):
            return SceneType.OUTDOOR
        
        return SceneType.UNKNOWN
    
    def _calculate_confidence(self, features: SceneFeatures, scene_type: SceneType) -> float:
        """计算分类置信度"""
        # 基于特征一致性计算置信度
        confidence_factors = []
        
        # 亮度一致性
        if 50 <= features.brightness <= 200:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # 对比度合理性
        if 20 <= features.contrast <= 80:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # 纹理复杂度
        if 10 <= features.texture_variance <= 1000:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # 场景特定规则
        if scene_type != SceneType.UNKNOWN:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    def _assess_image_quality(self, frame: np.ndarray, features: SceneFeatures) -> float:
        """评估图像质量"""
        quality_factors = []
        
        # 亮度质量
        if 80 <= features.brightness <= 180:
            quality_factors.append(1.0)
        elif 50 <= features.brightness <= 220:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # 对比度质量
        if features.contrast > 30:
            quality_factors.append(1.0)
        elif features.contrast > 15:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # 清晰度质量
        sharpness = features.texture_variance
        if sharpness > 100:
            quality_factors.append(1.0)
        elif sharpness > 50:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # 噪声水平（基于边缘密度异常）
        if features.edge_density < 0.3:
            quality_factors.append(1.0)
        elif features.edge_density < 0.5:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)
        
        return np.mean(quality_factors)
    
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人员
        
        Args:
            frame: 输入图像帧
            
        Returns:
            List[Tuple[int, int, int, int]]: 检测到的人员边界框列表 (x, y, w, h)
        """
        if not self.people_detector_enabled:
            return []
        
        try:
            # 调整图像大小以提高检测速度
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            else:
                resized_frame = frame
            
            # 使用HOG检测器
            rects, weights = self.hog.detectMultiScale(
                resized_frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05
            )
            
            # 调整边界框坐标到原始图像尺寸
            if scale != 1.0:
                rects = [(x/scale, y/scale, w/scale, h/scale) for (x, y, w, h) in rects]
            
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
            
        except Exception as e:
            self.logger.error(f"人员检测失败: {e}")
            return []
    
    def get_scene_statistics(self) -> Dict[str, Any]:
        """获取场景处理统计信息"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            "total_frames_processed": self.frame_count,
            "average_processing_time": avg_processing_time,
            "processing_fps": 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            "motion_detection_enabled": self.previous_frame is not None,
            "people_detection_enabled": self.people_detector_enabled,
            "processing_time_history": len(self.processing_times)
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.frame_count = 0
        self.processing_times.clear()
        self.previous_frame = None


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='场景处理系统')
    parser.add_argument('--input', type=str, help='输入视频文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--output', type=str, help='输出视频文件路径')
    
    args = parser.parse_args()
    
    # 初始化场景处理器
    processor = SceneProcessor()
    
    # 选择输入源
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("无法打开输入源")
        exit(1)
    
    # 输出设置
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 20.0, (640, 480))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = processor.process_frame(frame)
            
            # 分析环境
            scene_info = processor.analyze_environment(frame)
            
            # 检测人员
            people = processor.detect_people(processed_frame)
            
            # 在图像上标注信息
            # 场景类型
            info_text = f"场景: {scene_info.scene_type.value} ({scene_info.confidence:.2f})"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 质量评分
            quality_text = f"质量: {scene_info.quality_score:.2f}"
            cv2.putText(processed_frame, quality_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 特征信息
            features = scene_info.features
            feature_text = f"亮度: {features.brightness:.1f}, 对比度: {features.contrast:.1f}"
            cv2.putText(processed_frame, feature_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 标注检测到的人员
            for (x, y, w, h) in people:
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(processed_frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 显示处理统计
            stats = processor.get_scene_statistics()
            fps_text = f"FPS: {stats['processing_fps']:.1f}"
            cv2.putText(processed_frame, fps_text, (10, processed_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示图像
            cv2.imshow('场景处理系统', processed_frame)
            
            # 保存输出
            if args.output:
                out.write(processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        cap.release()
        if args.output:
            out.release()
        cv2.destroyAllWindows()
        print("程序结束")
        
        # 输出统计信息
        stats = processor.get_scene_statistics()
        print(f"处理统计: {stats}")