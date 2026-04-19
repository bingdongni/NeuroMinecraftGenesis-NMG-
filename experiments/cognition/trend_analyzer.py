"""
趋势分析器
=========

该模块实现了六维认知能力发展趋势的深度分析，包括：
- 长期趋势识别
- 模式检测与分类
- 拐点检测
- 趋势强度评估
- 预测建模
- 可视化分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import logging
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """趋势方向枚举"""
    STRONG_UP = "强烈上升"
    MODERATE_UP = "温和上升"
    STABLE = "稳定"
    MODERATE_DOWN = "温和下降"
    STRONG_DOWN = "强烈下降"
    VOLATILE = "波动"

class TrendPattern(Enum):
    """趋势模式枚举"""
    LINEAR = "线性趋势"
    EXPONENTIAL = "指数趋势"
    LOGARITHMIC = "对数趋势"
    POLYNOMIAL = "多项式趋势"
    CYCLIC = "周期性模式"
    IRREGULAR = "不规则模式"

@dataclass
class TrendAnalysis:
    """趋势分析结果数据类"""
    dimension: str
    direction: TrendDirection
    pattern: TrendPattern
    strength: float  # 趋势强度 (0-1)
    slope: float     # 斜率
    r_squared: float # 拟合度
    inflection_points: List[int]  # 拐点
    forecast_next_6h: List[float]  # 未来6小时预测
    confidence: float  # 预测置信度
    significance: float  # 统计显著性

class TrendAnalyzer:
    """认知能力趋势分析器"""
    
    def __init__(self, min_data_points: int = 5):
        """
        初始化趋势分析器
        
        Args:
            min_data_points: 最少数据点数
        """
        self.min_data_points = min_data_points
        self.analysis_cache: Dict[str, TrendAnalysis] = {}
        self.analysis_history: List[Dict] = []
        
        # 分析参数
        self.params = {
            'smoothing_window': 3,      # 平滑窗口
            'inflection_sensitivity': 0.1,  # 拐点敏感度
            'trend_threshold': 0.05,    # 趋势阈值
            'volatility_threshold': 0.15,  # 波动阈值
            'forecast_horizon': 6,      # 预测时间范围
            'confidence_level': 0.95    # 置信水平
        }
        
        logger.info(f"趋势分析器初始化完成 - 最小数据点: {min_data_points}")
    
    def _smooth_data(self, data: List[float], window_size: int = None) -> np.ndarray:
        """
        数据平滑处理
        
        Args:
            data: 原始数据
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的数据数组
        """
        if len(data) < self.min_data_points:
            return np.array(data)
        
        window_size = window_size or self.params['smoothing_window']
        window_size = min(window_size, len(data) // 2)
        
        if window_size < 2:
            return np.array(data)
        
        # 使用Savitzky-Golay滤波器
        try:
            if window_size % 2 == 0:
                window_size += 1
            smoothed = savgol_filter(data, window_size, 2)
            return smoothed
        except:
            # 如果Savitzky-Golay失败，使用简单移动平均
            return self._moving_average(data, window_size)
    
    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        """计算移动平均"""
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            smoothed.append(np.mean(data[start_idx:end_idx]))
        return np.array(smoothed)
    
    def _detect_inflection_points(self, data: np.ndarray) -> List[int]:
        """
        检测拐点
        
        Args:
            data: 数据数组
            
        Returns:
            拐点索引列表
        """
        if len(data) < 3:
            return []
        
        # 计算一阶导数
        first_derivative = np.diff(data)
        
        # 计算二阶导数
        second_derivative = np.diff(first_derivative)
        
        # 找到符号变化的点
        inflection_points = []
        sensitivity = self.params['inflection_sensitivity']
        
        for i in range(1, len(second_derivative)):
            if abs(second_derivative[i]) > sensitivity * np.std(data):
                if second_derivative[i-1] * second_derivative[i] < 0:  # 符号变化
                    inflection_points.append(i + 1)
        
        return inflection_points
    
    def _classify_trend_direction(self, slope: float, r_squared: float) -> TrendDirection:
        """
        分类趋势方向
        
        Args:
            slope: 斜率
            r_squared: 拟合度
            
        Returns:
            趋势方向
        """
        # 根据斜率和拟合度分类
        if r_squared < 0.3:  # 拟合度低，可能是波动
            return TrendDirection.VOLATILE
        
        abs_slope = abs(slope)
        if abs_slope < self.params['trend_threshold']:
            return TrendDirection.STABLE
        
        if slope > 0:
            if abs_slope > 0.3:
                return TrendDirection.STRONG_UP
            else:
                return TrendDirection.MODERATE_UP
        else:
            if abs_slope > 0.3:
                return TrendDirection.STRONG_DOWN
            else:
                return TrendDirection.MODERATE_DOWN
    
    def _identify_pattern_type(self, x: np.ndarray, y: np.ndarray) -> TrendPattern:
        """
        识别趋势模式类型
        
        Args:
            x: x坐标
            y: y坐标
            
        Returns:
            趋势模式
        """
        if len(x) < 4:
            return TrendPattern.LINEAR
        
        try:
            # 线性拟合
            linear_r2 = LinearRegression().fit(x.reshape(-1, 1), y).score(x.reshape(-1, 1), y)
            
            # 多项式拟合（二次）
            poly_features = np.column_stack([x, x**2])
            poly_model = LinearRegression().fit(poly_features, y)
            poly_r2 = poly_model.score(poly_features, y)
            
            # 对数拟合
            log_x = np.log(x + 1)  # 避免log(0)
            log_r2 = LinearRegression().fit(log_x.reshape(-1, 1), y).score(log_x.reshape(-1, 1), y)
            
            # 指数拟合
            exp_x = np.exp(x / max(x))
            exp_r2 = LinearRegression().fit(exp_x.reshape(-1, 1), y).score(exp_x.reshape(-1, 1), y)
            
            # 选择最佳拟合
            r2_scores = {
                TrendPattern.LINEAR: linear_r2,
                TrendPattern.POLYNOMIAL: poly_r2,
                TrendPattern.LOGARITHMIC: log_r2,
                TrendPattern.EXPONENTIAL: exp_r2
            }
            
            best_pattern = max(r2_scores, key=r2_scores.get)
            return best_pattern
            
        except:
            return TrendPattern.LINEAR
    
    def _calculate_trend_strength(self, r_squared: float, slope: float, 
                                volatility: float) -> float:
        """
        计算趋势强度
        
        Args:
            r_squared: 拟合度
            slope: 斜率
            volatility: 数据波动性
            
        Returns:
            趋势强度 (0-1)
        """
        # 基础强度来自拟合度
        strength = r_squared
        
        # 根据斜率调整
        if abs(slope) > 0.1:
            strength *= 1.2
        
        # 根据波动性惩罚
        strength *= (1 - min(0.5, volatility))
        
        return max(0, min(1, strength))
    
    def _forecast_trend(self, x: np.ndarray, y: np.ndarray, 
                       pattern: TrendPattern, steps: int) -> Tuple[List[float], float]:
        """
        基于趋势模式预测未来值
        
        Args:
            x: x坐标
            y: y坐标
            pattern: 趋势模式
            steps: 预测步数
            
        Returns:
            预测值列表和置信度
        """
        try:
            if pattern == TrendPattern.LINEAR:
                # 线性预测
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                future_x = np.arange(x[-1] + 1, x[-1] + steps + 1)
                forecast = model.predict(future_x.reshape(-1, 1))
                confidence = model.score(x.reshape(-1, 1), y)
                
            elif pattern == TrendPattern.POLYNOMIAL:
                # 多项式预测
                poly_features = np.column_stack([x, x**2])
                model = LinearRegression()
                model.fit(poly_features, y)
                future_x = np.arange(x[-1] + 1, x[-1] + steps + 1)
                future_features = np.column_stack([future_x, future_x**2])
                forecast = model.predict(future_features)
                confidence = model.score(poly_features, y)
                
            else:
                # 简单线性外推作为默认
                linear_model = LinearRegression()
                linear_model.fit(x.reshape(-1, 1), y)
                future_x = np.arange(x[-1] + 1, x[-1] + steps + 1)
                forecast = linear_model.predict(future_x.reshape(-1, 1))
                confidence = linear_model.score(x.reshape(-1, 1), y)
            
            # 确保预测值在合理范围内
            forecast = np.clip(forecast, 0, 100)
            
            return forecast.tolist(), max(0, min(1, confidence))
            
        except Exception as e:
            logger.warning(f"趋势预测失败: {e}")
            # 返回基于最后几个点的平均值
            recent_avg = np.mean(y[-min(3, len(y)):])
            forecast = [recent_avg] * steps
            return forecast, 0.1
    
    def _calculate_significance(self, slope: float, r_squared: float, 
                              n_points: int) -> float:
        """
        计算统计显著性
        
        Args:
            slope: 斜率
            r_squared: 拟合度
            n_points: 数据点数
            
        Returns:
            显著性值 (0-1)
        """
        if n_points < 3:
            return 0.0
        
        # 使用t检验检验斜率的显著性
        try:
            # 简化的显著性计算
            effect_size = abs(slope) * np.sqrt(n_points - 2)
            t_statistic = effect_size / np.sqrt(1 - r_squared + 1e-10)
            
            # 转换为p值
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_statistic), n_points - 2))
            significance = max(0, 1 - p_value)
            
            return significance
        except:
            # 备用计算
            significance = min(1, abs(slope) * r_squared * np.sqrt(n_points) / 10)
            return significance
    
    def analyze_dimension_trend(self, scores: List[float], 
                              timestamps: List[datetime],
                              dimension_name: str) -> TrendAnalysis:
        """
        分析指定维度的趋势
        
        Args:
            scores: 分数列表
            timestamps: 时间戳列表
            dimension_name: 维度名称
            
        Returns:
            趋势分析结果
        """
        if len(scores) < self.min_data_points:
            raise ValueError(f"数据点不足，需要至少 {self.min_data_points} 个数据点")
        
        # 准备数据
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # 数据平滑
        y_smooth = self._smooth_data(scores)
        
        # 计算趋势统计
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_smooth)
        r_squared = r_value ** 2
        
        # 计算波动性
        volatility = np.std(y) / (np.mean(y) + 1e-10)
        
        # 分类趋势
        direction = self._classify_trend_direction(slope, r_squared)
        
        # 识别模式
        pattern = self._identify_pattern_type(x, y_smooth)
        
        # 计算趋势强度
        strength = self._calculate_trend_strength(r_squared, slope, volatility)
        
        # 检测拐点
        inflection_points = self._detect_inflection_points(y_smooth)
        
        # 预测未来趋势
        forecast_values, confidence = self._forecast_trend(x, y_smooth, pattern, 
                                                         self.params['forecast_horizon'])
        
        # 计算统计显著性
        significance = self._calculate_significance(slope, r_squared, len(scores))
        
        # 创建分析结果
        analysis = TrendAnalysis(
            dimension=dimension_name,
            direction=direction,
            pattern=pattern,
            strength=strength,
            slope=slope,
            r_squared=r_squared,
            inflection_points=inflection_points,
            forecast_next_6h=forecast_values,
            confidence=confidence,
            significance=significance
        )
        
        # 缓存结果
        cache_key = f"{dimension_name}_{len(scores)}"
        self.analysis_cache[cache_key] = analysis
        
        logger.info(f"维度 '{dimension_name}' 趋势分析完成 - 方向: {direction.value}, 强度: {strength:.3f}")
        
        return analysis
    
    def analyze_all_dimensions(self, metrics_history: List, hours: int = 24) -> Dict[str, TrendAnalysis]:
        """
        分析所有六个维度的趋势
        
        Args:
            metrics_history: 认知指标历史列表
            hours: 分析时间范围
            
        Returns:
            各维度趋势分析结果字典
        """
        if not metrics_history:
            raise ValueError("指标历史为空")
        
        # 提取时间范围内的数据
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in metrics_history if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < self.min_data_points:
            logger.warning(f"时间范围内数据点不足，使用全部数据")
            recent_metrics = metrics_history
        
        results = {}
        dimensions = ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']
        
        for dim in dimensions:
            try:
                # 提取指定维度的分数
                scores = [getattr(m, f"{dim}_score") for m in recent_metrics]
                timestamps = [m.timestamp for m in recent_metrics]
                
                # 分析趋势
                analysis = self.analyze_dimension_trend(scores, timestamps, dim)
                results[dim] = analysis
                
            except Exception as e:
                logger.error(f"维度 {dim} 趋势分析失败: {e}")
        
        # 记录分析历史
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'hours_analyzed': hours,
            'dimensions_analyzed': len(results),
            'data_points': len(recent_metrics)
        })
        
        return results
    
    def compare_trends(self, analysis1: Dict[str, TrendAnalysis], 
                      analysis2: Dict[str, TrendAnalysis]) -> Dict:
        """
        比较两组趋势分析结果
        
        Args:
            analysis1: 第一组分析结果
            analysis2: 第二组分析结果
            
        Returns:
            比较结果字典
        """
        comparison = {}
        
        for dim in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
            if dim in analysis1 and dim in analysis2:
                trend1 = analysis1[dim]
                trend2 = analysis2[dim]
                
                # 比较方向
                direction_change = self._compare_directions(trend1.direction, trend2.direction)
                
                # 比较强度
                strength_diff = trend2.strength - trend1.strength
                
                # 比较斜率
                slope_diff = trend2.slope - trend1.slope
                
                # 计算整体改进
                overall_improvement = (trend2.overall_score - trend1.overall_score 
                                     if hasattr(trend2, 'overall_score') and hasattr(trend1, 'overall_score')
                                     else 0)
                
                comparison[dim] = {
                    'direction_change': direction_change,
                    'strength_change': strength_diff,
                    'slope_change': slope_diff,
                    'trend1_direction': trend1.direction.value,
                    'trend2_direction': trend2.direction.value,
                    'trend1_strength': trend1.strength,
                    'trend2_strength': trend2.strength
                }
        
        return comparison
    
    def _compare_directions(self, dir1: TrendDirection, dir2: TrendDirection) -> str:
        """比较两个趋势方向"""
        direction_scores = {
            TrendDirection.STRONG_DOWN: 1,
            TrendDirection.MODERATE_DOWN: 2,
            TrendDirection.STABLE: 3,
            TrendDirection.MODERATE_UP: 4,
            TrendDirection.STRONG_UP: 5,
            TrendDirection.VOLATILE: 3  # 波动中性
        }
        
        score1 = direction_scores.get(dir1, 3)
        score2 = direction_scores.get(dir2, 3)
        
        if score2 > score1:
            return "改善"
        elif score2 < score1:
            return "恶化"
        else:
            return "稳定"
    
    def get_trend_summary(self, analysis: Dict[str, TrendAnalysis]) -> Dict:
        """
        获取趋势摘要
        
        Args:
            analysis: 趋势分析结果
            
        Returns:
            趋势摘要字典
        """
        if not analysis:
            return {}
        
        # 统计各方向的数量
        direction_counts = {}
        strength_values = []
        r_squared_values = []
        
        for dim, trend in analysis.items():
            direction = trend.direction.value
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            strength_values.append(trend.strength)
            r_squared_values.append(trend.r_squared)
        
        # 计算平均强度和拟合度
        avg_strength = np.mean(strength_values) if strength_values else 0
        avg_r_squared = np.mean(r_squared_values) if r_squared_values else 0
        
        # 识别最强和最弱趋势
        strongest_dim = max(analysis.keys(), key=lambda k: analysis[k].strength) if analysis else None
        weakest_dim = min(analysis.keys(), key=lambda k: analysis[k].strength) if analysis else None
        
        # 识别主导趋势模式
        pattern_counts = {}
        for trend in analysis.values():
            pattern = trend.pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        dominant_pattern = max(pattern_counts, key=pattern_counts.get) if pattern_counts else None
        
        return {
            'total_dimensions': len(analysis),
            'direction_distribution': direction_counts,
            'average_strength': avg_strength,
            'average_r_squared': avg_r_squared,
            'strongest_dimension': strongest_dim,
            'strongest_strength': analysis[strongest_dim].strength if strongest_dim else 0,
            'weakest_dimension': weakest_dim,
            'weakest_strength': analysis[weakest_dim].strength if weakest_dim else 0,
            'dominant_pattern': dominant_pattern,
            'pattern_distribution': pattern_counts
        }
    
    def export_analysis(self, analysis: Dict[str, TrendAnalysis], 
                       filepath: str) -> bool:
        """
        导出趋势分析结果
        
        Args:
            analysis: 趋势分析结果
            filepath: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'dimension_analyses': {},
                'summary': self.get_trend_summary(analysis)
            }
            
            for dim, trend in analysis.items():
                export_data['dimension_analyses'][dim] = {
                    'direction': trend.direction.value,
                    'pattern': trend.pattern.value,
                    'strength': trend.strength,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared,
                    'inflection_points': trend.inflection_points,
                    'forecast_next_6h': trend.forecast_next_6h,
                    'confidence': trend.confidence,
                    'significance': trend.significance
                }
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"趋势分析结果已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出趋势分析失败: {e}")
            return False
    
    def clear_cache(self) -> None:
        """清除分析缓存"""
        self.analysis_cache.clear()
        self.analysis_history.clear()
        logger.info("趋势分析缓存已清除")

if __name__ == "__main__":
    # 测试趋势分析器
    from datetime import datetime, timedelta
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 生成24小时数据
    hours = 24
    timestamps = [datetime.now() - timedelta(hours=hours-i) for i in range(hours)]
    
    # 模拟记忆分数上升趋势
    memory_scores = [50 + i * 1.5 + np.random.normal(0, 2) for i in range(hours)]
    
    # 模拟思维分数波动
    thinking_scores = [60 + np.sin(i * 0.5) * 5 + np.random.normal(0, 3) for i in range(hours)]
    
    # 模拟创造力分数下降趋势
    creativity_scores = [70 - i * 0.8 + np.random.normal(0, 2) for i in range(hours)]
    
    # 创建模拟认知指标
    class MockMetrics:
        def __init__(self, timestamp, memory, thinking, creativity):
            self.timestamp = timestamp
            self.memory_score = memory
            self.thinking_score = thinking
            self.creativity_score = creativity
            self.observation_score = np.random.normal(65, 5)
            self.attention_score = np.random.normal(60, 4)
            self.imagination_score = np.random.normal(55, 6)
    
    metrics_history = []
    for i in range(hours):
        metrics = MockMetrics(timestamps[i], memory_scores[i], thinking_scores[i], creativity_scores[i])
        metrics_history.append(metrics)
    
    # 创建趋势分析器
    analyzer = TrendAnalyzer(min_data_points=5)
    
    # 测试单维度分析
    print("=== 测试记忆维度趋势分析 ===")
    memory_analysis = analyzer.analyze_dimension_trend(memory_scores, timestamps, "memory")
    print(f"方向: {memory_analysis.direction.value}")
    print(f"强度: {memory_analysis.strength:.3f}")
    print(f"斜率: {memory_analysis.slope:.3f}")
    print(f"R²: {memory_analysis.r_squared:.3f}")
    print(f"未来6小时预测: {[f'{x:.1f}' for x in memory_analysis.forecast_next_6h]}")
    
    # 测试全维度分析
    print("\n=== 测试全维度趋势分析 ===")
    all_analysis = analyzer.analyze_all_dimensions(metrics_history, hours=24)
    
    for dim, analysis in all_analysis.items():
        print(f"{dim}: {analysis.direction.value} (强度: {analysis.strength:.3f})")
    
    # 获取趋势摘要
    print("\n=== 趋势摘要 ===")
    summary = analyzer.get_trend_summary(all_analysis)
    print(f"主导模式: {summary['dominant_pattern']}")
    print(f"最强维度: {summary['strongest_dimension']} (强度: {summary['strongest_strength']:.3f})")
    print(f"方向分布: {summary['direction_distribution']}")
    
    # 导出分析结果
    analyzer.export_analysis(all_analysis, "test_trend_analysis.json")
    print("\n趋势分析结果已导出")
    
    print("趋势分析器测试完成")