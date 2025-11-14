"""
学习曲线分析器

该模块实现了学习曲线分析功能，包括：
1. 学习速度评估
2. 收敛性分析
3. 稳定性检测
4. 学习曲线特征提取

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
import json
from scipy import stats, signal, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class LearningCurveMetrics:
    """学习曲线指标数据结构"""
    task_id: int
    learning_speed: float
    convergence_rate: float
    final_performance: float
    plateau_duration: float
    oscillation_amplitude: float
    stability_score: float
    improvement_efficiency: float
    learning_curve_type: str  # "exponential", "linear", "sigmoid", "oscillating", "plateau"
    fit_quality: float  # R²决定系数
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CurveAnalysisResult:
    """曲线分析结果数据结构"""
    curve_type: str
    parameters: Dict[str, float]
    goodness_of_fit: float
    convergence_point: Optional[float]
    learning_rate_estimate: float
    plateau_point: Optional[float]
    stability_metrics: Dict[str, float]
    anomaly_detection: Dict[str, Any]


class LearningCurveAnalyzer:
    """
    学习曲线分析器
    
    该类负责分析学习曲线的各种特征，包括：
    - 学习速度评估（前期斜率）
    - 收敛性分析（后期稳定性）
    - 稳定性检测（波动性）
    - 学习曲线类型识别
    - 异常检测和预警
    """
    
    def __init__(self, window_size: int = 10, convergence_threshold: float = 0.01):
        """
        初始化学习曲线分析器
        
        Args:
            window_size: 分析窗口大小
            convergence_threshold: 收敛阈值
        """
        self.window_size = window_size
        self.threshold = convergence_threshold
        self.logger = logging.getLogger("curve_analyzer")
        
        # 学习曲线数据存储
        self.learning_curves: Dict[int, List[float]] = {}
        self.curve_metrics: List[LearningCurveMetrics] = []
        self.curve_analyses: Dict[int, CurveAnalysisResult] = {}
        
        # 统计分析
        self.curve_statistics: Dict[str, Any] = {}
        self.curve_comparison: Dict[str, float] = {}
        
        # 异常检测
        self.anomaly_thresholds = {
            'sudden_drop': 0.2,  # 突然下降阈值
            'plateau_length': 0.7,  # 平台期长度阈值
            'oscillation_amplitude': 0.3,  # 振荡幅度阈值
            'learning_stagnation': 0.05  # 学习停滞阈值
        }
        
        self.logger.info(f"学习曲线分析器初始化完成，收敛阈值: {convergence_threshold:.3f}")
    
    def analyze_learning_curves(self, performance_history: Dict[int, List[float]], 
                              current_task_count: int) -> Dict[str, Any]:
        """
        分析学习曲线
        
        Args:
            performance_history: 性能历史数据
            current_task_count: 当前任务数量
            
        Returns:
            学习曲线分析结果
        """
        self.logger.info(f"分析 {current_task_count} 个任务的学习曲线")
        
        # 清空之前的数据
        self.learning_curves.clear()
        self.curve_metrics.clear()
        self.curve_analyses.clear()
        
        # 为每个任务分析学习曲线
        for task_id in range(current_task_count):
            if task_id in performance_history:
                episode_rewards = performance_history[task_id]
                
                if len(episode_rewards) >= 10:  # 至少需要10个数据点
                    self.learning_curves[task_id] = episode_rewards.copy()
                    
                    # 分析单个学习曲线
                    metrics = self._analyze_single_curve(task_id, episode_rewards)
                    self.curve_metrics.append(metrics)
                    
                    # 详细曲线分析
                    analysis = self._analyze_curve_characteristics(task_id, episode_rewards)
                    self.curve_analyses[task_id] = analysis
        
        # 生成综合分析报告
        analysis_report = self._generate_comprehensive_analysis(current_task_count)
        
        return analysis_report
    
    def _analyze_single_curve(self, task_id: int, rewards: List[float]) -> LearningCurveMetrics:
        """分析单个学习曲线"""
        rewards_array = np.array(rewards)
        
        # 1. 学习速度分析（前期斜率）
        early_window = min(self.window_size, len(rewards) // 3)
        if early_window >= 2:
            x = np.arange(early_window)
            y = rewards_array[:early_window]
            
            # 线性回归计算斜率
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            learning_speed = slope
        else:
            learning_speed = 0.0
        
        # 2. 收敛率分析（后期稳定性）
        late_window = min(self.window_size, len(rewards) // 3)
        if late_window >= 2:
            late_rewards = rewards_array[-late_window:]
            convergence_rate = 1.0 - np.std(late_rewards) / (np.mean(late_rewards) + 1e-8)
        else:
            convergence_rate = 0.0
        
        # 3. 最终性能
        final_performance = np.mean(rewards_array[-self.window_size:]) if len(rewards_array) >= self.window_size else np.mean(rewards_array)
        
        # 4. 平台期检测
        plateau_duration = self._detect_plateau_duration(rewards_array)
        
        # 5. 振荡幅度
        oscillation_amplitude = self._calculate_oscillation_amplitude(rewards_array)
        
        # 6. 稳定性分数
        stability_score = self._calculate_stability_score(rewards_array)
        
        # 7. 改进效率
        improvement_efficiency = self._calculate_improvement_efficiency(rewards_array)
        
        # 8. 学习曲线类型识别
        curve_type = self._identify_curve_type(rewards_array)
        
        # 9. 拟合质量
        fit_quality = self._calculate_fit_quality(rewards_array, curve_type)
        
        return LearningCurveMetrics(
            task_id=task_id,
            learning_speed=learning_speed,
            convergence_rate=convergence_rate,
            final_performance=final_performance,
            plateau_duration=plateau_duration,
            oscillation_amplitude=oscillation_amplitude,
            stability_score=stability_score,
            improvement_efficiency=improvement_efficiency,
            learning_curve_type=curve_type,
            fit_quality=fit_quality
        )
    
    def _analyze_curve_characteristics(self, task_id: int, rewards: List[float]) -> CurveAnalysisResult:
        """分析学习曲线特征"""
        rewards_array = np.array(rewards)
        
        # 曲线类型识别和参数拟合
        curve_type, parameters, goodness_of_fit = self._fit_curve_model(rewards_array)
        
        # 收敛点分析
        convergence_point = self._find_convergence_point(rewards_array)
        
        # 学习率估计
        learning_rate_estimate = self._estimate_learning_rate(rewards_array)
        
        # 平台点分析
        plateau_point = self._find_plateau_point(rewards_array)
        
        # 稳定性指标
        stability_metrics = self._calculate_stability_metrics(rewards_array)
        
        # 异常检测
        anomaly_detection = self._detect_curve_anomalies(rewards_array)
        
        return CurveAnalysisResult(
            curve_type=curve_type,
            parameters=parameters,
            goodness_of_fit=goodness_of_fit,
            convergence_point=convergence_point,
            learning_rate_estimate=learning_rate_estimate,
            plateau_point=plateau_point,
            stability_metrics=stability_metrics,
            anomaly_detection=anomaly_detection
        )
    
    def _detect_plateau_duration(self, rewards: np.ndarray) -> float:
        """检测平台期持续时间"""
        if len(rewards) < 2 * self.window_size:
            return 0.0
        
        # 计算滑动标准差
        rolling_std = []
        window = min(self.window_size, len(rewards) // 4)
        
        for i in range(window, len(rewards)):
            std_val = np.std(rewards[i-window:i])
            rolling_std.append(std_val)
        
        # 寻找低方差区域（平台期）
        threshold = np.percentile(rolling_std, 20)  # 20百分位数
        plateau_regions = [std < threshold for std in rolling_std]
        
        # 计算连续平台期长度
        max_plateau = 0
        current_plateau = 0
        
        for is_plateau in plateau_regions:
            if is_plateau:
                current_plateau += 1
                max_plateau = max(max_plateau, current_plateau)
            else:
                current_plateau = 0
        
        return max_plateau / len(rewards)  # 归一化为比例
    
    def _calculate_oscillation_amplitude(self, rewards: np.ndarray) -> float:
        """计算振荡幅度"""
        if len(rewards) < 3:
            return 0.0
        
        # 使用滑动窗口计算局部极值
        extrema = []
        for i in range(1, len(rewards) - 1):
            if (rewards[i] > rewards[i-1] and rewards[i] > rewards[i+1]) or \
               (rewards[i] < rewards[i-1] and rewards[i] < rewards[i+1]):
                extrema.append(rewards[i])
        
        if len(extrema) < 2:
            return 0.0
        
        # 计算极值之间的平均差异
        amplitude = np.std(extrema)
        return amplitude / (np.mean(rewards) + 1e-8)  # 归一化
    
    def _calculate_stability_score(self, rewards: np.ndarray) -> float:
        """计算稳定性分数"""
        if len(rewards) < 2:
            return 0.0
        
        # 计算变异系数
        cv = np.std(rewards) / (np.mean(rewards) + 1e-8)
        
        # 计算趋势稳定性（线性回归的R²）
        if len(rewards) >= 3:
            x = np.arange(len(rewards))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
            trend_stability = r_value ** 2
        else:
            trend_stability = 0.0
        
        # 综合稳定性分数
        stability_score = (1 - min(1, cv)) * 0.7 + trend_stability * 0.3
        return max(0, min(1, stability_score))
    
    def _calculate_improvement_efficiency(self, rewards: np.ndarray) -> float:
        """计算改进效率"""
        if len(rewards) < 2:
            return 0.0
        
        # 总体改进幅度
        total_improvement = rewards[-1] - rewards[0]
        
        # 理论最大改进（假设指数增长）
        max_possible_improvement = rewards[0] * 0.5  # 简化假设
        
        if max_possible_improvement > 0:
            efficiency = total_improvement / max_possible_improvement
        else:
            efficiency = 0.0
        
        return max(0, min(1, efficiency))
    
    def _identify_curve_type(self, rewards: np.ndarray) -> str:
        """识别学习曲线类型"""
        if len(rewards) < 5:
            return "insufficient_data"
        
        # 标准化数据
        normalized_rewards = (rewards - rewards[0]) / (rewards[-1] - rewards[0] + 1e-8)
        
        # 尝试拟合不同的函数模型
        models = {
            'exponential': self._fit_exponential,
            'linear': self._fit_linear,
            'sigmoid': self._fit_sigmoid,
            'logarithmic': self._fit_logarithmic
        }
        
        best_fit = None
        best_score = -np.inf
        x = np.arange(len(rewards))
        
        for model_name, fit_func in models.items():
            try:
                score, _ = fit_func(x, rewards)
                if score > best_score:
                    best_score = score
                    best_fit = model_name
            except:
                continue
        
        # 基于模式特征进行分类
        if best_fit is None:
            # 检查是否振荡
            oscillations = self._count_oscillations(normalized_rewards)
            if oscillations > len(normalized_rewards) * 0.3:
                return "oscillating"
            
            # 检查是否平台
            variance = np.var(normalized_rewards)
            if variance < 0.01:
                return "plateau"
            
            return "irregular"
        
        return best_fit
    
    def _fit_exponential(self, x, y):
        """拟合指数函数"""
        try:
            # y = a * exp(b * x) + c
            popt, pcov = optimize.curve_fit(
                lambda t, a, b, c: a * np.exp(b * t) + c,
                x, y, p0=[1, 0.1, y[0]], maxfev=1000
            )
            y_pred = popt[0] * np.exp(popt[1] * x) + popt[2]
            r2 = r2_score(y, y_pred)
            return r2, popt
        except:
            return -np.inf, None
    
    def _fit_linear(self, x, y):
        """拟合线性函数"""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            y_pred = slope * x + intercept
            r2 = r2_score(y, y_pred)
            return r2, {'slope': slope, 'intercept': intercept}
        except:
            return -np.inf, None
    
    def _fit_sigmoid(self, x, y):
        """拟合Sigmoid函数"""
        try:
            # y = a / (1 + exp(-b * (x - c))) + d
            popt, pcov = optimize.curve_fit(
                lambda t, a, b, c, d: a / (1 + np.exp(-b * (t - c))) + d,
                x, y, p0=[y[-1] - y[0], 1, len(x)/2, y[0]], maxfev=1000
            )
            y_pred = popt[0] / (1 + np.exp(-popt[1] * (x - popt[2]))) + popt[3]
            r2 = r2_score(y, y_pred)
            return r2, popt
        except:
            return -np.inf, None
    
    def _fit_logarithmic(self, x, y):
        """拟合对数函数"""
        try:
            # y = a * log(x + b) + c
            popt, pcov = optimize.curve_fit(
                lambda t, a, b, c: a * np.log(t + b) + c,
                x, y, p0=[1, 1, y[0]], maxfev=1000
            )
            y_pred = popt[0] * np.log(x + popt[1]) + popt[2]
            r2 = r2_score(y, y_pred)
            return r2, popt
        except:
            return -np.inf, None
    
    def _count_oscillations(self, rewards: np.ndarray) -> int:
        """计算振荡次数"""
        if len(rewards) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(rewards) - 1):
            if (rewards[i] > rewards[i-1] and rewards[i] > rewards[i+1]) or \
               (rewards[i] < rewards[i-1] and rewards[i] < rewards[i+1]):
                oscillations += 1
        
        return oscillations
    
    def _calculate_fit_quality(self, rewards: np.ndarray, curve_type: str) -> float:
        """计算拟合质量"""
        if curve_type == "insufficient_data":
            return 0.0
        
        x = np.arange(len(rewards))
        
        # 根据曲线类型计算R²
        if curve_type == "linear":
            try:
                slope, intercept, r_value, _, _ = stats.linregress(x, rewards)
                return r_value ** 2
            except:
                return 0.0
        
        # 简化处理：基于数据趋势计算拟合质量
        if len(rewards) >= 3:
            # 计算决定系数
            y_mean = np.mean(rewards)
            ss_res = np.sum((rewards - y_mean) ** 2)
            ss_tot = np.sum((rewards - y_mean) ** 2)
            
            # 简化R²
            variance_explained = 1 - (np.var(rewards) / (np.mean(rewards) ** 2 + 1e-8))
            return max(0, min(1, variance_explained))
        
        return 0.0
    
    def _fit_curve_model(self, rewards: np.ndarray) -> Tuple[str, Dict[str, float], float]:
        """拟合曲线模型"""
        x = np.arange(len(rewards))
        best_type = "linear"
        best_params = {}
        best_r2 = 0.0
        
        # 尝试线性拟合
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, rewards)
            if r_value ** 2 > best_r2:
                best_r2 = r_value ** 2
                best_type = "linear"
                best_params = {"slope": slope, "intercept": intercept}
        except:
            pass
        
        return best_type, best_params, best_r2
    
    def _find_convergence_point(self, rewards: np.ndarray) -> Optional[float]:
        """寻找收敛点"""
        if len(rewards) < 10:
            return None
        
        # 使用滑动窗口检测收敛
        window_size = min(10, len(rewards) // 4)
        
        for i in range(window_size, len(rewards)):
            window_data = rewards[i-window_size:i]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            
            # 如果窗口内的变异系数很小，认为已收敛
            if std_val / (abs(mean_val) + 1e-8) < self.threshold:
                return i / len(rewards)  # 归一化位置
        
        return None
    
    def _estimate_learning_rate(self, rewards: np.ndarray) -> float:
        """估计学习率"""
        if len(rewards) < 2:
            return 0.0
        
        # 使用指数移动平均估计学习率
        alpha = 0.3  # 移动平均参数
        ema = rewards[0]
        learning_rate = 0.0
        
        for reward in rewards[1:]:
            learning_rate += alpha * abs(reward - ema)
            ema = alpha * reward + (1 - alpha) * ema
        
        return learning_rate / len(rewards)
    
    def _find_plateau_point(self, rewards: np.ndarray) -> Optional[float]:
        """寻找平台点"""
        if len(rewards) < 5:
            return None
        
        # 寻找性能不再显著提升的点
        improvements = []
        for i in range(1, len(rewards)):
            improvement = rewards[i] - rewards[i-1]
            improvements.append(improvement)
        
        # 寻找改进接近零的点
        for i, improvement in enumerate(improvements):
            if abs(improvement) < self.threshold * np.std(improvements):
                return (i + 1) / len(rewards)  # 归一化位置
        
        return None
    
    def _calculate_stability_metrics(self, rewards: np.ndarray) -> Dict[str, float]:
        """计算稳定性指标"""
        return {
            'variance': float(np.var(rewards)),
            'coefficient_of_variation': float(np.std(rewards) / (np.mean(rewards) + 1e-8)),
            'trend_stability': self._calculate_trend_stability(rewards),
            'local_stability': self._calculate_local_stability(rewards)
        }
    
    def _calculate_trend_stability(self, rewards: np.ndarray) -> float:
        """计算趋势稳定性"""
        if len(rewards) < 3:
            return 0.0
        
        x = np.arange(len(rewards))
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, rewards)
            return abs(r_value)  # R²值表示趋势的稳定性
        except:
            return 0.0
    
    def _calculate_local_stability(self, rewards: np.ndarray) -> float:
        """计算局部稳定性"""
        if len(rewards) < 3:
            return 0.0
        
        # 计算相邻点之间的变化
        local_changes = [abs(rewards[i] - rewards[i-1]) for i in range(1, len(rewards))]
        
        # 局部稳定性与变化成反比
        avg_change = np.mean(local_changes)
        stability = 1.0 / (1.0 + avg_change)
        
        return stability
    
    def _detect_curve_anomalies(self, rewards: np.ndarray) -> Dict[str, Any]:
        """检测曲线异常"""
        anomalies = {
            'sudden_drops': [],
            'unusual_plateau': False,
            'excessive_oscillation': False,
            'learning_stagnation': False
        }
        
        if len(rewards) < 3:
            return anomalies
        
        # 检测突然下降
        for i in range(1, len(rewards)):
            drop = rewards[i-1] - rewards[i]
            if drop > self.anomaly_thresholds['sudden_drop']:
                anomalies['sudden_drops'].append(i)
        
        # 检测不寻常的平台期
        plateau_length = self._detect_plateau_duration(rewards)
        if plateau_length > self.anomaly_thresholds['plateau_length']:
            anomalies['unusual_plateau'] = True
        
        # 检测过度振荡
        oscillation = self._calculate_oscillation_amplitude(rewards)
        if oscillation > self.anomaly_thresholds['oscillation_amplitude']:
            anomalies['excessive_oscillation'] = True
        
        # 检测学习停滞
        if len(rewards) >= 10:
            recent_improvements = [rewards[i] - rewards[i-1] for i in range(-5, 0)]
            avg_recent_improvement = np.mean([abs(imp) for imp in recent_improvements])
            if avg_recent_improvement < self.anomaly_thresholds['learning_stagnation']:
                anomalies['learning_stagnation'] = True
        
        return anomalies
    
    def _generate_comprehensive_analysis(self, task_count: int) -> Dict[str, Any]:
        """生成综合分析报告"""
        if not self.curve_metrics:
            return {"error": "没有足够的学习曲线数据"}
        
        self.logger.info(f"生成 {task_count} 个任务的综合学习曲线分析报告")
        
        # 基础统计
        learning_speeds = [m.learning_speed for m in self.curve_metrics]
        convergence_rates = [m.convergence_rate for m in self.curve_metrics]
        final_performances = [m.final_performance for m in self.curve_metrics]
        stability_scores = [m.stability_score for m in self.curve_metrics]
        improvement_efficiencies = [m.improvement_efficiency for m in self.curve_metrics]
        
        # 曲线类型分布
        curve_types = [m.learning_curve_type for m in self.curve_metrics]
        curve_type_counts = {}
        for curve_type in curve_types:
            curve_type_counts[curve_type] = curve_type_counts.get(curve_type, 0) + 1
        
        # 学习趋势分析
        learning_trend = self._analyze_learning_trends()
        
        # 任务间比较
        task_comparison = self._compare_task_curves()
        
        # 预测和预警
        predictions = self._generate_predictions()
        
        # 生成可视化
        self._create_curve_visualizations(task_count)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tasks_analyzed': task_count,
            'summary_statistics': {
                'learning_speed': {
                    'mean': np.mean(learning_speeds),
                    'std': np.std(learning_speeds),
                    'min': np.min(learning_speeds),
                    'max': np.max(learning_speeds)
                },
                'convergence_rate': {
                    'mean': np.mean(convergence_rates),
                    'std': np.std(convergence_rates)
                },
                'final_performance': {
                    'mean': np.mean(final_performances),
                    'std': np.std(final_performances),
                    'improvement_trend': self._calculate_performance_trend(final_performances)
                },
                'stability': {
                    'mean': np.mean(stability_scores),
                    'consistency': 1.0 - np.std(stability_scores)
                },
                'efficiency': {
                    'mean': np.mean(improvement_efficiencies),
                    'distribution': improvement_efficiencies
                }
            },
            'curve_type_distribution': curve_type_counts,
            'learning_trends': learning_trends,
            'task_comparison': task_comparison,
            'predictions': predictions,
            'recommendations': self._generate_learning_recommendations()
        }
        
        return report
    
    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """分析学习趋势"""
        if len(self.curve_metrics) < 2:
            return {}
        
        # 分析学习速度趋势
        task_ids = [m.task_id for m in self.curve_metrics]
        learning_speeds = [m.learning_speed for m in self.curve_metrics]
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(task_ids, learning_speeds)
            trend_analysis = {
                'speed_trend_slope': slope,
                'speed_trend_r2': r_value ** 2,
                'speed_trend_significance': p_value < 0.05
            }
        except:
            trend_analysis = {'error': '无法计算学习速度趋势'}
        
        return trend_analysis
    
    def _compare_task_curves(self) -> Dict[str, Any]:
        """比较任务间的学习曲线"""
        if len(self.curve_metrics) < 2:
            return {}
        
        # 找到最佳和最差的任务
        best_task = max(self.curve_metrics, key=lambda m: m.final_performance)
        worst_task = min(self.curve_metrics, key=lambda m: m.final_performance)
        most_stable = max(self.curve_metrics, key=lambda m: m.stability_score)
        fastest_learning = max(self.curve_metrics, key=lambda m: m.learning_speed)
        
        return {
            'best_performing_task': best_task.task_id,
            'worst_performing_task': worst_task.task_id,
            'most_stable_task': most_stable.task_id,
            'fastest_learning_task': fastest_learning.task_id,
            'performance_gap': best_task.final_performance - worst_task.final_performance,
            'stability_variance': np.std([m.stability_score for m in self.curve_metrics])
        }
    
    def _calculate_performance_trend(self, performances: List[float]) -> float:
        """计算性能趋势"""
        if len(performances) < 2:
            return 0.0
        
        # 简单线性趋势
        x = np.arange(len(performances))
        slope, _, _, _, _ = stats.linregress(x, performances)
        
        return slope
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """生成预测"""
        if not self.curve_metrics:
            return {}
        
        predictions = {}
        
        # 基于当前趋势预测下一个任务的性能
        if len(self.curve_metrics) >= 3:
            recent_performances = [m.final_performance for m in self.curve_metrics[-3:]]
            recent_tasks = list(range(len(self.curve_metrics) - 3, len(self.curve_metrics)))
            
            try:
                slope, intercept, _, _, _ = stats.linregress(recent_tasks, recent_performances)
                next_task_prediction = slope * len(self.curve_metrics) + intercept
                predictions['next_task_performance'] = max(0, next_task_prediction)
            except:
                predictions['next_task_performance'] = np.mean(recent_performances)
        
        # 学习稳定性预测
        recent_stabilities = [m.stability_score for m in self.curve_metrics[-5:]]
        predictions['expected_stability'] = np.mean(recent_stabilities)
        
        return predictions
    
    def _generate_learning_recommendations(self) -> List[str]:
        """生成学习建议"""
        recommendations = []
        
        if not self.curve_metrics:
            return ["需要更多数据进行分析"]
        
        # 基于学习速度的建议
        avg_learning_speed = np.mean([m.learning_speed for m in self.curve_metrics])
        if avg_learning_speed < 0:
            recommendations.append("学习速度偏慢，建议调整学习率或优化算法")
        elif avg_learning_speed > 0.1:
            recommendations.append("学习速度较快但可能不稳定，建议平衡探索与利用")
        
        # 基于收敛性的建议
        avg_convergence_rate = np.mean([m.convergence_rate for m in self.curve_metrics])
        if avg_convergence_rate < 0.5:
            recommendations.append("收敛性较差，建议延长训练时间或调整收敛条件")
        
        # 基于稳定性的建议
        avg_stability = np.mean([m.stability_score for m in self.curve_metrics])
        if avg_stability < 0.6:
            recommendations.append("学习不够稳定，建议增加正则化或改进探索策略")
        
        # 基于曲线类型的建议
        curve_types = [m.learning_curve_type for m in self.curve_metrics]
        if "oscillating" in curve_types:
            recommendations.append("检测到振荡模式，建议增加探索控制或学习率衰减")
        
        if "plateau" in curve_types:
            recommendations.append("检测到平台期，建议调整奖励函数或增加探索多样性")
        
        return recommendations
    
    def _create_curve_visualizations(self, task_count: int):
        """创建学习曲线可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'学习曲线分析报告 (共 {task_count} 个任务)', fontsize=16)
        
        # 1. 所有学习曲线
        for task_id in self.learning_curves:
            rewards = self.learning_curves[task_id]
            axes[0, 0].plot(rewards, alpha=0.7, label=f'任务 {task_id}')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].set_title('学习曲线总览')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习速度分布
        learning_speeds = [m.learning_speed for m in self.curve_metrics]
        if learning_speeds:
            axes[0, 1].hist(learning_speeds, bins=10, alpha=0.7, color='skyblue')
            axes[0, 1].set_xlabel('学习速度')
            axes[0, 1].set_ylabel('频率')
            axes[0, 1].set_title('学习速度分布')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最终性能趋势
        task_ids = [m.task_id for m in self.curve_metrics]
        final_performances = [m.final_performance for m in self.curve_metrics]
        if final_performances:
            axes[0, 2].plot(task_ids, final_performances, 'b-', linewidth=2, marker='o')
            axes[0, 2].set_xlabel('任务ID')
            axes[0, 2].set_ylabel('最终性能')
            axes[0, 2].set_title('最终性能趋势')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 稳定性分布
        stability_scores = [m.stability_score for m in self.curve_metrics]
        if stability_scores:
            axes[1, 0].boxplot(stability_scores)
            axes[1, 0].set_ylabel('稳定性分数')
            axes[1, 0].set_title('稳定性分布')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 曲线类型分布
        curve_types = [m.learning_curve_type for m in self.curve_metrics]
        if curve_types:
            type_counts = {}
            for ct in curve_types:
                type_counts[ct] = type_counts.get(ct, 0) + 1
            
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            axes[1, 1].pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('曲线类型分布')
        
        # 6. 学习效率对比
        improvement_efficiencies = [m.improvement_efficiency for m in self.curve_metrics]
        if improvement_efficiencies:
            axes[1, 2].scatter(task_ids, improvement_efficiencies, alpha=0.7)
            axes[1, 2].set_xlabel('任务ID')
            axes[1, 2].set_ylabel('改进效率')
            axes[1, 2].set_title('学习效率分布')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'learning_curve_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self, filepath: str):
        """保存分析结果到文件
        
        Args:
            filepath: 文件路径
        """
        results = {
            'learning_curves': self.learning_curves,
            'curve_metrics': [
                {
                    'task_id': m.task_id,
                    'learning_speed': m.learning_speed,
                    'convergence_rate': m.convergence_rate,
                    'final_performance': m.final_performance,
                    'plateau_duration': m.plateau_duration,
                    'oscillation_amplitude': m.oscillation_amplitude,
                    'stability_score': m.stability_score,
                    'improvement_efficiency': m.improvement_efficiency,
                    'learning_curve_type': m.learning_curve_type,
                    'fit_quality': m.fit_quality,
                    'timestamp': m.timestamp
                }
                for m in self.curve_metrics
            ],
            'curve_analyses': {
                k: {
                    'curve_type': v.curve_type,
                    'parameters': v.parameters,
                    'goodness_of_fit': v.goodness_of_fit,
                    'convergence_point': v.convergence_point,
                    'learning_rate_estimate': v.learning_rate_estimate,
                    'plateau_point': v.plateau_point,
                    'stability_metrics': v.stability_metrics,
                    'anomaly_detection': v.anomaly_detection
                }
                for k, v in self.curve_analyses.items()
            },
            'statistics': self.curve_statistics,
            'comparison': self.curve_comparison
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"学习曲线分析结果已保存到: {filepath}")


def main():
    """主函数 - 演示学习曲线分析器"""
    # 创建分析器
    analyzer = LearningCurveAnalyzer()
    
    # 模拟性能历史数据
    performance_history = {}
    
    # 生成不同类型的学习曲线
    np.random.seed(42)  # 确保可重复性
    
    for task_id in range(10):
        # 模拟不同类型的学习曲线
        if task_id % 4 == 0:  # 指数型
            base = 0.3
            rewards = [base * (1 - np.exp(-0.1 * i)) + np.random.normal(0, 0.05) for i in range(100)]
        elif task_id % 4 == 1:  # 线性型
            rewards = [0.3 + 0.005 * i + np.random.normal(0, 0.03) for i in range(100)]
        elif task_id % 4 == 2:  # 振荡型
            rewards = [0.5 + 0.3 * np.sin(0.1 * i) + np.random.normal(0, 0.1) for i in range(100)]
        else:  # 平台型
            rewards = [0.7 + np.random.normal(0, 0.02) if i > 30 else 0.4 + i * 0.01 + np.random.normal(0, 0.03) 
                      for i in range(100)]
        
        performance_history[task_id] = rewards[:50]  # 限制长度用于演示
    
    # 分析学习曲线
    report = analyzer.analyze_learning_curves(performance_history, 10)
    
    print("学习曲线分析报告:")
    print("=" * 40)
    
    if 'summary_statistics' in report:
        summary = report['summary_statistics']
        print(f"学习速度 - 平均: {summary['learning_speed']['mean']:.4f}, 标准差: {summary['learning_speed']['std']:.4f}")
        print(f"收敛率 - 平均: {summary['convergence_rate']['mean']:.4f}")
        print(f"最终性能 - 平均: {summary['final_performance']['mean']:.4f}, 趋势: {summary['final_performance']['improvement_trend']:.4f}")
        print(f"稳定性 - 平均: {summary['stability']['mean']:.4f}")
        print(f"效率 - 平均: {summary['efficiency']['mean']:.4f}")
    
    if 'curve_type_distribution' in report:
        print(f"\n曲线类型分布:")
        for curve_type, count in report['curve_type_distribution'].items():
            print(f"  {curve_type}: {count} 个任务")
    
    # 保存结果
    analyzer.save_analysis_results("demo_learning_curve_analysis.json")
    print("\n分析结果已保存")


if __name__ == "__main__":
    main()