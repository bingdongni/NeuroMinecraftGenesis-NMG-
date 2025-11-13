#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析器
负责策略迁移系统的性能监控、分析和优化建议
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
from scipy import stats
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    """
    性能分析器类
    
    功能：
    1. 实时监控迁移系统性能指标
    2. 分析性能趋势和模式
    3. 检测性能异常和问题
    4. 生成性能优化建议
    5. 预测系统性能走向
    
    分析维度：
    - 时间序列分析：性能随时间的变化
    - 统计分布分析：性能数据的统计特征
    - 相关性分析：不同指标间的关系
    - 异常检测：识别性能异常点
    - 趋势预测：预测未来性能表现
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化性能分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger('PerformanceAnalyzer')
        
        # 分析配置
        self.window_size = self.config.get('window_size', 100)
        self.analysis_frequency = self.config.get('analysis_frequency', 10)
        self.alert_threshold = self.config.get('alert_threshold', 0.8)
        
        # 性能数据存储
        self.performance_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.session_performance = defaultdict(list)
        self.metric_aggregates = defaultdict(lambda: defaultdict(list))
        
        # 分析模型
        self.trend_analyzer = self._init_trend_analyzer()
        self.anomaly_detector = self._init_anomaly_detector()
        self.correlation_analyzer = self._init_correlation_analyzer()
        self.forecaster = self._init_forecaster()
        
        # 分析缓存
        self.analysis_cache = {}
        self.last_analysis_time = {}
        
        self.logger.info("性能分析器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'window_size': 100,
            'analysis_frequency': 10,
            'alert_threshold': 0.8,
            'analysis_methods': ['trend', 'anomaly', 'correlation', 'forecast'],
            'performance_metrics': [
                'accuracy', 'success_rate', 'execution_time', 'stability', 'adaptability'
            ],
            'anomaly_detection': {
                'method': 'statistical',
                'threshold_multiplier': 2.0,
                'min_data_points': 10
            },
            'trend_analysis': {
                'linear_regression': True,
                'polynomial_fitting': False,
                'moving_average_window': 5
            },
            'forecasting': {
                'method': 'linear_regression',
                'horizon': 10,
                'confidence_intervals': True
            }
        }
    
    def _init_trend_analyzer(self) -> Dict[str, Any]:
        """初始化趋势分析器"""
        return {
            'trend_methods': ['linear', 'exponential', 'polynomial'],
            'trend_indicators': ['slope', 'direction', 'strength', 'significance'],
            'moving_average': {
                'short_window': 5,
                'long_window': 20
            }
        }
    
    def _init_anomaly_detector(self) -> Dict[str, Any]:
        """初始化异常检测器"""
        return {
            'detection_methods': ['z_score', 'iqr', 'isolation_forest', 'statistical'],
            'sensitivity_levels': {
                'low': 3.0,
                'medium': 2.0,
                'high': 1.5
            },
            'anomaly_types': ['point', 'contextual', 'collective']
        }
    
    def _init_correlation_analyzer(self) -> Dict[str, Any]:
        """初始化相关性分析器"""
        return {
            'correlation_methods': ['pearson', 'spearman', 'kendall'],
            'significance_threshold': 0.05,
            'lag_analysis': True,
            'partial_correlation': False
        }
    
    def _init_forecaster(self) -> Dict[str, Any]:
        """初始化预测器"""
        return {
            'forecasting_methods': ['linear', 'arima', 'exponential_smoothing'],
            'prediction_horizon': 10,
            'uncertainty_estimation': True,
            'model_selection': 'aic'
        }
    
    def analyze_performance(self, transfer_history: List[Dict[str, Any]], 
                          evaluation_report: Dict[str, Any],
                          current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        执行性能分析
        
        Args:
            transfer_history: 迁移历史记录
            evaluation_report: 评估报告
            current_metrics: 当前性能指标
            
        Returns:
            Dict: 分析结果，包含性能趋势、异常检测、优化建议等
        """
        try:
            self.logger.info("开始性能分析")
            
            # 更新性能数据
            self._update_performance_data(transfer_history, evaluation_report, current_metrics)
            
            # 执行多维度分析
            trend_analysis = self._perform_trend_analysis()
            anomaly_detection = self._perform_anomaly_detection()
            correlation_analysis = self._perform_correlation_analysis()
            performance_forecast = self._perform_performance_forecast()
            
            # 生成优化建议
            optimization_recommendations = self._generate_optimization_recommendations(
                trend_analysis, anomaly_detection, correlation_analysis
            )
            
            # 构建完整分析结果
            analysis_result = {
                'analysis_id': f"analysis_{datetime.now().timestamp()}",
                'analysis_timestamp': datetime.now().isoformat(),
                'trend_analysis': trend_analysis,
                'anomaly_detection': anomaly_detection,
                'correlation_analysis': correlation_analysis,
                'performance_forecast': performance_forecast,
                'optimization_recommendations': optimization_recommendations,
                'overall_performance_assessment': self._assess_overall_performance(),
                'optimization_confidence': self._calculate_analysis_confidence(),
                'analysis_metadata': {
                    'data_points_analyzed': len(current_metrics),
                    'analysis_methods_used': ['trend', 'anomaly', 'correlation', 'forecast'],
                    'time_span_analyzed': self._calculate_analysis_timespan()
                }
            }
            
            # 缓存分析结果
            self.analysis_cache['latest_analysis'] = analysis_result
            self.last_analysis_time = datetime.now()
            
            self.logger.info("性能分析完成")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"性能分析失败: {str(e)}")
            raise
    
    def _update_performance_data(self, transfer_history: List[Dict[str, Any]], 
                               evaluation_report: Dict[str, Any],
                               current_metrics: Dict[str, float]):
        """更新性能数据"""
        # 更新当前指标
        timestamp = datetime.now()
        
        for metric_name, metric_value in current_metrics.items():
            self.performance_history[metric_name].append({
                'timestamp': timestamp,
                'value': metric_value,
                'session_id': evaluation_report.get('session_id', 'unknown')
            })
        
        # 更新会话性能
        session_id = evaluation_report.get('session_id', 'unknown')
        self.session_performance[session_id].append({
            'timestamp': timestamp,
            'metrics': current_metrics,
            'overall_score': evaluation_report.get('overall_score', 0.0)
        })
        
        # 更新指标聚合数据
        for metric_name, metric_value in current_metrics.items():
            self.metric_aggregates[metric_name]['values'].append(metric_value)
            self.metric_aggregates[metric_name]['timestamps'].append(timestamp)
    
    def _perform_trend_analysis(self) -> Dict[str, Any]:
        """执行趋势分析"""
        trend_results = {}
        
        for metric_name in self.config.get('performance_metrics', []):
            if metric_name in self.performance_history:
                metric_data = list(self.performance_history[metric_name])
                
                if len(metric_data) < 3:
                    continue
                
                # 提取时间序列
                values = [data['value'] for data in metric_data]
                timestamps = [data['timestamp'] for data in metric_data]
                
                # 转换为数值时间序列
                time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
                
                # 线性回归分析
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
                
                # 计算移动平均
                moving_averages = self._calculate_moving_averages(values)
                
                # 趋势强度和方向
                trend_strength = abs(r_value)
                trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                trend_significance = p_value < 0.05
                
                # 趋势评估
                trend_assessment = self._assess_trend_quality(trend_strength, trend_significance, slope)
                
                trend_results[metric_name] = {
                    'linear_trend': {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'standard_error': std_err
                    },
                    'trend_indicators': {
                        'direction': trend_direction,
                        'strength': trend_strength,
                        'significance': trend_significance,
                        'quality': trend_assessment
                    },
                    'moving_averages': {
                        'short_ma': moving_averages['short'],
                        'long_ma': moving_averages['long']
                    },
                    'trend_prediction': self._predict_trend_continuation(slope, trend_strength),
                    'data_points': len(values),
                    'time_span': (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0
                }
        
        return trend_results
    
    def _calculate_moving_averages(self, values: List[float]) -> Dict[str, List[float]]:
        """计算移动平均"""
        short_window = self.trend_analyzer['moving_average']['short_window']
        long_window = self.trend_analyzer['moving_average']['long_window']
        
        short_ma = []
        long_ma = []
        
        for i in range(len(values)):
            # 短期移动平均
            if i >= short_window - 1:
                short_avg = np.mean(values[i - short_window + 1:i + 1])
                short_ma.append(short_avg)
            else:
                short_ma.append(np.mean(values[:i + 1]))
            
            # 长期移动平均
            if i >= long_window - 1:
                long_avg = np.mean(values[i - long_window + 1:i + 1])
                long_ma.append(long_avg)
            else:
                long_ma.append(np.mean(values[:i + 1]))
        
        return {'short': short_ma, 'long': long_ma}
    
    def _assess_trend_quality(self, strength: float, significance: bool, slope: float) -> str:
        """评估趋势质量"""
        if significance and strength > 0.7:
            return 'strong'
        elif significance and strength > 0.4:
            return 'moderate'
        elif strength > 0.2:
            return 'weak'
        else:
            return 'negligible'
    
    def _predict_trend_continuation(self, slope: float, strength: float) -> Dict[str, Any]:
        """预测趋势延续"""
        # 基于趋势强度和斜率预测未来走向
        if strength < 0.3:
            prediction = 'stable'
            confidence = 0.5
        elif slope > 0 and strength > 0.6:
            prediction = 'continuing_improvement'
            confidence = min(0.9, strength)
        elif slope < 0 and strength > 0.6:
            prediction = 'continuing_decline'
            confidence = min(0.9, strength)
        else:
            prediction = 'uncertain'
            confidence = 0.3
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': 'low' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'high'
        }
    
    def _perform_anomaly_detection(self) -> Dict[str, Any]:
        """执行异常检测"""
        anomaly_results = {}
        
        for metric_name in self.config.get('performance_metrics', []):
            if metric_name in self.performance_history:
                metric_data = list(self.performance_history[metric_name])
                
                if len(metric_data) < self.anomaly_detector['anomaly_detection']['min_data_points']:
                    continue
                
                values = [data['value'] for data in metric_data]
                timestamps = [data['timestamp'] for data in metric_data]
                
                # Z-score异常检测
                z_scores = np.abs(stats.zscore(values))
                z_threshold = self.anomaly_detector['sensitivity_levels']['medium']
                z_anomalies = np.where(z_scores > z_threshold)[0]
                
                # IQR异常检测
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                iqr_threshold_low = q1 - 1.5 * iqr
                iqr_threshold_high = q3 + 1.5 * iqr
                
                iqr_anomalies = []
                for i, value in enumerate(values):
                    if value < iqr_threshold_low or value > iqr_threshold_high:
                        iqr_anomalies.append(i)
                
                # 统计异常检测
                statistical_anomalies = self._detect_statistical_anomalies(values)
                
                anomaly_results[metric_name] = {
                    'z_score_detection': {
                        'anomalous_indices': z_anomalies.tolist(),
                        'anomaly_count': len(z_anomalies),
                        'threshold': z_threshold,
                        'max_z_score': np.max(z_scores)
                    },
                    'iqr_detection': {
                        'anomalous_indices': iqr_anomalies,
                        'anomaly_count': len(iqr_anomalies),
                        'iqr_range': [iqr_threshold_low, iqr_threshold_high]
                    },
                    'statistical_detection': statistical_anomalies,
                    'anomaly_summary': {
                        'total_anomalies': len(set(z_anomalies) | set(iqr_anomalies)),
                        'anomaly_rate': len(set(z_anomalies) | set(iqr_anomalies)) / len(values),
                        'severity_distribution': self._calculate_anomaly_severity(values, z_anomalies)
                    },
                    'last_anomaly_time': timestamps[max(z_anomalies)] if len(z_anomalies) > 0 else None
                }
        
        return anomaly_results
    
    def _detect_statistical_anomalies(self, values: List[float]) -> Dict[str, Any]:
        """检测统计异常"""
        if len(values) < 10:
            return {'detection_available': False, 'reason': 'insufficient_data'}
        
        # 简单的分位数分析
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        
        # 检测极端值
        extreme_low = np.percentile(values, 5)
        extreme_high = np.percentile(values, 95)
        
        # 检测分布偏斜
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        
        return {
            'quartile_analysis': {
                'q1': q1,
                'median': median,
                'q3': q3,
                'iqr': q3 - q1
            },
            'extreme_values': {
                'low_extreme': extreme_low,
                'high_extreme': extreme_high
            },
            'distribution_properties': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': abs(skewness) < 0.5 and abs(kurtosis) < 3
            }
        }
    
    def _calculate_anomaly_severity(self, values: List[float], anomaly_indices: List[int]) -> Dict[str, int]:
        """计算异常严重程度"""
        if not anomaly_indices:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for idx in anomaly_indices:
            value = values[idx]
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            
            if z_score > 3.0:
                severity_counts['high'] += 1
            elif z_score > 2.0:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        return severity_counts
    
    def _perform_correlation_analysis(self) -> Dict[str, Any]:
        """执行相关性分析"""
        correlation_results = {}
        metrics = list(self.performance_history.keys())
        
        if len(metrics) < 2:
            return {'insufficient_metrics': True}
        
        # 计算两两相关性
        correlation_matrix = {}
        for i, metric1 in enumerate(metrics):
            correlation_matrix[metric1] = {}
            for j, metric2 in enumerate(metrics):
                if i == j:
                    correlation_matrix[metric1][metric2] = 1.0
                else:
                    # 获取共同时间点的数据
                    data1, data2 = self._align_metric_data(metric1, metric2)
                    
                    if len(data1) > 2 and len(data2) > 2:
                        # 计算皮尔逊相关系数
                        correlation, p_value = stats.pearsonr(data1, data2)
                        correlation_matrix[metric1][metric2] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'is_significant': p_value < 0.05,
                            'strength': self._interpret_correlation_strength(abs(correlation))
                        }
                    else:
                        correlation_matrix[metric1][metric2] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'is_significant': False,
                            'strength': 'none'
                        }
        
        # 识别强相关性
        strong_correlations = self._identify_strong_correlations(correlation_matrix)
        
        # 相关性聚类分析
        correlation_clusters = self._analyze_correlation_clusters(correlation_matrix)
        
        correlation_results = {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations,
            'correlation_clusters': correlation_clusters,
            'analysis_summary': {
                'total_pairs': len(metrics) * (len(metrics) - 1) // 2,
                'significant_pairs': sum(
                    1 for row in correlation_matrix.values()
                    for col in row.values()
                    if isinstance(col, dict) and col.get('is_significant', False)
                ),
                'strong_pairs': sum(
                    1 for row in correlation_matrix.values()
                    for col in row.values()
                    if isinstance(col, dict) and col.get('strength') in ['strong', 'very_strong']
                )
            }
        }
        
        return correlation_results
    
    def _align_metric_data(self, metric1: str, metric2: str) -> Tuple[List[float], List[float]]:
        """对齐两个指标的数据"""
        data1 = list(self.performance_history[metric1])
        data2 = list(self.performance_history[metric2])
        
        # 找到时间重叠的数据点
        aligned_data1 = []
        aligned_data2 = []
        
        for d1 in data1:
            # 寻找时间最接近的d2数据点
            closest_d2 = None
            min_time_diff = float('inf')
            
            for d2 in data2:
                time_diff = abs((d1['timestamp'] - d2['timestamp']).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_d2 = d2
            
            # 如果时间差小于5秒，认为是同一时间点的数据
            if closest_d2 and min_time_diff < 5.0:
                aligned_data1.append(d1['value'])
                aligned_data2.append(closest_d2['value'])
        
        return aligned_data1, aligned_data2
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """解释相关性强度"""
        if correlation >= 0.8:
            return 'very_strong'
        elif correlation >= 0.6:
            return 'strong'
        elif correlation >= 0.4:
            return 'moderate'
        elif correlation >= 0.2:
            return 'weak'
        else:
            return 'none'
    
    def _identify_strong_correlations(self, correlation_matrix: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """识别强相关性"""
        strong_correlations = []
        
        for metric1, row in correlation_matrix.items():
            for metric2, correlation_info in row.items():
                if (metric1 != metric2 and isinstance(correlation_info, dict) and
                    correlation_info.get('strength') in ['strong', 'very_strong']):
                    strong_correlations.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation_info['correlation'],
                        'strength': correlation_info['strength'],
                        'significance': correlation_info['is_significant']
                    })
        
        # 按相关性强度排序
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return strong_correlations
    
    def _analyze_correlation_clusters(self, correlation_matrix: Dict[str, Dict]) -> Dict[str, Any]:
        """分析相关性聚类"""
        metrics = list(correlation_matrix.keys())
        
        # 简单的聚类分析：基于相关性强度
        clusters = []
        processed_metrics = set()
        
        for metric in metrics:
            if metric in processed_metrics:
                continue
            
            # 找到与当前指标强相关的其他指标
            cluster = [metric]
            processed_metrics.add(metric)
            
            for other_metric in metrics:
                if other_metric != metric and other_metric not in processed_metrics:
                    correlation_info = correlation_matrix[metric][other_metric]
                    if (isinstance(correlation_info, dict) and 
                        correlation_info.get('strength') in ['strong', 'very_strong']):
                        cluster.append(other_metric)
                        processed_metrics.add(other_metric)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return {
            'clusters': clusters,
            'cluster_count': len(clusters),
            'unclustered_metrics': [m for m in metrics if m not in processed_metrics]
        }
    
    def _perform_performance_forecast(self) -> Dict[str, Any]:
        """执行性能预测"""
        forecast_results = {}
        
        for metric_name in self.config.get('performance_metrics', []):
            if metric_name in self.performance_history:
                metric_data = list(self.performance_history[metric_name])
                
                if len(metric_data) < 5:
                    continue
                
                values = [data['value'] for data in metric_data]
                timestamps = [data['timestamp'] for data in metric_data]
                
                # 转换为数值时间序列
                time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
                
                # 线性回归预测
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
                
                # 预测未来值
                horizon = self.forecaster['prediction_horizon']
                future_times = [time_numeric[-1] + i * 60 for i in range(1, horizon + 1)]  # 每分钟一个点
                future_predictions = [slope * t + intercept for t in future_times]
                
                # 计算预测不确定性
                prediction_std = std_err * np.sqrt(1 + 1/len(values))  # 预测区间标准误差
                confidence_intervals = []
                for pred in future_predictions:
                    ci_lower = pred - 1.96 * prediction_std
                    ci_upper = pred + 1.96 * prediction_std
                    confidence_intervals.append((ci_lower, ci_upper))
                
                # 预测质量评估
                forecast_quality = self._assess_forecast_quality(r_value, len(values), p_value)
                
                forecast_results[metric_name] = {
                    'forecast_method': 'linear_regression',
                    'model_quality': forecast_quality,
                    'predictions': {
                        'future_times': future_times,
                        'predicted_values': future_predictions,
                        'confidence_intervals': confidence_intervals
                    },
                    'model_parameters': {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value
                    },
                    'forecast_horizon': horizon,
                    'uncertainty_level': 'high' if r_value < 0.5 else 'medium' if r_value < 0.7 else 'low'
                }
        
        return forecast_results
    
    def _assess_forecast_quality(self, r_value: float, data_points: int, p_value: float) -> Dict[str, Any]:
        """评估预测质量"""
        if data_points < 10:
            return {'quality': 'poor', 'reason': 'insufficient_data'}
        
        r_squared = r_value**2
        
        if r_squared > 0.7 and p_value < 0.05:
            quality = 'excellent'
        elif r_squared > 0.5 and p_value < 0.1:
            quality = 'good'
        elif r_squared > 0.3:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'r_squared': r_squared,
            'p_value': p_value,
            'data_adequacy': 'sufficient' if data_points >= 10 else 'insufficient'
        }
    
    def _generate_optimization_recommendations(self, trend_analysis: Dict[str, Any],
                                             anomaly_detection: Dict[str, Any],
                                             correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化建议"""
        recommendations = {
            'parameter_recommendations': [],
            'strategy_improvements': [],
            'configuration_updates': [],
            'learning_optimizations': [],
            'expected_improvements': []
        }
        
        # 基于趋势分析生成建议
        for metric_name, trend_info in trend_analysis.items():
            trend_direction = trend_info['trend_indicators']['direction']
            trend_strength = trend_info['trend_indicators']['strength']
            
            if trend_direction == 'decreasing' and trend_strength > 0.6:
                recommendations['parameter_recommendations'].append({
                    'type': 'performance_recovery',
                    'metric': metric_name,
                    'recommendation': f'调整{metric_name}相关参数以阻止性能下降',
                    'urgency': 'high',
                    'expected_impact': 'medium'
                })
        
        # 基于异常检测生成建议
        for metric_name, anomaly_info in anomaly_detection.items():
            anomaly_count = anomaly_info['anomaly_summary']['anomaly_count']
            anomaly_rate = anomaly_info['anomaly_summary']['anomaly_rate']
            
            if anomaly_rate > 0.1:  # 异常率超过10%
                recommendations['strategy_improvements'].append({
                    'type': 'anomaly_reduction',
                    'metric': metric_name,
                    'recommendation': f'提高{metric_name}的稳定性，减少异常波动',
                    'urgency': 'medium',
                    'expected_impact': 'high'
                })
        
        # 基于相关性分析生成建议
        strong_correlations = correlation_analysis.get('strong_correlations', [])
        for corr in strong_correlations:
            if corr['significance']:
                recommendations['learning_optimizations'].append({
                    'type': 'correlated_optimization',
                    'metrics': [corr['metric1'], corr['metric2']],
                    'recommendation': f'同时优化{corr["metric1"]}和{corr["metric2"]}以获得协同改进',
                    'urgency': 'low',
                    'expected_impact': 'medium'
                })
        
        # 通用优化建议
        recommendations['configuration_updates'].append({
            'type': 'general_optimization',
            'recommendation': '根据当前性能分析结果微调系统配置参数',
            'urgency': 'low',
            'expected_impact': 'low'
        })
        
        # 计算预期改进效果
        total_recommendations = (
            len(recommendations['parameter_recommendations']) +
            len(recommendations['strategy_improvements']) +
            len(recommendations['configuration_updates']) +
            len(recommendations['learning_optimizations'])
        )
        
        recommendations['expected_improvements'] = {
            'total_recommendations': total_recommendations,
            'high_priority_count': sum(
                1 for rec_type in recommendations.values()
                if isinstance(rec_type, list)
                for rec in rec_type
                if rec.get('urgency') == 'high'
            ),
            'estimated_overall_improvement': min(0.3, total_recommendations * 0.05),  # 最多30%改进
            'implementation_timeline': '1-2 weeks' if total_recommendations > 5 else 'few days'
        }
        
        return recommendations
    
    def _assess_overall_performance(self) -> Dict[str, Any]:
        """评估整体性能"""
        overall_score = 0.0
        metric_count = 0
        performance_details = {}
        
        for metric_name in self.config.get('performance_metrics', []):
            if metric_name in self.performance_history:
                recent_data = list(self.performance_history[metric_name])[-10:]  # 最近10个数据点
                
                if recent_data:
                    values = [data['value'] for data in recent_data]
                    avg_performance = np.mean(values)
                    stability = 1.0 - (np.std(values) / (avg_performance + 1e-8))  # 稳定性指标
                    
                    performance_details[metric_name] = {
                        'average': avg_performance,
                        'stability': max(0, stability),
                        'trend': self._calculate_recent_trend(values),
                        'data_points': len(values)
                    }
                    
                    overall_score += avg_performance * stability
                    metric_count += 1
        
        if metric_count > 0:
            overall_score /= metric_count
        
        # 性能等级评定
        if overall_score >= 0.9:
            performance_grade = 'excellent'
        elif overall_score >= 0.8:
            performance_grade = 'good'
        elif overall_score >= 0.7:
            performance_grade = 'satisfactory'
        elif overall_score >= 0.6:
            performance_grade = 'needs_improvement'
        else:
            performance_grade = 'poor'
        
        return {
            'overall_score': overall_score,
            'performance_grade': performance_grade,
            'metric_details': performance_details,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_recent_trend(self, values: List[float]) -> str:
        """计算最近趋势"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # 比较前半段和后半段的平均值
        mid_point = len(values) // 2
        early_avg = np.mean(values[:mid_point])
        late_avg = np.mean(values[mid_point:])
        
        change_percent = (late_avg - early_avg) / (early_avg + 1e-8) * 100
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_analysis_confidence(self) -> float:
        """计算分析置信度"""
        # 基于数据充足性、分析方法适用性等计算置信度
        total_metrics = len(self.config.get('performance_metrics', []))
        analyzed_metrics = len([m for m in self.performance_history if m in 
                              self.config.get('performance_metrics', [])])
        
        data_adequacy = analyzed_metrics / total_metrics if total_metrics > 0 else 0
        
        # 基于缓存新鲜度
        if self.last_analysis_time:
            time_since_last = (datetime.now() - self.last_analysis_time).total_seconds()
            freshness_factor = max(0.5, 1.0 - time_since_last / 3600)  # 1小时内置信度较高
        else:
            freshness_factor = 0.5
        
        confidence = 0.7 * data_adequacy + 0.3 * freshness_factor
        return min(1.0, max(0.0, confidence))
    
    def _calculate_analysis_timespan(self) -> float:
        """计算分析时间跨度"""
        all_timestamps = []
        
        for metric_history in self.performance_history.values():
            for data_point in metric_history:
                all_timestamps.append(data_point['timestamp'])
        
        if len(all_timestamps) < 2:
            return 0.0
        
        return (max(all_timestamps) - min(all_timestamps)).total_seconds()
    
    def generate_performance_report(self, session_id: str = None) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            report_timestamp = datetime.now().isoformat()
            
            # 获取最新的分析结果
            latest_analysis = self.analysis_cache.get('latest_analysis')
            if not latest_analysis:
                return {'status': 'no_analysis_available', 'message': '需要先执行性能分析'}
            
            # 生成报告
            performance_report = {
                'report_id': f"report_{datetime.now().timestamp()}",
                'report_timestamp': report_timestamp,
                'session_id': session_id,
                'executive_summary': self._generate_executive_summary(latest_analysis),
                'detailed_analysis': latest_analysis,
                'key_findings': self._extract_key_findings(latest_analysis),
                'action_items': self._extract_action_items(latest_analysis['optimization_recommendations']),
                'next_steps': self._generate_next_steps(latest_analysis)
            }
            
            self.logger.info(f"性能报告生成完成，报告ID: {performance_report['report_id']}")
            
            return performance_report
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {str(e)}")
            raise
    
    def _generate_executive_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        overall_assessment = analysis_result.get('overall_performance_assessment', {})
        
        summary = {
            'overall_score': overall_assessment.get('overall_score', 0.0),
            'performance_grade': overall_assessment.get('performance_grade', 'unknown'),
            'key_metrics_status': self._summarize_metric_status(analysis_result),
            'critical_issues': self._identify_critical_issues(analysis_result),
            'major_opportunities': self._identify_opportunities(analysis_result)
        }
        
        return summary
    
    def _summarize_metric_status(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """总结指标状态"""
        trend_analysis = analysis_result.get('trend_analysis', {})
        status_summary = {}
        
        for metric_name, trend_info in trend_analysis.items():
            direction = trend_info.get('trend_indicators', {}).get('direction', 'unknown')
            strength = trend_info.get('trend_indicators', {}).get('strength', 'unknown')
            
            if direction == 'increasing':
                status = 'improving'
            elif direction == 'decreasing':
                status = 'declining'
            elif strength == 'strong':
                status = 'stable_strong'
            else:
                status = 'stable_weak'
            
            status_summary[metric_name] = status
        
        return status_summary
    
    def _identify_critical_issues(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别关键问题"""
        issues = []
        
        # 基于异常检测识别问题
        anomaly_detection = analysis_result.get('anomaly_detection', {})
        for metric_name, anomaly_info in anomaly_detection.items():
            anomaly_rate = anomaly_info.get('anomaly_summary', {}).get('anomaly_rate', 0)
            if anomaly_rate > 0.2:  # 异常率超过20%
                issues.append({
                    'type': 'high_anomaly_rate',
                    'metric': metric_name,
                    'severity': 'high',
                    'description': f'{metric_name}异常率过高({anomaly_rate:.1%})',
                    'impact': '可能影响系统稳定性'
                })
        
        # 基于趋势分析识别问题
        trend_analysis = analysis_result.get('trend_analysis', {})
        for metric_name, trend_info in trend_analysis.items():
            direction = trend_info.get('trend_indicators', {}).get('direction')
            strength = trend_info.get('trend_indicators', {}).get('strength')
            
            if direction == 'decreasing' and strength in ['strong', 'very_strong']:
                issues.append({
                    'type': 'performance_decline',
                    'metric': metric_name,
                    'severity': 'medium',
                    'description': f'{metric_name}呈现下降趋势',
                    'impact': '可能影响整体性能'
                })
        
        return issues
    
    def _identify_opportunities(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别机会"""
        opportunities = []
        
        # 基于优化建议识别机会
        optimization_recommendations = analysis_result.get('optimization_recommendations', {})
        
        high_impact_recommendations = []
        for rec_type, recommendations in optimization_recommendations.items():
            if isinstance(recommendations, list):
                for rec in recommendations:
                    if rec.get('expected_impact') == 'high':
                        high_impact_recommendations.append(rec)
        
        for rec in high_impact_recommendations:
            opportunities.append({
                'type': 'optimization_opportunity',
                'description': rec.get('recommendation', ''),
                'potential_benefit': rec.get('expected_impact', 'medium'),
                'effort_level': 'medium'  # 简化处理
            })
        
        return opportunities
    
    def _extract_key_findings(self, analysis_result: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 基于整体评估
        overall_assessment = analysis_result.get('overall_performance_assessment', {})
        performance_grade = overall_assessment.get('performance_grade', 'unknown')
        
        findings.append(f"系统整体性能评级为: {performance_grade}")
        
        # 基于趋势分析
        trend_analysis = analysis_result.get('trend_analysis', {})
        improving_metrics = [name for name, info in trend_analysis.items()
                           if info.get('trend_indicators', {}).get('direction') == 'increasing']
        
        if improving_metrics:
            findings.append(f"性能改善的指标: {', '.join(improving_metrics)}")
        
        declining_metrics = [name for name, info in trend_analysis.items()
                           if info.get('trend_indicators', {}).get('direction') == 'decreasing']
        
        if declining_metrics:
            findings.append(f"性能下降的指标: {', '.join(declining_metrics)}")
        
        return findings
    
    def _extract_action_items(self, optimization_recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取行动项"""
        action_items = []
        
        # 从优化建议中提取高优先级行动项
        for rec_type, recommendations in optimization_recommendations.items():
            if isinstance(recommendations, list):
                for rec in recommendations:
                    if rec.get('urgency') in ['high', 'medium']:
                        action_items.append({
                            'priority': rec.get('urgency', 'low'),
                            'description': rec.get('recommendation', ''),
                            'type': rec_type,
                            'timeline': 'immediate' if rec.get('urgency') == 'high' else 'within_week'
                        })
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        action_items.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return action_items
    
    def _generate_next_steps(self, analysis_result: Dict[str, Any]) -> List[str]:
        """生成后续步骤"""
        next_steps = []
        
        # 基于分析结果生成具体步骤
        overall_assessment = analysis_result.get('overall_performance_assessment', {})
        performance_grade = overall_assessment.get('performance_grade', 'unknown')
        
        if performance_grade in ['poor', 'needs_improvement']:
            next_steps.append("立即进行参数调优和策略调整")
            next_steps.append("加强系统监控和异常处理")
        
        if performance_grade == 'satisfactory':
            next_steps.append("持续监控系统性能变化")
            next_steps.append("实施预防性优化措施")
        
        # 基于异常检测结果
        anomaly_detection = analysis_result.get('anomaly_detection', {})
        high_anomaly_metrics = [name for name, info in anomaly_detection.items()
                              if info.get('anomaly_summary', {}).get('anomaly_rate', 0) > 0.15]
        
        if high_anomaly_metrics:
            next_steps.append(f"重点关注高异常率指标: {', '.join(high_anomaly_metrics)}")
        
        return next_steps
    
    def predict_performance_trend(self, performance_history: Dict[str, List[float]],
                                 prediction_horizon: int = 10,
                                 confidence_level: float = 0.95,
                                 trend_type: str = "comprehensive") -> Dict[str, Any]:
        """预测性能趋势
        
        Args:
            performance_history: 性能历史数据字典
            prediction_horizon: 预测时间范围
            confidence_level: 置信水平
            trend_type: 趋势类型 ('linear', 'polynomial', 'comprehensive')
            
        Returns:
            Dict: 详细的趋势预测报告
        """
        try:
            self.logger.info(f"开始预测性能趋势，预测范围: {prediction_horizon}")
            
            if not performance_history:
                raise ValueError("性能历史数据不能为空")
            
            prediction_results = {}
            overall_predictions = {}
            
            # 为每个指标执行趋势预测
            for metric_name, historical_values in performance_history.items():
                if len(historical_values) < 3:
                    self.logger.warning(f"指标 {metric_name} 数据不足，跳过预测")
                    continue
                
                metric_prediction = self._predict_single_metric_trend(
                    metric_name, historical_values, prediction_horizon, confidence_level, trend_type
                )
                prediction_results[metric_name] = metric_prediction
            
            # 生成综合趋势预测
            overall_trend = self._generate_overall_trend_prediction(prediction_results)
            
            # 预测关键事件和转折点
            critical_events = self._predict_critical_events(prediction_results)
            
            # 风险评估和建议
            risk_assessment = self._assess_prediction_risks(prediction_results)
            future_recommendations = self._generate_future_recommendations(prediction_results, risk_assessment)
            
            # 计算预测置信度
            prediction_confidence = self._calculate_prediction_confidence(prediction_results)
            
            trend_prediction_report = {
                'prediction_id': f"trend_prediction_{datetime.now().timestamp()}",
                'prediction_timestamp': datetime.now().isoformat(),
                'prediction_parameters': {
                    'horizon': prediction_horizon,
                    'confidence_level': confidence_level,
                    'trend_type': trend_type
                },
                'metric_predictions': prediction_results,
                'overall_trend': overall_trend,
                'critical_events': critical_events,
                'risk_assessment': risk_assessment,
                'future_recommendations': future_recommendations,
                'prediction_confidence': prediction_confidence,
                'prediction_metadata': {
                    'metrics_predicted': len(prediction_results),
                    'prediction_quality': self._assess_prediction_quality(prediction_results),
                    'data_sufficiency': self._assess_data_sufficiency_for_prediction(performance_history)
                }
            }
            
            self.logger.info(f"性能趋势预测完成，预测指标数: {len(prediction_results)}")
            return trend_prediction_report
            
        except Exception as e:
            self.logger.error(f"性能趋势预测失败: {str(e)}")
            raise
    
    def identify_bottlenecks(self, performance_metrics: Dict[str, Any],
                           resource_utilization: Dict[str, float] = None,
                           system_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """识别系统瓶颈
        
        Args:
            performance_metrics: 性能指标数据
            resource_utilization: 资源利用率数据
            system_constraints: 系统约束条件
            
        Returns:
            Dict: 详细的瓶颈分析报告
        """
        try:
            self.logger.info("开始识别系统瓶颈")
            
            # 分析性能指标瓶颈
            performance_bottlenecks = self._analyze_performance_bottlenecks(performance_metrics)
            
            # 分析资源利用瓶颈
            resource_bottlenecks = self._analyze_resource_bottlenecks(resource_utilization or {})
            
            # 分析系统约束瓶颈
            constraint_bottlenecks = self._analyze_constraint_bottlenecks(system_constraints or {})
            
            # 计算瓶颈严重程度
            bottleneck_severity = self._calculate_bottleneck_severity(
                performance_bottlenecks, resource_bottlenecks, constraint_bottlenecks
            )
            
            # 生成瓶颈优先级排序
            prioritized_bottlenecks = self._prioritize_bottlenecks(
                performance_bottlenecks, resource_bottlenecks, constraint_bottlenecks, bottleneck_severity
            )
            
            # 分析瓶颈间的关系和影响
            bottleneck_interactions = self._analyze_bottleneck_interactions(prioritized_bottlenecks)
            
            # 生成瓶颈解决方案
            solution_recommendations = self._generate_bottleneck_solutions(prioritized_bottlenecks)
            
            # 评估解决效果
            solution_impact = self._estimate_solution_impact(solution_recommendations, bottleneck_severity)
            
            bottleneck_report = {
                'bottleneck_id': f"bottleneck_analysis_{datetime.now().timestamp()}",
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_bottlenecks': performance_bottlenecks,
                'resource_bottlenecks': resource_bottlenecks,
                'constraint_bottlenecks': constraint_bottlenecks,
                'bottleneck_severity': bottleneck_severity,
                'prioritized_bottlenecks': prioritized_bottlenecks,
                'bottleneck_interactions': bottleneck_interactions,
                'solution_recommendations': solution_recommendations,
                'solution_impact': solution_impact,
                'analysis_confidence': self._calculate_bottleneck_confidence(
                    performance_metrics, resource_utilization, system_constraints
                ),
                'analysis_summary': {
                    'total_bottlenecks_identified': len(prioritized_bottlenecks),
                    'critical_bottlenecks': len([b for b in prioritized_bottlenecks if b.get('severity') == 'critical']),
                    'immediate_action_required': len([b for b in prioritized_bottlenecks if b.get('urgency') == 'immediate']),
                    'estimated_improvement_potential': self._estimate_total_improvement_potential(prioritized_bottlenecks)
                }
            }
            
            self.logger.info(f"瓶颈识别完成，发现 {len(prioritized_bottlenecks)} 个瓶颈")
            return bottleneck_report
            
        except Exception as e:
            self.logger.error(f"瓶颈识别失败: {str(e)}")
            raise
    
    def optimize_resource_allocation(self, current_allocation: Dict[str, Dict[str, float]],
                                   performance_requirements: Dict[str, float],
                                   resource_constraints: Dict[str, float],
                                   optimization_objective: str = "performance") -> Dict[str, Any]:
        """优化资源分配建议
        
        Args:
            current_allocation: 当前资源分配
            performance_requirements: 性能需求
            resource_constraints: 资源约束
            optimization_objective: 优化目标 ('performance', 'efficiency', 'cost', 'balanced')
            
        Returns:
            Dict: 详细的资源优化建议报告
        """
        try:
            self.logger.info(f"开始资源分配优化，目标: {optimization_objective}")
            
            if not current_allocation:
                raise ValueError("当前资源分配数据不能为空")
            
            # 分析当前资源分配效率
            allocation_efficiency = self._analyze_current_allocation_efficiency(current_allocation, performance_requirements)
            
            # 识别资源分配不平衡
            allocation_imbalances = self._identify_allocation_imbalances(current_allocation, performance_requirements)
            
            # 执行资源优化计算
            optimization_results = self._compute_resource_optimization(
                current_allocation, performance_requirements, resource_constraints, optimization_objective
            )
            
            # 生成优化后的分配方案
            optimized_allocation = optimization_results.get('optimized_allocation', {})
            
            # 计算优化效果
            optimization_impact = self._calculate_optimization_impact(
                current_allocation, optimized_allocation, performance_requirements
            )
            
            # 生成实施计划
            implementation_plan = self._create_optimization_implementation_plan(
                optimized_allocation, optimization_impact
            )
            
            # 风险评估和缓解策略
            optimization_risks = self._assess_optimization_risks(
                optimized_allocation, resource_constraints
            )
            risk_mitigation = self._generate_risk_mitigation_strategies(optimization_risks)
            
            # 长期规划建议
            long_term_recommendations = self._generate_long_term_resource_recommendations(
                optimized_allocation, performance_requirements, optimization_impact
            )
            
            optimization_report = {
                'optimization_id': f"resource_optimization_{datetime.now().timestamp()}",
                'optimization_timestamp': datetime.now().isoformat(),
                'optimization_objective': optimization_objective,
                'current_allocation_analysis': {
                    'efficiency': allocation_efficiency,
                    'imbalances': allocation_imbalances
                },
                'optimization_results': optimization_results,
                'optimized_allocation': optimized_allocation,
                'optimization_impact': optimization_impact,
                'implementation_plan': implementation_plan,
                'risk_assessment': {
                    'risks': optimization_risks,
                    'mitigation_strategies': risk_mitigation
                },
                'long_term_recommendations': long_term_recommendations,
                'optimization_confidence': self._calculate_optimization_confidence(
                    current_allocation, performance_requirements, optimization_results
                ),
                'optimization_summary': {
                    'total_resource_reallocation': self._calculate_total_reallocation_amount(current_allocation, optimized_allocation),
                    'expected_performance_improvement': optimization_impact.get('expected_performance_gain', 0),
                    'implementation_complexity': self._assess_implementation_complexity(optimized_allocation),
                    'time_to_full_optimization': optimization_impact.get('implementation_timeframe', 'unknown')
                }
            }
            
            self.logger.info("资源分配优化完成")
            return optimization_report
            
        except Exception as e:
            self.logger.error(f"资源分配优化失败: {str(e)}")
            raise
    
    # 新增的辅助方法实现
    def _predict_single_metric_trend(self, metric_name: str, historical_values: List[float],
                                   prediction_horizon: int, confidence_level: float,
                                   trend_type: str) -> Dict[str, Any]:
        """预测单个指标的趋势"""
        # 简化的趋势预测（实际应用中可以使用更复杂的模型）
        values = np.array(historical_values)
        x = np.arange(len(values))
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # 生成预测值
        future_x = np.arange(len(values), len(values) + prediction_horizon)
        predicted_values = slope * future_x + intercept
        
        # 计算置信区间
        prediction_std = np.std(values) * np.sqrt(1 + 1/len(values) + 
                                                 (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(values) - 2)
        margin_of_error = t_value * prediction_std
        
        lower_bound = predicted_values - margin_of_error
        upper_bound = predicted_values + margin_of_error
        
        return {
            'metric_name': metric_name,
            'historical_data_points': len(values),
            'linear_model': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value
            },
            'predictions': {
                'horizon': prediction_horizon,
                'values': predicted_values.tolist(),
                'confidence_intervals': {
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist(),
                    'confidence_level': confidence_level
                }
            },
            'trend_assessment': {
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'strength': abs(r_value),
                'significance': 'significant' if p_value < 0.05 else 'not_significant',
                'prediction_quality': 'high' if r_value**2 > 0.8 else 'medium' if r_value**2 > 0.5 else 'low'
            }
        }
    
    def _generate_overall_trend_prediction(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合趋势预测"""
        if not prediction_results:
            return {}
        
        # 收集所有预测结果
        all_trends = []
        all_slopes = []
        all_r_squared = []
        
        for metric_name, result in prediction_results.items():
            linear_model = result.get('linear_model', {})
            trend_assessment = result.get('trend_assessment', {})
            
            all_trends.append(trend_assessment.get('direction', 'stable'))
            all_slopes.append(linear_model.get('slope', 0))
            all_r_squared.append(linear_model.get('r_squared', 0))
        
        # 计算整体趋势
        positive_trends = len([t for t in all_trends if t == 'increasing'])
        negative_trends = len([t for t in all_trends if t == 'decreasing'])
        stable_trends = len([t for t in all_trends if t == 'stable'])
        
        overall_direction = 'improving' if positive_trends > negative_trends else 'declining' if negative_trends > positive_trends else 'stable'
        overall_confidence = np.mean(all_r_squared)
        
        return {
            'overall_direction': overall_direction,
            'confidence': overall_confidence,
            'trend_distribution': {
                'improving': positive_trends,
                'declining': negative_trends,
                'stable': stable_trends
            },
            'average_slope': np.mean(all_slopes),
            'prediction_reliability': 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.5 else 'low'
        }
    
    def _predict_critical_events(self, prediction_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测关键事件和转折点"""
        critical_events = []
        
        for metric_name, result in prediction_results.items():
            linear_model = result.get('linear_model', {})
            trend_assessment = result.get('trend_assessment', {})
            
            slope = linear_model.get('slope', 0)
            r_squared = linear_model.get('r_squared', 0)
            
            # 预测大幅变化
            if abs(slope) > 0.1 and r_squared > 0.7:
                direction = 'improvement' if slope > 0 else 'degradation'
                critical_events.append({
                    'metric': metric_name,
                    'event_type': f'significant_{direction}',
                    'expected_magnitude': abs(slope),
                    'confidence': r_squared,
                    'timeline': 'within_prediction_horizon'
                })
            
            # 预测性能突破或低谷
            predictions = result.get('predictions', {}).get('values', [])
            if predictions:
                max_predicted = max(predictions)
                min_predicted = min(predictions)
                
                if max_predicted > 1.0:  # 预测性能超过100%
                    critical_events.append({
                        'metric': metric_name,
                        'event_type': 'performance_breakthrough',
                        'expected_value': max_predicted,
                        'confidence': 0.7,
                        'timeline': 'mid_prediction_horizon'
                    })
                
                if min_predicted < 0.3:  # 预测性能低于30%
                    critical_events.append({
                        'metric': metric_name,
                        'event_type': 'performance_low',
                        'expected_value': min_predicted,
                        'confidence': 0.7,
                        'timeline': 'mid_prediction_horizon'
                    })
        
        return critical_events
    
    def _assess_prediction_risks(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估预测风险"""
        risks = {
            'model_uncertainty': [],
            'data_insufficiency': [],
            'trend_reversal': [],
            'external_factors': []
        }
        
        for metric_name, result in prediction_results.items():
            linear_model = result.get('linear_model', {})
            trend_assessment = result.get('trend_assessment', {})
            
            r_squared = linear_model.get('r_squared', 0)
            p_value = linear_model.get('p_value', 1)
            
            # 模型不确定性风险
            if r_squared < 0.5:
                risks['model_uncertainty'].append({
                    'metric': metric_name,
                    'risk_level': 'high',
                    'description': f'{metric_name}的模型拟合度低（r²={r_squared:.2f}）'
                })
            
            # 数据不足风险
            data_points = result.get('historical_data_points', 0)
            if data_points < 5:
                risks['data_insufficiency'].append({
                    'metric': metric_name,
                    'risk_level': 'medium',
                    'description': f'{metric_name}历史数据点较少（{data_points}个）'
                })
            
            # 趋势逆转风险
            if trend_assessment.get('direction') == 'increasing' and p_value > 0.3:
                risks['trend_reversal'].append({
                    'metric': metric_name,
                    'risk_level': 'medium',
                    'description': f'{metric_name}上升趋势显著性不足，可能逆转'
                })
        
        # 计算整体风险等级
        total_risks = sum(len(risk_list) for risk_list in risks.values())
        risk_level = 'high' if total_risks > 5 else 'medium' if total_risks > 2 else 'low'
        
        return {
            'risk_categories': risks,
            'overall_risk_level': risk_level,
            'total_risk_count': total_risks
        }
    
    def _generate_future_recommendations(self, prediction_results: Dict[str, Any],
                                       risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成未来建议"""
        recommendations = []
        
        # 基于趋势预测的建议
        for metric_name, result in prediction_results.items():
            trend_assessment = result.get('trend_assessment', {})
            linear_model = result.get('linear_model', {})
            
            direction = trend_assessment.get('direction', 'stable')
            slope = linear_model.get('slope', 0)
            r_squared = linear_model.get('r_squared', 0)
            
            if direction == 'declining' and r_squared > 0.6:
                recommendations.append({
                    'type': 'preventive_action',
                    'metric': metric_name,
                    'recommendation': f'提前干预{metric_name}的下降趋势',
                    'urgency': 'high',
                    'timeline': 'immediate'
                })
            
            elif direction == 'increasing' and r_squared > 0.7:
                recommendations.append({
                    'type': 'performance_optimization',
                    'metric': metric_name,
                    'recommendation': f'利用{metric_name}的上升趋势，进一步优化性能',
                    'urgency': 'medium',
                    'timeline': 'within_month'
                })
        
        # 基于风险评估的建议
        risk_level = risk_assessment.get('overall_risk_level', 'low')
        if risk_level == 'high':
            recommendations.append({
                'type': 'risk_mitigation',
                'recommendation': '收集更多历史数据以提高预测可靠性',
                'urgency': 'high',
                'timeline': 'immediate'
            })
        
        return recommendations
    
    def _calculate_prediction_confidence(self, prediction_results: Dict[str, Any]) -> float:
        """计算预测置信度"""
        if not prediction_results:
            return 0.0
        
        confidence_scores = []
        for result in prediction_results.values():
            linear_model = result.get('linear_model', {})
            r_squared = linear_model.get('r_squared', 0)
            
            # 模型质量评分
            model_quality = r_squared
            confidence_scores.append(model_quality)
        
        return np.mean(confidence_scores)
    
    def _assess_prediction_quality(self, prediction_results: Dict[str, Any]) -> str:
        """评估预测质量"""
        if not prediction_results:
            return 'unknown'
        
        avg_r_squared = np.mean([
            result.get('linear_model', {}).get('r_squared', 0)
            for result in prediction_results.values()
        ])
        
        if avg_r_squared > 0.8:
            return 'excellent'
        elif avg_r_squared > 0.6:
            return 'good'
        elif avg_r_squared > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_data_sufficiency_for_prediction(self, performance_history: Dict[str, List[float]]) -> str:
        """评估预测数据充分性"""
        data_counts = [len(values) for values in performance_history.values()]
        min_data_points = min(data_counts) if data_counts else 0
        avg_data_points = np.mean(data_counts) if data_counts else 0
        
        if min_data_points >= 20 and avg_data_points >= 30:
            return 'excellent'
        elif min_data_points >= 10 and avg_data_points >= 15:
            return 'good'
        elif min_data_points >= 5 and avg_data_points >= 10:
            return 'adequate'
        else:
            return 'insufficient'
    
    # 瓶颈分析相关的辅助方法
    def _analyze_performance_bottlenecks(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析性能指标瓶颈"""
        bottlenecks = []
        
        for metric_name, metric_data in performance_metrics.items():
            if isinstance(metric_data, dict):
                current_value = metric_data.get('current_value', 0)
                target_value = metric_data.get('target_value', 1.0)
                historical_trend = metric_data.get('trend', 'stable')
                
                # 计算性能缺口
                performance_gap = max(0, target_value - current_value) / target_value if target_value > 0 else 0
                
                # 识别瓶颈
                if performance_gap > 0.2 or current_value < target_value * 0.8:
                    bottlenecks.append({
                        'type': 'performance_gap',
                        'metric': metric_name,
                        'current_value': current_value,
                        'target_value': target_value,
                        'gap_percentage': performance_gap * 100,
                        'severity': 'critical' if performance_gap > 0.4 else 'high' if performance_gap > 0.25 else 'medium',
                        'trend': historical_trend,
                        'impact_score': performance_gap * 10  # 10分制评分
                    })
                
                # 检查趋势瓶颈
                elif historical_trend == 'declining':
                    bottlenecks.append({
                        'type': 'trend_degradation',
                        'metric': metric_name,
                        'current_value': current_value,
                        'trend': historical_trend,
                        'severity': 'medium',
                        'impact_score': 5
                    })
        
        return bottlenecks
    
    def _analyze_resource_bottlenecks(self, resource_utilization: Dict[str, float]) -> List[Dict[str, Any]]:
        """分析资源利用瓶颈"""
        bottlenecks = []
        
        for resource_name, utilization in resource_utilization.items():
            if utilization > 0.9:  # 90%以上利用率认为是瓶颈
                bottlenecks.append({
                    'type': 'resource_overutilization',
                    'resource': resource_name,
                    'utilization_rate': utilization,
                    'severity': 'critical' if utilization > 0.95 else 'high',
                    'impact_score': utilization * 10,
                    'recommended_action': f'增加{resource_name}资源或优化使用'
                })
            elif utilization < 0.1:  # 10%以下利用率可能是资源浪费
                bottlenecks.append({
                    'type': 'resource_underutilization',
                    'resource': resource_name,
                    'utilization_rate': utilization,
                    'severity': 'low',
                    'impact_score': (1 - utilization) * 5,
                    'recommended_action': f'重新分配{resource_name}资源'
                })
        
        return bottlenecks
    
    def _analyze_constraint_bottlenecks(self, system_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析系统约束瓶颈"""
        bottlenecks = []
        
        for constraint_name, constraint_data in system_constraints.items():
            if isinstance(constraint_data, dict):
                current_limit = constraint_data.get('current_limit', float('inf'))
                required_capacity = constraint_data.get('required_capacity', 0)
                
                # 检查容量约束
                if current_limit < required_capacity:
                    capacity_shortage = (required_capacity - current_limit) / required_capacity
                    bottlenecks.append({
                        'type': 'capacity_constraint',
                        'constraint': constraint_name,
                        'current_limit': current_limit,
                        'required_capacity': required_capacity,
                        'shortage_percentage': capacity_shortage * 100,
                        'severity': 'critical' if capacity_shortage > 0.5 else 'high',
                        'impact_score': capacity_shortage * 15
                    })
        
        return bottlenecks
    
    def _calculate_bottleneck_severity(self, performance_bottlenecks: List[Dict[str, Any]],
                                     resource_bottlenecks: List[Dict[str, Any]],
                                     constraint_bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算瓶颈严重程度"""
        all_bottlenecks = performance_bottlenecks + resource_bottlenecks + constraint_bottlenecks
        
        if not all_bottlenecks:
            return {'overall_severity': 'none', 'total_bottlenecks': 0}
        
        # 计算严重程度分布
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        total_impact_score = 0
        
        for bottleneck in all_bottlenecks:
            severity = bottleneck.get('severity', 'low')
            if severity in severity_counts:
                severity_counts[severity] += 1
            total_impact_score += bottleneck.get('impact_score', 0)
        
        # 确定整体严重程度
        if severity_counts['critical'] > 0:
            overall_severity = 'critical'
        elif severity_counts['high'] > 2:
            overall_severity = 'high'
        elif severity_counts['high'] > 0 or severity_counts['medium'] > 3:
            overall_severity = 'medium'
        elif severity_counts['medium'] > 0 or severity_counts['low'] > 0:
            overall_severity = 'low'
        else:
            overall_severity = 'minimal'
        
        return {
            'overall_severity': overall_severity,
            'severity_distribution': severity_counts,
            'total_bottlenecks': len(all_bottlenecks),
            'total_impact_score': total_impact_score,
            'average_impact_score': total_impact_score / len(all_bottlenecks) if all_bottlenecks else 0
        }
    
    def _prioritize_bottlenecks(self, performance_bottlenecks: List[Dict[str, Any]],
                              resource_bottlenecks: List[Dict[str, Any]],
                              constraint_bottlenecks: List[Dict[str, Any]],
                              bottleneck_severity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成瓶颈优先级排序"""
        all_bottlenecks = performance_bottlenecks + resource_bottlenecks + constraint_bottlenecks
        
        # 按影响评分和严重程度排序
        severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        def priority_key(bottleneck):
            severity = bottleneck.get('severity', 'low')
            impact_score = bottleneck.get('impact_score', 0)
            return (severity_weights.get(severity, 0), impact_score)
        
        # 排序并添加优先级
        sorted_bottlenecks = sorted(all_bottlenecks, key=priority_key, reverse=True)
        
        for i, bottleneck in enumerate(sorted_bottlenecks):
            bottleneck['priority'] = i + 1
            bottleneck['urgency'] = 'immediate' if bottleneck.get('severity') == 'critical' else 'high' if bottleneck.get('severity') == 'high' else 'medium'
        
        return sorted_bottlenecks
    
    def _analyze_bottleneck_interactions(self, prioritized_bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析瓶颈间的关系和影响"""
        if len(prioritized_bottlenecks) < 2:
            return {'interaction_type': 'none', 'details': '瓶颈数量不足，无法分析相互作用'}
        
        interactions = []
        
        # 简化的相互作用分析（实际应用中会更复杂）
        for i, bottleneck1 in enumerate(prioritized_bottlenecks[:3]):  # 只分析前3个瓶颈
            for j, bottleneck2 in enumerate(prioritized_bottlenecks[i+1:], i+1):
                # 检查类型相互作用
                type1 = bottleneck1.get('type', '')
                type2 = bottleneck2.get('type', '')
                
                if ('resource' in type1 and 'performance' in type2) or ('performance' in type1 and 'resource' in type2):
                    interactions.append({
                        'bottleneck_1': bottleneck1.get('metric', bottleneck1.get('resource', 'unknown')),
                        'bottleneck_2': bottleneck2.get('metric', bottleneck2.get('resource', 'unknown')),
                        'interaction_type': 'cascading',
                        'description': '资源瓶颈导致性能瓶颈',
                        'severity_amplification': 1.2  # 20%放大效应
                    })
                
                elif type1 == type2:
                    interactions.append({
                        'bottleneck_1': bottleneck1.get('metric', bottleneck1.get('resource', 'unknown')),
                        'bottleneck_2': bottleneck2.get('metric', bottleneck2.get('resource', 'unknown')),
                        'interaction_type': 'compound',
                        'description': '同类瓶颈的复合效应',
                        'severity_amplification': 1.5  # 50%放大效应
                    })
        
        interaction_summary = {
            'total_interactions': len(interactions),
            'interaction_types': list(set([i['interaction_type'] for i in interactions])),
            'max_severity_amplification': max([i['severity_amplification'] for i in interactions]) if interactions else 1.0
        }
        
        return {
            'interactions': interactions,
            'summary': interaction_summary
        }
    
    def _generate_bottleneck_solutions(self, prioritized_bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成瓶颈解决方案"""
        solutions = []
        
        for bottleneck in prioritized_bottlenecks:
            bottleneck_type = bottleneck.get('type', '')
            
            if bottleneck_type == 'performance_gap':
                solutions.append({
                    'bottleneck': bottleneck,
                    'solutions': [
                        '优化算法和参数',
                        '增加计算资源',
                        '改进系统架构'
                    ],
                    'estimated_improvement': 0.3,  # 30%改进潜力
                    'implementation_complexity': 'medium'
                })
            
            elif bottleneck_type == 'resource_overutilization':
                solutions.append({
                    'bottleneck': bottleneck,
                    'solutions': [
                        '增加资源配置',
                        '优化资源使用',
                        '实施负载均衡'
                    ],
                    'estimated_improvement': 0.4,
                    'implementation_complexity': 'low'
                })
            
            elif bottleneck_type == 'capacity_constraint':
                solutions.append({
                    'bottleneck': bottleneck,
                    'solutions': [
                        '升级硬件配置',
                        '优化存储结构',
                        '采用分布式架构'
                    ],
                    'estimated_improvement': 0.5,
                    'implementation_complexity': 'high'
                })
        
        return solutions
    
    def _estimate_solution_impact(self, solution_recommendations: List[Dict[str, Any]],
                                bottleneck_severity: Dict[str, Any]) -> Dict[str, Any]:
        """评估解决方案效果"""
        total_improvement = 0
        implementation_complexities = []
        
        for solution in solution_recommendations:
            improvement = solution.get('estimated_improvement', 0)
            complexity = solution.get('implementation_complexity', 'medium')
            
            total_improvement += improvement
            implementation_complexities.append(complexity)
        
        # 计算整体改进潜力
        overall_improvement_potential = min(1.0, total_improvement)
        
        # 评估实施复杂度
        complexity_weights = {'low': 1, 'medium': 2, 'high': 3}
        avg_complexity_weight = np.mean([complexity_weights.get(c, 2) for c in implementation_complexities])
        
        implementation_difficulty = 'low' if avg_complexity_weight < 1.5 else 'medium' if avg_complexity_weight < 2.5 else 'high'
        
        return {
            'total_improvement_potential': overall_improvement_potential,
            'expected_performance_gain': overall_improvement_potential * 100,  # 百分比
            'implementation_difficulty': implementation_difficulty,
            'estimated_implementation_time': '2-6周' if implementation_difficulty == 'low' else '1-3个月' if implementation_difficulty == 'medium' else '3-6个月',
            'cost_estimate': 'low' if implementation_difficulty == 'low' else 'medium' if implementation_difficulty == 'medium' else 'high'
        }
    
    def _calculate_bottleneck_confidence(self, performance_metrics: Dict[str, Any],
                                       resource_utilization: Dict[str, float] = None,
                                       system_constraints: Dict[str, Any] = None) -> float:
        """计算瓶颈分析置信度"""
        confidence_factors = []
        
        # 数据充分性因子
        if performance_metrics:
            confidence_factors.append(0.8)  # 性能数据可用
        
        if resource_utilization:
            confidence_factors.append(0.7)  # 资源数据可用
        
        if system_constraints:
            confidence_factors.append(0.6)  # 约束数据可用
        
        # 数据质量和完整性
        if len(performance_metrics) >= 3:
            confidence_factors.append(0.2)
        
        return min(1.0, sum(confidence_factors))
    
    def _estimate_total_improvement_potential(self, prioritized_bottlenecks: List[Dict[str, Any]]) -> float:
        """估算总改进潜力"""
        if not prioritized_bottlenecks:
            return 0.0
        
        # 简单估算：前3个瓶颈的平均改进潜力
        top_bottlenecks = prioritized_bottlenecks[:3]
        improvement_scores = []
        
        for bottleneck in top_bottlenecks:
            severity = bottleneck.get('severity', 'low')
            if severity == 'critical':
                improvement_scores.append(0.4)  # 40%改进潜力
            elif severity == 'high':
                improvement_scores.append(0.3)
            elif severity == 'medium':
                improvement_scores.append(0.2)
            else:
                improvement_scores.append(0.1)
        
        return np.mean(improvement_scores) if improvement_scores else 0.0
    
    # 资源优化相关的辅助方法
    def _analyze_current_allocation_efficiency(self, current_allocation: Dict[str, Dict[str, float]],
                                             performance_requirements: Dict[str, float]) -> Dict[str, Any]:
        """分析当前资源分配效率"""
        efficiency_scores = {}
        overall_efficiency = 0
        
        for resource_type, allocation_data in current_allocation.items():
            allocated_amount = sum(allocation_data.values()) if isinstance(allocation_data, dict) else allocation_data
            required_amount = performance_requirements.get(resource_type, allocated_amount)
            
            # 计算效率比
            efficiency_ratio = min(1.0, allocated_amount / required_amount) if required_amount > 0 else 1.0
            efficiency_scores[resource_type] = {
                'allocated': allocated_amount,
                'required': required_amount,
                'efficiency_ratio': efficiency_ratio,
                'efficiency_grade': self._grade_efficiency(efficiency_ratio)
            }
            
            overall_efficiency += efficiency_ratio
        
        overall_efficiency /= len(current_allocation) if current_allocation else 1
        
        return {
            'resource_efficiency': efficiency_scores,
            'overall_efficiency': overall_efficiency,
            'efficiency_grade': self._grade_efficiency(overall_efficiency)
        }
    
    def _identify_allocation_imbalances(self, current_allocation: Dict[str, Dict[str, float]],
                                      performance_requirements: Dict[str, float]) -> List[Dict[str, Any]]:
        """识别资源分配不平衡"""
        imbalances = []
        
        for resource_type, allocation_data in current_allocation.items():
            allocated_amount = sum(allocation_data.values()) if isinstance(allocation_data, dict) else allocation_data
            required_amount = performance_requirements.get(resource_type, allocated_amount)
            
            imbalance_ratio = abs(allocated_amount - required_amount) / required_amount if required_amount > 0 else 0
            
            if imbalance_ratio > 0.2:  # 20%以上的不平衡被认为是显著的
                imbalance_type = 'over_allocation' if allocated_amount > required_amount else 'under_allocation'
                
                imbalances.append({
                    'resource_type': resource_type,
                    'imbalance_type': imbalance_type,
                    'allocated_amount': allocated_amount,
                    'required_amount': required_amount,
                    'imbalance_ratio': imbalance_ratio,
                    'severity': 'high' if imbalance_ratio > 0.5 else 'medium',
                    'rebalance_action': f'减少{resource_type}资源' if imbalance_type == 'over_allocation' else f'增加{resource_type}资源'
                })
        
        return imbalances
    
    def _compute_resource_optimization(self, current_allocation: Dict[str, Dict[str, float]],
                                     performance_requirements: Dict[str, float],
                                     resource_constraints: Dict[str, float],
                                     optimization_objective: str) -> Dict[str, Any]:
        """执行资源优化计算"""
        # 简化的优化算法（实际应用中会使用更复杂的优化方法）
        optimized_allocation = {}
        
        for resource_type in current_allocation.keys():
            current_amount = sum(current_allocation[resource_type].values()) if isinstance(current_allocation[resource_type], dict) else current_allocation[resource_type]
            required_amount = performance_requirements.get(resource_type, current_amount)
            max_constraint = resource_constraints.get(resource_type, float('inf'))
            
            # 根据优化目标调整
            if optimization_objective == "performance":
                # 优先满足性能需求
                optimized_amount = min(required_amount * 1.1, max_constraint)
            elif optimization_objective == "efficiency":
                # 优化资源效率
                optimized_amount = min(required_amount, max_constraint)
            elif optimization_objective == "cost":
                # 成本优化（减少资源）
                optimized_amount = min(required_amount * 0.9, max_constraint)
            else:  # balanced
                # 平衡各方面
                optimized_amount = min(required_amount * 1.05, max_constraint)
            
            optimized_allocation[resource_type] = {
                'optimized_amount': optimized_amount,
                'current_amount': current_amount,
                'change_amount': optimized_amount - current_amount,
                'change_percentage': ((optimized_amount - current_amount) / current_amount * 100) if current_amount > 0 else 0
            }
        
        return {
            'optimized_allocation': optimized_allocation,
            'optimization_method': 'simplified_linear',
            'objective_met': True  # 简化处理
        }
    
    def _calculate_optimization_impact(self, current_allocation: Dict[str, Dict[str, float]],
                                     optimized_allocation: Dict[str, Dict[str, float]],
                                     performance_requirements: Dict[str, float]) -> Dict[str, Any]:
        """计算优化效果"""
        total_current_cost = 0
        total_optimized_cost = 0
        total_performance_gain = 0
        
        for resource_type in current_allocation.keys():
            current_amount = sum(current_allocation[resource_type].values()) if isinstance(current_allocation[resource_type], dict) else current_allocation[resource_type]
            optimized_data = optimized_allocation.get(resource_type, {})
            optimized_amount = optimized_data.get('optimized_amount', current_amount)
            required_amount = performance_requirements.get(resource_type, current_amount)
            
            # 假设成本与资源量成正比
            current_cost = current_amount
            optimized_cost = optimized_amount
            
            # 计算性能增益（基于需求满足度）
            current_performance = min(1.0, current_amount / required_amount) if required_amount > 0 else 1.0
            optimized_performance = min(1.0, optimized_amount / required_amount) if required_amount > 0 else 1.0
            performance_gain = optimized_performance - current_performance
            
            total_current_cost += current_cost
            total_optimized_cost += optimized_cost
            total_performance_gain += performance_gain
        
        # 计算整体指标
        cost_change = (total_optimized_cost - total_current_cost) / total_current_cost if total_current_cost > 0 else 0
        average_performance_gain = total_performance_gain / len(current_allocation) if current_allocation else 0
        
        return {
            'cost_impact': {
                'current_total_cost': total_current_cost,
                'optimized_total_cost': total_optimized_cost,
                'cost_change_percentage': cost_change * 100,
                'cost_efficiency': 'improved' if cost_change < 0 else 'increased'
            },
            'performance_impact': {
                'expected_performance_gain': average_performance_gain * 100,  # 百分比
                'performance_improvement': average_performance_gain
            },
            'implementation_timeframe': '2-4周',  # 简化估算
            'risk_level': 'low' if abs(cost_change) < 0.1 else 'medium' if abs(cost_change) < 0.2 else 'high'
        }
    
    def _create_optimization_implementation_plan(self, optimized_allocation: Dict[str, Dict[str, float]],
                                               optimization_impact: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化实施计划"""
        phases = []
        
        # 第一阶段：立即行动
        immediate_actions = []
        for resource_type, data in optimized_allocation.items():
            change_amount = data.get('change_amount', 0)
            if abs(change_amount) > data.get('current_amount', 1) * 0.3:  # 30%以上变化
                immediate_actions.append({
                    'action': f'调整{resource_type}资源配置',
                    'description': f'从{data.get("current_amount", 0)}调整到{data.get("optimized_amount", 0)}',
                    'timeline': 'immediate',
                    'priority': 'high'
                })
        
        phases.append({
            'phase': 1,
            'name': '立即执行',
            'timeline': '1-2周',
            'actions': immediate_actions
        })
        
        # 第二阶段：逐步优化
        gradual_actions = []
        for resource_type, data in optimized_allocation.items():
            change_amount = data.get('change_amount', 0)
            if 0.1 <= abs(change_amount) / data.get('current_amount', 1) <= 0.3:  # 10-30%变化
                gradual_actions.append({
                    'action': f'逐步优化{resource_type}',
                    'description': f'按计划逐步调整至目标值',
                    'timeline': '2-4周',
                    'priority': 'medium'
                })
        
        if gradual_actions:
            phases.append({
                'phase': 2,
                'name': '逐步优化',
                'timeline': '2-4周',
                'actions': gradual_actions
            })
        
        # 第三阶段：监控和调整
        phases.append({
            'phase': 3,
            'name': '监控调整',
            'timeline': '4-8周',
            'actions': [
                '监控系统性能变化',
                '根据实际情况微调配置',
                '评估优化效果'
            ]
        })
        
        return {
            'implementation_phases': phases,
            'total_timeline': '4-8周',
            'key_milestones': [
                '完成主要资源调整',
                '性能指标达到预期',
                '系统稳定运行'
            ]
        }
    
    def _assess_optimization_risks(self, optimized_allocation: Dict[str, Dict[str, float]],
                                 resource_constraints: Dict[str, float]) -> List[Dict[str, Any]]:
        """评估优化风险"""
        risks = []
        
        for resource_type, data in optimized_allocation.items():
            optimized_amount = data.get('optimized_amount', 0)
            constraint_limit = resource_constraints.get(resource_type, float('inf'))
            
            # 检查约束风险
            if optimized_amount > constraint_limit:
                risks.append({
                    'risk_type': 'constraint_violation',
                    'resource_type': resource_type,
                    'risk_level': 'high',
                    'description': f'{resource_type}优化后数量({optimized_amount})超过约束限制({constraint_limit})',
                    'mitigation': f'调整优化目标或重新评估约束条件'
                })
            
            # 检查大幅变化风险
            change_percentage = abs(data.get('change_percentage', 0))
            if change_percentage > 50:
                risks.append({
                    'risk_type': 'large_change',
                    'resource_type': resource_type,
                    'risk_level': 'medium',
                    'description': f'{resource_type}资源配置变化超过50%，可能引起不稳定',
                    'mitigation': '采用渐进式调整策略'
                })
        
        return risks
    
    def _generate_risk_mitigation_strategies(self, optimization_risks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """生成风险缓解策略"""
        mitigation_strategies = defaultdict(list)
        
        for risk in optimization_risks:
            risk_type = risk.get('risk_type', '')
            risk_level = risk.get('risk_level', 'medium')
            
            if risk_type == 'constraint_violation':
                mitigation_strategies['constraint_violation'].extend([
                    '重新评估和调整优化目标',
                    '分阶段实施优化方案',
                    '增加约束条件的灵活性'
                ])
            
            elif risk_type == 'large_change':
                mitigation_strategies['large_change'].extend([
                    '采用渐进式调整策略',
                    '增加监控系统监控频率',
                    '准备回退方案'
                ])
            
            # 通用缓解策略
            if risk_level == 'high':
                mitigation_strategies['high_level_risks'].extend([
                    '增加测试和验证阶段',
                    '设置更多的检查点',
                    '准备应急预案'
                ])
        
        # 去重
        for key in mitigation_strategies:
            mitigation_strategies[key] = list(set(mitigation_strategies[key]))
        
        return dict(mitigation_strategies)
    
    def _generate_long_term_resource_recommendations(self, optimized_allocation: Dict[str, Dict[str, float]],
                                                   performance_requirements: Dict[str, float],
                                                   optimization_impact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成长期资源规划建议"""
        recommendations = []
        
        # 基于优化效果的建议
        performance_gain = optimization_impact.get('performance_impact', {}).get('expected_performance_gain', 0)
        if performance_gain > 20:  # 20%以上性能提升
            recommendations.append({
                'type': 'performance_optimization',
                'title': '扩大优化成果',
                'description': '当前优化效果显著，建议扩大应用范围',
                'timeline': '1-3个月',
                'priority': 'high'
            })
        
        # 基于资源变化趋势的建议
        total_change = 0
        for resource_type, data in optimized_allocation.items():
            total_change += abs(data.get('change_percentage', 0))
        
        if total_change > 50:  # 总变化超过50%
            recommendations.append({
                'type': 'infrastructure_upgrade',
                'title': '基础设施升级',
                'description': '资源需求变化较大，建议考虑基础设施升级',
                'timeline': '3-6个月',
                'priority': 'medium'
            })
        
        # 基于监控需求的建议
        recommendations.append({
            'type': 'monitoring_enhancement',
            'title': '监控体系完善',
            'description': '建立更完善的资源监控和预警体系',
            'timeline': '2-4周',
            'priority': 'high'
        })
        
        return recommendations
    
    def _calculate_optimization_confidence(self, current_allocation: Dict[str, Dict[str, float]],
                                         performance_requirements: Dict[str, float],
                                         optimization_results: Dict[str, Any]) -> float:
        """计算优化置信度"""
        confidence_factors = []
        
        # 数据完整性因子
        if len(current_allocation) >= 3:
            confidence_factors.append(0.3)
        
        if len(performance_requirements) == len(current_allocation):
            confidence_factors.append(0.3)
        
        # 优化结果合理性因子
        optimized_allocation = optimization_results.get('optimized_allocation', {})
        if optimized_allocation:
            reasonable_changes = 0
            total_resources = len(optimized_allocation)
            
            for resource_type, data in optimized_allocation.items():
                change_percentage = abs(data.get('change_percentage', 0))
                if change_percentage < 50:  # 合理的变化范围
                    reasonable_changes += 1
            
            if total_resources > 0:
                confidence_factors.append(reasonable_changes / total_resources * 0.4)
        
        return min(1.0, sum(confidence_factors))
    
    def _calculate_total_reallocation_amount(self, current_allocation: Dict[str, Dict[str, float]],
                                           optimized_allocation: Dict[str, Dict[str, float]]) -> float:
        """计算总重新分配量"""
        total_change = 0
        
        for resource_type in current_allocation.keys():
            current_data = current_allocation[resource_type]
            current_amount = sum(current_data.values()) if isinstance(current_data, dict) else current_data
            
            optimized_data = optimized_allocation.get(resource_type, {})
            optimized_amount = optimized_data.get('optimized_amount', current_amount)
            
            total_change += abs(optimized_amount - current_amount)
        
        return total_change
    
    def _assess_implementation_complexity(self, optimized_allocation: Dict[str, Dict[str, float]]) -> str:
        """评估实施复杂度"""
        large_changes = 0
        total_resources = len(optimized_allocation)
        
        for resource_type, data in optimized_allocation.items():
            change_percentage = abs(data.get('change_percentage', 0))
            if change_percentage > 30:  # 30%以上变化被认为是大变化
                large_changes += 1
        
        if total_resources == 0:
            return 'low'
        
        large_change_ratio = large_changes / total_resources
        
        if large_change_ratio > 0.5:
            return 'high'
        elif large_change_ratio > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _grade_efficiency(self, efficiency_score: float) -> str:
        """效率等级评估"""
        if efficiency_score >= 0.9:
            return '优秀'
        elif efficiency_score >= 0.8:
            return '良好'
        elif efficiency_score >= 0.7:
            return '中等'
        elif efficiency_score >= 0.6:
            return '一般'
        else:
            return '差'