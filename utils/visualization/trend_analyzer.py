"""
性能基准展示面板系统 - 趋势分析器
Performance Benchmark System - Trend Analyzer

该模块提供了性能趋势分析、预测和可视化功能，支持时间序列分析、
性能模式识别和未来性能预测。

This module provides performance trend analysis, prediction, and visualization, 
supporting time series analysis, performance pattern recognition, and future performance forecasting.

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import math

class TrendAnalyzer:
    """
    性能趋势分析器
    
    功能特性:
    - 时间序列趋势分析
    - 性能模式识别
    - 未来性能预测
    - 异常检测和报警
    - 趋势可视化
    
    Features:
    - Time series trend analysis
    - Performance pattern recognition
    - Future performance prediction
    - Anomaly detection and alerting
    - Trend visualization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化趋势分析器
        Initialize the trend analyzer
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger('TrendAnalyzer')
        self.config = config or self._default_config()
        
        # 趋势类型定义
        self.trend_types = {
            'increasing': {
                'name': '上升趋势',
                'description': '性能指标随时间呈上升趋势',
                'interpretation': '算法学习效果良好，性能持续提升'
            },
            'decreasing': {
                'name': '下降趋势',
                'description': '性能指标随时间呈下降趋势',
                'interpretation': '可能存在过拟合或性能退化'
            },
            'stable': {
                'name': '稳定趋势',
                'description': '性能指标保持相对稳定',
                'interpretation': '算法已达到收敛状态'
            },
            'volatile': {
                'name': '波动趋势',
                'description': '性能指标波动较大，不稳定',
                'interpretation': '学习过程不稳定，需要调整超参数'
            },
            'seasonal': {
                'name': '周期性趋势',
                'description': '性能指标呈现周期性变化',
                'interpretation': '可能受外部因素或任务特性影响'
            }
        }
        
        # 预测模型配置
        self.prediction_models = {
            'linear': {
                'name': '线性回归',
                'description': '基于线性趋势的预测',
                'suitable_for': '稳定上升或下降趋势'
            },
            'polynomial': {
                'name': '多项式回归',
                'description': '基于多项式拟合的预测',
                'suitable_for': '非线性增长趋势'
            },
            'exponential': {
                'name': '指数拟合',
                'description': '基于指数函数的预测',
                'suitable_for': '快速变化趋势'
            }
        }
        
        # 趋势分析历史
        self.trend_history = {}
        
        self.logger.info("趋势分析器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'min_data_points': 10,
            'trend_threshold': 0.1,  # 趋势显著性阈值
            'anomaly_threshold': 2.0,  # 异常检测标准差倍数
            'prediction_horizon': 30,  # 预测时间窗口（数据点）
            'confidence_level': 0.95,
            'seasonal_period': 7  # 周期性分析周期
        }
    
    def analyze_performance_trend(self, 
                                history: List[Dict[str, Any]], 
                                algorithm: str, 
                                task: str,
                                analysis_window: int = None) -> Dict[str, Any]:
        """
        分析性能趋势
        Analyze performance trends
        
        Args:
            history: 历史性能数据
            algorithm: 算法名称
            task: 任务名称
            analysis_window: 分析窗口大小
            
        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        try:
            if analysis_window is None:
                analysis_window = self.config.get('min_data_points', 10)
                
            # 数据预处理
            processed_data = self._preprocess_trend_data(history, analysis_window)
            
            if len(processed_data) < self.config.get('min_data_points', 10):
                return {
                    'error': f'数据点不足，需要至少 {self.config.get("min_data_points", 10)} 个数据点'
                }
            
            # 多维度趋势分析
            trend_analysis = {
                'algorithm': algorithm,
                'task': task,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(processed_data),
                'trends': {},
                'overall_trend': {},
                'predictions': {},
                'anomalies': {},
                'confidence_metrics': {}
            }
            
            # 提取性能指标
            metrics = self._extract_metrics(processed_data)
            
            # 逐个指标进行趋势分析
            for metric_name, values in metrics.items():
                metric_trend = self._analyze_single_metric_trend(
                    values, processed_data, metric_name
                )
                trend_analysis['trends'][metric_name] = metric_trend
            
            # 总体趋势分析
            overall_trend = self._analyze_overall_trend(metrics, processed_data)
            trend_analysis['overall_trend'] = overall_trend
            
            # 未来预测
            predictions = self._generate_predictions(metrics, processed_data)
            trend_analysis['predictions'] = predictions
            
            # 异常检测
            anomalies = self._detect_anomalies(metrics, processed_data)
            trend_analysis['anomalies'] = anomalies
            
            # 置信度指标
            confidence_metrics = self._calculate_confidence_metrics(trend_analysis)
            trend_analysis['confidence_metrics'] = confidence_metrics
            
            # 保存分析历史
            self._save_trend_analysis(algorithm, task, trend_analysis)
            
            self.logger.info(f"趋势分析完成: {algorithm} - {task}")
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return {'error': str(e)}
    
    def _preprocess_trend_data(self, 
                             history: List[Dict[str, Any]], 
                             window_size: int) -> List[Dict[str, Any]]:
        """
        预处理趋势数据
        Preprocess trend data for analysis
        """
        # 按时间排序
        sorted_history = sorted(history, key=lambda x: x['timestamp'])
        
        # 如果数据点太多，取最近的数据
        if len(sorted_history) > window_size * 2:
            sorted_history = sorted_history[-window_size:]
        
        # 补充时间戳信息
        processed_data = []
        base_time = datetime.now() - timedelta(days=len(sorted_history))
        
        for i, entry in enumerate(sorted_history):
            processed_entry = entry.copy()
            processed_entry['time_index'] = i
            processed_entry['timestamp'] = entry['timestamp']
            processed_data.append(processed_entry)
        
        return processed_data
    
    def _extract_metrics(self, processed_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        提取性能指标
        Extract performance metrics from data
        """
        metrics = {}
        
        if not processed_data:
            return metrics
        
        # 找到所有可能的指标
        sample_metrics = processed_data[0]['metrics']
        
        for metric_name in sample_metrics:
            values = []
            for entry in processed_data:
                if 'metrics' in entry and metric_name in entry['metrics']:
                    values.append(float(entry['metrics'][metric_name]))
                else:
                    values.append(0.0)
            metrics[metric_name] = values
        
        return metrics
    
    def _analyze_single_metric_trend(self, 
                                   values: List[float], 
                                   processed_data: List[Dict[str, Any]], 
                                   metric_name: str) -> Dict[str, Any]:
        """
        分析单个指标的趋势
        Analyze trend for a single metric
        """
        if len(values) < 3:
            return {'error': '数据点不足'}
        
        # 转换为numpy数组
        x = np.arange(len(values))
        y = np.array(values)
        
        # 线性趋势分析
        linear_slope, linear_intercept, linear_r, p_value, std_err = stats.linregress(x, y)
        
        # 趋势分类
        trend_type = self._classify_trend(linear_slope, p_value, linear_r)
        
        # 计算趋势强度
        trend_strength = abs(linear_r)
        
        # 计算趋势的方向性和稳定性
        direction = 'increasing' if linear_slope > 0 else 'decreasing' if linear_slope < 0 else 'stable'
        
        # 分析波动性
        volatility = np.std(y) / np.mean(y) if np.mean(y) != 0 else float('inf')
        
        # 计算最近变化率
        recent_change_rate = self._calculate_recent_change_rate(values)
        
        return {
            'metric_name': metric_name,
            'trend_type': trend_type,
            'direction': direction,
            'slope': float(linear_slope),
            'r_squared': float(linear_r ** 2),
            'p_value': float(p_value),
            'trend_strength': float(trend_strength),
            'volatility': float(volatility),
            'recent_change_rate': float(recent_change_rate),
            'interpretation': self._interpret_trend(trend_type, direction, trend_strength),
            'statistics': {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y)),
                'range': float(np.max(y) - np.min(y))
            }
        }
    
    def _classify_trend(self, slope: float, p_value: float, r_value: float) -> str:
        """
        分类趋势类型
        Classify trend type
        """
        significance_level = 0.05
        
        if p_value > significance_level:
            return 'stable'
        
        r_squared = r_value ** 2
        
        # 基于斜率和R²值分类
        if slope > self.config.get('trend_threshold', 0.1) and r_squared > 0.3:
            return 'increasing'
        elif slope < -self.config.get('trend_threshold', 0.1) and r_squared > 0.3:
            return 'decreasing'
        elif r_squared < 0.2:
            return 'volatile'
        else:
            return 'stable'
    
    def _calculate_recent_change_rate(self, values: List[float], window: int = 5) -> float:
        """
        计算最近变化率
        Calculate recent change rate
        """
        if len(values) < window:
            return 0.0
        
        recent_values = values[-window:]
        if len(recent_values) < 2:
            return 0.0
        
        # 计算最近区间的平均变化率
        changes = [(recent_values[i] - recent_values[i-1]) / max(recent_values[i-1], 0.001) 
                  for i in range(1, len(recent_values))]
        
        return np.mean(changes) if changes else 0.0
    
    def _interpret_trend(self, trend_type: str, direction: str, strength: float) -> str:
        """
        解释趋势
        Interpret trend
        """
        strength_desc = "强" if strength > 0.7 else "中等" if strength > 0.4 else "弱"
        
        if trend_type == 'increasing':
            return f"{strength_desc}上升趋势，性能持续改善"
        elif trend_type == 'decreasing':
            return f"{strength_desc}下降趋势，性能有所下降"
        elif trend_type == 'stable':
            return f"性能相对稳定，在可控范围内波动"
        elif trend_type == 'volatile':
            return f"性能波动较大，学习过程不稳定"
        else:
            return "趋势不明显，需要更多数据进行分析"
    
    def _analyze_overall_trend(self, 
                             metrics: Dict[str, List[float]], 
                             processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析总体趋势
        Analyze overall trend
        """
        if not metrics:
            return {'error': '没有可分析的性能指标'}
        
        # 计算综合性能指数
        overall_scores = []
        for i in range(len(processed_data)):
            score = 0
            valid_metrics = 0
            for metric_name, values in metrics.items():
                if i < len(values) and values[i] > 0:
                    # 标准化指标值（这里使用简单的0-1标准化）
                    normalized_value = min(1.0, values[i] / 1000)  # 假设1000为满分
                    score += normalized_value
                    valid_metrics += 1
            
            if valid_metrics > 0:
                overall_scores.append(score / valid_metrics)
        
        if len(overall_scores) < 3:
            return {'error': '数据不足无法进行总体趋势分析'}
        
        # 分析总体趋势
        x = np.arange(len(overall_scores))
        y = np.array(overall_scores)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trend_type = self._classify_trend(slope, p_value, r_value)
        
        return {
            'overall_trend_type': trend_type,
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'trend_strength': float(abs(r_value)),
            'average_performance': float(np.mean(overall_scores)),
            'performance_improvement': float((overall_scores[-1] - overall_scores[0]) / max(overall_scores[0], 0.001)),
            'stability_score': 1.0 - (np.std(overall_scores) / max(np.mean(overall_scores), 0.001))
        }
    
    def _generate_predictions(self, 
                            metrics: Dict[str, List[float]], 
                            processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成未来预测
        Generate future predictions
        """
        predictions = {}
        prediction_horizon = self.config.get('prediction_horizon', 30)
        
        for metric_name, values in metrics.items():
            if len(values) < 5:
                continue
            
            # 使用线性回归进行预测
            try:
                x = np.arange(len(values)).reshape(-1, 1)
                y = np.array(values)
                
                model = LinearRegression()
                model.fit(x, y)
                
                # 预测未来值
                future_x = np.arange(len(values), len(values) + prediction_horizon).reshape(-1, 1)
                future_predictions = model.predict(future_x)
                
                # 计算预测置信区间（简化版本）
                residuals = y - model.predict(x)
                prediction_std = np.std(residuals)
                confidence_interval = 1.96 * prediction_std  # 95%置信区间
                
                predictions[metric_name] = {
                    'model_type': 'linear',
                    'predictions': future_predictions.tolist(),
                    'confidence_interval': confidence_interval,
                    'prediction_quality': self._assess_prediction_quality(model.score(x, y)),
                    'trend_forecast': 'increasing' if model.coef_[0] > 0 else 'decreasing' if model.coef_[0] < 0 else 'stable'
                }
                
            except Exception as e:
                self.logger.warning(f"预测生成失败 ({metric_name}): {e}")
                continue
        
        return predictions
    
    def _assess_prediction_quality(self, r_squared: float) -> str:
        """
        评估预测质量
        Assess prediction quality
        """
        if r_squared >= 0.8:
            return 'excellent'
        elif r_squared >= 0.6:
            return 'good'
        elif r_squared >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _detect_anomalies(self, 
                        metrics: Dict[str, List[float]], 
                        processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检测异常值
        Detect anomalies
        """
        anomalies = {}
        anomaly_threshold = self.config.get('anomaly_threshold', 2.0)
        
        for metric_name, values in metrics.items():
            if len(values) < 5:
                continue
            
            # 使用Z-score方法检测异常
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                continue
            
            z_scores = [(val - mean_val) / std_val for val in values]
            
            # 找到异常点
            anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > anomaly_threshold]
            
            if anomaly_indices:
                anomalies[metric_name] = {
                    'anomaly_indices': anomaly_indices,
                    'anomaly_values': [values[i] for i in anomaly_indices],
                    'z_scores': [z_scores[i] for i in anomaly_indices],
                    'anomaly_count': len(anomaly_indices),
                    'anomaly_rate': len(anomaly_indices) / len(values),
                    'severity': 'high' if len(anomaly_indices) / len(values) > 0.1 else 'medium'
                }
        
        return anomalies
    
    def _calculate_confidence_metrics(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算置信度指标
        Calculate confidence metrics
        """
        trends = trend_analysis.get('trends', {})
        predictions = trend_analysis.get('predictions', {})
        anomalies = trend_analysis.get('anomalies', {})
        
        # 基于多种因素计算总体置信度
        confidence_factors = []
        
        # 1. 趋势显著性
        significant_trends = sum(1 for t in trends.values() 
                               if t.get('p_value', 1.0) < 0.05)
        if len(trends) > 0:
            confidence_factors.append(significant_trends / len(trends))
        
        # 2. 预测质量
        good_predictions = sum(1 for p in predictions.values() 
                             if p.get('prediction_quality', 'poor') in ['good', 'excellent'])
        if len(predictions) > 0:
            confidence_factors.append(good_predictions / len(predictions))
        
        # 3. 异常率（越低越好）
        total_anomalies = sum(len(a.get('anomaly_indices', [])) for a in anomalies.values())
        total_data_points = sum(len(t.get('statistics', {}).get('min', [0])) for t in trends.values())
        anomaly_rate = total_anomalies / max(total_data_points, 1)
        confidence_factors.append(1.0 - min(anomaly_rate, 1.0))
        
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        return {
            'overall_confidence': float(overall_confidence),
            'confidence_level': self._interpret_confidence(overall_confidence),
            'data_quality_score': float(1.0 - anomaly_rate),
            'analysis_reliability': 'high' if overall_confidence > 0.7 else 'medium' if overall_confidence > 0.4 else 'low'
        }
    
    def _interpret_confidence(self, confidence: float) -> str:
        """
        解释置信度
        Interpret confidence level
        """
        if confidence >= 0.8:
            return '非常高'
        elif confidence >= 0.6:
            return '高'
        elif confidence >= 0.4:
            return '中等'
        else:
            return '低'
    
    def _save_trend_analysis(self, algorithm: str, task: str, analysis: Dict[str, Any]):
        """
        保存趋势分析结果
        Save trend analysis results
        """
        key = f"{algorithm}_{task}"
        if key not in self.trend_history:
            self.trend_history[key] = []
        
        self.trend_history[key].append({
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis
        })
        
        # 保持历史记录在合理范围内
        if len(self.trend_history[key]) > 50:
            self.trend_history[key] = self.trend_history[key][-50:]
    
    def get_trend_summary(self, algorithm: str, task: str) -> Dict[str, Any]:
        """
        获取趋势摘要
        Get trend summary
        """
        key = f"{algorithm}_{task}"
        history = self.trend_history.get(key, [])
        
        if not history:
            return {'error': '没有找到趋势分析历史'}
        
        latest_analysis = history[-1]['analysis']
        
        summary = {
            'algorithm': algorithm,
            'task': task,
            'latest_analysis_time': history[-1]['timestamp'],
            'overall_trend': latest_analysis.get('overall_trend', {}),
            'key_metrics_trends': {},
            'prediction_summary': {},
            'recommendations': []
        }
        
        # 总结关键指标的趋势
        trends = latest_analysis.get('trends', {})
        for metric, trend_info in trends.items():
            if trend_info.get('trend_strength', 0) > 0.5:  # 只包含显著趋势
                summary['key_metrics_trends'][metric] = {
                    'type': trend_info.get('trend_type'),
                    'direction': trend_info.get('direction'),
                    'strength': trend_info.get('trend_strength'),
                    'interpretation': trend_info.get('interpretation')
                }
        
        # 预测摘要
        predictions = latest_analysis.get('predictions', {})
        if predictions:
            summary['prediction_summary'] = {
                'predicted_metrics': list(predictions.keys()),
                'average_trend_forecast': self._calculate_average_trend_forecast(predictions),
                'prediction_quality': 'good' if all(
                    p.get('prediction_quality', 'poor') in ['good', 'excellent'] 
                    for p in predictions.values()
                ) else 'fair'
            }
        
        # 生成建议
        summary['recommendations'] = self._generate_trend_recommendations(latest_analysis)
        
        return summary
    
    def _calculate_average_trend_forecast(self, predictions: Dict[str, Any]) -> str:
        """
        计算平均趋势预测
        Calculate average trend forecast
        """
        forecasts = [p.get('trend_forecast', 'stable') for p in predictions.values()]
        
        increasing_count = forecasts.count('increasing')
        decreasing_count = forecasts.count('decreasing')
        stable_count = forecasts.count('stable')
        
        if increasing_count > decreasing_count and increasing_count > stable_count:
            return 'increasing'
        elif decreasing_count > increasing_count and decreasing_count > stable_count:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_trend_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        生成趋势建议
        Generate trend recommendations
        """
        recommendations = []
        
        overall_trend = analysis.get('overall_trend', {})
        trends = analysis.get('trends', {})
        anomalies = analysis.get('anomalies', {})
        
        # 基于总体趋势的建议
        trend_type = overall_trend.get('overall_trend_type', 'stable')
        if trend_type == 'decreasing':
            recommendations.append("检测到性能下降趋势，建议检查学习率设置和训练数据质量")
        elif trend_type == 'increasing':
            recommendations.append("性能持续改善，当前训练策略效果良好")
        
        # 基于异常的建议
        if anomalies:
            recommendations.append(f"检测到 {len(anomalies)} 个指标的异常值，建议调查原因并调整超参数")
        
        # 基于稳定性的建议
        stability_score = overall_trend.get('stability_score', 0)
        if stability_score < 0.5:
            recommendations.append("性能波动较大，建议增加训练数据或调整网络架构")
        
        # 基于预测的建议
        predictions = analysis.get('predictions', {})
        for metric, pred_info in predictions.items():
            quality = pred_info.get('prediction_quality', 'poor')
            if quality == 'poor':
                recommendations.append(f"指标 {metric} 的预测质量较低，建议收集更多历史数据")
        
        if not recommendations:
            recommendations.append("当前性能表现良好，继续保持现有训练策略")
        
        return recommendations