#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势分析器模块
=============

这个模块负责分析智能体性能数据的趋势和模式。
通过统计学方法、机器学习技术和时间序列分析，
识别性能变化模式、预测未来趋势、检测异常行为。

核心功能：
- 时间序列趋势分析
- 模式识别和分类
- 异常检测和预警
- 预测模型和预报
- 统计显著性检验

作者：AI研究团队
日期：2025-11-13
"""

import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns


class TrendAnalyzer:
    """
    趋势分析器类
    
    负责对性能数据进行深度分析和趋势预测。
    支持多种分析算法，包括统计方法、机器学习模型等。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化趋势分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 分析配置
        self.analysis_config = {
            'trend_window_size': self.config.get('trend_window_size', 7),  # 趋势窗口大小（天）
            'seasonal_period': self.config.get('seasonal_period', 7),     # 季节性周期（天）
            'anomaly_threshold': self.config.get('anomaly_threshold', 2.0),  # 异常检测阈值
            'min_data_points': self.config.get('min_data_points', 10),    # 最少数据点
            'confidence_level': self.config.get('confidence_level', 0.95) # 置信水平
        }
        
        # 分析结果缓存
        self.analysis_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 缓存时间（秒）
        
        # 预测模型
        self.models = {}
        
        self.logger.info("趋势分析器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'trend_window_size': 7,
            'seasonal_period': 7,
            'anomaly_threshold': 2.0,
            'min_data_points': 10,
            'confidence_level': 0.95,
            'cache_ttl': 3600,
            'enable_ml_models': True,
            'auto_save_plots': True,
            'plot_dpi': 100,
            'output_dir': '/workspace/worlds/real/analysis_results'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TrendAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_dir = Path('/workspace/worlds/real/logs')
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f'trend_analyzer_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def analyze_trends(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析性能趋势
        
        这是趋势分析器的核心方法，对性能数据进行全面的趋势分析。
        
        Args:
            performance_data: 性能数据
            
        Returns:
            趋势分析结果
        """
        try:
            self.logger.info("开始执行性能趋势分析")
            start_time = datetime.now()
            
            # 验证数据
            if not self._validate_data(performance_data):
                return {'error': '数据验证失败'}
            
            # 数据预处理
            processed_data = self._preprocess_data(performance_data)
            
            # 执行各项分析
            analysis_results = {
                'timestamp': start_time.isoformat(),
                'data_summary': self._summarize_data(processed_data),
                'trend_analysis': self._analyze_overall_trend(processed_data),
                'seasonal_analysis': self._analyze_seasonal_patterns(processed_data),
                'anomaly_detection': self._detect_anomalies(processed_data),
                'correlation_analysis': self._analyze_correlations(processed_data),
                'stability_analysis': self._analyze_stability(processed_data),
                'prediction': self._generate_predictions(processed_data),
                'recommendations': self._generate_recommendations(processed_data)
            }
            
            # 生成可视化图表
            if self.config.get('auto_save_plots', True):
                self._generate_visualizations(processed_data, analysis_results)
            
            # 缓存结果
            self._cache_analysis_results(analysis_results)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            analysis_results['analysis_time'] = analysis_time
            
            self.logger.info(f"趋势分析完成，耗时 {analysis_time:.2f} 秒")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        if not isinstance(data, dict):
            self.logger.error("数据格式错误：期望字典类型")
            return False
        
        if 'timestamp' not in data:
            self.logger.error("缺少时间戳信息")
            return False
        
        if 'test_summary' not in data and 'environment_details' not in data:
            self.logger.error("缺少性能数据")
            return False
        
        return True
    
    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """预处理数据"""
        records = []
        
        # 处理环境详细数据
        if 'environment_details' in data:
            for env_name, env_data in data['environment_details'].items():
                if isinstance(env_data, dict) and 'metrics' in env_data:
                    record = {
                        'timestamp': data['timestamp'],
                        'environment': env_name,
                        **env_data['metrics']
                    }
                    records.append(record)
        
        # 处理测试汇总数据
        if 'test_summary' in data:
            summary = data['test_summary']
            record = {
                'timestamp': data['timestamp'],
                'environment': 'overall',
                **summary
            }
            records.append(record)
        
        if not records:
            raise ValueError("没有找到有效的性能数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # 排序
        df = df.sort_index()
        
        self.logger.info(f"数据预处理完成，共 {len(df)} 条记录")
        return df
    
    def _summarize_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数据摘要分析"""
        summary = {
            'total_records': len(df),
            'time_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat(),
                'duration_days': (df.index.max() - df.index.min()).days
            },
            'environments': df['environment'].unique().tolist() if 'environment' in df.columns else [],
            'metrics': df.select_dtypes(include=[np.number]).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # 基本统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['basic_statistics'] = df[numeric_cols].describe().to_dict()
        
        return summary
    
    def _analyze_overall_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """整体趋势分析"""
        trend_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        for col in numeric_cols:
            try:
                # 线性趋势分析
                linear_trend = self._calculate_linear_trend(df[col])
                
                # 移动平均趋势
                moving_avg_trend = self._calculate_moving_average_trend(df[col])
                
                # 趋势强度
                trend_strength = self._calculate_trend_strength(df[col])
                
                trend_results[col] = {
                    'linear_trend': linear_trend,
                    'moving_average_trend': moving_avg_trend,
                    'trend_strength': trend_strength,
                    'overall_direction': self._determine_trend_direction(linear_trend)
                }
                
            except Exception as e:
                self.logger.warning(f"趋势分析失败 {col}: {e}")
                trend_results[col] = {'error': str(e)}
        
        return trend_results
    
    def _calculate_linear_trend(self, series: pd.Series) -> Dict[str, Any]:
        """计算线性趋势"""
        if len(series) < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1}
        
        # 准备数据
        x = np.arange(len(series))
        y = series.values
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'standard_error': float(std_err),
            'significant': p_value < (1 - self.analysis_config['confidence_level'])
        }
    
    def _calculate_moving_average_trend(self, series: pd.Series, window: int = 5) -> Dict[str, Any]:
        """计算移动平均趋势"""
        if len(series) < window:
            return {'trend': 0, 'stability': 0}
        
        ma = series.rolling(window=window).mean()
        
        # 计算趋势（最后几个点的平均变化）
        recent_ma = ma.tail(window//2).mean()
        previous_ma = ma.head(window//2).mean()
        trend = recent_ma - previous_ma
        
        # 计算稳定性（移动标准差）
        stability = 1 / (1 + ma.std()) if ma.std() > 0 else 1
        
        return {
            'trend': float(trend),
            'stability': float(stability),
            'moving_average': ma.tolist()
        }
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """计算趋势强度"""
        if len(series) < 3:
            return 0.0
        
        # 使用Theil-Sen估算器
        try:
            from scipy.stats import theilslopes
            slope, intercept, lower, upper = theilslopes(series.values)
            
            # 趋势强度基于斜率的绝对值和置信区间宽度
            interval_width = upper - lower
            strength = abs(slope) / (1 + interval_width)
            return float(strength)
            
        except ImportError:
            # 备用方法：基于方差
            return float(abs(np.polyfit(range(len(series)), series.values, 1)[0]))
    
    def _determine_trend_direction(self, linear_trend: Dict[str, Any]) -> str:
        """确定趋势方向"""
        slope = linear_trend['slope']
        p_value = linear_trend['p_value']
        
        if p_value >= (1 - self.analysis_config['confidence_level']):
            return 'stable'
        
        if slope > 0:
            return 'improving'
        elif slope < 0:
            return 'declining'
        else:
            return 'stable'
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """季节性模式分析"""
        seasonal_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        for col in numeric_cols:
            try:
                # 周期性分析
                autocorrelation = self._calculate_autocorrelation(df[col])
                
                # 日周期性
                daily_pattern = self._analyze_daily_pattern(df[col])
                
                # 周周期性
                weekly_pattern = self._analyze_weekly_pattern(df[col])
                
                seasonal_results[col] = {
                    'autocorrelation': autocorrelation,
                    'daily_pattern': daily_pattern,
                    'weekly_pattern': weekly_pattern,
                    'has_seasonality': self._detect_seasonality(df[col])
                }
                
            except Exception as e:
                self.logger.warning(f"季节性分析失败 {col}: {e}")
                seasonal_results[col] = {'error': str(e)}
        
        return seasonal_results
    
    def _calculate_autocorrelation(self, series: pd.Series, max_lag: int = 20) -> Dict[str, Any]:
        """计算自相关性"""
        if len(series) < max_lag:
            max_lag = len(series) // 2
        
        autocorr = []
        for lag in range(1, min(max_lag, len(series))):
            if len(series) > lag:
                corr = series.autocorr(lag=lag)
                autocorr.append(float(corr) if not np.isnan(corr) else 0.0)
        
        # 找到最强的自相关
        max_autocorr = max(autocorr) if autocorr else 0.0
        optimal_lag = autocorr.index(max_autocorr) + 1 if autocorr else 0
        
        return {
            'autocorrelation_series': autocorr,
            'max_autocorrelation': max_autocorr,
            'optimal_lag': optimal_lag
        }
    
    def _analyze_daily_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """分析日周期模式"""
        if not isinstance(series.index, pd.DatetimeIndex):
            return {'pattern_strength': 0, 'peak_hours': []}
        
        # 按小时分组
        hourly_data = series.groupby(series.index.hour).mean()
        
        if len(hourly_data) < 2:
            return {'pattern_strength': 0, 'peak_hours': []}
        
        # 计算模式强度
        pattern_strength = hourly_data.std() / hourly_data.mean() if hourly_data.mean() > 0 else 0
        
        # 找到峰值时间
        peak_threshold = hourly_data.mean() + hourly_data.std()
        peak_hours = hourly_data[hourly_data >= peak_threshold].index.tolist()
        
        return {
            'pattern_strength': float(pattern_strength),
            'peak_hours': peak_hours,
            'hourly_values': hourly_data.to_dict()
        }
    
    def _analyze_weekly_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """分析周周期模式"""
        if not isinstance(series.index, pd.DatetimeIndex):
            return {'pattern_strength': 0, 'peak_days': []}
        
        # 按星期几分组
        weekday_data = series.groupby(series.index.dayofweek).mean()
        
        if len(weekday_data) < 2:
            return {'pattern_strength': 0, 'peak_days': []}
        
        # 计算模式强度
        pattern_strength = weekday_data.std() / weekday_data.mean() if weekday_data.mean() > 0 else 0
        
        # 找到峰值天数（0=Monday, 6=Sunday）
        peak_threshold = weekday_data.mean() + weekday_data.std()
        peak_days = weekday_data[weekday_data >= peak_threshold].index.tolist()
        
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_day_names = [weekday_names[day] for day in peak_days]
        
        return {
            'pattern_strength': float(pattern_strength),
            'peak_days': peak_day_names,
            'daily_values': weekday_data.to_dict()
        }
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """检测季节性"""
        # 使用自相关检测
        autocorr = series.autocorr(lag=7)  # 周周期
        if abs(autocorr) > 0.3:
            return True
        
        autocorr = series.autocorr(lag=1)  # 日周期
        if abs(autocorr) > 0.5:
            return True
        
        return False
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """异常检测"""
        anomaly_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        for col in numeric_cols:
            try:
                # 统计异常检测
                statistical_anomalies = self._detect_statistical_anomalies(df[col])
                
                # 基于模型异常检测
                model_anomalies = self._detect_model_anomalies(df[col])
                
                # 时间序列异常检测
                timeseries_anomalies = self._detect_timeseries_anomalies(df[col])
                
                anomaly_results[col] = {
                    'statistical_anomalies': statistical_anomalies,
                    'model_anomalies': model_anomalies,
                    'timeseries_anomalies': timeseries_anomalies,
                    'anomaly_score': self._calculate_anomaly_score(statistical_anomalies, model_anomalies)
                }
                
            except Exception as e:
                self.logger.warning(f"异常检测失败 {col}: {e}")
                anomaly_results[col] = {'error': str(e)}
        
        return anomaly_results
    
    def _detect_statistical_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """统计异常检测"""
        threshold = self.analysis_config['anomaly_threshold']
        
        # Z-score异常检测
        z_scores = np.abs(stats.zscore(series.dropna()))
        z_anomalies = np.where(z_scores > threshold)[0].tolist()
        
        # IQR异常检测
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
        return {
            'z_score_anomalies': z_anomalies,
            'iqr_anomalies': [idx.isoformat() for idx in iqr_anomalies],
            'z_score_threshold': threshold,
            'iqr_bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
        }
    
    def _detect_model_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """基于模型的异常检测"""
        if len(series) < 10:
            return {'anomalies': [], 'model_type': 'insufficient_data'}
        
        # 使用Isolation Forest思想（简化版）
        # 基于滚动窗口的统计特征
        window_size = min(10, len(series) // 3)
        
        anomalies = []
        for i in range(window_size, len(series)):
            window = series.iloc[i-window_size:i]
            current_value = series.iloc[i]
            
            # 计算当前值在窗口中的分位数
            percentile = stats.percentileofscore(window, current_value)
            
            # 如果分位数过低或过高，标记为异常
            if percentile < 5 or percentile > 95:
                anomalies.append({
                    'index': i,
                    'value': float(current_value),
                    'percentile': float(percentile),
                    'severity': abs(50 - percentile) / 50  # 异常严重程度
                })
        
        return {
            'anomalies': anomalies,
            'model_type': 'percentile_based',
            'window_size': window_size
        }
    
    def _detect_timeseries_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """时间序列异常检测"""
        if len(series) < 5:
            return {'anomalies': [], 'method': 'insufficient_data'}
        
        # 基于变化率检测
        changes = series.pct_change().dropna()
        change_threshold = self.analysis_config['anomaly_threshold'] * changes.std()
        
        significant_changes = changes[abs(changes) > change_threshold]
        
        anomalies = []
        for idx, change in significant_changes.items():
            anomalies.append({
                'timestamp': idx.isoformat(),
                'change_rate': float(change),
                'absolute_change': float(series[idx] - series.shift(1)[idx])
            })
        
        return {
            'anomalies': anomalies,
            'method': 'change_rate',
            'threshold': float(change_threshold)
        }
    
    def _calculate_anomaly_score(self, stat_anomalies: Dict, model_anomalies: Dict) -> float:
        """计算综合异常分数"""
        score = 0.0
        
        # 统计异常分数
        stat_score = len(stat_anomalies.get('z_score_anomalies', [])) + len(stat_anomalies.get('iqr_anomalies', []))
        score += stat_score * 0.4
        
        # 模型异常分数
        model_score = len(model_anomalies.get('anomalies', []))
        score += model_score * 0.6
        
        # 归一化到0-1范围
        normalized_score = min(score / 10.0, 1.0)
        
        return normalized_score
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """相关性分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        if len(numeric_cols) < 2:
            return {'correlation_matrix': {}, 'significant_correlations': []}
        
        # 计算相关系数矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 找到显著相关性
        significant_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # 避免重复
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.7:  # 显著相关性阈值
                        significant_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'significant_correlations': significant_correlations
        }
    
    def _analyze_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """稳定性分析"""
        stability_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        for col in numeric_cols:
            try:
                # 计算变异系数
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else float('inf')
                
                # 计算稳定性指数（基于移动标准差）
                rolling_std = df[col].rolling(window=5).std().mean()
                stability_index = 1 / (1 + rolling_std) if not np.isnan(rolling_std) else 0
                
                # 检测趋势断点
                breakpoints = self._detect_breakpoints(df[col])
                
                stability_results[col] = {
                    'coefficient_of_variation': float(cv) if cv != float('inf') else 0.0,
                    'stability_index': float(stability_index),
                    'trend_breakpoints': breakpoints,
                    'stability_rating': self._rate_stability(cv, stability_index)
                }
                
            except Exception as e:
                self.logger.warning(f"稳定性分析失败 {col}: {e}")
                stability_results[col] = {'error': str(e)}
        
        return stability_results
    
    def _detect_breakpoints(self, series: pd.Series) -> List[Dict[str, Any]]:
        """检测趋势断点"""
        if len(series) < 10:
            return []
        
        breakpoints = []
        
        # 使用滑动窗口检测均值变化
        window_size = max(3, len(series) // 5)
        
        for i in range(window_size, len(series) - window_size):
            before_window = series.iloc[i-window_size:i]
            after_window = series.iloc[i:i+window_size]
            
            # t检验
            try:
                t_stat, p_value = stats.ttest_ind(before_window, after_window)
                
                if p_value < 0.05:  # 显著性水平5%
                    mean_change = after_window.mean() - before_window.mean()
                    breakpoints.append({
                        'index': i,
                        'timestamp': series.index[i].isoformat(),
                        'mean_change': float(mean_change),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value)
                    })
                    
            except Exception:
                continue
        
        return breakpoints
    
    def _rate_stability(self, cv: float, stability_index: float) -> str:
        """评级稳定性"""
        if cv < 0.1 and stability_index > 0.8:
            return 'very_stable'
        elif cv < 0.2 and stability_index > 0.6:
            return 'stable'
        elif cv < 0.5 and stability_index > 0.4:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _generate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成预测"""
        prediction_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        for col in numeric_cols:
            try:
                # 简单预测模型
                linear_prediction = self._linear_regression_prediction(df[col])
                
                # 移动平均预测
                ma_prediction = self._moving_average_prediction(df[col])
                
                # 趋势外推
                trend_extrapolation = self._trend_extrapolation(df[col])
                
                prediction_results[col] = {
                    'linear_prediction': linear_prediction,
                    'moving_average_prediction': ma_prediction,
                    'trend_extrapolation': trend_extrapolation,
                    'confidence': self._calculate_prediction_confidence(df[col])
                }
                
            except Exception as e:
                self.logger.warning(f"预测生成失败 {col}: {e}")
                prediction_results[col] = {'error': str(e)}
        
        return prediction_results
    
    def _linear_regression_prediction(self, series: pd.Series, steps: int = 5) -> Dict[str, Any]:
        """线性回归预测"""
        if len(series) < 2:
            return {'predicted_values': [], 'model_quality': 'insufficient_data'}
        
        x = np.arange(len(series))
        y = series.values
        
        # 拟合线性模型
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 生成预测
        future_x = np.arange(len(series), len(series) + steps)
        predicted_y = slope * future_x + intercept
        
        return {
            'predicted_values': predicted_y.tolist(),
            'slope': float(slope),
            'r_squared': float(r_value**2),
            'model_quality': 'good' if r_value**2 > 0.7 else 'poor'
        }
    
    def _moving_average_prediction(self, series: pd.Series, window: int = 3, steps: int = 5) -> Dict[str, Any]:
        """移动平均预测"""
        if len(series) < window:
            return {'predicted_values': [], 'model_quality': 'insufficient_data'}
        
        # 计算最近几个点的平均值
        recent_values = series.tail(window).values
        predicted_value = np.mean(repeated for _ in range(steps))
        
        return {
            'predicted_values': [float(predicted_value)] * steps,
            'window_size': window,
            'base_value': float(np.mean(recent_values)),
            'model_quality': 'simple'
        }
    
    def _trend_extrapolation(self, series: pd.Series, steps: int = 5) -> Dict[str, Any]:
        """趋势外推预测"""
        if len(series) < 2:
            return {'predicted_values': [], 'trend': 'insufficient_data'}
        
        # 计算最近趋势
        recent_trend = (series.iloc[-1] - series.iloc[-min(5, len(series))]) / min(5, len(series))
        
        # 外推预测
        last_value = series.iloc[-1]
        predicted_values = [last_value + recent_trend * (i + 1) for i in range(steps)]
        
        trend_direction = 'increasing' if recent_trend > 0 else 'decreasing' if recent_trend < 0 else 'stable'
        
        return {
            'predicted_values': predicted_values,
            'trend_per_step': float(recent_trend),
            'trend_direction': trend_direction
        }
    
    def _calculate_prediction_confidence(self, series: pd.Series) -> float:
        """计算预测置信度"""
        if len(series) < 5:
            return 0.0
        
        # 基于数据稳定性和趋势一致性
        cv = series.std() / series.mean() if series.mean() != 0 else 1.0
        
        # 趋势一致性
        recent_trend = series.iloc[-1] - series.iloc[-min(3, len(series))]
        trend_consistency = min(1.0, abs(recent_trend) / series.std()) if series.std() > 0 else 0.5
        
        # 综合置信度
        confidence = (1 - min(cv, 1.0)) * trend_consistency
        
        return float(confidence)
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []
        
        # 基于趋势分析的建议
        trend_analysis = self._analyze_overall_trend(df)
        
        for metric, analysis in trend_analysis.items():
            if 'error' in analysis:
                continue
                
            direction = analysis.get('overall_direction', 'stable')
            
            if direction == 'declining':
                recommendations.append({
                    'type': 'trend_alert',
                    'metric': metric,
                    'priority': 'high',
                    'message': f'{metric} 指标呈现下降趋势，建议检查相关配置和算法参数',
                    'suggested_actions': [
                        '检查数据质量',
                        '审查算法参数',
                        '分析外部因素影响'
                    ]
                })
            
            elif direction == 'improving':
                recommendations.append({
                    'type': 'optimization_opportunity',
                    'metric': metric,
                    'priority': 'medium',
                    'message': f'{metric} 指标表现良好，可以考虑将此配置应用到其他环境',
                    'suggested_actions': [
                        '记录当前配置',
                        '扩展到其他场景',
                        '持续监控稳定性'
                    ]
                })
        
        # 基于异常检测的建议
        anomaly_results = self._detect_anomalies(df)
        high_anomaly_metrics = [
            metric for metric, result in anomaly_results.items()
            if 'error' not in result and result.get('anomaly_score', 0) > 0.5
        ]
        
        if high_anomaly_metrics:
            recommendations.append({
                'type': 'anomaly_alert',
                'metrics': high_anomaly_metrics,
                'priority': 'high',
                'message': '检测到多个指标存在异常行为，建议进行深入调查',
                'suggested_actions': [
                    '检查系统日志',
                    '验证数据源',
                    '分析异常发生时间点的外部事件'
                ]
            })
        
        # 基于稳定性的建议
        stability_analysis = self._analyze_stability(df)
        unstable_metrics = [
            metric for metric, analysis in stability_analysis.items()
            if 'error' not in analysis and analysis.get('stability_rating') == 'unstable'
        ]
        
        if unstable_metrics:
            recommendations.append({
                'type': 'stability_improvement',
                'metrics': unstable_metrics,
                'priority': 'medium',
                'message': f'指标 {unstable_metrics} 稳定性较差，建议优化算法参数',
                'suggested_actions': [
                    '增加正则化参数',
                    '改进数据预处理',
                    '调整学习率'
                ]
            })
        
        return recommendations
    
    def _generate_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]):
        """生成可视化图表"""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 时间序列图
            self._plot_time_series(df, output_dir)
            
            # 趋势分析图
            self._plot_trend_analysis(df, analysis_results.get('trend_analysis', {}), output_dir)
            
            # 异常检测图
            self._plot_anomaly_detection(df, analysis_results.get('anomaly_detection', {}), output_dir)
            
            self.logger.info(f"可视化图表已保存到 {output_dir}")
            
        except Exception as e:
            self.logger.error(f"生成可视化图表失败: {e}")
    
    def _plot_time_series(self, df: pd.DataFrame, output_dir: Path):
        """绘制时间序列图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 6 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            axes[i].plot(df.index, df[col], marker='o', markersize=4)
            axes[i].set_title(f'{col} 时间序列')
            axes[i].set_xlabel('时间')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series.png', dpi=self.config['plot_dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_trend_analysis(self, df: pd.DataFrame, trend_analysis: Dict[str, Any], output_dir: Path):
        """绘制趋势分析图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 6 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            if col in trend_analysis and 'linear_trend' in trend_analysis[col]:
                # 原始数据
                axes[i].plot(df.index, df[col], 'b-', alpha=0.7, label='实际值')
                
                # 趋势线
                trend = trend_analysis[col]['linear_trend']
                x_range = range(len(df))
                trend_line = [trend['slope'] * x + trend['intercept'] for x in x_range]
                axes[i].plot(df.index, trend_line, 'r--', alpha=0.8, label='趋势线')
                
                axes[i].set_title(f'{col} 趋势分析 (R²={trend["r_squared"]:.3f})')
                axes[i].set_xlabel('时间')
                axes[i].set_ylabel(col)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trend_analysis.png', dpi=self.config['plot_dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_detection(self, df: pd.DataFrame, anomaly_detection: Dict[str, Any], output_dir: Path):
        """绘制异常检测图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'environment']
        
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 6 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            if col in anomaly_detection and 'statistical_anomalies' in anomaly_detection[col]:
                # 原始数据
                axes[i].plot(df.index, df[col], 'b-', alpha=0.7, label='数据')
                
                # 异常点
                anomalies = anomaly_detection[col]['statistical_anomalies']['z_score_anomalies']
                if anomalies:
                    anomaly_values = [df[col].iloc[idx] for idx in anomalies]
                    anomaly_times = [df.index[idx] for idx in anomalies]
                    axes[i].scatter(anomaly_times, anomaly_values, color='red', s=50, label='异常点', zorder=5)
                
                axes[i].set_title(f'{col} 异常检测')
                axes[i].set_xlabel('时间')
                axes[i].set_ylabel(col)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_detection.png', dpi=self.config['plot_dpi'], bbox_inches='tight')
        plt.close()
    
    def _cache_analysis_results(self, results: Dict[str, Any]):
        """缓存分析结果"""
        cache_key = hash(str(results.get('data_summary', {})))
        self.analysis_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now().timestamp()
        }
    
    def get_cached_analysis(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """获取缓存的分析结果"""
        if data_hash in self.analysis_cache:
            cache_entry = self.analysis_cache[data_hash]
            if datetime.now().timestamp() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['results']
        return None
    
    def clear_cache(self):
        """清理缓存"""
        self.analysis_cache.clear()
        self.logger.info("分析结果缓存已清理")


if __name__ == "__main__":
    # 示例用法
    analyzer = TrendAnalyzer()
    
    # 模拟性能数据
    import random
    from datetime import datetime, timedelta
    
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'accuracy': random.uniform(0.85, 0.95),
            'precision': random.uniform(0.80, 0.92),
            'recall': random.uniform(0.82, 0.94)
        },
        'environment_details': {
            f'env_{i}': {
                'metrics': {
                    'accuracy': random.uniform(0.80, 0.90),
                    'processing_time': random.uniform(1.0, 3.0)
                }
            } for i in range(3)
        }
    }
    
    # 执行趋势分析
    results = analyzer.analyze_trends(test_data)
    print(f"趋势分析结果: {json.dumps(results, indent=2, ensure_ascii=False)}")