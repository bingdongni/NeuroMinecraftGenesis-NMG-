"""
灾难性遗忘分析器

该模块实现了灾难性遗忘分析功能，包括：
1. 灾难性遗忘率计算
2. 性能保持评估
3. 遗忘模式识别
4. 正则化效果分析

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass
class ForgettingEvent:
    """遗忘事件数据结构"""
    task_id: int
    current_performance: float
    baseline_performance: float
    forgetting_rate: float
    severity: str  # "mild", "moderate", "severe", "catastrophic"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cause_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class ForgettingPattern:
    """遗忘模式数据结构"""
    pattern_type: str  # "gradual", "sudden", "oscillating", "stable"
    trend_slope: float
    volatility: float
    persistence_score: float
    affected_tasks: List[int]
    severity_distribution: Dict[str, float]


class CatastrophicForgettingAnalyzer:
    """
    灾难性遗忘分析器
    
    该类负责分析连续学习过程中的灾难性遗忘问题，包括：
    - 遗忘率计算和监控
    - 性能保持评估
    - 遗忘模式识别和分析
    - EWC和MAS正则化效果评估
    - 遗忘预警和干预建议
    """
    
    def __init__(self, threshold: float = 0.05, window_size: int = 5):
        """
        初始化灾难性遗忘分析器
        
        Args:
            threshold: 遗忘率阈值（5%）
            window_size: 分析窗口大小
        """
        self.threshold = threshold
        self.window_size = window_size
        self.logger = logging.getLogger("forgetting_analyzer")
        
        # 性能基准数据
        self.performance_baselines: Dict[int, List[float]] = {}
        self.performance_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.forgetting_events: List[ForgettingEvent] = []
        
        # 分析结果缓存
        self.patterns: Dict[str, ForgettingPattern] = {}
        self.anomaly_scores: Dict[int, float] = {}
        self.intervention_points: List[Dict[str, Any]] = []
        
        # 正则化效果监控
        self.ewc_effectiveness: List[float] = []
        self.mas_effectiveness: List[float] = []
        
        # 统计分析
        self.statistical_tests = {}
        
        self.logger.info(f"灾难性遗忘分析器初始化完成，阈值: {self.threshold:.3f}")
    
    def track_performance(self, task_id: int, performance_scores: List[float], 
                         metadata: Dict[str, Any] = None):
        """
        跟踪任务性能
        
        Args:
            task_id: 任务ID
            performance_scores: 性能分数列表
            metadata: 元数据信息
        """
        if task_id not in self.performance_baselines:
            # 首次记录，建立基线
            self.performance_baselines[task_id] = performance_scores.copy()
            self.logger.info(f"为任务 {task_id} 建立性能基线")
        
        # 更新性能历史
        self.performance_history[task_id].extend(performance_scores)
        
        # 记录元数据
        if metadata:
            self._update_metadata(task_id, metadata)
    
    def _update_metadata(self, task_id: int, metadata: Dict[str, Any]):
        """更新元数据信息"""
        if 'metadata' not in locals():
            metadata = {}
        
        # 存储正则化参数
        if 'ewc_lambda' in metadata:
            self.ewc_effectiveness.append(metadata.get('ewc_effectiveness', 0))
        
        if 'mas_lambda' in metadata:
            self.mas_effectiveness.append(metadata.get('mas_effectiveness', 0))
    
    def analyze_forgetting(self, current_task_id: int, 
                          current_performance: float,
                          all_performances: Dict[int, List[float]]) -> float:
        """
        分析灾难性遗忘
        
        Args:
            current_task_id: 当前任务ID
            current_performance: 当前任务性能
            all_performances: 所有任务的性能历史
            
        Returns:
            灾难性遗忘率
        """
        self.logger.info(f"分析任务 {current_task_id} 的灾难性遗忘")
        
        total_forgetting = 0.0
        forgetting_tasks = 0
        
        # 计算对之前所有任务的影响
        for task_id in range(current_task_id):
            if task_id in all_performances:
                # 获取基线性能（任务完成后立即的性能）
                baseline_performance = all_performances[task_id][-1] if all_performances[task_id] else 0
                
                # 重新测试当前性能
                current_test_performance = self._re_evaluate_task_performance(task_id)
                
                if baseline_performance > 0:
                    # 计算遗忘率
                    forgetting_rate = (baseline_performance - current_test_performance) / baseline_performance
                    
                    # 记录遗忘事件
                    event = self._create_forgetting_event(
                        task_id, current_test_performance, baseline_performance, forgetting_rate
                    )
                    self.forgetting_events.append(event)
                    
                    if forgetting_rate > 0:
                        total_forgetting += forgetting_rate
                        forgetting_tasks += 1
                        
                        # 触发预警
                        if forgetting_rate > self.threshold:
                            self._trigger_forgetting_alert(task_id, forgetting_rate)
        
        # 计算平均遗忘率
        avg_forgetting_rate = total_forgetting / max(1, forgetting_tasks)
        
        # 更新异常检测
        self._update_anomaly_detection(current_task_id, avg_forgetting_rate)
        
        # 分析遗忘模式
        self._analyze_forgetting_pattern(current_task_id, avg_forgetting_rate)
        
        self.logger.info(f"任务 {current_task_id} 平均遗忘率: {avg_forgetting_rate:.3f}")
        
        return avg_forgetting_rate
    
    def _create_forgetting_event(self, task_id: int, current: float, 
                               baseline: float, rate: float) -> ForgettingEvent:
        """创建遗忘事件"""
        # 确定严重程度
        if rate < 0.02:
            severity = "mild"
        elif rate < 0.05:
            severity = "moderate"
        elif rate < 0.15:
            severity = "severe"
        else:
            severity = "catastrophic"
        
        # 分析原因
        cause_analysis = self._analyze_forgetting_causes(task_id, rate)
        
        return ForgettingEvent(
            task_id=task_id,
            current_performance=current,
            baseline_performance=baseline,
            forgetting_rate=rate,
            severity=severity,
            cause_analysis=cause_analysis
        )
    
    def _analyze_forgetting_causes(self, task_id: int, forgetting_rate: float) -> Dict[str, float]:
        """分析遗忘原因"""
        causes = {}
        
        # 基于时间间隔的影响
        time_factor = min(1.0, task_id / 20.0)  # 时间越长遗忘越严重
        causes['temporal_decay'] = time_factor * 0.3
        
        # 基于任务复杂度的影响
        complexity_factor = task_id % 5 / 4.0  # 简单周期模式
        causes['complexity_interference'] = complexity_factor * 0.2
        
        # 基于相似性干扰
        similarity_factor = np.sin(task_id * 0.1) * 0.1 + 0.5  # 模拟相似性
        causes['similarity_interference'] = similarity_factor * 0.2
        
        # 基于学习强度的不足
        learning_intensity_factor = max(0, 0.5 - task_id * 0.01)
        causes['insufficient_rehearsal'] = learning_intensity_factor * 0.3
        
        return causes
    
    def _re_evaluate_task_performance(self, task_id: int) -> float:
        """重新评估任务性能
        
        实际应用中这里应该从存储中重新加载模型并在任务上测试
        这里使用模拟数据
        """
        if task_id not in self.performance_history:
            return 0.0
        
        # 获取最近的性能表现
        recent_performances = list(self.performance_history[task_id])[-10:]
        return np.mean(recent_performances) if recent_performances else 0.0
    
    def _trigger_forgetting_alert(self, task_id: int, forgetting_rate: float):
        """触发遗忘预警"""
        alert = {
            'task_id': task_id,
            'forgetting_rate': forgetting_rate,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if forgetting_rate > 0.1 else 'medium',
            'recommended_actions': self._get_intervention_recommendations(task_id, forgetting_rate)
        }
        
        self.intervention_points.append(alert)
        self.logger.warning(f"遗忘预警: 任务 {task_id} 遗忘率 {forgetting_rate:.3f}")
    
    def _get_intervention_recommendations(self, task_id: int, forgetting_rate: float) -> List[str]:
        """获取干预建议"""
        recommendations = []
        
        if forgetting_rate > 0.1:
            recommendations.append("立即实施回忆强化训练")
            recommendations.append("增加EWC正则化强度")
            recommendations.append("考虑使用记忆重放机制")
        elif forgetting_rate > 0.05:
            recommendations.append("增加MAS正则化参数")
            recommendations.append("延长旧任务复习间隔")
            recommendations.append("调整学习率策略")
        else:
            recommendations.append("继续监控性能")
            recommendations.append("保持当前参数设置")
        
        if task_id > 50:
            recommendations.append("考虑实施渐进式网络")
            recommendations.append("使用注意力机制增强")
        
        return recommendations
    
    def _update_anomaly_detection(self, current_task_id: int, forgetting_rate: float):
        """更新异常检测"""
        # 使用滑动窗口检测异常
        if current_task_id >= self.window_size:
            recent_forgetting_rates = []
            for i in range(max(0, current_task_id - self.window_size), current_task_id):
                recent_forgetting_rates.append(
                    self.forgetting_events[i].forgetting_rate 
                    if i < len(self.forgetting_events) else 0
                )
            
            if len(recent_forgetting_rates) > 1:
                # 计算Z分数
                mean_rate = np.mean(recent_forgetting_rates)
                std_rate = np.std(recent_forgetting_rates)
                
                if std_rate > 0:
                    anomaly_score = abs(forgetting_rate - mean_rate) / std_rate
                    self.anomaly_scores[current_task_id] = anomaly_score
                    
                    if anomaly_score > 2.0:  # 2个标准差
                        self.logger.warning(f"检测到异常遗忘模式: 任务 {current_task_id}")
    
    def _analyze_forgetting_pattern(self, current_task_id: int, avg_forgetting_rate: float):
        """分析遗忘模式"""
        if current_task_id < 5:
            return
        
        # 收集最近的遗忘率数据
        recent_forgetting_rates = []
        for event in self.forgetting_events[-20:]:
            recent_forgetting_rates.append(event.forgetting_rate)
        
        if len(recent_forgetting_rates) < 3:
            return
        
        # 趋势分析
        x = np.arange(len(recent_forgetting_rates))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_forgetting_rates)
        
        # 波动性分析
        volatility = np.std(recent_forgetting_rates)
        
        # 模式识别
        if abs(slope) < 0.01 and volatility < 0.02:
            pattern_type = "stable"
        elif slope > 0.01:
            pattern_type = "gradual" if volatility < 0.05 else "oscillating"
        elif slope < -0.01:
            pattern_type = "improving"
        else:
            pattern_type = "oscillating"
        
        # 计算持久性分数
        persistence_score = 1.0 - min(1.0, avg_forgetting_rate)
        
        # 严重程度分布
        severity_counts = {}
        for event in self.forgetting_events[-20:]:
            severity = event.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 归一化分布
        total = sum(severity_counts.values())
        severity_distribution = {k: v/total for k, v in severity_counts.items()}
        
        pattern = ForgettingPattern(
            pattern_type=pattern_type,
            trend_slope=slope,
            volatility=volatility,
            persistence_score=persistence_score,
            affected_tasks=list(range(max(0, current_task_id - 20), current_task_id + 1)),
            severity_distribution=severity_distribution
        )
        
        self.patterns[current_task_id] = pattern
        
        self.logger.info(f"遗忘模式分析: {pattern_type}, 趋势斜率: {slope:.3f}, 波动性: {volatility:.3f}")
    
    def evaluate_regularization_effectiveness(self, ewc_scores: List[float], 
                                            mas_scores: List[float]) -> Dict[str, float]:
        """评估正则化算法效果
        
        Args:
            ewc_scores: EWC效果分数列表
            mas_scores: MAS效果分数列表
            
        Returns:
            效果评估结果
        """
        self.logger.info("评估正则化算法效果")
        
        results = {}
        
        # EWC效果分析
        if ewc_scores:
            ewc_mean = np.mean(ewc_scores)
            ewc_std = np.std(ewc_scores)
            ewc_trend = np.polyfit(range(len(ewc_scores)), ewc_scores, 1)[0]
            
            results['ewc_mean_effectiveness'] = ewc_mean
            results['ewc_consistency'] = 1.0 - (ewc_std / max(ewc_mean, 1e-8))
            results['ewc_improvement_trend'] = ewc_trend
            results['ewc_recommended_lambda'] = self._calculate_optimal_lambda(ewc_scores, 'ewc')
        
        # MAS效果分析
        if mas_scores:
            mas_mean = np.mean(mas_scores)
            mas_std = np.std(mas_scores)
            mas_trend = np.polyfit(range(len(mas_scores)), mas_scores, 1)[0]
            
            results['mas_mean_effectiveness'] = mas_mean
            results['mas_consistency'] = 1.0 - (mas_std / max(mas_mean, 1e-8))
            results['mas_improvement_trend'] = mas_trend
            results['mas_recommended_lambda'] = self._calculate_optimal_lambda(mas_scores, 'mas')
        
        # 组合效果分析
        if ewc_scores and mas_scores:
            combined_effectiveness = np.mean([np.mean(ewc_scores), np.mean(mas_scores)])
            results['combined_effectiveness'] = combined_effectiveness
            
            # 推荐最佳正则化策略
            if combined_effectiveness > 0.7:
                results['recommended_strategy'] = "保持当前正则化设置"
            elif ewc_mean > mas_mean:
                results['recommended_strategy'] = "增加EWC权重，减少MAS权重"
            else:
                results['recommended_strategy'] = "增加MAS权重，减少EWC权重"
        
        return results
    
    def _calculate_optimal_lambda(self, scores: List[float], method: str) -> float:
        """计算最优正则化参数"""
        # 基于效果分数的二次曲线拟合
        lambdas = np.linspace(0.1, 10.0, len(scores))  # 模拟lambda值
        coeffs = np.polyfit(lambdas, scores, 2)
        
        # 找到最大值点
        if len(coeffs) >= 2:
            optimal_lambda = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 1.0
            return max(0.1, min(10.0, optimal_lambda))
        
        return 1.0  # 默认值
    
    def generate_forgetting_report(self, current_task_id: int) -> Dict[str, Any]:
        """生成遗忘分析报告
        
        Args:
            current_task_id: 当前任务ID
            
        Returns:
            分析报告
        """
        self.logger.info(f"生成任务 {current_task_id} 的遗忘分析报告")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_task': current_task_id,
            'summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # 总结统计
        total_events = len(self.forgetting_events)
        severe_events = sum(1 for e in self.forgetting_events 
                           if e.severity in ['severe', 'catastrophic'])
        
        report['summary'] = {
            'total_forgetting_events': total_events,
            'severe_forgetting_events': severe_events,
            'severe_event_rate': severe_events / max(1, total_events),
            'average_forgetting_rate': np.mean([e.forgetting_rate for e in self.forgetting_events]) if self.forgetting_events else 0,
            'current_threshold_exceeded': sum(1 for e in self.forgetting_events if e.forgetting_rate > self.threshold)
        }
        
        # 详细分析
        if current_task_id in self.patterns:
            pattern = self.patterns[current_task_id]
            report['detailed_analysis'] = {
                'current_pattern': pattern.pattern_type,
                'trend_slope': pattern.trend_slope,
                'volatility': pattern.volatility,
                'persistence_score': pattern.persistence_score,
                'severity_distribution': pattern.severity_distribution
            }
        
        # 干预建议
        recent_alerts = [alert for alert in self.intervention_points 
                        if alert['task_id'] > current_task_id - 10]
        
        for alert in recent_alerts:
            report['recommendations'].extend(alert['recommended_actions'])
        
        # 基于正则化效果的建议
        if self.ewc_effectiveness and self.mas_effectiveness:
            ewc_effect = np.mean(self.ewc_effectiveness[-10:])
            mas_effect = np.mean(self.mas_effectiveness[-10:])
            
            if ewc_effect > mas_effect:
                report['recommendations'].append("EWC算法表现更好，建议增加其权重")
            else:
                report['recommendations'].append("MAS算法表现更好，建议增加其权重")
        
        # 生成报告可视化
        self._create_forgetting_visualizations(report)
        
        return report
    
    def _create_forgetting_visualizations(self, report: Dict[str, Any]):
        """创建遗忘分析可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('灾难性遗忘分析报告', fontsize=16)
        
        # 1. 遗忘率随时间变化
        if self.forgetting_events:
            task_ids = [e.task_id for e in self.forgetting_events]
            forgetting_rates = [e.forgetting_rate for e in self.forgetting_events]
            
            axes[0, 0].plot(task_ids, forgetting_rates, 'r-', linewidth=2, label='遗忘率')
            axes[0, 0].axhline(y=self.threshold, color='orange', linestyle='--', label=f'阈值({self.threshold})')
            axes[0, 0].set_xlabel('任务ID')
            axes[0, 0].set_ylabel('遗忘率')
            axes[0, 0].set_title('遗忘率变化趋势')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 2. 严重程度分布
        if report['detailed_analysis'].get('severity_distribution'):
            severity_dist = report['detailed_analysis']['severity_distribution']
            labels = list(severity_dist.keys())
            values = list(severity_dist.values())
            
            axes[0, 1].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('遗忘严重程度分布')
        
        # 3. 正则化效果对比
        if self.ewc_effectiveness and self.mas_effectiveness:
            recent_tasks = range(min(len(self.ewc_effectiveness), len(self.mas_effectiveness)))
            
            axes[1, 0].plot(recent_tasks, self.ewc_effectiveness[:len(recent_tasks)], 
                           'b-', linewidth=2, label='EWC效果')
            axes[1, 0].plot(recent_tasks, self.mas_effectiveness[:len(recent_tasks)], 
                           'g-', linewidth=2, label='MAS效果')
            axes[1, 0].set_xlabel('任务')
            axes[1, 0].set_ylabel('效果分数')
            axes[1, 0].set_title('正则化算法效果对比')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. 异常检测结果
        if self.anomaly_scores:
            anomaly_tasks = list(self.anomaly_scores.keys())
            anomaly_scores = list(self.anomaly_scores.values())
            
            axes[1, 1].bar(anomaly_tasks, anomaly_scores, alpha=0.7)
            axes[1, 1].axhline(y=2.0, color='red', linestyle='--', label='异常阈值')
            axes[1, 1].set_xlabel('任务ID')
            axes[1, 1].set_ylabel('异常分数')
            axes[1, 1].set_title('遗忘异常检测')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'forgetting_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self, filepath: str):
        """保存分析结果到文件
        
        Args:
            filepath: 文件路径
        """
        results = {
            'forgetting_events': [
                {
                    'task_id': e.task_id,
                    'current_performance': e.current_performance,
                    'baseline_performance': e.baseline_performance,
                    'forgetting_rate': e.forgetting_rate,
                    'severity': e.severity,
                    'timestamp': e.timestamp,
                    'cause_analysis': e.cause_analysis
                }
                for e in self.forgetting_events
            ],
            'patterns': {
                k: {
                    'pattern_type': v.pattern_type,
                    'trend_slope': v.trend_slope,
                    'volatility': v.volatility,
                    'persistence_score': v.persistence_score,
                    'affected_tasks': v.affected_tasks,
                    'severity_distribution': v.severity_distribution
                }
                for k, v in self.patterns.items()
            },
            'intervention_points': self.intervention_points,
            'regularization_effectiveness': {
                'ewc': self.ewc_effectiveness,
                'mas': self.mas_effectiveness
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"遗忘分析结果已保存到: {filepath}")


def main():
    """主函数 - 演示灾难性遗忘分析器"""
    # 创建分析器
    analyzer = CatastrophicForgettingAnalyzer(threshold=0.05)
    
    # 模拟数据
    for task_id in range(20):
        # 模拟性能历史
        baseline_performance = np.random.uniform(0.7, 0.9)
        current_performance = baseline_performance * np.random.uniform(0.8, 1.0)
        
        # 跟踪性能
        performance_scores = [baseline_performance] * 10
        analyzer.track_performance(task_id, performance_scores)
        
        # 分析遗忘
        all_performances = {task_id: [baseline_performance]}
        forgetting_rate = analyzer.analyze_forgetting(
            task_id, current_performance, all_performances
        )
        
        print(f"任务 {task_id}: 遗忘率 {forgetting_rate:.3f}")
    
    # 生成报告
    report = analyzer.generate_forgetting_report(19)
    
    print("\n遗忘分析报告摘要:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_forgetting_analysis.json")
    print("\n分析结果已保存")


if __name__ == "__main__":
    main()