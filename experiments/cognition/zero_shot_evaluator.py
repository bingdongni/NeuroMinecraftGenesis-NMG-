#!/usr/bin/env python3
"""
零样本评估器模块
==============

该模块实现了零样本学习评估器的核心功能，用于评估智能体在未见任务上的泛化能力和适应性能。

主要功能：
- 计算零样本和少样本性能指标
- 评估适应速度和学习效率
- 分析跨域迁移能力
- 生成性能比较报告

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

# 日志配置
import logging
logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """评估指标类型"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SUCCESS_RATE = "success_rate"
    ADAPTATION_SPEED = "adaptation_speed"
    LEARNING_EFFICIENCY = "learning_efficiency"
    TRANSFER_ABILITY = "transfer_ability"


class DomainType(Enum):
    """领域类型"""
    MINECRAFT = "minecraft"
    PYBULLET = "pybullet"
    REDDIT = "reddit"
    GENERAL = "general"


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    task_name: str
    domain: DomainType
    zero_shot_score: float
    few_shot_score: float
    adaptation_speed: float
    learning_efficiency: float
    success_rate: float
    error_count: int
    completion_time: float
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """评估报告数据类"""
    overall_performance: float
    domain_comparison: Dict[str, float]
    adaptation_analysis: Dict[str, Any]
    cross_domain_transfer: Dict[str, float]
    improvement_trajectory: List[float]
    recommendations: List[str]
    timestamp: str


class ZeroShotEvaluator:
    """
    零样本学习评估器
    
    负责评估智能体在不同领域任务上的零样本泛化能力
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 min_sample_size: int = 30):
        """
        初始化零样本评估器
        
        Args:
            confidence_level: 置信水平 (默认95%)
            min_sample_size: 最小样本大小
        """
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        
        # 评估权重配置
        self.metric_weights = {
            EvaluationMetric.ACCURACY: 0.25,
            EvaluationMetric.SUCCESS_RATE: 0.20,
            EvaluationMetric.ADAPTATION_SPEED: 0.20,
            EvaluationMetric.LEARNING_EFFICIENCY: 0.15,
            EvaluationMetric.TRANSFER_ABILITY: 0.20
        }
        
        # 基准性能数据（用于比较）
        self.baseline_performance = {
            DomainType.MINECRAFT: 0.65,
            DomainType.PYBULLET: 0.45,
            DomainType.REDDIT: 0.70,
            DomainType.GENERAL: 0.60
        }
        
        logger.info(f"零样本评估器初始化完成，置信水平: {confidence_level}")
    
    def calculate_adaptation_speed(self, 
                                 zero_shot_scores: Union[float, List[float]], 
                                 few_shot_scores: Union[float, List[float]], 
                                 max_attempts: int = 50) -> float:
        """
        计算适应速度指标
        
        适应速度 = (少样本性能 - 零样本性能) / 最大尝试次数
        
        Args:
            zero_shot_scores: 零样本性能分数
            few_shot_scores: 少样本性能分数
            max_attempts: 最大尝试次数
            
        Returns:
            float: 适应速度指标 (0.0-1.0)
        """
        if isinstance(zero_shot_scores, list):
            zero_shot_scores = statistics.mean(zero_shot_scores)
        
        if isinstance(few_shot_scores, list):
            few_shot_scores = statistics.mean(few_shot_scores)
        
        # 确保数值在有效范围内
        zero_shot_scores = max(0.0, min(1.0, zero_shot_scores))
        few_shot_scores = max(0.0, min(1.0, few_shot_scores))
        
        # 计算性能提升
        improvement = few_shot_scores - zero_shot_scores
        
        # 计算标准化适应速度
        adaptation_speed = improvement / max_attempts
        
        # 确保适应速度在合理范围内
        return max(0.0, min(adaptation_speed, 0.1))  # 最大理论值约0.1
    
    def calculate_learning_efficiency(self, 
                                    adaptation_speed: float, 
                                    consistency_score: float = 0.8) -> float:
        """
        计算学习效率指标
        
        学习效率考虑适应速度和一致性
        
        Args:
            adaptation_speed: 适应速度
            consistency_score: 一致性分数
            
        Returns:
            float: 学习效率指标 (0.0-1.0)
        """
        # 学习效率 = 适应速度 × 一致性 × 100 (归一化)
        efficiency = adaptation_speed * consistency_score * 100
        
        # 确保在合理范围内
        return max(0.0, min(efficiency, 1.0))
    
    def evaluate_cross_domain_transfer(self, 
                                     performance_by_domain: Dict[DomainType, float]) -> Dict[str, float]:
        """
        评估跨域迁移能力
        
        分析智能体在不同领域间的知识迁移效果
        
        Args:
            performance_by_domain: 各领域性能字典
            
        Returns:
            Dict[str, float]: 跨域迁移能力指标
        """
        transfer_metrics = {}
        
        # 计算领域间相关性
        domain_pairs = [
            (DomainType.MINECRAFT, DomainType.PYBULLET, "空间推理迁移"),
            (DomainType.MINECRAFT, DomainType.REDDIT, "策略应用迁移"),
            (DomainType.PYBULLET, DomainType.REDDIT, "逻辑推理迁移"),
            (DomainType.MINECRAFT, DomainType.GENERAL, "通用能力评估")
        ]
        
        for domain1, domain2, description in domain_pairs:
            if domain1 in performance_by_domain and domain2 in performance_by_domain:
                perf1 = performance_by_domain[domain1]
                perf2 = performance_by_domain[domain2]
                
                # 计算迁移相关性 (皮尔逊相关系数)
                correlation = self._calculate_correlation(perf1, perf2)
                transfer_score = min(correlation, 1.0)
                
                transfer_metrics[description] = max(0.0, transfer_score)
        
        # 计算总体迁移能力
        overall_transfer = statistics.mean(transfer_metrics.values()) if transfer_metrics else 0.0
        transfer_metrics["overall_transfer"] = overall_transfer
        
        return transfer_metrics
    
    def _calculate_correlation(self, x: float, y: float) -> float:
        """计算两个值的相关性（简化为差值）"""
        # 在单一场景中，我们使用相对差异作为相关性代理
        max_val = max(x, y, 0.001)  # 避免除零
        correlation = 1.0 - abs(x - y) / max_val
        return max(0.0, min(correlation, 1.0))
    
    def calculate_confidence_interval(self, scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        计算置信区间
        
        Args:
            scores: 分数列表
            confidence_level: 置信水平
            
        Returns:
            Tuple[float, float]: (下界, 上界)
        """
        if len(scores) < 2:
            mean_score = scores[0] if scores else 0.0
            return (mean_score, mean_score)
        
        mean_score = statistics.mean(scores)
        std_error = statistics.stdev(scores) / (len(scores) ** 0.5)
        
        # 简化的置信区间计算（实际应使用t分布）
        margin = std_error * 1.96  # 95%置信区间
        return (max(0.0, mean_score - margin), min(1.0, mean_score + margin))
    
    def evaluate_single_task(self, 
                           task_name: str,
                           domain: DomainType,
                           zero_shot_results: List[float],
                           few_shot_results: List[float],
                           max_attempts: int = 50,
                           error_count: int = 0,
                           completion_time: float = 0.0,
                           metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """
        评估单个任务的性能
        
        Args:
            task_name: 任务名称
            domain: 任务领域
            zero_shot_results: 零样本测试结果列表
            few_shot_results: 少样本测试结果列表
            max_attempts: 最大尝试次数
            error_count: 错误次数
            completion_time: 完成时间
            metadata: 元数据
            
        Returns:
            PerformanceMetrics: 性能指标对象
        """
        # 计算基本性能指标
        zero_shot_score = statistics.mean(zero_shot_results) if zero_shot_results else 0.0
        few_shot_score = statistics.mean(few_shot_results) if few_shot_results else 0.0
        
        # 计算适应速度
        adaptation_speed = self.calculate_adaptation_speed(
            zero_shot_score, few_shot_score, max_attempts
        )
        
        # 计算学习效率
        consistency_score = 1.0 - (error_count / max(max_attempts, 1))  # 基于错误率的反向一致性
        learning_efficiency = self.calculate_learning_efficiency(
            adaptation_speed, consistency_score
        )
        
        # 计算成功率
        success_rate = few_shot_score  # 简化处理
        
        # 计算置信区间
        all_scores = zero_shot_results + few_shot_results
        confidence_interval = self.calculate_confidence_interval(all_scores, self.confidence_level)
        
        return PerformanceMetrics(
            task_name=task_name,
            domain=domain,
            zero_shot_score=zero_shot_score,
            few_shot_score=few_shot_score,
            adaptation_speed=adaptation_speed,
            learning_efficiency=learning_efficiency,
            success_rate=success_rate,
            error_count=error_count,
            completion_time=completion_time,
            confidence_interval=confidence_interval,
            metadata=metadata or {}
        )
    
    def generate_improvement_trajectory(self, 
                                      metrics_history: List[PerformanceMetrics]) -> List[float]:
        """
        生成性能改进轨迹
        
        Args:
            metrics_history: 历史性能指标列表
            
        Returns:
            List[float]: 性能改进轨迹
        """
        if not metrics_history:
            return []
        
        trajectory = []
        baseline_score = self.baseline_performance.get(DomainType.GENERAL, 0.6)
        
        for i, metrics in enumerate(metrics_history):
            # 计算相对于基线的改进
            improvement = (metrics.few_shot_score - baseline_score) * 100
            trajectory.append(improvement)
        
        return trajectory
    
    def comprehensive_evaluation(self, 
                               task_metrics: List[PerformanceMetrics]) -> EvaluationReport:
        """
        进行综合评估
        
        Args:
            task_metrics: 任务性能指标列表
            
        Returns:
            EvaluationReport: 综合评估报告
        """
        if not task_metrics:
            raise ValueError("任务指标列表不能为空")
        
        # 计算总体性能
        overall_performance = statistics.mean([m.few_shot_score for m in task_metrics])
        
        # 领域性能对比
        domain_performance = {}
        domain_groups = {}
        
        for metric in task_metrics:
            if metric.domain not in domain_groups:
                domain_groups[metric.domain] = []
            domain_groups[metric.domain].append(metric.few_shot_score)
        
        for domain, scores in domain_groups.items():
            domain_performance[domain.value] = statistics.mean(scores)
        
        # 适应分析
        adaptation_analysis = {
            "average_adaptation_speed": statistics.mean([m.adaptation_speed for m in task_metrics]),
            "fastest_adaptation_task": max(task_metrics, key=lambda x: x.adaptation_speed).task_name,
            "slowest_adaptation_task": min(task_metrics, key=lambda x: x.adaptation_speed).task_name,
            "efficiency_distribution": self._categorize_efficiency([m.learning_efficiency for m in task_metrics])
        }
        
        # 跨域迁移评估
        performance_by_domain = {m.domain: m.few_shot_score for m in task_metrics}
        cross_domain_transfer = self.evaluate_cross_domain_transfer(performance_by_domain)
        
        # 生成改进轨迹
        improvement_trajectory = self.generate_improvement_trajectory(task_metrics)
        
        # 生成建议
        recommendations = self._generate_recommendations(task_metrics, overall_performance)
        
        return EvaluationReport(
            overall_performance=overall_performance,
            domain_comparison=domain_performance,
            adaptation_analysis=adaptation_analysis,
            cross_domain_transfer=cross_domain_transfer,
            improvement_trajectory=improvement_trajectory,
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _categorize_efficiency(self, efficiency_scores: List[float]) -> Dict[str, int]:
        """对效率分数进行分类"""
        categories = {"high": 0, "medium": 0, "low": 0}
        
        for score in efficiency_scores:
            if score >= 0.7:
                categories["high"] += 1
            elif score >= 0.4:
                categories["medium"] += 1
            else:
                categories["low"] += 1
        
        return categories
    
    def _generate_recommendations(self, 
                                task_metrics: List[PerformanceMetrics], 
                                overall_performance: float) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于整体性能的建议
        if overall_performance < 0.4:
            recommendations.append("需要大幅提升基础能力和学习效率")
        elif overall_performance < 0.6:
            recommendations.append("需要重点改善适应速度和学习质量")
        else:
            recommendations.append("整体表现良好，可进一步优化细节")
        
        # 基于适应速度的建议
        avg_adaptation_speed = statistics.mean([m.adaptation_speed for m in task_metrics])
        if avg_adaptation_speed < 0.001:
            recommendations.append("适应速度较慢，建议增加样例数量或改善学习策略")
        
        # 基于错误率的建议
        avg_error_count = statistics.mean([m.error_count for m in task_metrics])
        if avg_error_count > 10:
            recommendations.append("错误率较高，需要提升任务执行的精确度")
        
        # 基于跨域迁移的建议
        domain_variations = []
        domain_groups = {}
        for metric in task_metrics:
            if metric.domain not in domain_groups:
                domain_groups[metric.domain] = []
            domain_groups[metric.domain].append(metric.few_shot_score)
        
        for domain, scores in domain_groups.items():
            if len(scores) > 1:
                domain_variations.append(statistics.stdev(scores))
        
        if domain_variations and statistics.mean(domain_variations) > 0.2:
            recommendations.append("跨域性能差异较大，建议平衡各领域训练")
        
        return recommendations
    
    def benchmark_comparison(self, 
                           evaluation_report: EvaluationReport) -> Dict[str, Any]:
        """
        与基准性能进行对比
        
        Args:
            evaluation_report: 评估报告
            
        Returns:
            Dict[str, Any]: 对比结果
        """
        comparison_results = {}
        
        # 与总体基准对比
        benchmark_overall = self.baseline_performance[DomainType.GENERAL]
        comparison_results["overall_vs_benchmark"] = {
            "current": evaluation_report.overall_performance,
            "benchmark": benchmark_overall,
            "difference": evaluation_report.overall_performance - benchmark_overall,
            "performance_ratio": evaluation_report.overall_performance / benchmark_overall
        }
        
        # 领域对比
        for domain_name, performance in evaluation_report.domain_comparison.items():
            if domain_name in [d.value for d in DomainType]:
                domain_enum = DomainType(domain_name)
                benchmark = self.baseline_performance.get(domain_enum, 0.6)
                
                comparison_results[f"{domain_name}_vs_benchmark"] = {
                    "current": performance,
                    "benchmark": benchmark,
                    "difference": performance - benchmark,
                    "is_above_benchmark": performance > benchmark
                }
        
        return comparison_results
    
    def export_evaluation_data(self, 
                             task_metrics: List[PerformanceMetrics], 
                             evaluation_report: EvaluationReport,
                             output_file: str) -> str:
        """
        导出评估数据
        
        Args:
            task_metrics: 任务指标列表
            evaluation_report: 评估报告
            output_file: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        export_data = {
            "evaluation_metadata": {
                "confidence_level": self.confidence_level,
                "min_sample_size": self.min_sample_size,
                "evaluation_timestamp": evaluation_report.timestamp
            },
            "task_performance": [
                {
                    "task_name": m.task_name,
                    "domain": m.domain.value,
                    "zero_shot_score": m.zero_shot_score,
                    "few_shot_score": m.few_shot_score,
                    "adaptation_speed": m.adaptation_speed,
                    "learning_efficiency": m.learning_efficiency,
                    "success_rate": m.success_rate,
                    "confidence_interval": m.confidence_interval,
                    "metadata": m.metadata
                }
                for m in task_metrics
            ],
            "comprehensive_evaluation": {
                "overall_performance": evaluation_report.overall_performance,
                "domain_comparison": evaluation_report.domain_comparison,
                "adaptation_analysis": evaluation_report.adaptation_analysis,
                "cross_domain_transfer": evaluation_report.cross_domain_transfer,
                "improvement_trajectory": evaluation_report.improvement_trajectory,
                "recommendations": evaluation_report.recommendations
            },
            "benchmark_comparison": self.benchmark_comparison(evaluation_report)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估数据已导出至: {output_file}")
        return output_file


def main():
    """演示函数"""
    print("=" * 60)
    print("零样本评估器演示")
    print("=" * 60)
    
    # 创建评估器
    evaluator = ZeroShotEvaluator(confidence_level=0.95)
    
    # 模拟测试结果
    mock_metrics = [
        evaluator.evaluate_single_task(
            task_name="modded_minecraft_terralith",
            domain=DomainType.MINECRAFT,
            zero_shot_results=[0.45, 0.52, 0.38],
            few_shot_results=[0.78, 0.82, 0.75],
            max_attempts=50,
            error_count=3,
            completion_time=120.5
        ),
        evaluator.evaluate_single_task(
            task_name="pybullet_physics_simulation",
            domain=DomainType.PYBULLET,
            zero_shot_results=[0.32, 0.28, 0.35],
            few_shot_results=[0.65, 0.68, 0.62],
            max_attempts=50,
            error_count=7,
            completion_time=180.2
        ),
        evaluator.evaluate_single_task(
            task_name="reddit_dialogue_askscience",
            domain=DomainType.REDDIT,
            zero_shot_results=[0.58, 0.61, 0.55],
            few_shot_results=[0.85, 0.88, 0.82],
            max_attempts=50,
            error_count=2,
            completion_time=95.8
        )
    ]
    
    # 生成综合评估报告
    report = evaluator.comprehensive_evaluation(mock_metrics)
    
    print(f"整体性能: {report.overall_performance:.3f}")
    print(f"适应分析: {report.adaptation_analysis}")
    print(f"跨域迁移: {report.cross_domain_transfer}")
    print(f"改进轨迹: {report.improvement_trajectory}")
    print(f"优化建议: {report.recommendations}")
    
    # 基准对比
    benchmark_comparison = evaluator.benchmark_comparison(report)
    print(f"\n基准对比:")
    for key, value in benchmark_comparison.items():
        print(f"  {key}: {value}")
    
    # 导出数据
    output_file = "/workspace/NeuroMinecraftGenesis/reports/zero_shot_evaluation.json"
    evaluator.export_evaluation_data(mock_metrics, report, output_file)


if __name__ == "__main__":
    main()