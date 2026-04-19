#!/usr/bin/env python3
"""
泛化能力压力测试主系统
===================

该模块实现了泛化能力压力测试的核心框架，用于测试智能体在未见任务上的零样本迁移能力。

主要功能：
- 管理多个泛化测试任务
- 协调零样本和少样本测试
- 计算适应速度指标
- 生成详细的性能报告

作者: NeuroMinecraftGenesis Team
创建时间: 2025-11-13
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .modded_minecraft_test import ModdedMinecraftTest
from .pybullet_test import PyBulletTest
from .reddit_dialogue_test import RedditDialogueTest
from .zero_shot_evaluator import ZeroShotEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """测试类型枚举"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    ADAPTIVE = "adaptive"


@dataclass
class GeneralizationResult:
    """泛化测试结果数据类"""
    task_name: str
    test_type: TestType
    zero_shot_score: float
    few_shot_score: float
    adaptation_speed: float
    success_rate: float
    completion_time: float
    error_count: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class GeneralizationTest:
    """
    泛化能力测试主类
    
    负责管理三个未见任务测试：Modded Minecraft、PyBullet物理模拟器、Reddit对话任务
    通过零样本和少样本测试评估智能体的迁移能力
    """
    
    def __init__(self, 
                 output_dir: str = "/workspace/NeuroMinecraftGenesis/reports",
                 max_few_shot_attempts: int = 50):
        """
        初始化泛化测试系统
        
        Args:
            output_dir: 测试结果输出目录
            max_few_shot_attempts: 最大少样本适应尝试次数
        """
        self.output_dir = output_dir
        self.max_few_shot_attempts = max_few_shot_attempts
        self.results: List[GeneralizationResult] = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化测试任务
        self.test_tasks = {
            'modded_minecraft': ModdedMinecraftTest(),
            'pybullet': PyBulletTest(),
            'reddit_dialogue': RedditDialogueTest()
        }
        
        # 初始化评估器
        self.evaluator = ZeroShotEvaluator()
        
        logger.info(f"泛化测试系统初始化完成，输出目录: {output_dir}")
    
    def run_zero_shot_tests(self) -> Dict[str, float]:
        """
        运行零样本测试
        
        在未见任务上直接测试智能体的迁移能力，不提供任何示例或训练数据
        
        Returns:
            Dict[str, float]: 任务名称到零样本分数的映射
        """
        logger.info("开始零样本泛化测试...")
        zero_shot_scores = {}
        
        for task_name, test_task in self.test_tasks.items():
            logger.info(f"正在测试任务: {task_name}")
            
            try:
                start_time = time.time()
                
                # 运行零样本测试
                score = test_task.run_zero_shot_test()
                completion_time = time.time() - start_time
                
                zero_shot_scores[task_name] = score
                
                logger.info(f"{task_name} 零样本测试完成: {score:.3f} (用时: {completion_time:.2f}秒)")
                
            except Exception as e:
                logger.error(f"零样本测试 {task_name} 失败: {str(e)}")
                zero_shot_scores[task_name] = 0.0
        
        return zero_shot_scores
    
    def run_few_shot_tests(self, zero_shot_scores: Dict[str, float]) -> Dict[str, float]:
        """
        运行少样本适应测试
        
        在零样本测试基础上，允许智能体进行有限次数的适应学习
        
        Args:
            zero_shot_scores: 零样本测试结果
            
        Returns:
            Dict[str, float]: 任务名称到少样本分数的映射
        """
        logger.info("开始少样本适应测试...")
        few_shot_scores = {}
        
        for task_name, test_task in self.test_tasks.items():
            logger.info(f"正在适应任务: {task_name}")
            
            try:
                start_time = time.time()
                
                # 运行少样本测试
                score = test_task.run_few_shot_test(
                    max_attempts=self.max_few_shot_attempts,
                    baseline_score=zero_shot_scores.get(task_name, 0.0)
                )
                completion_time = time.time() - start_time
                
                few_shot_scores[task_name] = score
                
                logger.info(f"{task_name} 少样本测试完成: {score:.3f} (用时: {completion_time:.2f}秒)")
                
            except Exception as e:
                logger.error(f"少样本测试 {task_name} 失败: {str(e)}")
                few_shot_scores[task_name] = 0.0
        
        return few_shot_scores
    
    def calculate_adaptation_metrics(self, 
                                   zero_shot_scores: Dict[str, float],
                                   few_shot_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        计算适应性能指标
        
        计算每个任务的适应速度、成功率等关键指标
        
        Args:
            zero_shot_scores: 零样本测试结果
            few_shot_scores: 少样本测试结果
            
        Returns:
            Dict[str, Dict[str, float]]: 任务名称到适应指标的映射
        """
        logger.info("计算适应性能指标...")
        
        adaptation_metrics = {}
        
        for task_name in self.test_tasks.keys():
            zero_score = zero_shot_scores.get(task_name, 0.0)
            few_score = few_shot_scores.get(task_name, 0.0)
            
            # 计算适应速度：(少样本性能 - 零样本性能) / 尝试次数
            adaptation_speed = (few_score - zero_score) / self.max_few_shot_attempts
            
            # 计算成功率提升
            success_rate_improvement = few_score - zero_score
            
            adaptation_metrics[task_name] = {
                'zero_shot_score': zero_score,
                'few_shot_score': few_score,
                'adaptation_speed': adaptation_speed,
                'success_rate_improvement': success_rate_improvement,
                'relative_improvement': (few_score - zero_score) / max(zero_score, 0.001) * 100
            }
            
            logger.info(f"{task_name} 适应速度: {adaptation_speed:.6f}, 相对改进: {success_rate_improvement:.1%}")
        
        return adaptation_metrics
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行综合泛化测试
        
        执行完整的零样本和少样本测试流程，生成详细的性能报告
        
        Returns:
            Dict[str, Any]: 完整的测试结果报告
        """
        logger.info("=" * 50)
        logger.info("开始综合泛化能力测试")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # 阶段1: 零样本测试
        logger.info("阶段1: 零样本迁移测试")
        zero_shot_scores = self.run_zero_shot_tests()
        
        # 阶段2: 少样本适应测试
        logger.info("\n阶段2: 少样本适应测试")
        few_shot_scores = self.run_few_shot_tests(zero_shot_scores)
        
        # 阶段3: 性能指标计算
        logger.info("\n阶段3: 性能指标计算")
        adaptation_metrics = self.calculate_adaptation_metrics(zero_shot_scores, few_shot_scores)
        
        # 生成测试结果
        total_time = time.time() - start_time
        
        # 更新结果记录
        for task_name in self.test_tasks.keys():
            result = GeneralizationResult(
                task_name=task_name,
                test_type=TestType.ADAPTIVE,
                zero_shot_score=zero_shot_scores.get(task_name, 0.0),
                few_shot_score=few_shot_scores.get(task_name, 0.0),
                adaptation_speed=adaptation_metrics[task_name]['adaptation_speed'],
                success_rate=few_shot_scores.get(task_name, 0.0),
                completion_time=total_time,
                error_count=0,  # TODO: 实际错误计数
                timestamp=datetime.now().isoformat()
            )
            self.results.append(result)
        
        # 生成完整报告
        report = {
            'test_metadata': {
                'test_name': '泛化能力压力测试',
                'test_time': datetime.now().isoformat(),
                'total_duration': total_time,
                'max_few_shot_attempts': self.max_few_shot_attempts,
                'tested_tasks': list(self.test_tasks.keys())
            },
            'zero_shot_results': zero_shot_scores,
            'few_shot_results': few_shot_scores,
            'adaptation_metrics': adaptation_metrics,
            'detailed_results': [result.to_dict() for result in self.results]
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"generalization_test_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n测试完成！总用时: {total_time:.2f}秒")
        logger.info(f"详细报告已保存至: {report_file}")
        
        return report
    
    def generate_performance_summary(self, report: Dict[str, Any]) -> str:
        """
        生成性能摘要报告
        
        Args:
            report: 测试报告数据
            
        Returns:
            str: 格式化的性能摘要
        """
        summary = []
        summary.append("=" * 60)
        summary.append("泛化能力压力测试 - 性能摘要")
        summary.append("=" * 60)
        
        # 测试信息
        metadata = report['test_metadata']
        summary.append(f"测试时间: {metadata['test_time']}")
        summary.append(f"总耗时: {metadata['total_duration']:.2f}秒")
        summary.append(f"测试任务: {len(metadata['tested_tasks'])}个")
        summary.append("")
        
        # 任务性能对比
        summary.append("任务性能对比:")
        summary.append("-" * 60)
        summary.append(f"{'任务名称':<20} {'零样本':<10} {'少样本':<10} {'适应速度':<12} {'相对改进'}")
        summary.append("-" * 60)
        
        for task_name in metadata['tested_tasks']:
            metrics = report['adaptation_metrics'][task_name]
            summary.append(
                f"{task_name:<20} "
                f"{metrics['zero_shot_score']:<10.3f} "
                f"{metrics['few_shot_score']:<10.3f} "
                f"{metrics['adaptation_speed']:<12.6f} "
                f"{metrics['relative_improvement']:>8.1f}%"
            )
        
        summary.append("-" * 60)
        
        # 整体评估
        avg_zero_shot = sum(report['zero_shot_scores'].values()) / len(report['zero_shot_scores'])
        avg_few_shot = sum(report['few_shot_results'].values()) / len(report['few_shot_results'])
        avg_adaptation = sum(
            metrics['adaptation_speed'] for metrics in report['adaptation_metrics'].values()
        ) / len(report['adaptation_metrics'])
        
        summary.append("")
        summary.append("整体性能:")
        summary.append(f"平均零样本性能: {avg_zero_shot:.3f}")
        summary.append(f"平均少样本性能: {avg_few_shot:.3f}")
        summary.append(f"平均适应速度: {avg_adaptation:.6f}")
        
        if avg_zero_shot > 0.5:
            assessment = "优秀 - 表现出强大的零样本迁移能力"
        elif avg_zero_shot > 0.3:
            assessment = "良好 - 具有一定的迁移能力"
        elif avg_zero_shot > 0.1:
            assessment = "一般 - 迁移能力有限"
        else:
            assessment = "较差 - 零样本迁移能力不足"
        
        summary.append(f"总体评估: {assessment}")
        
        return "\n".join(summary)
    
    def export_results(self, format_type: str = "json") -> str:
        """
        导出测试结果
        
        Args:
            format_type: 导出格式 ("json", "csv", "txt")
            
        Returns:
            str: 导出的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            filename = f"generalization_results_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([result.to_dict() for result in self.results], f, 
                         ensure_ascii=False, indent=2)
        
        elif format_type == "csv":
            filename = f"generalization_results_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            import pandas as pd
            df = pd.DataFrame([result.to_dict() for result in self.results])
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        elif format_type == "txt":
            filename = f"generalization_summary_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # 生成报告摘要
                report = self.run_comprehensive_test()
                summary = self.generate_performance_summary(report)
                f.write(summary)
        
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
        
        logger.info(f"结果已导出至: {filepath}")
        return filepath


def main():
    """主函数 - 演示泛化测试系统"""
    # 创建泛化测试实例
    generalization_test = GeneralizationTest(
        output_dir="/workspace/NeuroMinecraftGenesis/reports",
        max_few_shot_attempts=50
    )
    
    # 运行综合测试
    report = generalization_test.run_comprehensive_test()
    
    # 生成并打印摘要
    summary = generalization_test.generate_performance_summary(report)
    print(summary)
    
    # 导出结果
    json_file = generalization_test.export_results("json")
    csv_file = generalization_test.export_results("csv")
    txt_file = generalization_test.export_results("txt")
    
    print(f"\n测试完成！结果文件:")
    print(f"- JSON: {json_file}")
    print(f"- CSV: {csv_file}")
    print(f"- TXT: {txt_file}")


if __name__ == "__main__":
    main()