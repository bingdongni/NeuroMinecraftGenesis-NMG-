"""
前后迁移分析器

该模块实现了前后迁移性能分析功能，包括：
1. 向前迁移（Forward Transfer）评估
2. 向后迁移（Backward Transfer）评估
3. 迁移矩阵构建和分析
4. 元学习能力评估

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TransferMetrics:
    """迁移指标数据结构"""
    source_task: int
    target_task: int
    forward_transfer_score: float
    backward_transfer_score: float
    transfer_efficiency: float
    transfer_direction: str  # "forward", "bidirectional", "none"
    skill_transfer_score: float
    knowledge_preservation: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TransferPattern:
    """迁移模式数据结构"""
    pattern_type: str  # "increasing", "decreasing", "stable", "oscillating"
    learning_acceleration: float
    knowledge_retention: float
    meta_learning_score: float
    transferred_skills: List[str]
    affected_task_pairs: List[Tuple[int, int]]
    transfer_effectiveness: float


class ForwardBackwardTransfer:
    """
    前后迁移分析器
    
    该类负责分析连续学习过程中的迁移性能，包括：
    - 向前迁移评估（新任务学习速度提升）
    - 向后迁移评估（旧任务性能保持）
    - 迁移矩阵构建和分析
    - 元学习能力评估
    - 技能迁移检测
    """
    
    def __init__(self, baseline_window: int = 10, transfer_threshold: float = 0.05):
        """
        初始化前后迁移分析器
        
        Args:
            baseline_window: 基线窗口大小
            transfer_threshold: 迁移阈值
        """
        self.baseline_window = baseline_window
        self.threshold = transfer_threshold
        self.logger = logging.getLogger("transfer_analyzer")
        
        # 迁移数据存储
        self.transfer_matrix: np.ndarray = np.zeros((100, 100))  # 任务间迁移矩阵
        self.transfer_metrics: List[TransferMetrics] = []
        self.learning_progress: Dict[int, List[float]] = {}
        self.skill_evolution: Dict[str, List[float]] = defaultdict(list)
        
        # 性能基线
        self.baseline_performance: Dict[int, float] = {}
        self.performance_evolution: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 迁移模式分析
        self.transfer_patterns: Dict[str, TransferPattern] = {}
        self.meta_learning_scores: List[float] = []
        
        # 技能迁移跟踪
        self.skill_transfer_matrix: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.common_skills: Set[str] = set()
        
        self.logger.info(f"前后迁移分析器初始化完成，阈值: {self.threshold:.3f}")
    
    def record_learning_progress(self, task_id: int, episode_rewards: List[float],
                               skills_used: List[str] = None):
        """
        记录学习进度
        
        Args:
            task_id: 任务ID
            episode_rewards: 回合奖励列表
            skills_used: 使用的技能列表
        """
        self.learning_progress[task_id] = episode_rewards.copy()
        
        # 更新性能演化
        self.performance_evolution[task_id].extend(episode_rewards)
        
        # 建立基线性能（首次任务的平均性能）
        if task_id == 0:
            self.baseline_performance[task_id] = np.mean(episode_rewards)
        else:
            # 计算相对于基线的改进
            if 0 in self.baseline_performance:
                baseline_avg = self.baseline_performance[0]
                current_avg = np.mean(episode_rewards)
                improvement = (current_avg - baseline_avg) / (abs(baseline_avg) + 1e-8)
                self.baseline_performance[task_id] = current_avg
                
                # 记录元学习进展
                self.meta_learning_scores.append(improvement)
        
        # 跟踪技能使用
        if skills_used:
            for skill in skills_used:
                if skill not in self.common_skills:
                    self.common_skills.add(skill)
                self.skill_evolution[skill].append(task_id)
    
    def calculate_forward_transfer(self, current_task_id: int, 
                                 episode_rewards: List[float]) -> float:
        """
        计算向前迁移分数
        
        向前迁移衡量新任务的学习速度是否比之前任务更快
        
        Args:
            current_task_id: 当前任务ID
            episode_rewards: 当前任务的奖励序列
            
        Returns:
            向前迁移分数
        """
        if current_task_id == 0:
            return 0.0  # 第一个任务没有前向迁移
        
        self.logger.info(f"计算任务 {current_task_id} 的向前迁移")
        
        # 计算当前任务的学习速度（前期奖励的平均值）
        early_performance = np.mean(episode_rewards[:self.baseline_window])
        
        # 计算之前任务的平均学习速度
        previous_speeds = []
        for prev_task_id in range(current_task_id):
            if prev_task_id in self.learning_progress:
                prev_early_performance = np.mean(
                    self.learning_progress[prev_task_id][:self.baseline_window]
                )
                previous_speeds.append(prev_early_performance)
        
        if not previous_speeds:
            return 0.0
        
        avg_previous_speed = np.mean(previous_speeds)
        
        # 计算前向迁移分数
        if avg_previous_speed != 0:
            forward_transfer_score = (early_performance - avg_previous_speed) / abs(avg_previous_speed)
        else:
            forward_transfer_score = 0.0
        
        # 记录到迁移矩阵
        for prev_task_id in range(current_task_id):
            if prev_task_id < self.transfer_matrix.shape[0] and current_task_id < self.transfer_matrix.shape[1]:
                self.transfer_matrix[prev_task_id, current_task_id] = forward_transfer_score
        
        self.logger.info(f"任务 {current_task_id} 前向迁移分数: {forward_transfer_score:.3f}")
        
        return forward_transfer_score
    
    def calculate_backward_transfer(self, current_task_id: int,
                                  re_evaluated_scores: Dict[int, float]) -> List[float]:
        """
        计算向后迁移分数
        
        向后迁移衡量学习新任务后，旧任务的性能是否保持
        
        Args:
            current_task_id: 当前任务ID
            re_evaluated_scores: 重新评估的旧任务分数
            
        Returns:
            各个旧任务的向后迁移分数列表
        """
        if current_task_id == 0:
            return []
        
        self.logger.info(f"计算任务 {current_task_id} 的向后迁移")
        
        backward_transfer_scores = []
        
        for prev_task_id in range(current_task_id):
            if prev_task_id not in re_evaluated_scores:
                continue
            
            # 获取基线性能
            if prev_task_id not in self.baseline_performance:
                continue
            
            baseline_performance = self.baseline_performance[prev_task_id]
            current_performance = re_evaluated_scores[prev_task_id]
            
            # 计算向后迁移分数
            if baseline_performance != 0:
                backward_transfer_score = (current_performance - baseline_performance) / abs(baseline_performance)
            else:
                backward_transfer_score = 0.0
            
            backward_transfer_scores.append(backward_transfer_score)
            
            # 更新迁移矩阵
            if prev_task_id < self.transfer_matrix.shape[0] and current_task_id < self.transfer_matrix.shape[1]:
                self.transfer_matrix[prev_task_id, current_task_id] = backward_transfer_score
        
        self.logger.info(f"任务 {current_task_id} 向后迁移分数: {np.mean(backward_transfer_scores):.3f}")
        
        return backward_transfer_scores
    
    def analyze_transfer_effectiveness(self, task_range: range = None) -> Dict[str, Any]:
        """
        分析迁移有效性
        
        Args:
            task_range: 分析的任务范围
            
        Returns:
            迁移效果分析结果
        """
        self.logger.info("分析迁移有效性")
        
        if task_range is None:
            task_range = range(len(self.learning_progress))
        
        # 计算整体迁移统计
        forward_transfers = []
        backward_transfers = []
        
        for i in task_range:
            # 前向迁移
            for j in range(i + 1, len(self.learning_progress)):
                if i < self.transfer_matrix.shape[0] and j < self.transfer_matrix.shape[1]:
                    forward_score = self.transfer_matrix[i, j]
                    if forward_score != 0:
                        forward_transfers.append(forward_score)
            
            # 后向迁移
            for j in range(i):
                if i < self.transfer_matrix.shape[0] and j < self.transfer_matrix.shape[1]:
                    backward_score = self.transfer_matrix[j, i]
                    if backward_score != 0:
                        backward_transfers.append(backward_score)
        
        results = {}
        
        if forward_transfers:
            results['forward_transfer_stats'] = {
                'mean': np.mean(forward_transfers),
                'std': np.std(forward_transfers),
                'min': np.min(forward_transfers),
                'max': np.max(forward_transfers),
                'positive_rate': np.mean([score > 0 for score in forward_transfers]),
                'significant_rate': np.mean([abs(score) > self.threshold for score in forward_transfers])
            }
        
        if backward_transfers:
            results['backward_transfer_stats'] = {
                'mean': np.mean(backward_transfers),
                'std': np.std(backward_transfers),
                'min': np.min(backward_transfers),
                'max': np.max(backward_transfers),
                'positive_rate': np.mean([score > 0 for score in backward_transfers]),
                'retention_rate': np.mean([score > -self.threshold for score in backward_transfers])
            }
        
        # 元学习评估
        if len(self.meta_learning_scores) > 1:
            meta_learning_trend = np.polyfit(
                range(len(self.meta_learning_scores)), 
                self.meta_learning_scores, 1
            )[0]
            
            results['meta_learning_analysis'] = {
                'current_score': self.meta_learning_scores[-1],
                'improvement_trend': meta_learning_trend,
                'learning_efficiency': np.mean(self.meta_learning_scores),
                'consistency': 1.0 - np.std(self.meta_learning_scores)
            }
        
        return results
    
    def detect_skill_transfer(self, skill_representations: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        检测技能迁移
        
        Args:
            skill_representations: 各任务的技能表示
            
        Returns:
            技能迁移检测结果
        """
        self.logger.info("检测技能迁移")
        
        skill_transfers = {}
        task_ids = list(skill_representations.keys())
        
        if len(task_ids) < 2:
            return skill_transfers
        
        # 计算任务间的技能相似性
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids[i+1:], i+1):
                representation_i = skill_representations[task_i]
                representation_j = skill_representations[task_j]
                
                # 计算余弦相似性
                if representation_i.shape == representation_j.shape:
                    similarity = cosine_similarity([representation_i], [representation_j])[0, 0]
                    
                    # 如果相似性超过阈值，认为存在技能迁移
                    if abs(similarity) > 0.3:
                        transfer_key = f"task_{task_i}_to_task_{task_j}"
                        skill_transfers[transfer_key] = similarity
        
        return skill_transfers
    
    def analyze_transfer_patterns(self, current_task_id: int) -> TransferPattern:
        """
        分析迁移模式
        
        Args:
            current_task_id: 当前任务ID
            
        Returns:
            迁移模式分析结果
        """
        if current_task_id < 5:
            # 数据不足，返回默认模式
            return TransferPattern(
                pattern_type="stable",
                learning_acceleration=0.0,
                knowledge_retention=1.0,
                meta_learning_score=0.0,
                transferred_skills=[],
                affected_task_pairs=[],
                transfer_effectiveness=0.0
            )
        
        # 收集最近的迁移数据
        recent_forward_scores = []
        recent_backward_scores = []
        affected_pairs = []
        
        # 获取最近的迁移分数
        for i in range(max(0, current_task_id - 10), current_task_id):
            for j in range(i + 1, current_task_id):
                if i < self.transfer_matrix.shape[0] and j < self.transfer_matrix.shape[1]:
                    score = self.transfer_matrix[i, j]
                    if score != 0:
                        if j > i:  # 前向迁移
                            recent_forward_scores.append(score)
                        else:  # 后向迁移
                            recent_backward_scores.append(score)
                        affected_pairs.append((i, j))
        
        # 计算学习加速度
        learning_acceleration = 0.0
        if recent_forward_scores:
            learning_acceleration = np.mean(recent_forward_scores)
        
        # 计算知识保持
        knowledge_retention = 1.0
        if recent_backward_scores:
            # 平均性能保持率
            retention_scores = [max(0, 1 + score) for score in recent_backward_scores]
            knowledge_retention = np.mean(retention_scores)
        
        # 计算元学习分数
        meta_learning_score = 0.0
        if len(self.meta_learning_scores) > 1:
            meta_learning_score = self.meta_learning_scores[-1]
        
        # 识别模式类型
        if len(recent_forward_scores) >= 3:
            x = np.arange(len(recent_forward_scores))
            slope, _, r_value, p_value, _ = stats.linregress(x, recent_forward_scores)
            
            if abs(r_value) > 0.5 and p_value < 0.05:
                if slope > 0:
                    pattern_type = "increasing"
                else:
                    pattern_type = "decreasing"
            else:
                pattern_type = "stable"
        else:
            pattern_type = "stable"
        
        # 计算转化的技能
        transferred_skills = list(self.common_skills)
        
        # 计算迁移有效性
        transfer_effectiveness = 0.0
        if recent_forward_scores and recent_backward_scores:
            positive_forward = np.mean([score > 0 for score in recent_forward_scores])
            good_retention = np.mean([score > -self.threshold for score in recent_backward_scores])
            transfer_effectiveness = (positive_forward + good_retention) / 2.0
        
        pattern = TransferPattern(
            pattern_type=pattern_type,
            learning_acceleration=learning_acceleration,
            knowledge_retention=knowledge_retention,
            meta_learning_score=meta_learning_score,
            transferred_skills=transferred_skills,
            affected_task_pairs=affected_pairs,
            transfer_effectiveness=transfer_effectiveness
        )
        
        self.transfer_patterns[current_task_id] = pattern
        
        self.logger.info(f"迁移模式分析: {pattern_type}, 有效性: {transfer_effectiveness:.3f}")
        
        return pattern
    
    def generate_transfer_report(self, current_task_id: int) -> Dict[str, Any]:
        """
        生成迁移分析报告
        
        Args:
            current_task_id: 当前任务ID
            
        Returns:
            迁移分析报告
        """
        self.logger.info(f"生成任务 {current_task_id} 的迁移分析报告")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_task': current_task_id,
            'summary': {},
            'transfer_matrix': self.transfer_matrix[:current_task_id, :current_task_id].tolist(),
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # 汇总统计
        total_pairs = current_task_id * (current_task_id - 1) // 2
        if total_pairs > 0:
            # 前向迁移统计
            forward_scores = self.transfer_matrix[:current_task_id, :current_task_id]
            forward_scores = forward_scores[np.triu_indices_from(forward_scores, k=1)]
            forward_scores = forward_scores[forward_scores != 0]
            
            # 后向迁移统计
            backward_scores = self.transfer_matrix[:current_task_id, :current_task_id]
            backward_scores = backward_scores[np.tril_indices_from(backward_scores, k=-1)]
            backward_scores = backward_scores[backward_scores != 0]
            
            report['summary'] = {
                'total_task_pairs': total_pairs,
                'forward_transfer_pairs': len(forward_scores),
                'backward_transfer_pairs': len(backward_scores),
                'positive_forward_rate': np.mean(forward_scores > 0) if len(forward_scores) > 0 else 0,
                'good_backward_rate': np.mean(backward_scores > -self.threshold) if len(backward_scores) > 0 else 0,
                'average_forward_transfer': np.mean(forward_scores) if len(forward_scores) > 0 else 0,
                'average_backward_transfer': np.mean(backward_scores) if len(backward_scores) > 0 else 0
            }
        
        # 详细分析
        if current_task_id in self.transfer_patterns:
            pattern = self.transfer_patterns[current_task_id]
            report['detailed_analysis'] = {
                'pattern_type': pattern.pattern_type,
                'learning_acceleration': pattern.learning_acceleration,
                'knowledge_retention': pattern.knowledge_retention,
                'meta_learning_score': pattern.meta_learning_score,
                'transfer_effectiveness': pattern.transfer_effectiveness,
                'transferred_skills_count': len(pattern.transferred_skills)
            }
        
        # 生成建议
        if report['summary'].get('positive_forward_rate', 0) < 0.5:
            report['recommendations'].append("前向迁移效果不佳，建议优化元学习算法")
        
        if report['summary'].get('good_backward_rate', 0) < 0.8:
            report['recommendations'].append("后向保持率较低，建议加强正则化")
        
        # 技能迁移建议
        if len(self.common_skills) > 5:
            report['recommendations'].append("检测到多种技能，建议设计技能组合策略")
        
        # 元学习建议
        if len(self.meta_learning_scores) > 5:
            recent_trend = np.polyfit(range(max(0, len(self.meta_learning_scores) - 5), len(self.meta_learning_scores)), 
                                    self.meta_learning_scores[-5:], 1)[0]
            if recent_trend < 0:
                report['recommendations'].append("元学习趋势下降，建议调整学习策略")
        
        # 创建可视化
        self._create_transfer_visualizations(report)
        
        return report
    
    def _create_transfer_visualizations(self, report: Dict[str, Any]):
        """创建迁移分析可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('前后迁移分析报告', fontsize=16)
        
        # 1. 迁移矩阵热图
        transfer_matrix = np.array(report['transfer_matrix'])
        if transfer_matrix.size > 0:
            im = axes[0, 0].imshow(transfer_matrix, cmap='RdYlBu', aspect='auto', vmin=-0.5, vmax=0.5)
            axes[0, 0].set_xlabel('目标任务')
            axes[0, 0].set_ylabel('源任务')
            axes[0, 0].set_title('迁移矩阵')
            plt.colorbar(im, ax=axes[0, 0])
        
        # 2. 向前迁移趋势
        forward_data = []
        task_ids = []
        for i in range(len(self.learning_progress)):
            for j in range(i + 1, len(self.learning_progress)):
                if i < self.transfer_matrix.shape[0] and j < self.transfer_matrix.shape[1]:
                    score = self.transfer_matrix[i, j]
                    if score != 0:
                        forward_data.append(score)
                        task_ids.append(j)
        
        if forward_data:
            axes[0, 1].plot(task_ids, forward_data, 'b-', linewidth=2, label='前向迁移')
            axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[0, 1].set_xlabel('目标任务')
            axes[0, 1].set_ylabel('迁移分数')
            axes[0, 1].set_title('前向迁移趋势')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. 元学习进展
        if self.meta_learning_scores:
            axes[1, 0].plot(self.meta_learning_scores, 'g-', linewidth=2, label='元学习分数')
            axes[1, 0].set_xlabel('任务序列')
            axes[1, 0].set_ylabel('元学习分数')
            axes[1, 0].set_title('元学习进展')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. 技能转移分析
        if self.skill_evolution:
            skill_counts = [len(tasks) for tasks in self.skill_evolution.values()]
            skill_names = list(self.skill_evolution.keys())[:10]  # 显示前10个技能
            
            axes[1, 1].barh(skill_names, skill_counts[:10])
            axes[1, 1].set_xlabel('使用频率')
            axes[1, 1].set_ylabel('技能')
            axes[1, 1].set_title('技能使用频率')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'transfer_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self, filepath: str):
        """保存分析结果到文件
        
        Args:
            filepath: 文件路径
        """
        results = {
            'transfer_matrix': self.transfer_matrix.tolist(),
            'transfer_metrics': [
                {
                    'source_task': m.source_task,
                    'target_task': m.target_task,
                    'forward_transfer_score': m.forward_transfer_score,
                    'backward_transfer_score': m.backward_transfer_score,
                    'transfer_efficiency': m.transfer_efficiency,
                    'transfer_direction': m.transfer_direction,
                    'skill_transfer_score': m.skill_transfer_score,
                    'knowledge_preservation': m.knowledge_preservation,
                    'timestamp': m.timestamp
                }
                for m in self.transfer_metrics
            ],
            'transfer_patterns': {
                k: {
                    'pattern_type': v.pattern_type,
                    'learning_acceleration': v.learning_acceleration,
                    'knowledge_retention': v.knowledge_retention,
                    'meta_learning_score': v.meta_learning_score,
                    'transferred_skills': v.transferred_skills,
                    'affected_task_pairs': v.affected_task_pairs,
                    'transfer_effectiveness': v.transfer_effectiveness
                }
                for k, v in self.transfer_patterns.items()
            },
            'meta_learning_scores': self.meta_learning_scores,
            'skill_evolution': dict(self.skill_evolution),
            'baseline_performance': self.baseline_performance
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"迁移分析结果已保存到: {filepath}")


def main():
    """主函数 - 演示前后迁移分析器"""
    # 创建分析器
    analyzer = ForwardBackwardTransfer()
    
    # 模拟学习进度
    for task_id in range(15):
        # 模拟回合奖励（逐渐改进但有波动）
        base_performance = 0.5 + task_id * 0.02  # 基础性能提升
        noise = np.random.normal(0, 0.1, 50)  # 噪声
        episode_rewards = [base_performance + n for n in noise]
        
        # 记录学习进度
        analyzer.record_learning_progress(task_id, episode_rewards)
        
        # 计算前向迁移
        forward_score = analyzer.calculate_forward_transfer(task_id, episode_rewards)
        
        # 模拟重新评估的旧任务分数
        re_evaluated_scores = {i: analyzer.baseline_performance[i] * 0.9 for i in range(task_id)}
        backward_scores = analyzer.calculate_backward_transfer(task_id, re_evaluated_scores)
        
        print(f"任务 {task_id}: 前向迁移 {forward_score:.3f}, 平均后向迁移 {np.mean(backward_scores) if backward_scores else 0:.3f}")
        
        # 分析迁移模式
        pattern = analyzer.analyze_transfer_patterns(task_id)
        print(f"  迁移模式: {pattern.pattern_type}, 有效性: {pattern.transfer_effectiveness:.3f}")
    
    # 生成报告
    report = analyzer.generate_transfer_report(14)
    
    print("\n迁移分析报告摘要:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    # 保存结果
    analyzer.save_analysis_results("demo_transfer_analysis.json")
    print("\n分析结果已保存")


if __name__ == "__main__":
    main()