# -*- coding: utf-8 -*-
"""
跨域学习能力评估系统 - 主类
Cross-Domain Learning Assessment System - Main Class

该模块实现了跨域学习能力评估的核心功能，用于评估智能体在不同领域间的
知识迁移和学习能力。支持游戏、物理、社会等多个不同领域的学习评估。

主要功能：
- 跨域学习能力综合评估
- 多领域知识迁移分析  
- 学习策略优化
- 性能监控和反馈

作者: AI系统
日期: 2025-11-13
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .domain_adapter import DomainAdapter
from .transfer_analyzer import TransferAnalyzer  
from .learning_efficiency import LearningEfficiency
from .adaptation_metrics import AdaptationMetrics


@dataclass
class LearningResult:
    """学习结果数据类"""
    domain: str                    # 学习领域
    accuracy: float               # 准确率
    efficiency: float             # 学习效率
    transfer_speed: float         # 迁移速度
    adaptation_time: float        # 适应时间
    success_rate: float           # 成功率
    knowledge_retention: float    # 知识保持率
    transfer_quality: float       # 迁移质量
    learning_curve: List[float]   # 学习曲线
    metadata: Dict[str, Any]      # 额外元数据


class CrossDomainLearner:
    """
    跨域学习者主类
    
    该类负责协调各个组件，实现跨域学习能力的综合评估。
    通过集成领域适配器、迁移分析器、学习效率和适应指标等组件，
    提供完整的跨域学习评估解决方案。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化跨域学习者
        
        Args:
            config: 配置参数，包含各个组件的初始化参数
        """
        self.logger = self._setup_logger()
        self.config = config or self._get_default_config()
        
        # 初始化各组件
        self.domain_adapter = DomainAdapter(self.config.get('domain_adapter', {}))
        self.transfer_analyzer = TransferAnalyzer(self.config.get('transfer_analyzer', {}))
        self.learning_efficiency = LearningEfficiency(self.config.get('learning_efficiency', {}))
        self.adaptation_metrics = AdaptationMetrics(self.config.get('adaptation_metrics', {}))
        
        # 学习状态跟踪
        self.learning_history = []
        self.current_session = None
        self.domain_models = {}
        
        # 预定义领域定义
        self.defined_domains = {
            'game': {
                'name': '游戏领域',
                'description': '策略游戏、解谜游戏等认知任务',
                'characteristics': ['策略思维', '模式识别', '反应速度'],
                'complexity_level': 'medium'
            },
            'physics': {
                'name': '物理领域', 
                'description': '物理规律理解和应用',
                'characteristics': ['因果推理', '空间想象', '数学建模'],
                'complexity_level': 'high'
            },
            'social': {
                'name': '社会领域',
                'description': '人际交往和社会认知',
                'characteristics': ['情感理解', '沟通能力', '情境判断'],
                'complexity_level': 'medium'
            },
            'language': {
                'name': '语言领域',
                'description': '语言理解和生成',
                'characteristics': ['语义理解', '语法规则', '语言生成'],
                'complexity_level': 'high'
            },
            'spatial': {
                'name': '空间领域',
                'description': '空间认知和几何推理',
                'characteristics': ['空间定位', '几何变换', '路径规划'],
                'complexity_level': 'medium'
            }
        }
        
        self.logger.info("跨域学习者初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('CrossDomainLearner')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'domains': ['game', 'physics', 'social', 'language', 'spatial'],
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100,
            'transfer_threshold': 0.7,
            'efficiency_weight': 0.3,
            'speed_weight': 0.3,
            'accuracy_weight': 0.4,
            'parallel_processing': True,
            'max_workers': 4,
            'enable_async': True,
            'monitor_interval': 10,
            'save_intermediate': True
        }
    
    async def assess_cross_domain_learning(self, 
                                          source_domains: List[str], 
                                          target_domains: List[str],
                                          evaluation_tasks: Dict[str, Any],
                                          learner_agent: Any = None) -> Dict[str, Any]:
        """
        评估跨域学习能力
        
        这是系统的核心方法，用于评估智能体在不同领域间的知识迁移和学习能力。
        
        Args:
            source_domains: 源领域列表
            target_domains: 目标领域列表  
            evaluation_tasks: 评估任务配置
            learner_agent: 被评估的学习智能体
            
        Returns:
            Dict: 跨域学习评估结果
        """
        self.logger.info(f"开始跨域学习评估: {source_domains} -> {target_domains}")
        
        start_time = datetime.now()
        
        try:
            # 1. 领域相似性分析
            similarity_scores = await self.analyze_domain_similarity(source_domains, target_domains)
            
            # 2. 并行评估各目标领域的学习效果
            learning_results = {}
            
            if self.config['parallel_processing'] and self.config['enable_async']:
                # 异步并行处理
                tasks = []
                for target_domain in target_domains:
                    task = self._evaluate_single_domain_transfer(
                        source_domains, target_domain, evaluation_tasks, learner_agent
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, target_domain in enumerate(target_domains):
                    if isinstance(results[i], Exception):
                        self.logger.error(f"领域 {target_domain} 评估失败: {results[i]}")
                        learning_results[target_domain] = {'error': str(results[i])}
                    else:
                        learning_results[target_domain] = results[i]
            else:
                # 串行处理
                for target_domain in target_domains:
                    result = await self._evaluate_single_domain_transfer(
                        source_domains, target_domain, evaluation_tasks, learner_agent
                    )
                    learning_results[target_domain] = result
            
            # 3. 综合分析
            overall_score = self._calculate_overall_performance(learning_results, similarity_scores)
            
            # 4. 生成优化建议
            optimization_suggestions = await self.optimize_cross_domain_strategy(
                source_domains, target_domains, learning_results, similarity_scores
            )
            
            # 5. 构建最终结果
            assessment_result = {
                'evaluation_id': f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'source_domains': source_domains,
                'target_domains': target_domains,
                'domain_similarity': similarity_scores,
                'learning_results': learning_results,
                'overall_performance': overall_score,
                'optimization_suggestions': optimization_suggestions,
                'evaluation_duration': (datetime.now() - start_time).total_seconds(),
                'metadata': {
                    'config': self.config,
                    'domains_defined': len(self.defined_domains)
                }
            }
            
            # 保存评估结果
            self.learning_history.append(assessment_result)
            if self.config['save_intermediate']:
                await self._save_assessment_result(assessment_result)
            
            self.logger.info(f"跨域学习评估完成，总体得分: {overall_score['overall_score']:.3f}")
            
            return assessment_result
            
        except Exception as e:
            self.logger.error(f"跨域学习评估过程中发生错误: {str(e)}")
            raise
    
    async def _evaluate_single_domain_transfer(self,
                                             source_domains: List[str],
                                             target_domain: str,
                                             evaluation_tasks: Dict[str, Any],
                                             learner_agent: Any) -> Dict[str, Any]:
        """评估单个领域的学习迁移效果"""
        
        # 1. 领域适配
        adapted_knowledge = await self.domain_adapter.adapt_knowledge(
            source_domains, target_domain, learner_agent
        )
        
        # 2. 迁移效率测量
        transfer_metrics = await self.transfer_analyzer.measure_transfer_efficiency(
            source_domains, target_domain, adapted_knowledge
        )
        
        # 3. 学习效率评估
        efficiency_metrics = await self.learning_efficiency.evaluate_learning_efficiency(
            target_domain, adapted_knowledge, evaluation_tasks.get(target_domain, {})
        )
        
        # 4. 适应速度评估
        adaptation_metrics = await self.adaptation_metrics.evaluate_adaptation_speed(
            target_domain, adapted_knowledge, evaluation_tasks.get(target_domain, {})
        )
        
        # 5. 整合结果
        result = LearningResult(
            domain=target_domain,
            accuracy=efficiency_metrics.get('accuracy', 0.0),
            efficiency=efficiency_metrics.get('efficiency_score', 0.0),
            transfer_speed=transfer_metrics.get('transfer_speed', 0.0),
            adaptation_time=adaptation_metrics.get('adaptation_time', 0.0),
            success_rate=adaptation_metrics.get('success_rate', 0.0),
            knowledge_retention=transfer_metrics.get('knowledge_retention', 0.0),
            transfer_quality=transfer_metrics.get('transfer_quality', 0.0),
            learning_curve=efficiency_metrics.get('learning_curve', []),
            metadata={
                'source_domains': source_domains,
                'adaptation_process': adapted_knowledge.get('process_metrics', {}),
                'transfer_analysis': transfer_metrics,
                'efficiency_analysis': efficiency_metrics,
                'adaptation_analysis': adaptation_metrics
            }
        )
        
        return asdict(result)
    
    async def analyze_domain_similarity(self,
                                      source_domains: List[str],
                                      target_domains: List[str]) -> Dict[str, Dict[str, float]]:
        """
        分析领域相似性
        
        使用多种相似性度量方法计算不同领域间的相似度，包括：
        - 特征相似性
        - 语义相似性  
        - 复杂度相似性
        - 认知负载相似性
        
        Args:
            source_domains: 源领域列表
            target_domains: 目标领域列表
            
        Returns:
            Dict: 领域相似性矩阵
        """
        self.logger.info("开始分析领域相似性")
        
        similarity_matrix = {}
        
        for target in target_domains:
            similarity_matrix[target] = {}
            
            for source in source_domains:
                if source not in self.defined_domains or target not in self.defined_domains:
                    continue
                
                # 计算特征相似性
                source_chars = self.defined_domains[source]['characteristics']
                target_chars = self.defined_domains[target]['characteristics']
                feature_similarity = self._calculate_characteristic_similarity(source_chars, target_chars)
                
                # 计算复杂度相似性
                source_complexity = self._get_complexity_score(self.defined_domains[source]['complexity_level'])
                target_complexity = self._get_complexity_score(self.defined_domains[target]['complexity_level'])
                complexity_similarity = 1.0 - abs(source_complexity - target_complexity)
                
                # 计算领域语义相似性
                semantic_similarity = await self._calculate_semantic_similarity(source, target)
                
                # 综合相似性评分
                overall_similarity = (
                    feature_similarity * 0.4 +
                    complexity_similarity * 0.3 +
                    semantic_similarity * 0.3
                )
                
                similarity_matrix[target][source] = {
                    'overall_similarity': overall_similarity,
                    'feature_similarity': feature_similarity,
                    'complexity_similarity': complexity_similarity,
                    'semantic_similarity': semantic_similarity
                }
        
        self.logger.info("领域相似性分析完成")
        return similarity_matrix
    
    def _calculate_characteristic_similarity(self, 
                                           source_chars: List[str], 
                                           target_chars: List[str]) -> float:
        """计算特征相似性"""
        if not source_chars or not target_chars:
            return 0.0
        
        # 计算特征交集比例
        source_set = set(source_chars)
        target_set = set(target_chars)
        intersection = len(source_set.intersection(target_set))
        union = len(source_set.union(target_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_complexity_score(self, complexity_level: str) -> float:
        """获取复杂度评分"""
        complexity_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'very_high': 1.0
        }
        return complexity_map.get(complexity_level, 0.5)
    
    async def _calculate_semantic_similarity(self, domain1: str, domain2: str) -> float:
        """计算领域语义相似性"""
        # 简化的语义相似性计算
        # 在实际应用中可以使用词向量、预训练模型等更复杂的方法
        
        domain_keywords = {
            'game': ['策略', '规则', '胜利', '竞争'],
            'physics': ['力', '运动', '能量', '规律'],
            'social': ['人际', '交流', '情感', '合作'],
            'language': ['词汇', '语法', '语义', '表达'],
            'spatial': ['空间', '位置', '几何', '方向']
        }
        
        keywords1 = domain_keywords.get(domain1, [])
        keywords2 = domain_keywords.get(domain2, [])
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # 计算关键词重叠度
        common_keywords = len(set(keywords1).intersection(set(keywords2)))
        total_keywords = len(set(keywords1).union(set(keywords2)))
        
        return common_keywords / total_keywords if total_keywords > 0 else 0.0
    
    def _calculate_overall_performance(self, 
                                     learning_results: Dict[str, Any],
                                     similarity_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算总体性能得分"""
        
        # 提取所有领域的性能数据
        all_accuracies = []
        all_efficiencies = []
        all_transfer_speeds = []
        all_success_rates = []
        
        for domain, result in learning_results.items():
            if 'error' not in result:
                all_accuracies.append(result['accuracy'])
                all_efficiencies.append(result['efficiency'])
                all_transfer_speeds.append(result['transfer_speed'])
                all_success_rates.append(result['success_rate'])
        
        # 计算平均性能
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        avg_efficiency = np.mean(all_efficiencies) if all_efficiencies else 0.0
        avg_speed = np.mean(all_transfer_speeds) if all_transfer_speeds else 0.0
        avg_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
        
        # 计算总体得分
        weights = self.config
        overall_score = (
            avg_accuracy * weights['accuracy_weight'] +
            avg_efficiency * weights['efficiency_weight'] +
            avg_speed * weights['speed_weight']
        )
        
        return {
            'overall_score': overall_score,
            'accuracy_score': avg_accuracy,
            'efficiency_score': avg_efficiency,
            'speed_score': avg_speed,
            'success_rate': avg_success_rate,
            'domain_count': len(learning_results),
            'successful_domains': len([r for r in learning_results.values() if 'error' not in r]),
            'performance_summary': {
                'excellent': len([s for s in all_accuracies if s >= 0.9]),
                'good': len([s for s in all_accuracies if 0.7 <= s < 0.9]),
                'fair': len([s for s in all_accuracies if 0.5 <= s < 0.7]),
                'poor': len([s for s in all_accuracies if s < 0.5])
            }
        }
    
    async def optimize_cross_domain_strategy(self,
                                           source_domains: List[str],
                                           target_domains: List[str], 
                                           learning_results: Dict[str, Any],
                                           similarity_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        优化跨域学习策略
        
        基于评估结果生成个性化的跨域学习优化建议。
        
        Args:
            source_domains: 源领域
            target_domains: 目标领域
            learning_results: 学习结果
            similarity_scores: 相似性评分
            
        Returns:
            Dict: 优化策略建议
        """
        suggestions = {
            'domain_selection': [],
            'learning_path': [],
            'parameter_adjustment': [],
            'improvement_areas': [],
            'best_practices': []
        }
        
        # 1. 领域选择优化
        for target_domain in target_domains:
            if target_domain in similarity_scores:
                best_sources = sorted(
                    similarity_scores[target_domain].items(),
                    key=lambda x: x[1]['overall_similarity'],
                    reverse=True
                )
                
                if best_sources:
                    best_source = best_sources[0]
                    suggestions['domain_selection'].append({
                        'target_domain': target_domain,
                        'recommended_source': best_source[0],
                        'similarity_score': best_source[1]['overall_similarity'],
                        'reason': f"与{best_source[0]}领域相似性最高({best_source[1]['overall_similarity']:.3f})"
                    })
        
        # 2. 学习路径优化
        sorted_domains = sorted(
            learning_results.items(),
            key=lambda x: x[1].get('accuracy', 0.0),
            reverse=True
        )
        
        for i, (domain, result) in enumerate(sorted_domains):
            suggestions['learning_path'].append({
                'sequence': i + 1,
                'domain': domain,
                'reason': f"性能得分 {result.get('accuracy', 0.0):.3f}"
            })
        
        # 3. 参数调整建议
        low_performance_domains = [
            domain for domain, result in learning_results.items()
            if result.get('accuracy', 0.0) < 0.6
        ]
        
        if low_performance_domains:
            suggestions['parameter_adjustment'].extend([
                "增加学习率以提高收敛速度",
                "扩大训练批次大小以稳定学习过程", 
                "增加训练轮数以确保充分学习"
            ])
        
        # 4. 改进领域识别
        improvement_areas = []
        for domain, result in learning_results.items():
            if result.get('transfer_speed', 0.0) < 0.5:
                improvement_areas.append(f"{domain}领域的迁移速度")
            if result.get('efficiency', 0.0) < 0.6:
                improvement_areas.append(f"{domain}领域的学习效率")
        
        suggestions['improvement_areas'] = improvement_areas
        
        # 5. 最佳实践建议
        suggestions['best_practices'] = [
            "优先选择相似性高的领域进行迁移学习",
            "建立领域间的知识映射关系",
            "监控学习过程，及时调整学习策略",
            "保留高价值知识的记忆痕迹",
            "定期评估和更新跨域学习策略"
        ]
        
        return suggestions
    
    async def measure_transfer_efficiency(self,
                                        source_domains: List[str],
                                        target_domains: List[str],
                                        knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """
        测量迁移效率
        
        Args:
            source_domains: 源领域列表
            target_domains: 目标领域列表  
            knowledge_base: 知识库
            
        Returns:
            Dict: 迁移效率测量结果
        """
        # 委托给TransferAnalyzer组件
        return await self.transfer_analyzer.measure_transfer_efficiency(
            source_domains, target_domains, knowledge_base
        )
    
    async def evaluate_adaptation_speed(self,
                                      target_domain: str,
                                      transferred_knowledge: Dict[str, Any],
                                      adaptation_tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估适应速度
        
        Args:
            target_domain: 目标领域
            transferred_knowledge: 迁移的知识
            adaptation_tasks: 适应任务
            
        Returns:
            Dict: 适应速度评估结果
        """
        # 委托给AdaptationMetrics组件
        return await self.adaptation_metrics.evaluate_adaptation_speed(
            target_domain, transferred_knowledge, adaptation_tasks
        )
    
    async def _save_assessment_result(self, result: Dict[str, Any]) -> None:
        """保存评估结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cross_domain_assessment_{timestamp}.json"
        filepath = f"worlds/real/results/{filename}"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"评估结果已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存评估结果失败: {str(e)}")
    
    def get_learning_history(self) -> List[Dict[str, Any]]:
        """获取学习历史记录"""
        return self.learning_history.copy()
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """获取领域统计信息"""
        if not self.learning_history:
            return {}
        
        domain_stats = {}
        for record in self.learning_history:
            for domain in record.get('target_domains', []):
                if domain not in domain_stats:
                    domain_stats[domain] = {
                        'count': 0,
                        'total_accuracy': 0.0,
                        'total_efficiency': 0.0,
                        'avg_performance': 0.0
                    }
                
                domain_stats[domain]['count'] += 1
                if 'learning_results' in record and domain in record['learning_results']:
                    result = record['learning_results'][domain]
                    if 'error' not in result:
                        domain_stats[domain]['total_accuracy'] += result.get('accuracy', 0.0)
                        domain_stats[domain]['total_efficiency'] += result.get('efficiency', 0.0)
        
        # 计算平均值
        for domain, stats in domain_stats.items():
            if stats['count'] > 0:
                stats['avg_accuracy'] = stats['total_accuracy'] / stats['count']
                stats['avg_efficiency'] = stats['total_efficiency'] / stats['count']
                stats['avg_performance'] = (stats['avg_accuracy'] + stats['avg_efficiency']) / 2
        
        return domain_stats
    
    def reset_session(self) -> None:
        """重置当前学习会话"""
        self.current_session = None
        self.logger.info("学习会话已重置")


# 便捷函数
def create_cross_domain_learner(config: Optional[Dict[str, Any]] = None) -> CrossDomainLearner:
    """创建跨域学习者实例的便捷函数"""
    return CrossDomainLearner(config)


async def run_cross_domain_assessment(source_domains: List[str],
                                    target_domains: List[str],
                                    learner_agent: Any = None,
                                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """运行跨域学习评估的便捷函数"""
    learner = create_cross_domain_learner(config)
    
    # 模拟评估任务
    evaluation_tasks = {
        domain: {'tasks': ['classification', 'reasoning', 'prediction']}
        for domain in target_domains
    }
    
    return await learner.assess_cross_domain_learning(
        source_domains, target_domains, evaluation_tasks, learner_agent
    )


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建跨域学习者
        learner = CrossDomainLearner({
            'domains': ['game', 'physics', 'social'],
            'learning_rate': 0.01,
            'parallel_processing': True
        })
        
        # 运行评估
        result = await learner.assess_cross_domain_learning(
            source_domains=['game'],
            target_domains=['physics', 'social'],
            evaluation_tasks={'physics': {'tasks': ['mechanics']}},
            learner_agent=None
        )
        
        print(f"评估结果: {result['overall_performance']['overall_score']:.3f}")
    
    # 运行示例
    # asyncio.run(main())