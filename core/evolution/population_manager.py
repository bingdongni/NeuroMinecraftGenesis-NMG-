"""
种群管理器 - 负责种群初始化、状态跟踪和管理

本模块实现了智能体种群的完整生命周期管理，包括初始化、状态跟踪、
性能监控和统计分析。支持动态环境复杂度调整和性能优化。

主要功能:
- 智能体种群初始化和生成
- 实时状态监控和性能跟踪
- 统计分析和报告生成
- 环境复杂度动态调整
- 种群多样性维护

作者: NeuroMinecraftGenesis Team
日期: 2025-11-13
"""

import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from datetime import datetime


@dataclass
class Agent:
    """
    智能体数据类
    
    存储单个智能体的完整信息，包括学习规则、适应度、统计信息等。
    
    Attributes:
        id: 智能体唯一标识符
        learning_rule: 学习规则参数（50维浮点向量）
        fitness: 适应度元组（游戏得分、学习速度、泛化能力）
        generation: 当前代数
        performance_history: 历史性能记录
        complexity_level: 环境复杂度级别
        age: 智能体年龄（代数）
    """
    id: int
    learning_rule: np.ndarray
    fitness: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    generation: int = 0
    performance_history: List[float] = field(default_factory=list)
    complexity_level: int = 1
    age: int = 0
    
    def update_performance(self, fitness: Tuple[float, float, float]):
        """更新智能体性能记录"""
        self.fitness = fitness
        avg_performance = np.mean(fitness)
        self.performance_history.append(avg_performance)
        self.age += 1
    
    def get_avg_performance(self) -> float:
        """获取平均性能"""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history)
    
    def is_elite(self, threshold_percentile: float = 0.8) -> bool:
        """判断是否为精英个体"""
        return self.get_avg_performance() >= threshold_percentile


class PopulationManager:
    """
    智能体种群管理器
    
    负责管理整个种群的初始化、状态跟踪、统计分析和性能监控。
    支持动态环境复杂度调整和实时性能监控。
    
    Attributes:
        population_size: 种群大小（默认16）
        rule_dim: 学习规则维度（默认50）
        agents: 智能体列表
        generation: 当前代数
        elite_agents: 精英个体集合
        complexity_manager: 环境复杂度管理器
        performance_history: 性能历史记录
        diversity_tracker: 多样性跟踪器
        logger: 日志记录器
    """
    
    def __init__(self, population_size: int = 16, rule_dim: int = 50):
        """
        初始化种群管理器
        
        Args:
            population_size: 种群大小，默认16个智能体
            rule_dim: 学习规则向量维度，默认50维
        """
        self.population_size = population_size
        self.rule_dim = rule_dim
        self.agents: List[Agent] = []
        self.generation = 0
        self.elite_agents = set()
        
        # 环境复杂度管理
        self.current_complexity = 1
        self.complexity_progression = {
            10: 2,   # 第10代开始复杂度2
            20: 3,   # 第20代开始复杂度3
            30: 4,   # 第30代开始复杂度4
            40: 5,   # 第40代开始复杂度5
        }
        
        # 统计和监控
        self.performance_history = []
        self.diversity_history = []
        self.convergence_metrics = []
        
        # 多样性跟踪
        self.diversity_tracker = DiversityTracker(rule_dim)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录器"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def initialize_population(self) -> List[Agent]:
        """
        初始化智能体种群
        
        生成指定数量的智能体，每个智能体包含随机初始化的学习规则。
        学习规则向量使用正态分布初始化，支持不同的初始化策略。
        
        Returns:
            List[Agent]: 初始化后的智能体列表
            
        Raises:
            ValueError: 当种群大小或规则维度设置不合理时
        """
        if self.population_size <= 0:
            raise ValueError(f"种群大小必须大于0，当前值: {self.population_size}")
        
        if self.rule_dim <= 0:
            raise ValueError(f"规则维度必须大于0，当前值: {self.rule_dim}")
        
        self.logger.info(f"初始化种群 - 大小: {self.population_size}, 规则维度: {self.rule_dim}")
        
        agents = []
        
        for i in range(self.population_size):
            # 生成随机学习规则向量
            learning_rule = self._generate_initial_rule()
            
            # 创建智能体
            agent = Agent(
                id=i,
                learning_rule=learning_rule,
                generation=0,
                complexity_level=self.current_complexity
            )
            
            agents.append(agent)
            
            if i == 0:
                self.logger.info(f"第一个智能体学习规则示例: {learning_rule[:5]}...")
        
        self.agents = agents
        self._update_diversity_metrics()
        
        self.logger.info(f"种群初始化完成，包含 {len(agents)} 个智能体")
        return agents
    
    def _generate_initial_rule(self) -> np.ndarray:
        """
        生成初始学习规则向量
        
        使用正态分布生成50维学习规则向量，参数经过调优以保证
        初始种群的多样性和探索能力。
        
        Returns:
            np.ndarray: 50维学习规则向量
        """
        # 使用不同的初始化策略确保多样性
        strategies = ['uniform', 'normal', 'lognormal']
        strategy = random.choice(strategies)
        
        if strategy == 'uniform':
            # 均匀分布初始化
            rule = np.random.uniform(-1.0, 1.0, self.rule_dim)
        elif strategy == 'normal':
            # 正态分布初始化
            rule = np.random.normal(0.0, 0.5, self.rule_dim)
        else:  # lognormal
            # 对数正态分布初始化
            rule = np.random.lognormal(0.0, 0.3, self.rule_dim)
            # 调整到[-1, 1]范围
            rule = np.clip(rule * 2 - 1, -1.0, 1.0)
        
        return rule
    
    def get_agents_by_fitness(self) -> List[Tuple[Agent, float]]:
        """
        按适应度排序智能体
        
        Returns:
            List[Tuple[Agent, float]]: (智能体, 平均适应度) 按适应度降序排列
        """
        fitness_pairs = [(agent, agent.get_avg_performance()) for agent in self.agents]
        return sorted(fitness_pairs, key=lambda x: x[1], reverse=True)
    
    def get_elite_agents(self, elite_ratio: float = 0.25) -> List[Agent]:
        """
        获取精英智能体
        
        Args:
            elite_ratio: 精英比例，默认25%
            
        Returns:
            List[Agent]: 精英智能体列表
        """
        if not self.agents:
            return []
        
        sorted_agents = self.get_agents_by_fitness()
        num_elites = max(1, int(len(sorted_agents) * elite_ratio))
        
        elites = [agent for agent, _ in sorted_agents[:num_elites]]
        self.elite_agents = {agent.id for agent in elites}
        
        return elites
    
    def update_agent_fitness(self, agent_id: int, fitness: Tuple[float, float, float]):
        """
        更新智能体适应度
        
        Args:
            agent_id: 智能体ID
            fitness: 新的适应度元组
        """
        for agent in self.agents:
            if agent.id == agent_id:
                agent.update_performance(fitness)
                break
    
    def get_best_agents(self, count: int = 5) -> List[Agent]:
        """
        获取最佳智能体
        
        Args:
            count: 返回的智能体数量
            
        Returns:
            List[Agent]: 最佳智能体列表
        """
        sorted_agents = self.get_agents_by_fitness()
        return [agent for agent, _ in sorted_agents[:count]]
    
    def get_worst_agents(self, count: int = 5) -> List[Agent]:
        """
        获取最差智能体
        
        Args:
            count: 返回的智能体数量
            
        Returns:
            List[Agent]: 最差智能体列表
        """
        sorted_agents = self.get_agents_by_fitness()
        return [agent for agent, _ in sorted_agents[-count:]]
    
    def update_complexity_level(self):
        """更新环境复杂度级别"""
        new_complexity = self.complexity_progression.get(self.generation, self.current_complexity)
        
        if new_complexity != self.current_complexity:
            self.logger.info(f"环境复杂度提升: {self.current_complexity} -> {new_complexity}")
            self.current_complexity = new_complexity
            
            # 更新所有智能体的复杂度级别
            for agent in self.agents:
                agent.complexity_level = new_complexity
    
    def get_complexity_level(self) -> int:
        """获取当前环境复杂度级别"""
        return self.current_complexity
    
    def calculate_population_diversity(self) -> float:
        """
        计算种群多样性
        
        使用多维欧氏距离平均值来衡量种群在特征空间中的分散程度。
        
        Returns:
            float: 多样性指标（0-1之间，值越大越多样）
        """
        return self.diversity_tracker.calculate_diversity(self.agents)
    
    def _update_diversity_metrics(self):
        """更新多样性指标"""
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        return diversity
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        获取种群性能统计
        
        Returns:
            Dict[str, Any]: 包含各种统计指标的字典
        """
        if not self.agents:
            return {}
        
        # 计算各个目标的统计信息
        game_scores = [agent.fitness[0] for agent in self.agents]
        learning_speeds = [agent.fitness[1] for agent in self.agents]
        generalization = [agent.fitness[2] for agent in self.agents]
        avg_performance = [agent.get_avg_performance() for agent in self.agents]
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.agents),
            'diversity': self.calculate_population_diversity(),
            'complexity_level': self.current_complexity,
            'performance_stats': {
                'game_score': {
                    'mean': np.mean(game_scores),
                    'std': np.std(game_scores),
                    'min': np.min(game_scores),
                    'max': np.max(game_scores)
                },
                'learning_speed': {
                    'mean': np.mean(learning_speeds),
                    'std': np.std(learning_speeds),
                    'min': np.min(learning_speeds),
                    'max': np.max(learning_speeds)
                },
                'generalization': {
                    'mean': np.mean(generalization),
                    'std': np.std(generalization),
                    'min': np.min(generalization),
                    'max': np.max(generalization)
                },
                'overall_performance': {
                    'mean': np.mean(avg_performance),
                    'std': np.std(avg_performance),
                    'min': np.min(avg_performance),
                    'max': np.max(avg_performance)
                }
            },
            'elite_count': len(self.elite_agents),
            'avg_age': np.mean([agent.age for agent in self.agents])
        }
        
        # 添加收敛指标
        if len(self.performance_history) > 5:
            recent_performance = self.performance_history[-10:]
            convergence_slope = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            stats['convergence'] = {
                'trend_slope': convergence_slope,
                'is_converging': convergence_slope < 0.01,
                'improvement_rate': (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
            }
        
        return stats
    
    def log_statistics(self):
        """记录种群统计信息到日志"""
        stats = self.get_performance_statistics()
        
        self.logger.info(
            f"种群统计 - 代数: {stats['generation']}, "
            f"多样性: {stats['diversity']:.4f}, "
            f"复杂度: {stats['complexity_level']}, "
            f"精英数: {stats['elite_count']}, "
            f"平均性能: {stats['performance_stats']['overall_performance']['mean']:.4f}"
        )
        
        # 记录性能趋势
        if 'convergence' in stats:
            conv = stats['convergence']
            self.logger.info(
                f"收敛状态 - 斜率: {conv['trend_slope']:.6f}, "
                f"改善率: {conv['improvement_rate']:.6f}"
            )
    
    def save_population_state(self, filepath: str):
        """
        保存种群状态到文件
        
        Args:
            filepath: 保存路径
        """
        state = {
            'generation': self.generation,
            'population_size': self.population_size,
            'rule_dim': self.rule_dim,
            'current_complexity': self.current_complexity,
            'agents': [
                {
                    'id': agent.id,
                    'learning_rule': agent.learning_rule.tolist(),
                    'fitness': agent.fitness,
                    'generation': agent.generation,
                    'performance_history': agent.performance_history,
                    'complexity_level': agent.complexity_level,
                    'age': agent.age
                }
                for agent in self.agents
            ],
            'performance_history': self.performance_history,
            'diversity_history': self.diversity_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"种群状态已保存到: {filepath}")
    
    def load_population_state(self, filepath: str):
        """
        从文件加载种群状态
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.generation = state['generation']
        self.population_size = state['population_size']
        self.rule_dim = state['rule_dim']
        self.current_complexity = state['current_complexity']
        self.performance_history = state.get('performance_history', [])
        self.diversity_history = state.get('diversity_history', [])
        
        self.agents = []
        for agent_data in state['agents']:
            agent = Agent(
                id=agent_data['id'],
                learning_rule=np.array(agent_data['learning_rule']),
                fitness=tuple(agent_data['fitness']),
                generation=agent_data['generation'],
                performance_history=agent_data['performance_history'],
                complexity_level=agent_data['complexity_level'],
                age=agent_data['age']
            )
            self.agents.append(agent)
        
        self.logger.info(f"种群状态已从文件加载: {filepath}")


class DiversityTracker:
    """
    种群多样性跟踪器
    
    专门用于跟踪和计算种群在特征空间中的多样性指标。
    支持多种多样性计算方法。
    """
    
    def __init__(self, rule_dim: int):
        self.rule_dim = rule_dim
    
    def calculate_diversity(self, agents: List[Agent]) -> float:
        """
        计算种群多样性
        
        使用平均成对距离来衡量种群在特征空间中的分散程度。
        
        Args:
            agents: 智能体列表
            
        Returns:
            float: 多样性指标 [0, 1]
        """
        if len(agents) < 2:
            return 0.0
        
        # 提取所有学习规则
        rules = np.array([agent.learning_rule for agent in agents])
        
        # 计算成对距离
        distances = []
        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                dist = np.linalg.norm(rules[i] - rules[j])
                distances.append(dist)
        
        # 计算平均距离并标准化
        avg_distance = np.mean(distances)
        
        # 假设规则向量在[-1, 1]范围内，最大距离约为sqrt(rule_dim)
        max_possible_distance = np.sqrt(self.rule_dim) * 2  # 每个维度最大差异为2
        
        diversity = avg_distance / max_possible_distance
        return min(1.0, max(0.0, diversity))
    
    def calculate_fitness_diversity(self, fitnesses: List[Tuple[float, float, float]]) -> float:
        """
        计算适应度多样性
        
        Args:
            fitnesses: 适应度列表
            
        Returns:
            float: 适应度多样性指标
        """
        if len(fitnesses) < 2:
            return 0.0
        
        fitness_array = np.array(fitnesses)
        distances = []
        
        for i in range(len(fitness_array)):
            for j in range(i + 1, len(fitness_array)):
                dist = np.linalg.norm(fitness_array[i] - fitness_array[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # 标准化（假设适应度在合理范围内）
        normalized_diversity = avg_distance / 3.0  # 3个目标，假设每个目标范围为1
        return min(1.0, max(0.0, normalized_diversity))