#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiscoRL - 自主强化学习算法发现系统
DiscoRL - Autonomous Reinforcement Learning Algorithm Discovery System

该系统实现了完整的算法发现流程：
1. 数千个AI智能体的生存挑战赛机制
2. 自主算法发现和评估系统  
3. 达尔文式优胜劣汰进化机制
4. 算法交叉组合和变异操作
5. 泛化能力测试和评估（ProcGen环境）

Author: DiscoRL Team
Date: 2025-11-13
"""

import numpy as np
import random
import json
import pickle
import time
import copy
import threading
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用简化版本")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("警告: Matplotlib未安装，将禁用可视化功能")


@dataclass
class AgentConfig:
    """智能体配置"""
    id: str
    algorithm_type: str = "neural_network"  # neural_network, policy_gradient, q_learning, genetic
    neural_architecture: str = "mlp"        # mlp, cnn, rnn, transformer
    hidden_size: int = 64
    learning_rate: float = 0.001
    memory_size: int = 1000
    exploration_rate: float = 0.1
    mutation_rate: float = 0.05
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    
    def copy(self) -> 'AgentConfig':
        return copy.deepcopy(self)


@dataclass
class FitnessResult:
    """适应度评估结果"""
    agent_id: str
    score: float
    survival_time: float
    tasks_completed: List[str]
    efficiency: float
    exploration_score: float
    robustness_score: float
    generalization_score: float = 0.0
    
    def total_fitness(self) -> float:
        """计算总适应度分数"""
        return (
            self.score * 0.4 +
            self.survival_time * 0.2 +
            self.efficiency * 0.15 +
            self.exploration_score * 0.1 +
            self.robustness_score * 0.1 +
            self.generalization_score * 0.05
        )


class Environment:
    """挑战环境类 - 模拟各种生存挑战"""
    
    def __init__(self, env_type: str = "mixed"):
        self.env_type = env_type
        self.agents_in_env = []
        self.obstacles = self._generate_obstacles()
        self.resources = self._generate_resources()
        
    def _generate_obstacles(self) -> List[Dict]:
        """生成环境障碍物"""
        obstacles = []
        for i in range(random.randint(5, 15)):
            obstacles.append({
                'pos': (random.random() * 100, random.random() * 100),
                'size': random.uniform(5, 20),
                'type': random.choice(['danger', 'moving', 'temporary']),
                'damage': random.uniform(0.1, 0.5)
            })
        return obstacles
    
    def _generate_resources(self) -> List[Dict]:
        """生成环境资源"""
        resources = []
        for i in range(random.randint(10, 30)):
            resources.append({
                'pos': (random.random() * 100, random.random() * 100),
                'type': random.choice(['energy', 'intellect', 'health', 'speed']),
                'value': random.uniform(0.1, 1.0),
                'replenish_time': random.randint(50, 200)
            })
        return resources
    
    def get_state(self, agent_pos: Tuple[float, float, float] = None) -> np.ndarray:
        """获取环境状态"""
        if agent_pos is None:
            agent_pos = (50.0, 50.0, 0.0)
        
        # 简化状态表示
        nearby_obstacles = len([o for o in self.obstacles 
                               if np.linalg.norm(np.array(o['pos'][:2]) - np.array(agent_pos[:2])) < 10])
        nearby_resources = len([r for r in self.resources 
                              if np.linalg.norm(np.array(r['pos'][:2]) - np.array(agent_pos[:2])) < 10])
        
        return np.array([
            nearby_obstacles / 20.0,  # 标准化到[0,1]
            nearby_resources / 30.0,
            random.random()  # 随机因子
        ])
    
    def step(self, actions: Dict[str, str]) -> Dict[str, float]:
        """环境一步更新"""
        results = {}
        
        for agent_id, action in actions.items():
            # 简化的环境响应
            reward = 0.0
            new_pos = self._update_agent_position(agent_id, action)
            
            # 检查与障碍物的碰撞
            for obstacle in self.obstacles:
                if np.linalg.norm(np.array(obstacle['pos']) - np.array(new_pos)) < obstacle['size']:
                    reward -= obstacle['damage'] * 10
                    break
            
            # 检查资源收集
            for resource in self.resources[:]:
                if np.linalg.norm(np.array(resource['pos']) - np.array(new_pos)) < 5:
                    reward += resource['value'] * 20
                    resource['replenish_time'] -= 1
                    if resource['replenish_time'] <= 0:
                        self.resources.remove(resource)
            
            # 随机探索奖励
            if action == "explore":
                reward += random.uniform(0.1, 0.5)
            
            results[agent_id] = reward
        
        return results
    
    def _update_agent_position(self, agent_id: str, action: str) -> Tuple[float, float]:
        """更新智能体位置"""
        # 简化位置更新逻辑
        if not hasattr(self, f'agent_{agent_id}_pos'):
            setattr(self, f'agent_{agent_id}_pos', (50.0, 50.0))
        
        pos = getattr(self, f'agent_{agent_id}_pos')
        movement = {'forward': (0, 1), 'backward': (0, -1), 
                   'left': (-1, 0), 'right': (1, 0)}.get(action, (0, 0))
        
        new_pos = (pos[0] + movement[0], pos[1] + movement[1])
        setattr(self, f'agent_{agent_id}_pos', new_pos)
        return new_pos
    
    def is_done(self, steps: int) -> bool:
        """判断回合是否结束"""
        return steps >= 1000 or len(self.agents_in_env) <= 1


class Algorithm(ABC):
    """算法基类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.training_steps = 0
        self.performance_history = []
        
    @abstractmethod
    def get_action(self, state: np.ndarray, training: bool = True) -> str:
        """获取动作"""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray):
        """算法更新"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """获取算法参数"""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict):
        """设置算法参数"""
        pass
    
    def copy(self) -> 'Algorithm':
        """创建算法副本"""
        new_algo = self.__class__(self.config.copy())
        new_algo.training_steps = self.training_steps
        new_algo.performance_history = self.performance_history.copy()
        return new_algo


class NeuralNetworkAlgorithm(Algorithm):
    """神经网络算法实现"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        if TORCH_AVAILABLE:
            self.net = self._build_network()
            self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
            self.memory = deque(maxlen=config.memory_size)
        else:
            # 简化版本 - 无PyTorch时的替代实现
            self.weights = np.random.normal(0, 0.1, (100, 4))
            self.memory = []
    
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        if self.config.neural_architecture == "mlp":
            return nn.Sequential(
                nn.Linear(3, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size // 2, 4)
            )
        else:
            # 默认使用MLP
            return nn.Sequential(
                nn.Linear(3, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, 4)
            )
    
    def get_action(self, state: np.ndarray, training: bool = True) -> str:
        """获取动作"""
        actions = ['forward', 'backward', 'left', 'right', 'explore']
        
        if TORCH_AVAILABLE and hasattr(self, 'net'):
            with torch.no_grad():
                if len(state.shape) == 1:
                    state = torch.FloatTensor(state).unsqueeze(0)
                else:
                    state = torch.FloatTensor(state)
                
                q_values = self.net(state)
                if training and random.random() < self.config.exploration_rate:
                    action = random.choice(actions)
                else:
                    action_idx = q_values.argmax().item()
                    action = actions[action_idx % len(actions)]
        else:
            # 简化版本
            if training and random.random() < self.config.exploration_rate:
                action = random.choice(actions)
            else:
                # 简化的启发式
                q_values = np.dot(self.weights, state)
                action_idx = np.argmax(q_values) % len(actions)
                action = actions[action_idx]
        
        return action
    
    def update(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray):
        """算法更新"""
        self.training_steps += 1
        
        if TORCH_AVAILABLE and hasattr(self, 'net'):
            # 使用简化的Q-learning更新
            self.memory.append((state, action, reward, next_state))
            
            if len(self.memory) >= 32:
                batch = random.sample(self.memory, 32)
                
                states = torch.FloatTensor([b[0] for b in batch])
                actions = [b[1] for b in batch]
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                
                current_q = self.net(states).max(1)[0]
                next_q = self.net(next_states).max(1)[0]
                target = rewards + 0.99 * next_q
                
                loss = F.mse_loss(current_q, target.detach())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            # 简化版本更新
            self.memory.append((state, action, reward))
            
            if len(self.memory) >= 10:
                # 简化的梯度更新
                batch = random.sample(self.memory, 10)
                gradient = np.zeros_like(self.weights)
                
                for state, action, reward in batch:
                    gradient += np.outer(state, np.ones(4)) * reward
                
                self.weights += self.config.learning_rate * gradient / len(batch)
    
    def get_params(self) -> Dict:
        """获取算法参数"""
        if TORCH_AVAILABLE and hasattr(self, 'net'):
            return {
                'state_dict': self.net.state_dict(),
                'training_steps': self.training_steps,
                'config': self.config.__dict__
            }
        else:
            return {
                'weights': self.weights.tolist(),
                'training_steps': self.training_steps,
                'config': self.config.__dict__
            }
    
    def set_params(self, params: Dict):
        """设置算法参数"""
        if 'state_dict' in params and TORCH_AVAILABLE and hasattr(self, 'net'):
            self.net.load_state_dict(params['state_dict'])
            self.training_steps = params.get('training_steps', 0)
        elif 'weights' in params:
            self.weights = np.array(params['weights'])
            self.training_steps = params.get('training_steps', 0)


class GeneticAlgorithm(Algorithm):
    """遗传算法实现"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.genome = np.random.normal(0, 1, 100)  # 基因组长度
        self.fitness_history = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> str:
        """获取动作"""
        actions = ['forward', 'backward', 'left', 'right', 'explore']
        
        # 基于基因组的决策
        if training and random.random() < self.config.exploration_rate:
            return random.choice(actions)
        
        # 简化决策逻辑
        decision_vector = np.dot(self.genome[:20].reshape(4, 5), np.tile(state, (5, 1)))
        action_idx = np.argmax(np.sum(decision_vector, axis=1)) % len(actions)
        return actions[action_idx]
    
    def update(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray):
        """算法更新 - 遗传算法不需要在线更新"""
        self.performance_history.append(reward)
    
    def mutate(self, mutation_rate: float = None) -> 'GeneticAlgorithm':
        """变异操作"""
        if mutation_rate is None:
            mutation_rate = self.config.mutation_rate
        
        mutated = self.copy()
        mutation_mask = np.random.random(len(self.genome)) < mutation_rate
        mutated.genome[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return mutated
    
    def crossover(self, other: 'GeneticAlgorithm') -> Tuple['GeneticAlgorithm', 'GeneticAlgorithm']:
        """交叉操作"""
        child1 = self.copy()
        child2 = other.copy()
        
        # 单点交叉
        crossover_point = random.randint(1, len(self.genome) - 1)
        child1.genome = np.concatenate([self.genome[:crossover_point], other.genome[crossover_point:]])
        child2.genome = np.concatenate([other.genome[:crossover_point], self.genome[crossover_point:]])
        
        return child1, child2
    
    def get_params(self) -> Dict:
        """获取算法参数"""
        return {
            'genome': self.genome.tolist(),
            'performance_history': self.performance_history,
            'config': self.config.__dict__
        }
    
    def set_params(self, params: Dict):
        """设置算法参数"""
        self.genome = np.array(params['genome'])
        self.performance_history = params.get('performance_history', [])


class Agent:
    """AI智能体类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = config.id
        self.algo = self._create_algorithm()
        self.fitness_scores = []
        self.survival_time = 0
        self.tasks_completed = []
        self.generation = 0
        self.is_alive = True
        
    def _create_algorithm(self) -> Algorithm:
        """创建算法实例"""
        if self.config.algorithm_type == "neural_network":
            return NeuralNetworkAlgorithm(self.config)
        elif self.config.algorithm_type == "genetic":
            return GeneticAlgorithm(self.config)
        else:
            return NeuralNetworkAlgorithm(self.config)  # 默认
    
    def act(self, state: np.ndarray, training: bool = True) -> str:
        """获取动作"""
        if not self.is_alive:
            return "idle"
        
        return self.algo.get_action(state, training)
    
    def update(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray):
        """更新智能体"""
        self.algo.update(state, action, reward, next_state)
    
    def update_fitness(self, fitness: FitnessResult):
        """更新适应度"""
        self.fitness_scores.append(fitness.total_fitness())
        self.survival_time = max(self.survival_time, fitness.survival_time)
        self.tasks_completed.extend(fitness.tasks_completed)
    
    def evaluate_performance(self) -> float:
        """评估智能体性能"""
        if not self.fitness_scores:
            return 0.0
        
        recent_scores = self.fitness_scores[-10:]  # 最近10次评估
        return np.mean(recent_scores) * (1 + 0.01 * len(self.tasks_completed))
    
    def mutate(self) -> 'Agent':
        """智能体变异"""
        new_config = self.config.copy()
        new_config.mutation_rate = min(0.2, self.config.mutation_rate * 1.1)
        
        mutated_agent = Agent(new_config)
        mutated_agent.algo = self.algo.mutate() if hasattr(self.algo, 'mutate') else self.algo.copy()
        mutated_agent.generation = self.generation + 1
        
        return mutated_agent
    
    def crossover(self, other: 'Agent') -> Tuple['Agent', 'Agent']:
        """智能体交叉"""
        new_config1 = self.config.copy()
        new_config2 = other.config.copy()
        
        child1 = Agent(new_config1)
        child2 = Agent(new_config2)
        
        if hasattr(self.algo, 'crossover') and hasattr(other.algo, 'crossover'):
            child1_algo, child2_algo = self.algo.crossover(other.algo)
            child1.algo = child1_algo
            child2.algo = child2_algo
        else:
            child1.algo = self.algo.copy()
            child2.algo = other.algo.copy()
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        return child1, child2
    
    def get_params(self) -> Dict:
        """获取智能体参数"""
        return {
            'algo_params': self.algo.get_params(),
            'config': self.config.__dict__,
            'fitness_scores': self.fitness_scores,
            'generation': self.generation
        }
    
    def set_params(self, params: Dict):
        """设置智能体参数"""
        self.algo.set_params(params['algo_params'])
        self.fitness_scores = params.get('fitness_scores', [])
        self.generation = params.get('generation', 0)
    
    def copy(self) -> 'Agent':
        """创建智能体副本"""
        new_agent = Agent(self.config.copy())
        new_agent.algo = self.algo.copy()
        new_agent.fitness_scores = self.fitness_scores.copy()
        new_agent.survival_time = self.survival_time
        new_agent.tasks_completed = self.tasks_completed.copy()
        new_agent.generation = self.generation
        return new_agent


class Population:
    """种群管理类"""
    
    def __init__(self, size: int = 1000, environment_types: List[str] = None):
        self.size = size
        self.agents = []
        self.generation = 0
        self.best_agents = []
        self.environment_types = environment_types or ["mixed", "competitive", "cooperative"]
        self.evaluation_history = []
        
        # 初始化种群
        self._initialize_population()
    
    def _initialize_population(self):
        """初始化种群"""
        self.agents = []
        
        for i in range(self.size):
            config = AgentConfig(
                id=f"agent_{i}",
                algorithm_type=random.choice(["neural_network", "genetic"]),
                neural_architecture=random.choice(["mlp", "cnn"]),
                hidden_size=random.choice([32, 64, 128]),
                learning_rate=random.choice([0.001, 0.01, 0.0001]),
                exploration_rate=random.uniform(0.05, 0.3),
                mutation_rate=random.uniform(0.01, 0.1)
            )
            
            agent = Agent(config)
            self.agents.append(agent)
    
    def evaluate_population(self) -> List[FitnessResult]:
        """评估整个种群"""
        fitness_results = []
        environments = [Environment(env_type) for env_type in self.environment_types]
        
        for i, agent in enumerate(self.agents):
            print(f"评估智能体 {i+1}/{len(self.agents)}: {agent.id}", end='\r')
            
            # 多环境评估
            env_scores = []
            for env in environments:
                result = self._evaluate_agent_in_environment(agent, env)
                env_scores.append(result.total_fitness())
            
            # 泛化分数
            generalization_score = np.std(env_scores)  # 方差作为泛化指标
            overall_result = self._combine_environment_results(env_scores, generalization_score)
            overall_result.generalization_score = generalization_score
            
            # 更新智能体
            agent.update_fitness(overall_result)
            fitness_results.append(overall_result)
        
        print("\n种群评估完成")
        self.evaluation_history.append(fitness_results)
        return fitness_results
    
    def _evaluate_agent_in_environment(self, agent: Agent, env: Environment) -> FitnessResult:
        """在特定环境中评估智能体"""
        total_reward = 0
        steps = 0
        max_steps = 500
        exploration_count = 0
        tasks_completed = []
        
        # 初始位置
        state = np.array([50.0, 50.0, 0.0])  # 简化状态
        
        while steps < max_steps:
            # 获取动作
            action = agent.act(state, training=False)
            
            if action == "explore":
                exploration_count += 1
            
            # 环境步骤
            actions = {agent.id: action}
            rewards = env.step(actions)
            
            reward = rewards.get(agent.id, 0)
            total_reward += reward
            
            # 更新状态
            next_state = env.get_state(state)
            
            # 更新智能体
            agent.update(state, action, reward, next_state)
            
            state = next_state
            steps += 1
            
            # 检查任务完成
            if reward > 5:  # 假设奖励>5表示任务完成
                tasks_completed.append(f"task_{steps}")
        
        # 计算效率
        efficiency = total_reward / max_steps if max_steps > 0 else 0
        
        # 计算探索分数
        exploration_score = exploration_count / max_steps
        
        # 计算鲁棒性分数（基于奖励的稳定性）
        robustness_score = 1.0 / (1.0 + np.std([reward for _ in range(steps)]) )
        
        return FitnessResult(
            agent_id=agent.id,
            score=total_reward,
            survival_time=steps,
            tasks_completed=tasks_completed,
            efficiency=efficiency,
            exploration_score=exploration_score,
            robustness_score=robustness_score
        )
    
    def _combine_environment_results(self, env_scores: List[float], generalization_score: float) -> FitnessResult:
        """合并多环境评估结果"""
        return FitnessResult(
            agent_id="combined",
            score=np.mean(env_scores),
            survival_time=len(env_scores) * 100,  # 模拟
            tasks_completed=[],  # 合并的任务
            efficiency=np.mean(env_scores),
            exploration_score=generalization_score,
            robustness_score=np.mean(env_scores)
        )
    
    def select_parents(self, fitness_results: List[FitnessResult], num_parents: int) -> List[Agent]:
        """选择父代智能体 - 达尔文式选择"""
        # 按适应度排序
        sorted_results = sorted(fitness_results, key=lambda x: x.total_fitness(), reverse=True)
        
        # 精英选择
        elite_count = int(num_parents * self.agents[0].config.elite_ratio)
        elites = [self._get_agent_by_id(result.agent_id) for result in sorted_results[:elite_count]]
        
        # 轮盘赌选择其余部分
        remaining_parents = num_parents - elite_count
        remaining_parents = self._roulette_selection(fitness_results[elite_count:], remaining_parents)
        
        return elites + remaining_parents
    
    def _roulette_selection(self, fitness_results: List[FitnessResult], num_select: int) -> List[Agent]:
        """轮盘赌选择"""
        if not fitness_results:
            return []
        
        # 计算选择概率
        fitness_values = [max(0.001, result.total_fitness()) for result in fitness_results]
        total_fitness = sum(fitness_values)
        selection_probs = [f / total_fitness for f in fitness_values]
        
        selected = []
        for _ in range(num_select):
            idx = np.random.choice(len(fitness_results), p=selection_probs)
            selected.append(self._get_agent_by_id(fitness_results[idx].agent_id))
        
        return selected
    
    def _get_agent_by_id(self, agent_id: str) -> Agent:
        """根据ID获取智能体"""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise ValueError(f"智能体 {agent_id} 未找到")
    
    def reproduce(self, parents: List[Agent], num_children: int) -> List[Agent]:
        """繁殖新智能体"""
        children = []
        
        # 精英复制
        elite_count = int(num_children * 0.1)
        for i in range(elite_count):
            child = parents[i % len(parents)].copy()
            children.append(child)
        
        # 交叉繁殖
        crossover_count = int(num_children * 0.7)
        for _ in range(crossover_count):
            parent1, parent2 = random.sample(parents[:len(parents)//2], 2)
            child1, child2 = parent1.crossover(parent2)
            children.extend([child1, child2])
        
        # 变异
        mutation_count = num_children - elite_count - crossover_count * 2
        for _ in range(mutation_count):
            parent = random.choice(parents)
            child = parent.mutate()
            children.append(child)
        
        return children[:num_children]
    
    def evolve(self, fitness_results: List[FitnessResult]) -> 'Population':
        """种群进化"""
        print(f"\n第 {self.generation + 1} 代进化开始...")
        
        # 选择父代
        parents = self.select_parents(fitness_results, self.size)
        
        # 繁殖新代
        children = self.reproduce(parents, self.size)
        
        # 创建新种群
        new_population = Population(self.size, self.environment_types)
        new_population.agents = children
        new_population.generation = self.generation + 1
        
        # 更新最佳智能体
        sorted_results = sorted(fitness_results, key=lambda x: x.total_fitness(), reverse=True)
        self.best_agents = [self._get_agent_by_id(result.agent_id) for result in sorted_results[:10]]
        
        print(f"第 {self.generation + 1} 代进化完成")
        print(f"最佳适应度: {sorted_results[0].total_fitness():.3f}")
        print(f"平均适应度: {np.mean([r.total_fitness() for r in sorted_results]):.3f}")
        
        return new_population


class ProcGenTester:
    """ProcGen环境泛化能力测试器"""
    
    def __init__(self):
        self.test_environments = self._create_test_environments()
    
    def _create_test_environments(self) -> Dict[str, Environment]:
        """创建多种测试环境"""
        envs = {}
        
        # 环境类型：迷宫、资源收集、竞争、协作
        env_configs = [
            ("maze", {"type": "maze", "size": "large"}),
            ("resource_collect", {"type": "resource", "density": "high"}),
            ("competitive", {"type": "competitive", "agents": "many"}),
            ("cooperative", {"type": "cooperative", "team_size": "large"}),
            ("exploration", {"type": "exploration", "map_size": "huge"}),
            ("survival", {"type": "survival", "difficulty": "hard"})
        ]
        
        for name, config in env_configs:
            envs[name] = Environment(config.get("type", "mixed"))
        
        return envs
    
    def test_generalization(self, agent: Agent) -> Dict[str, float]:
        """测试智能体泛化能力"""
        results = {}
        
        for env_name, env in self.test_environments.items():
            print(f"在 {env_name} 环境中测试智能体 {agent.id}")
            
            scores = []
            for _ in range(5):  # 每环境5次测试
                score = self._evaluate_agent_in_env(agent, env)
                scores.append(score)
            
            results[env_name] = np.mean(scores)
        
        # 计算泛化指标
        scores = list(results.values())
        results['generalization_score'] = 1.0 / (1.0 + np.std(scores))  # 方差越小，泛化越好
        results['average_performance'] = np.mean(scores)
        results['consistency'] = 1.0 / (1.0 + np.std(scores) / np.mean(scores))
        
        return results
    
    def _evaluate_agent_in_env(self, agent: Agent, env: Environment) -> float:
        """在特定环境中评估智能体"""
        total_reward = 0
        steps = 0
        max_steps = 300
        
        # 简化状态
        state = np.array([0.0, 0.0, 0.0])
        
        while steps < max_steps:
            action = agent.act(state, training=False)
            
            # 简化环境响应
            reward = random.uniform(-1, 2)  # 随机奖励模拟
            total_reward += reward
            
            state = np.array([
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ])
            
            steps += 1
        
        return total_reward / max_steps


class DiscoRLSystem:
    """DiscoRL系统主类"""
    
    def __init__(self, population_size: int = 1000, num_generations: int = 50):
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = Population(population_size)
        self.procgen_tester = ProcGenTester()
        self.evolution_history = []
        self.best_algorithms = {}
        
    def run_evolution(self) -> Dict[str, Any]:
        """运行完整的进化过程"""
        print(f"开始DiscoRL算法发现系统")
        print(f"种群规模: {self.population_size}")
        print(f"进化代数: {self.num_generations}")
        print("=" * 50)
        
        start_time = time.time()
        
        for generation in range(self.num_generations):
            print(f"\n第 {generation + 1}/{self.num_generations} 代")
            
            # 1. 评估当前种群
            fitness_results = self.population.evaluate_population()
            
            # 2. 泛化能力测试
            print("进行泛化能力测试...")
            generalization_scores = []
            for agent in self.population.agents[:50]:  # 测试前50个智能体
                gen_scores = self.procgen_tester.test_generalization(agent)
                generalization_scores.append(gen_scores['generalization_score'])
            
            avg_gen_score = np.mean(generalization_scores) if generalization_scores else 0
            print(f"平均泛化分数: {avg_gen_score:.3f}")
            
            # 3. 记录历史
            evolution_record = {
                'generation': generation,
                'best_fitness': max(r.total_fitness() for r in fitness_results),
                'avg_fitness': np.mean([r.total_fitness() for r in fitness_results]),
                'generalization_score': avg_gen_score,
                'diversity': self._calculate_diversity(),
                'survival_rate': len([r for r in fitness_results if r.survival_time > 100]) / len(fitness_results)
            }
            self.evolution_history.append(evolution_record)
            
            # 4. 检查收敛
            if self._check_convergence():
                print("算法收敛，停止进化")
                break
            
            # 5. 进化下一代
            self.population = self.population.evolve(fitness_results)
        
        end_time = time.time()
        
        # 6. 最终结果
        final_results = self._generate_final_report(end_time - start_time)
        
        return final_results
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population.agents) < 2:
            return 0.0
        
        # 基于算法参数的多样性
        algo_types = [agent.config.algorithm_type for agent in self.population.agents]
        type_counts = {}
        for algo_type in algo_types:
            type_counts[algo_type] = type_counts.get(algo_type, 0) + 1
        
        # 计算类型分布的熵
        total = len(self.population.agents)
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy / np.log2(len(type_counts)) if len(type_counts) > 1 else 0.0
    
    def _check_convergence(self) -> bool:
        """检查算法是否收敛"""
        if len(self.evolution_history) < 10:
            return False
        
        recent_bests = [record['best_fitness'] for record in self.evolution_history[-10:]]
        
        # 检查最后10代是否改进很小
        improvement = (max(recent_bests) - min(recent_bests)) / max(1e-10, abs(min(recent_bests)))
        
        return improvement < 0.01  # 改进小于1%认为收敛
    
    def _generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """生成最终报告"""
        # 找到最佳算法
        best_agent = max(self.population.agents, key=lambda a: a.evaluate_performance())
        best_genetic = max(self.population.best_agents, key=lambda a: a.evaluate_performance())
        
        # 测试最佳算法的泛化能力
        print("\n测试最佳算法的泛化能力...")
        generalization_results = self.procgen_tester.test_generalization(best_agent)
        
        # 提取最佳算法
        self.best_algorithms = {
            'overall_best': best_agent.get_params(),
            'evolutionary_best': best_genetic.get_params(),
            'generalization_test': generalization_results
        }
        
        # 生成报告
        report = {
            'system_info': {
                'total_generations': self.population.generation + 1,
                'total_time': total_time,
                'population_size': self.population_size,
                'survivors': len([a for a in self.population.agents if a.is_alive])
            },
            'performance_summary': {
                'best_fitness': max(h['best_fitness'] for h in self.evolution_history),
                'average_improvement': np.mean([h['best_fitness'] for h in self.evolution_history]),
                'convergence_generation': self.population.generation + 1,
                'diversity_trend': [h['diversity'] for h in self.evolution_history]
            },
            'discovered_algorithms': self.best_algorithms,
            'evolution_history': self.evolution_history
        }
        
        print("\n" + "=" * 50)
        print("DiscoRL算法发现系统 - 最终报告")
        print("=" * 50)
        print(f"总进化代数: {report['system_info']['total_generations']}")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"最佳适应度: {report['performance_summary']['best_fitness']:.3f}")
        print(f"收敛代数: {report['performance_summary']['convergence_generation']}")
        print(f"平均泛化分数: {generalization_results.get('generalization_score', 0):.3f}")
        print("\n发现的算法类型:")
        print(f"- 总体最佳: {best_agent.config.algorithm_type}")
        print(f"- 进化最佳: {best_genetic.config.algorithm_type}")
        
        return report
    
    def save_results(self, filepath: str = "disco_rl_results.json"):
        """保存结果到文件"""
        results = {
            'best_algorithms': self.best_algorithms,
            'evolution_history': self.evolution_history,
            'final_population_size': len(self.population.agents),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {filepath}")
    
    def visualize_evolution(self, save_path: str = "evolution_plot.png"):
        """可视化进化过程"""
        if not PLOT_AVAILABLE:
            print("Matplotlib不可用，跳过可视化")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 适应度进化
        generations = [h['generation'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        avg_fitness = [h['avg_fitness'] for h in self.evolution_history]
        
        ax1.plot(generations, best_fitness, 'r-', label='最佳适应度', linewidth=2)
        ax1.plot(generations, avg_fitness, 'b-', label='平均适应度', linewidth=2)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('算法性能进化')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 多样性变化
        diversity = [h['diversity'] for h in self.evolution_history]
        ax2.plot(generations, diversity, 'g-', linewidth=2)
        ax2.set_xlabel('代数')
        ax2.set_ylabel('多样性')
        ax2.set_title('种群多样性变化')
        ax2.grid(True)
        
        # 3. 生存率
        survival_rate = [h['survival_rate'] for h in self.evolution_history]
        ax3.plot(generations, survival_rate, 'm-', linewidth=2)
        ax3.set_xlabel('代数')
        ax3.set_ylabel('生存率')
        ax3.set_title('达尔文式选择效果')
        ax3.grid(True)
        
        # 4. 泛化能力
        generalization = [h['generalization_score'] for h in self.evolution_history]
        ax4.plot(generations, generalization, 'c-', linewidth=2)
        ax4.set_xlabel('代数')
        ax4.set_ylabel('泛化分数')
        ax4.set_title('泛化能力进化')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"进化可视化已保存到: {save_path}")


def main():
    """主函数 - 运行DiscoRL系统演示"""
    print("DiscoRL - 自主强化学习算法发现系统")
    print("=" * 60)
    
    # 创建系统实例
    system = DiscoRLSystem(population_size=500, num_generations=20)
    
    try:
        # 运行进化
        results = system.run_evolution()
        
        # 保存结果
        system.save_results("disco_rl_results.json")
        
        # 可视化
        system.visualize_evolution("disco_rl_evolution.png")
        
        print("\nDiscoRL系统运行完成！")
        return results
        
    except KeyboardInterrupt:
        print("\n用户中断，保存当前进度...")
        system.save_results("disco_rl_partial_results.json")
        print("进度已保存")
        
    except Exception as e:
        print(f"\n运行出错: {e}")
        system.save_results("disco_rl_error_results.json")
        raise


if __name__ == "__main__":
    # 运行演示
    results = main()