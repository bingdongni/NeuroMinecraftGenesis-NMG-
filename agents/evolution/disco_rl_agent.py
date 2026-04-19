#!/usr/bin/env python3
"""
进化AI智能体 - 基于DISCO-RL的进化学习智能体
"""

import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional
import logging

class DiscoRLAgent:
    """基于DISCO-RL的进化AI智能体"""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"agent_{random.randint(1000, 9999)}"
        self.logger = logging.getLogger(__name__)
        
        # 核心智能体属性
        self.intelligence = 0.5
        self.adaptability = 0.5
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        
        # 经验和学习数据
        self.experience = []
        self.genome = self._initialize_genome()
        self.fitness_score = 0
        self.generation = 0
        
    def _initialize_genome(self) -> Dict:
        """初始化基因组参数"""
        return {
            'network_weights': np.random.normal(0, 0.1, (10, 10)).tolist(),
            'learning_parameters': {
                'alpha': np.random.uniform(0.01, 0.5),
                'beta': np.random.uniform(0.1, 2.0),
                'gamma': np.random.uniform(0.8, 0.99)
            },
            'behavioral_traits': {
                'aggression': np.random.uniform(0, 1),
                'cooperation': np.random.uniform(0, 1),
                'curiosity': np.random.uniform(0, 1)
            }
        }
    
    def perceive_environment(self, state: Dict) -> np.ndarray:
        """感知环境状态"""
        # 简化的环境感知
        state_vector = np.array([
            state.get('energy', 0),
            state.get('health', 0),
            state.get('safety', 0),
            state.get('resources', 0),
            state.get('social', 0),
            random.uniform(0, 1),  # 随机噪声
            random.uniform(0, 1),  # 随机噪声
            self.intelligence,
            self.adaptability,
            self.exploration_rate
        ])
        return state_vector
    
    def think(self, state_vector: np.ndarray) -> Dict:
        """智能体思考决策过程"""
        # 简化的决策逻辑
        decision_weights = np.array([
            0.2,  # energy
            0.25, # health  
            0.2,  # safety
            0.15, # resources
            0.2   # social
        ])
        
        priorities = decision_weights * state_vector[:5]
        top_priority = np.argmax(priorities)
        
        actions = {
            0: 'search_food',
            1: 'rest_heal',
            2: 'avoid_danger',
            3: 'gather_resources',
            4: 'social_interact'
        }
        
        return {
            'action': actions[top_priority],
            'confidence': priorities[top_priority],
            'reasoning': f"基于状态感知，选择行动: {actions[top_priority]}",
            'processing_time': random.uniform(0.01, 0.1)
        }
    
    def act(self, decision: Dict) -> Dict:
        """执行决策"""
        # 模拟行动执行
        action = decision['action']
        confidence = decision['confidence']
        
        # 根据动作和环境计算结果
        success_rate = confidence * self.intelligence
        success = random.random() < success_rate
        
        return {
            'action_executed': action,
            'success': success,
            'outcome': {
                'energy_change': random.uniform(-0.1, 0.3) if success else random.uniform(-0.2, -0.05),
                'health_change': random.uniform(0, 0.1) if success else random.uniform(-0.1, 0),
                'experience_gain': 0.1 * confidence
            },
            'timestamp': np.datetime64('now')
        }
    
    def learn(self, experience: Dict) -> Dict:
        """学习更新过程"""
        # 简化的学习算法
        reward = experience['outcome']['experience_gain']
        
        # 更新核心属性
        self.intelligence += reward * self.learning_rate
        self.adaptability += 0.05  # 基础适应增长
        
        # 更新经验
        self.experience.append(experience)
        
        # 更新适应度分数
        total_reward = sum([exp.get('outcome', {}).get('experience_gain', 0) 
                           for exp in self.experience])
        self.fitness_score = total_reward / max(1, len(self.experience))
        
        return {
            'learning_summary': {
                'intelligence_updated': self.intelligence,
                'adaptability_updated': self.adaptability,
                'total_experience': len(self.experience),
                'fitness_score': self.fitness_score
            }
        }
    
    def evolve(self, population: List) -> 'DiscoRLAgent':
        """进化过程（交配和变异）"""
        if random.random() > 0.1:  # 90%概率不进化
            return self
            
        # 选择交配对象
        mate = random.choice([p for p in population if p.agent_id != self.agent_id])
        
        # 创建子代
        offspring = DiscoRLAgent(f"{self.agent_id}_child_{random.randint(100, 999)}")
        
        # 混合基因组
        for key in self.genome.keys():
            if key == 'network_weights':
                # 混合网络权重
                parent_weights = np.array(self.genome[key])
                mate_weights = np.array(mate.genome[key])
                offspring.genome[key] = ((parent_weights + mate_weights) / 2).tolist()
            else:
                # 混合其他参数
                offspring.genome[key] = mate.genome[key] if random.random() < 0.5 else self.genome[key]
        
        # 随机变异
        if random.random() < 0.3:  # 30%概率变异
            offspring.genome['learning_parameters']['alpha'] *= random.uniform(0.8, 1.2)
            offspring.genome['behavioral_traits']['aggression'] *= random.uniform(0.9, 1.1)
        
        return offspring
    
    def get_agent_info(self) -> Dict:
        """获取智能体信息"""
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'intelligence': self.intelligence,
            'adaptability': self.adaptability,
            'learning_rate': self.learning_rate,
            'fitness_score': self.fitness_score,
            'experience_count': len(self.experience),
            'genome_summary': {
                'learning_parameters': self.genome['learning_parameters'],
                'behavioral_traits': self.genome['behavioral_traits']
            }
        }
    
    def save_agent(self, filepath: str) -> bool:
        """保存智能体状态"""
        try:
            agent_data = {
                'agent_info': self.get_agent_info(),
                'experience': self.experience,
                'genome': self.genome
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(agent_data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"保存智能体失败: {e}")
            return False

if __name__ == "__main__":
    # 测试代码
    agent = DiscoRLAgent("test_agent_001")
    
    # 模拟环境感知和决策循环
    for step in range(5):
        state = {
            'energy': random.uniform(0.3, 0.9),
            'health': random.uniform(0.6, 1.0),
            'safety': random.uniform(0.4, 0.8),
            'resources': random.uniform(0.2, 0.7),
            'social': random.uniform(0.1, 0.6)
        }
        
        # 感知 -> 思考 -> 行动 -> 学习
        state_vector = agent.perceive_environment(state)
        decision = agent.think(state_vector)
        outcome = agent.act(decision)
        learning_result = agent.learn(outcome)
        
        print(f"步骤 {step + 1}: {decision['action']} -> {'成功' if outcome['success'] else '失败'}")
    
    print("\n智能体最终状态:")
    info = agent.get_agent_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))