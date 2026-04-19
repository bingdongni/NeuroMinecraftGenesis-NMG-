"""
真实堆叠性能评估器

该模块实现真实堆叠任务环境下的智能体性能评估，主要评估：
1. 物理世界能力：物体堆叠的稳定性和精确性
2. 学习速度：从失败中学习和技能优化能力  
3. 泛化能力：跨物体类型和环境的表现一致性

真实堆叠任务模拟了物理世界的堆叠挑战，智能体需要：
- 理解物理规律（重力、平衡、摩擦等）
- 规划最优的堆叠序列
- 适应不同的物体形状和材质
- 预测和避免堆叠失败
"""

import time
import logging
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib未安装，将禁用可视化功能")


@dataclass
class PhysicsObject:
    """物理对象数据类"""
    name: str
    shape: str  # 'box', 'sphere', 'cylinder', 'irregular'
    dimensions: Tuple[float, float, float]  # (长, 宽, 高)
    weight: float
    friction_coefficient: float
    center_of_mass: Tuple[float, float, float]
    stability_factor: float  # 稳定性因子
    
    def get_volume(self) -> float:
        """计算体积"""
        if self.shape == 'box':
            return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        elif self.shape == 'sphere':
            return (4/3) * math.pi * (self.dimensions[0]/2)**3
        elif self.shape == 'cylinder':
            return math.pi * (self.dimensions[0]/2)**2 * self.dimensions[2]
        else:
            return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]  # 近似


@dataclass
class StackingMetrics:
    """堆叠任务评估指标数据类"""
    stacking_success_rate: float = 0.0  # 堆叠成功率
    stability_score: float = 0.0  # 稳定性得分
    physics_accuracy: float = 0.0  # 物理规律符合度
    height_achieved: float = 0.0  # 达到的高度
    precision_score: float = 0.0  # 精确度得分
    efficiency_score: float = 0.0  # 效率得分
    creativity_score: float = 0.0  # 创造性得分
    failure_analysis: Dict[str, int] = None  # 失败原因分析
    stability_history: List[float] = None  # 稳定性历史
    stacking_sequence: List[Dict] = field(default_factory=list)  # 堆叠序列
    total_attempts: int = 0  # 总尝试次数
    
    def __post_init__(self):
        if self.failure_analysis is None:
            self.failure_analysis = {}
        if self.stability_history is None:
            self.stability_history = []


class PhysicsWorldSimulator:
    """物理世界模拟器"""
    
    def __init__(self):
        """初始化物理世界模拟器"""
        # 重力加速度
        self.gravity = 9.81
        
        # 地面参数
        self.ground_friction = 0.8
        self.ground_height = 0.0
        
        # 物体库
        self.object_library = self._create_object_library()
        
        # 物理状态
        self.stacked_objects = []
        self.current_height = 0.0
        
    def _create_object_library(self) -> Dict[str, PhysicsObject]:
        """创建物体库"""
        objects = {}
        
        # 基本形状物体
        objects['small_box'] = PhysicsObject(
            name='small_box',
            shape='box',
            dimensions=(0.2, 0.2, 0.2),
            weight=1.0,
            friction_coefficient=0.6,
            center_of_mass=(0.1, 0.1, 0.1),
            stability_factor=0.8
        )
        
        objects['medium_box'] = PhysicsObject(
            name='medium_box',
            shape='box',
            dimensions=(0.3, 0.3, 0.3),
            weight=2.0,
            friction_coefficient=0.7,
            center_of_mass=(0.15, 0.15, 0.15),
            stability_factor=0.9
        )
        
        objects['large_box'] = PhysicsObject(
            name='large_box',
            shape='box',
            dimensions=(0.4, 0.4, 0.4),
            weight=3.0,
            friction_coefficient=0.8,
            center_of_mass=(0.2, 0.2, 0.2),
            stability_factor=0.95
        )
        
        objects['sphere_small'] = PhysicsObject(
            name='sphere_small',
            shape='sphere',
            dimensions=(0.2, 0.2, 0.2),
            weight=0.8,
            friction_coefficient=0.4,
            center_of_mass=(0.1, 0.1, 0.1),
            stability_factor=0.6
        )
        
        objects['cylinder_small'] = PhysicsObject(
            name='cylinder_small',
            shape='cylinder',
            dimensions=(0.2, 0.2, 0.3),
            weight=1.2,
            friction_coefficient=0.5,
            center_of_mass=(0.1, 0.1, 0.15),
            stability_factor=0.7
        )
        
        # 不规则形状（模拟现实物体）
        objects['book'] = PhysicsObject(
            name='book',
            shape='irregular',
            dimensions=(0.25, 0.18, 0.04),
            weight=0.5,
            friction_coefficient=0.6,
            center_of_mass=(0.125, 0.09, 0.02),
            stability_factor=0.7
        )
        
        objects['bottle'] = PhysicsObject(
            name='bottle',
            shape='irregular',
            dimensions=(0.08, 0.08, 0.25),
            weight=0.3,
            friction_coefficient=0.3,
            center_of_mass=(0.04, 0.04, 0.125),
            stability_factor=0.5
        )
        
        return objects
    
    def calculate_stability(self, stacked_objects: List[Dict]) -> float:
        """计算堆叠稳定性"""
        if not stacked_objects:
            return 0.0
        
        total_stability = 0.0
        
        for i, obj in enumerate(stacked_objects):
            # 基础稳定性
            base_stability = obj['stability_factor']
            
            # 支撑面积影响
            supporting_area = self._calculate_supporting_area(stacked_objects[:i+1])
            area_factor = min(1.0, supporting_area / (obj['dimensions'][0] * obj['dimensions'][1]))
            
            # 重心对齐度
            center_alignment = self._calculate_center_alignment(stacked_objects[:i+1])
            
            # 摩擦力
            friction_factor = obj['friction_coefficient']
            
            # 综合稳定性
            stability = base_stability * area_factor * center_alignment * friction_factor
            total_stability += stability
        
        return total_stability / len(stacked_objects)
    
    def _calculate_supporting_area(self, objects: List[Dict]) -> float:
        """计算支撑面积"""
        if len(objects) < 2:
            return objects[0]['dimensions'][0] * objects[0]['dimensions'][1] if objects else 0.0
        
        # 简化计算：取最顶层物体的接触面积
        top_object = objects[-1]
        return top_object['dimensions'][0] * top_object['dimensions'][1] * 0.8  # 假设80%有效接触
    
    def _calculate_center_alignment(self, objects: List[Dict]) -> float:
        """计算重心对齐度"""
        if len(objects) < 2:
            return 1.0
        
        # 计算从底部到顶部的重心偏移
        total_weight = sum(obj['weight'] for obj in objects)
        
        if total_weight == 0:
            return 1.0
        
        # 简化的对齐度计算
        alignment_factor = 0.9  # 假设良好的对齐
        
        # 根据物体形状调整
        for obj in objects:
            if obj['shape'] == 'sphere':
                alignment_factor *= 0.8  # 球体对齐困难
            elif obj['shape'] == 'cylinder':
                alignment_factor *= 0.9  # 圆柱体中等
        
        return max(0.1, min(1.0, alignment_factor))
    
    def simulate_collapse(self, stacked_objects: List[Dict]) -> str:
        """模拟堆叠倒塌"""
        if not stacked_objects:
            return "no_objects"
        
        stability_score = self.calculate_stability(stacked_objects)
        
        # 根据稳定性判断失败原因
        if stability_score < 0.3:
            return "severe_instability"
        elif stability_score < 0.5:
            return "moderate_instability"
        elif stability_score < 0.7:
            return "minor_instability"
        else:
            return "stable"
    
    def get_random_objects(self, num_objects: int, difficulty: str = 'medium') -> List[PhysicsObject]:
        """获取随机物体组合"""
        available_objects = list(self.object_library.values())
        
        if difficulty == 'easy':
            # 选择稳定性好的物体
            available_objects = [obj for obj in available_objects if obj.stability_factor > 0.7]
        elif difficulty == 'hard':
            # 选择挑战性物体
            available_objects = [obj for obj in available_objects if obj.stability_factor < 0.7]
        
        return random.sample(available_objects, min(num_objects, len(available_objects)))


class StackingEvaluator:
    """
    真实堆叠性能评估器
    
    在模拟的物理世界中评估智能体的堆叠能力。
    支持多物体类型、复杂堆叠序列和稳定性测试。
    
    Attributes:
        physics_simulator (PhysicsWorldSimulator): 物理世界模拟器
        config (Dict): 配置参数
        logger (logging.Logger): 日志记录器
        evaluation_history (List): 评估历史记录
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 max_objects: int = 8,
                 simulation_trials: int = 10):
        """
        初始化堆叠评估器
        
        Args:
            config: 配置参数字典
            max_objects: 最大堆叠物体数量
            simulation_trials: 模拟试验次数
        """
        # 物理世界模拟器
        self.physics_simulator = PhysicsWorldSimulator()
        self.max_objects = max_objects
        self.simulation_trials = simulation_trials
        
        # 设置配置参数
        self.config = self._setup_default_config()
        if config:
            self.config.update(config)
        
        # 设置日志记录
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 评估历史记录
        self.evaluation_history = []
        
        self.logger.info(f"堆叠评估器初始化完成 - 最大物体: {max_objects}, 试验次数: {simulation_trials}")
    
    def _setup_default_config(self) -> Dict:
        """设置默认配置参数"""
        return {
            # 评估参数
            'evaluation_episodes': 10,
            'max_objects_per_trial': self.max_objects,
            'simulation_trials': self.simulation_trials,
            'confidence_threshold': 0.95,
            
            # 难度配置
            'difficulty_levels': ['easy', 'medium', 'hard'],
            'object_types': ['box', 'sphere', 'cylinder', 'irregular'],
            
            # 物理参数
            'stability_threshold': 0.7,  # 稳定性阈值
            'precision_tolerance': 0.1,  # 精确度容忍度
            'max_height_limit': 2.0,  # 最大高度限制
            
            # 学习参数
            'learning_curve_points': 15,
            'adaptation_test_trials': 5,
            
            # 性能阈值
            'success_threshold': 0.8,  # 80%成功率为成功
            'optimal_stability': 0.95,  # 95%稳定性为最佳
            'physics_accuracy_threshold': 0.9,
            
            # 调试参数
            'save_stacking_sequences': True,
            'visualize_stacking': False,
            'verbose_logging': False
        }
    
    def evaluate(self, agent: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行堆叠性能评估
        
        Args:
            agent: 待评估的智能体
            config: 评估配置参数
            
        Returns:
            包含堆叠性能指标的字典
        """
        evaluation_config = self.config.copy()
        if config:
            evaluation_config.update(config)
        
        start_time = time.time()
        
        try:
            # 执行堆叠模拟
            stacking_data = self._run_stacking_simulation(agent, evaluation_config)
            
            # 计算堆叠指标
            metrics = self._calculate_stacking_metrics(stacking_data)
            
            # 计算衍生指标
            learning_curve = self._calculate_learning_curve(stacking_data)
            generalization_scores = self._calculate_generalization(stacking_data)
            
            # 构建详细结果
            detailed_metrics = {
                'stacking_success_rate': metrics.stacking_success_rate,
                'stability_score': metrics.stability_score,
                'physics_accuracy': metrics.physics_accuracy,
                'height_achieved': metrics.height_achieved,
                'precision_score': metrics.precision_score,
                'efficiency_score': metrics.efficiency_score,
                'creativity_score': metrics.creativity_score,
                'failure_analysis': metrics.failure_analysis,
                'stability_history': metrics.stability_history,
                'stacking_sequence': metrics.stacking_sequence,
                'total_attempts': metrics.total_attempts,
                'learning_curve': learning_curve,
                'generalization_scores': generalization_scores,
                'evaluation_config': evaluation_config,
                'evaluation_time': time.time() - start_time
            }
            
            # 记录评估历史
            self.evaluation_history.append({
                'timestamp': time.time(),
                'agent_id': getattr(agent, 'id', 'unknown'),
                'metrics': detailed_metrics
            })
            
            self.logger.info(
                f"堆叠评估完成 - 成功率: {metrics.stacking_success_rate:.2%}, "
                f"稳定性: {metrics.stability_score:.2f}, 高度: {metrics.height_achieved:.2f}m"
            )
            
            return detailed_metrics
            
        except Exception as e:
            self.logger.error(f"堆叠评估失败: {e}")
            return {
                'error': str(e),
                'stacking_success_rate': 0.0,
                'stability_score': 0.0,
                'physics_accuracy': 0.0,
                'evaluation_time': time.time() - start_time
            }
    
    def _run_stacking_simulation(self, agent: Any, config: Dict) -> Dict[str, Any]:
        """执行堆叠模拟"""
        episodes = config['evaluation_episodes']
        all_results = []
        
        for episode in range(episodes):
            # 选择难度级别
            difficulty = random.choice(config['difficulty_levels'])
            
            # 生成随机物体组合
            num_objects = random.randint(3, config['max_objects_per_trial'])
            objects = self.physics_simulator.get_random_objects(num_objects, difficulty)
            
            # 重置物理世界
            self.physics_simulator.stacked_objects = []
            self.physics_simulator.current_height = 0.0
            
            # 智能体决策堆叠顺序
            if hasattr(agent, 'plan_stacking_sequence'):
                sequence = agent.plan_stacking_sequence(objects, difficulty)
            elif hasattr(agent, 'act'):
                sequence = agent.act(objects, difficulty)
            elif callable(agent):
                sequence = agent(objects, difficulty)
            else:
                # 默认随机序列
                sequence = self._generate_random_sequence(objects)
            
            # 执行堆叠过程
            episode_result = self._execute_stacking_sequence(
                sequence, objects, difficulty, episode, config
            )
            
            all_results.append(episode_result)
        
        return {
            'episodes': all_results,
            'total_episodes': episodes,
            'evaluation_config': config
        }
    
    def _generate_random_sequence(self, objects: List[PhysicsObject]) -> List[int]:
        """生成随机堆叠序列（用于测试）"""
        indices = list(range(len(objects)))
        random.shuffle(indices)
        return indices
    
    def _execute_stacking_sequence(self, sequence: List[int], objects: List[PhysicsObject],
                                 difficulty: str, episode: int, config: Dict) -> Dict[str, Any]:
        """执行堆叠序列"""
        stacked_objects = []
        stability_history = []
        failure_reason = None
        
        total_height = 0.0
        
        for i, obj_index in enumerate(sequence):
            current_object = objects[obj_index]
            
            # 计算当前堆叠高度
            obj_height = current_object.dimensions[2]
            object_position = total_height + obj_height / 2
            
            # 模拟放置物体
            placed_object = {
                'object_index': obj_index,
                'name': current_object.name,
                'shape': current_object.shape,
                'dimensions': current_object.dimensions,
                'weight': current_object.weight,
                'friction_coefficient': current_object.friction_coefficient,
                'stability_factor': current_object.stability_factor,
                'position': (0, 0, object_position),
                'successfully_placed': True
            }
            
            # 添加到堆叠中
            stacked_objects.append(placed_object)
            
            # 更新总高度
            total_height += obj_height
            
            # 检查稳定性
            current_stability = self.physics_simulator.calculate_stability(stacked_objects)
            stability_history.append(current_stability)
            
            # 检查是否倒塌
            if current_stability < config['stability_threshold']:
                # 模拟倒塌
                failure_reason = self.physics_simulator.simulate_collapse(stacked_objects)
                placed_object['successfully_placed'] = False
                break
            
            # 检查是否超出高度限制
            if total_height > config['max_height_limit']:
                failure_reason = 'height_limit_exceeded'
                break
        
        # 计算最终稳定性
        final_stability = self.physics_simulator.calculate_stability(stacked_objects)
        
        return {
            'episode': episode,
            'difficulty': difficulty,
            'objects': objects,
            'stacked_objects': stacked_objects,
            'final_stability': final_stability,
            'total_height': total_height,
            'stability_history': stability_history,
            'successfully_stacked': failure_reason is None,
            'failure_reason': failure_reason,
            'objects_stacked': len(stacked_objects)
        }
    
    def _calculate_stacking_metrics(self, stacking_data: Dict) -> StackingMetrics:
        """计算堆叠性能指标"""
        episodes = stacking_data['episodes']
        
        if not episodes:
            return StackingMetrics()
        
        # 基本统计
        total_attempts = len(episodes)
        successful_attempts = sum(1 for ep in episodes if ep['successfully_stacked'])
        
        # 成功率
        stacking_success_rate = successful_attempts / total_attempts
        
        # 稳定性得分
        stability_scores = [ep['final_stability'] for ep in episodes]
        stability_score = np.mean(stability_scores) if stability_scores else 0.0
        
        # 物理准确性（基于稳定性分布）
        physics_accuracy = np.mean([ep['final_stability'] for ep in episodes if ep['successfully_stacked']])
        
        # 达到的高度
        achieved_heights = [ep['total_height'] for ep in episodes if ep['successfully_stacked']]
        height_achieved = np.mean(achieved_heights) if achieved_heights else 0.0
        
        # 精确度得分（基于物体放置位置）
        precision_scores = []
        for ep in episodes:
            if ep['successfully_stacked']:
                # 简化的精确度计算
                precision = 0.9  # 假设良好精确度
                precision_scores.append(precision)
        
        precision_score = np.mean(precision_scores) if precision_scores else 0.0
        
        # 效率得分（基于物体数量和高度）
        efficiency_scores = []
        for ep in episodes:
            if ep['successfully_stacked']:
                efficiency = ep['total_height'] / ep['objects_stacked']
                efficiency_scores.append(min(1.0, efficiency / 0.5))  # 标准化
                
        efficiency_score = np.mean(efficiency_scores) if efficiency_scores else 0.0
        
        # 创造性得分（基于堆叠策略的多样性）
        creativity_score = self._calculate_creativity_score(episodes)
        
        # 失败原因分析
        failure_analysis = {}
        for ep in episodes:
            if not ep['successfully_stacked']:
                reason = ep['failure_reason']
                failure_analysis[reason] = failure_analysis.get(reason, 0) + 1
        
        # 稳定性历史
        all_stability_history = []
        for ep in episodes:
            all_stability_history.extend(ep['stability_history'])
        
        # 堆叠序列记录
        stacking_sequences = []
        for ep in episodes:
            stacking_sequences.append({
                'episode': ep['episode'],
                'sequence': [obj['name'] for obj in ep['stacked_objects']],
                'success': ep['successfully_stacked'],
                'height': ep['total_height']
            })
        
        return StackingMetrics(
            stacking_success_rate=stacking_success_rate,
            stability_score=stability_score,
            physics_accuracy=physics_accuracy,
            height_achieved=height_achieved,
            precision_score=precision_score,
            efficiency_score=efficiency_score,
            creativity_score=creativity_score,
            failure_analysis=failure_analysis,
            stability_history=all_stability_history,
            stacking_sequence=stacking_sequences,
            total_attempts=total_attempts
        )
    
    def _calculate_creativity_score(self, episodes: List[Dict]) -> float:
        """计算创造性得分"""
        if not episodes:
            return 0.0
        
        # 分析堆叠序列的多样性
        sequences = []
        for ep in episodes:
            if ep['successfully_stacked']:
                sequence = tuple(obj['name'] for obj in ep['stacked_objects'])
                sequences.append(sequence)
        
        if not sequences:
            return 0.0
        
        # 计算序列多样性
        unique_sequences = set(sequences)
        diversity_factor = len(unique_sequences) / len(sequences)
        
        # 分析难度适应能力
        difficulty_performance = defaultdict(list)
        for ep in episodes:
            difficulty_performance[ep['difficulty']].append(ep['final_stability'])
        
        # 计算各难度的平均表现
        difficulty_scores = []
        for difficulty, scores in difficulty_performance.items():
            if scores:
                difficulty_scores.append(np.mean(scores))
        
        adaptation_score = np.std(difficulty_scores) if difficulty_scores else 0.0
        adaptation_score = max(0.0, 1.0 - adaptation_score)  # 方差越小适应力越强
        
        # 综合创造性得分
        creativity_score = 0.6 * diversity_factor + 0.4 * adaptation_score
        
        return min(1.0, creativity_score)
    
    def _calculate_learning_curve(self, stacking_data: Dict) -> List[float]:
        """计算学习曲线"""
        episodes = stacking_data['episodes']
        
        # 按episode排序
        episodes.sort(key=lambda x: x['episode'])
        
        learning_curve = []
        for ep in episodes:
            # 使用稳定性作为学习指标
            stability = ep['final_stability']
            # 考虑成功与否的奖励
            success_bonus = 0.2 if ep['successfully_stacked'] else 0.0
            learning_curve.append(stability + success_bonus)
        
        return learning_curve[:self.config['learning_curve_points']]
    
    def _calculate_generalization(self, stacking_data: Dict) -> Dict[str, float]:
        """计算泛化能力"""
        episodes = stacking_data['episodes']
        
        # 基于不同难度和物体类型的表现计算泛化能力
        difficulty_performances = defaultdict(list)
        shape_performances = defaultdict(list)
        
        for ep in episodes:
            difficulty = ep['difficulty']
            difficulty_performances[difficulty].append(ep['final_stability'])
            
            # 分析物体形状
            for obj in ep['objects']:
                shape = obj.shape
                shape_performances[shape].append(ep['final_stability'])
        
        generalization_scores = {}
        
        # 难度泛化得分
        difficulty_scores = []
        for difficulty, performances in difficulty_performances.items():
            if performances:
                difficulty_scores.append(np.mean(performances))
        
        if difficulty_scores:
            difficulty_variance = np.var(difficulty_scores)
            generalization_scores['difficulty_generation'] = max(0.0, 1.0 - difficulty_variance)
        
        # 形状泛化得分
        shape_scores = []
        for shape, performances in shape_performances.items():
            if performances:
                shape_scores.append(np.mean(performances))
        
        if shape_scores:
            shape_variance = np.var(shape_scores)
            generalization_scores['shape_generation'] = max(0.0, 1.0 - shape_variance)
        
        return generalization_scores
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """获取性能分析报告"""
        if not self.evaluation_history:
            return {"message": "暂无评估数据"}
        
        recent_evals = self.evaluation_history[-5:]  # 最近5次评估
        
        analysis_data = {
            'stacking_success_rates': [],
            'stability_scores': [],
            'heights_achieved': []
        }
        
        for eval_data in recent_evals:
            metrics = eval_data['metrics']
            analysis_data['stacking_success_rates'].append(metrics.get('stacking_success_rate', 0.0))
            analysis_data['stability_scores'].append(metrics.get('stability_score', 0.0))
            analysis_data['heights_achieved'].append(metrics.get('height_achieved', 0.0))
        
        return {
            '评估总数': len(self.evaluation_history),
            '平均成功率': np.mean(analysis_data['stacking_success_rates']),
            '平均稳定性': np.mean(analysis_data['stability_scores']),
            '平均高度': np.mean(analysis_data['heights_achieved']),
            '成功率趋势': np.polyfit(range(len(analysis_data['stacking_success_rates'])), 
                               analysis_data['stacking_success_rates'], 1)[0],
            '稳定性趋势': np.polyfit(range(len(analysis_data['stability_scores'])), 
                               analysis_data['stability_scores'], 1)[0]
        }
    
    def visualize_stacking_result(self, stacking_data: Dict, save_path: str = None):
        """可视化堆叠结果（如果matplotlib可用）"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib不可用，跳过可视化")
            return
        
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # 2D堆叠过程图
            ax1 = plt.subplot(2, 2, 1)
            episodes = stacking_data['episodes']
            
            # 绘制最近几个episode的堆叠过程
            for i, ep in enumerate(episodes[-3:]):
                stacked_objects = ep['stacked_objects']
                heights = [obj['position'][2] + obj['dimensions'][2]/2 for obj in stacked_objects]
                objects_names = [obj['name'] for obj in stacked_objects]
                
                y_pos = i * 1.5
                ax1.barh(y_pos, heights[-1] if heights else 0, height=0.3, 
                        alpha=0.7, label=f'Episode {ep["episode"]}')
                
                for j, (height, name) in enumerate(zip(heights, objects_names)):
                    ax1.text(height + 0.02, y_pos + (j - len(heights)/2) * 0.05, name, 
                            fontsize=8, ha='left', va='center')
            
            ax1.set_xlabel('堆叠高度')
            ax1.set_title('堆叠结果对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 稳定性历史
            ax2 = plt.subplot(2, 2, 2)
            if episodes:
                for ep in episodes[-3:]:
                    ax2.plot(ep['stability_history'], label=f'Episode {ep["episode"]}', linewidth=2)
            
            ax2.set_xlabel('堆叠步数')
            ax2.set_ylabel('稳定性得分')
            ax2.set_title('稳定性变化曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 成功率分布
            ax3 = plt.subplot(2, 2, 3)
            success_rates = [1 if ep['successfully_stacked'] else 0 for ep in episodes]
            ax3.hist(success_rates, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('成功堆叠')
            ax3.set_ylabel('频次')
            ax3.set_title('成功率分布')
            ax3.grid(True, alpha=0.3)
            
            # 性能指标雷达图
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            
            if episodes:
                latest_ep = episodes[-1]
                metrics = {
                    '稳定性': latest_ep['final_stability'],
                    '高度': latest_ep['total_height'] / 2.0,  # 标准化到0-1
                    '成功率': 1.0 if latest_ep['successfully_stacked'] else 0.0,
                    '精确度': 0.8  # 简化值
                }
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                values = list(metrics.values())
                angles = np.concatenate((angles, [angles[0]]))
                values = np.concatenate((values, [values[0]]))
                
                ax4.plot(angles, values, 'o-', linewidth=2, label='当前性能')
                ax4.fill(angles, values, alpha=0.25)
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(metrics.keys())
                ax4.set_title('性能指标雷达图')
                ax4.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"堆叠可视化已保存: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"堆叠可视化失败: {e}")
    
    def save_evaluation_data(self, filepath: str):
        """保存评估数据"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"堆叠评估数据已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存堆叠评估数据失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("堆叠评估器资源已清理")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()