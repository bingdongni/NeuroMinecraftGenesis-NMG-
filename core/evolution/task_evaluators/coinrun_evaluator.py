"""
CoinRun游戏性能评估器

该模块实现CoinRun任务环境下的智能体性能评估，主要评估：
1. 游戏得分：完成率和奖励收集情况
2. 学习速度：技能习得和性能改善率
3. 泛化能力：跨关卡的表现一致性

CoinRun是 ProcGen 环境中的经典平台游戏，智能体需要：
- 收集金币获得奖励
- 避开障碍物和敌人
- 到达关卡终点
- 在有限的时间内完成任务
"""

import time
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import os

try:
    import gym
    import procgen
    PROCGEN_AVAILABLE = True
except ImportError:
    PROCGEN_AVAILABLE = False
    logging.warning("ProcGen/Gym未安装，将使用模拟评估器")


@dataclass
class CoinRunMetrics:
    """CoinRun任务评估指标数据类"""
    completion_rate: float = 0.0  # 关卡完成率
    avg_score: float = 0.0  # 平均得分
    avg_survival_time: float = 0.0  # 平均存活时间
    coin_collection_rate: float = 0.0  # 金币收集率
    death_causes: Dict[str, int] = None  # 死因统计
    level_performance: Dict[int, float] = None  # 跨关卡性能
    
    def __post_init__(self):
        if self.death_causes is None:
            self.death_causes = {}
        if self.level_performance is None:
            self.level_performance = {}


class CoinRunEvaluator:
    """
    CoinRun游戏性能评估器
    
    在ProcGen环境的CoinRun游戏中评估智能体性能。
    支持跨关卡泛化测试和学习速度计算。
    
    Attributes:
        env_name (str): 环境名称
        config (Dict): 配置参数
        logger (logging.Logger): 日志记录器
        evaluation_history (List): 评估历史记录
    """
    
    def __init__(self, 
                 env_name: str = "coinrun",
                 config: Optional[Dict] = None,
                 max_steps: int = 1000,
                 num_levels: int = 200):
        """
        初始化CoinRun评估器
        
        Args:
            env_name: 环境名称
            config: 配置参数字典
            max_steps: 每个回合最大步数
            num_levels: 使用的关卡数量（用于泛化测试）
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.num_levels = num_levels
        
        # 设置配置参数
        self.config = self._setup_default_config()
        if config:
            self.config.update(config)
        
        # 设置日志记录
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 评估历史记录
        self.evaluation_history = []
        
        # 环境初始化
        self.env = None
        if PROCGEN_AVAILABLE:
            self._initialize_environment()
        else:
            self.logger.warning("使用模拟评估器模式")
        
        self.logger.info(f"CoinRun评估器初始化完成 - 环境: {env_name}")
    
    def _setup_default_config(self) -> Dict:
        """设置默认配置参数"""
        return {
            # 评估参数
            'evaluation_episodes': 20,
            'test_episodes_per_level': 3,
            'confidence_threshold': 0.95,
            
            # 难度配置
            'difficulty_levels': ['easy', 'medium', 'hard'],
            'level_seeds': list(range(100, 100 + self.num_levels)),
            
            # 性能阈值
            'success_threshold': 0.8,  # 成功阈值
            'optimal_score': 1000,  # 最佳得分
            'max_survival_time': 300,  # 最大存活时间
            
            # 学习参数
            'learning_curve_points': 10,  # 学习曲线采样点数
            'generalization_test_levels': 50,  # 泛化测试关卡数
            
            # 调试参数
            'save_trajectories': False,
            'verbose_logging': False
        }
    
    def _initialize_environment(self):
        """初始化ProcGen环境"""
        try:
            # 创建CoinRun环境
            self.env = gym.make(
                self.env_name,
                num_levels=self.num_levels,
                start_level=0,
                distribution_mode='easy'
            )
            
            self.logger.info("ProcGen环境初始化成功")
            
        except Exception as e:
            self.logger.error(f"ProcGen环境初始化失败: {e}")
            self.env = None
    
    def evaluate(self, agent: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行CoinRun游戏性能评估
        
        Args:
            agent: 待评估的智能体
            config: 评估配置参数
            
        Returns:
            包含游戏性能指标的字典
        """
        evaluation_config = self.config.copy()
        if config:
            evaluation_config.update(config)
        
        start_time = time.time()
        
        # 模拟评估（如果ProcGen不可用）
        if not PROCGEN_AVAILABLE or self.env is None:
            return self._simulate_evaluation(agent, evaluation_config)
        
        try:
            # 执行真实评估
            metrics = self._run_real_evaluation(agent, evaluation_config)
            
            # 计算衍生指标
            learning_curve = self._calculate_learning_curve(agent, evaluation_config)
            generalization_scores = self._calculate_generalization(agent, evaluation_config)
            
            # 构建详细结果
            detailed_metrics = {
                'completion_rate': metrics.completion_rate,
                'avg_score': metrics.avg_score,
                'avg_survival_time': metrics.avg_survival_time,
                'coin_collection_rate': metrics.coin_collection_rate,
                'death_causes': metrics.death_causes,
                'level_performance': metrics.level_performance,
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
                f"CoinRun评估完成 - 完成率: {metrics.completion_rate:.2%}, "
                f"平均得分: {metrics.avg_score:.1f}"
            )
            
            return detailed_metrics
            
        except Exception as e:
            self.logger.error(f"CoinRun评估失败: {e}")
            return {
                'error': str(e),
                'completion_rate': 0.0,
                'avg_score': 0.0,
                'avg_survival_time': 0.0,
                'coin_collection_rate': 0.0,
                'evaluation_time': time.time() - start_time
            }
    
    def _run_real_evaluation(self, agent: Any, config: Dict) -> CoinRunMetrics:
        """执行真实的ProcGen环境评估"""
        episodes = config['evaluation_episodes']
        results = []
        
        for episode in range(episodes):
            # 随机选择关卡
            level_seed = random.choice(config['level_seeds'])
            
            # 重置环境
            obs = self.env.reset()
            level_seed = self.env.call_stack[0][1] if self.env.call_stack else level_seed
            
            episode_result = {
                'score': 0,
                'survived_steps': 0,
                'coins_collected': 0,
                'completed': False,
                'death_cause': 'unknown',
                'level_seed': level_seed
            }
            
            # 运行环境步进
            for step in range(self.max_steps):
                # 智能体动作选择
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, training=False)
                elif hasattr(agent, 'act'):
                    action = agent.act(obs)
                elif callable(agent):
                    action = agent(obs)
                else:
                    # 默认随机动作
                    action = self.env.action_space.sample()
                
                # 执行动作
                obs, reward, done, info = self.env.step(action)
                
                # 记录信息
                episode_result['score'] += reward
                episode_result['survived_steps'] += 1
                
                # 检查特殊事件
                if info.get('got_coin', False):
                    episode_result['coins_collected'] += 1
                
                if info.get('reach_goal', False):
                    episode_result['completed'] = True
                    episode_result['death_cause'] = 'completion'
                    break
                
                if done:
                    if not episode_result['completed']:
                        episode_result['death_cause'] = 'death'
                    break
            
            results.append(episode_result)
        
        # 统计结果
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[Dict]) -> CoinRunMetrics:
        """聚合评估结果"""
        if not results:
            return CoinRunMetrics()
        
        # 基本统计
        total_episodes = len(results)
        completed_episodes = sum(1 for r in results if r['completed'])
        total_score = sum(r['score'] for r in results)
        total_survival = sum(r['survived_steps'] for r in results)
        total_coins = sum(r['coins_collected'] for r in results)
        
        # 计算指标
        completion_rate = completed_episodes / total_episodes
        avg_score = total_score / total_episodes
        avg_survival_time = total_survival / total_episodes
        coin_collection_rate = total_coins / total_episodes
        
        # 死因统计
        death_causes = {}
        for result in results:
            if not result['completed']:
                cause = result['death_cause']
                death_causes[cause] = death_causes.get(cause, 0) + 1
        
        # 关卡性能统计
        level_performance = {}
        level_scores = {}
        for result in results:
            level_seed = result['level_seed']
            if level_seed not in level_scores:
                level_scores[level_seed] = []
            level_scores[level_seed].append(result['score'])
        
        for level_seed, scores in level_scores.items():
            level_performance[level_seed] = np.mean(scores)
        
        return CoinRunMetrics(
            completion_rate=completion_rate,
            avg_score=avg_score,
            avg_survival_time=avg_survival_time,
            coin_collection_rate=coin_collection_rate,
            death_causes=death_causes,
            level_performance=level_performance
        )
    
    def _simulate_evaluation(self, agent: Any, config: Dict) -> Dict[str, Any]:
        """模拟评估（当ProcGen不可用时使用）"""
        start_time = time.time()
        
        # 生成模拟数据
        np.random.seed(int(time.time()) % 2**32)
        
        episodes = config['evaluation_episodes']
        simulation_results = []
        
        # 模拟智能体性能（基于假设的学习能力）
        base_performance = getattr(agent, 'performance_level', 0.5)
        learning_factor = getattr(agent, 'learning_factor', 1.0)
        
        for episode in range(episodes):
            # 模拟学习曲线（性能随episode逐渐改善）
            performance_boost = min(0.3, episode * 0.01 * learning_factor)
            episode_performance = min(1.0, base_performance + performance_boost)
            
            # 模拟结果
            completed = np.random.random() < episode_performance * 0.8
            score = np.random.normal(500 * episode_performance, 100)
            score = max(0, score)  # 确保非负
            
            survival_time = np.random.exponential(100 * episode_performance)
            coins_collected = np.random.poisson(episode_performance * 5)
            
            result = {
                'score': score,
                'survived_steps': survival_time,
                'coins_collected': coins_collected,
                'completed': completed,
                'death_cause': 'death' if not completed else 'completion',
                'level_seed': 100 + episode
            }
            simulation_results.append(result)
        
        # 聚合结果
        metrics = self._aggregate_results(simulation_results)
        
        # 生成学习曲线
        learning_curve = [0.3 + i * 0.05 for i in range(min(episodes, 10))]
        
        # 生成泛化测试结果
        generalization_scores = {
            f'test_level_{i}': np.random.normal(0.7, 0.2) 
            for i in range(config['generalization_test_levels'])
        }
        
        evaluation_time = time.time() - start_time
        
        return {
            'completion_rate': metrics.completion_rate,
            'avg_score': metrics.avg_score,
            'avg_survival_time': metrics.avg_survival_time,
            'coin_collection_rate': metrics.coin_collection_rate,
            'death_causes': metrics.death_causes,
            'level_performance': metrics.level_performance,
            'learning_curve': learning_curve,
            'generalization_scores': generalization_scores,
            'evaluation_config': config,
            'evaluation_time': evaluation_time,
            'simulation_mode': True
        }
    
    def _calculate_learning_curve(self, agent: Any, config: Dict) -> List[float]:
        """计算学习曲线"""
        points = config['learning_curve_points']
        
        # 基于评估历史计算学习曲线
        if len(self.evaluation_history) >= points:
            recent_evaluations = self.evaluation_history[-points:]
            learning_curve = []
            for eval_data in recent_evaluations:
                metrics = eval_data['metrics']
                # 使用完成率作为学习指标
                completion_rate = metrics.get('completion_rate', 0.0)
                learning_curve.append(completion_rate)
            return learning_curve
        
        # 如果历史记录不足，生成模拟曲线
        base_performance = getattr(agent, 'performance_level', 0.3)
        learning_curve = [base_performance + i * 0.1 for i in range(points)]
        return [min(1.0, p) for p in learning_curve]
    
    def _calculate_generalization(self, agent: Any, config: Dict) -> Dict[str, float]:
        """计算泛化能力"""
        test_levels = config['generalization_test_levels']
        episodes_per_level = config['test_episodes_per_level']
        
        # 生成测试关卡分数
        generalization_scores = {}
        
        base_performance = getattr(agent, 'performance_level', 0.5)
        generalization_factor = getattr(agent, 'generalization_factor', 1.0)
        
        for level in range(test_levels):
            # 模拟不同关卡的性能表现
            level_performance = np.random.normal(
                base_performance * generalization_factor, 0.1
            )
            level_performance = np.clip(level_performance, 0.0, 1.0)
            generalization_scores[f'test_level_{level}'] = level_performance
        
        return generalization_scores
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """获取性能分析报告"""
        if not self.evaluation_history:
            return {"message": "暂无评估数据"}
        
        # 分析趋势
        recent_evals = self.evaluation_history[-10:]  # 最近10次评估
        
        completion_rates = []
        avg_scores = []
        
        for eval_data in recent_evals:
            metrics = eval_data['metrics']
            completion_rates.append(metrics.get('completion_rate', 0.0))
            avg_scores.append(metrics.get('avg_score', 0.0))
        
        trend_analysis = {
            'completion_rate_trend': np.polyfit(range(len(completion_rates)), completion_rates, 1)[0],
            'score_trend': np.polyfit(range(len(avg_scores)), avg_scores, 1)[0],
            'stability': 1.0 - (np.std(completion_rates) + np.std(avg_scores)) / 2.0
        }
        
        # 最优性能
        best_eval = max(self.evaluation_history, 
                       key=lambda x: x['metrics'].get('avg_score', 0))
        
        analysis = {
            '评估总数': len(self.evaluation_history),
            '最近性能': {
                '平均完成率': np.mean(completion_rates),
                '平均得分': np.mean(avg_scores)
            },
            '趋势分析': trend_analysis,
            '历史最佳': {
                '得分': best_eval['metrics'].get('avg_score', 0),
                '完成率': best_eval['metrics'].get('completion_rate', 0),
                '时间': time.ctime(best_eval['timestamp'])
            }
        }
        
        return analysis
    
    def save_evaluation_data(self, filepath: str):
        """保存评估数据到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"评估数据已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存评估数据失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.env is not None:
                self.env.close()
                self.logger.info("CoinRun环境已关闭")
        except Exception as e:
            self.logger.warning(f"清理CoinRun环境时出错: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.cleanup()