"""
跨任务性能测试的适应度评估系统核心实现

该模块实现了FitnessEvaluator类，支持：
1. 跨任务性能测试（CoinRun、村庄交易、真实堆叠三个任务）
2. 使用Ray库进行并行评估
3. 多维度适应度计算（游戏得分、学习速度、泛化能力）
4. 实时性能监控和数据记录
5. 适应度地形可视化

主要功能：
- evaluate_game_performance(): CoinRun任务上的游戏得分测试
- evaluate_trading_performance(): 村庄交易任务上的交易技能测试  
- evaluate_real_world_performance(): 真实堆叠任务上的物理世界能力测试
- calculate_fitness_tuple(): 综合计算三维适应度值
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray未安装，将使用多线程模式替代并行评估")

from .task_evaluators.coinrun_evaluator import CoinRunEvaluator
from .task_evaluators.trading_evaluator import TradingEvaluator
from .task_evaluators.real_world_evaluator import StackingEvaluator

# 为了保持兼容性，使用别名
RealWorldEvaluator = StackingEvaluator


@dataclass
class FitnessResult:
    """适应度评估结果数据类"""
    game_score: float  # 游戏得分
    trading_score: float  # 交易得分
    real_world_score: float  # 现实世界得分
    learning_speed: float  # 学习速度
    generalization_ability: float  # 泛化能力
    total_fitness: float  # 总适应度
    evaluation_time: float  # 评估用时
    detailed_metrics: Dict[str, Any]  # 详细指标
    timestamp: float  # 评估时间戳

    def __post_init__(self):
        """验证分数范围并标准化"""
        self.game_score = max(0.0, min(1.0, self.game_score))
        self.trading_score = max(0.0, min(1.0, self.trading_score))
        self.real_world_score = max(0.0, min(1.0, self.real_world_score))
        self.learning_speed = max(0.0, min(1.0, self.learning_speed))
        self.generalization_ability = max(0.0, min(1.0, self.generalization_ability))
        
        # 计算总适应度（加权平均）
        weights = [0.3, 0.25, 0.25, 0.1, 0.1]  # 游戏、交易、现实、学习、泛化的权重
        scores = [self.game_score, self.trading_score, self.real_world_score, 
                 self.learning_speed, self.generalization_ability]
        self.total_fitness = sum(w * s for w, s in zip(weights, scores))


class FitnessEvaluator:
    """
    跨任务性能测试的适应度评估系统
    
    该类实现了多维度的适应度评估，支持并行计算和实时监控。
    通过在三个不同的任务域中评估智能体的性能来判断其适应度。
    
    Attributes:
        num_workers (int): 并行工作进程数
        ray_initialized (bool): Ray是否已初始化
        evaluators (Dict): 任务评估器字典
        performance_history (deque): 性能历史记录
        logger (logging.Logger): 日志记录器
        config (Dict): 配置参数
    """
    
    def __init__(self, 
                 num_workers: int = 4,
                 use_ray: bool = True,
                 config: Optional[Dict] = None,
                 log_dir: str = "evolution_logs"):
        """
        初始化适应度评估器
        
        Args:
            num_workers: 并行工作进程数
            use_ray: 是否使用Ray进行并行计算
            config: 配置参数字典
            log_dir: 日志目录
        """
        self.num_workers = min(num_workers, os.cpu_count() or 1)
        self.use_ray = use_ray and RAY_AVAILABLE
        self.ray_initialized = False
        
        # 设置默认配置
        self.config = self._setup_default_config()
        if config:
            self.config.update(config)
        
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志记录
        self._setup_logging()
        
        # 初始化评估器
        self.evaluators = {}
        self._initialize_evaluators()
        
        # 性能历史记录（用于趋势分析）
        self.performance_history = deque(maxlen=1000)
        
        # 性能统计
        self.stats_lock = threading.Lock()
        self.performance_stats = defaultdict(list)
        
        # Ray初始化
        if self.use_ray:
            self._initialize_ray()
        
        self.logger.info(f"适应度评估器初始化完成 - 并行模式: {'Ray' if self.use_ray else 'ThreadPool'}")
    
    def _setup_default_config(self) -> Dict:
        """设置默认配置参数"""
        return {
            # 评估参数
            'max_evaluation_steps': 10000,
            'evaluation_episodes': 5,
            'confidence_threshold': 0.95,
            
            # 并行评估参数
            'timeout_per_task': 300,  # 300秒超时
            'batch_size': 8,
            
            # 性能监控参数
            'monitoring_interval': 10,  # 每10次评估记录一次统计
            'save_interval': 50,  # 每50次评估保存一次数据
            
            # 适应度计算参数
            'fitness_weights': {
                'game_score': 0.3,
                'trading_score': 0.25,
                'real_world_score': 0.25,
                'learning_speed': 0.1,
                'generalization_ability': 0.1
            },
            
            # 可视化参数
            'visualization': {
                'save_plots': True,
                'plot_interval': 100,
                'fitness_landscape': True
            },
            
            # 数据记录参数
            'record_detailed_metrics': True,
            'performance_tracking': True
        }
    
    def _setup_logging(self):
        """设置日志记录系统"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加handler
        if not self.logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 文件输出
            log_file = os.path.join(self.log_dir, 'fitness_evaluator.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def _initialize_ray(self):
        """初始化Ray并行计算框架"""
        try:
            if not ray.is_initialized():
                ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
                self.ray_initialized = True
                self.logger.info("Ray框架初始化成功")
                
                # 动态添加ray.remote装饰器
                self._ray_evaluate_game = ray.remote(self._ray_evaluate_game)
                self._ray_evaluate_trading = ray.remote(self._ray_evaluate_trading)  
                self._ray_evaluate_real_world = ray.remote(self._ray_evaluate_real_world)
                
            else:
                self.ray_initialized = True
                self.logger.info("Ray框架已存在，直接使用")
        except Exception as e:
            self.logger.warning(f"Ray初始化失败，将使用多线程模式: {e}")
            self.use_ray = False
            self.ray_initialized = False
    
    def _initialize_evaluators(self):
        """初始化各任务评估器"""
        try:
            self.evaluators['coinrun'] = CoinRunEvaluator(
                config=self.config.get('coinrun', {})
            )
            self.evaluators['trading'] = TradingEvaluator(
                config=self.config.get('trading', {})
            )
            self.evaluators['real_world'] = RealWorldEvaluator(
                config=self.config.get('real_world', {})
            )
            self.logger.info("所有任务评估器初始化完成")
        except Exception as e:
            self.logger.error(f"评估器初始化失败: {e}")
            raise
    
    def evaluate_game_performance(self, agent: Any, 
                                evaluation_config: Optional[Dict] = None) -> Dict[str, float]:
        """
        在CoinRun任务上测试智能体的游戏得分能力
        
        Args:
            agent: 待评估的智能体
            evaluation_config: 评估配置参数
            
        Returns:
            包含游戏性能指标的字典
        """
        start_time = time.time()
        
        try:
            # 获取CoinRun评估器
            evaluator = self.evaluators.get('coinrun')
            if not evaluator:
                raise ValueError("CoinRun评估器未初始化")
            
            # 执行游戏性能评估
            game_metrics = evaluator.evaluate(
                agent, 
                config=evaluation_config or self.config.get('game_evaluation', {})
            )
            
            # 计算衍生指标
            game_score = self._calculate_game_score(game_metrics)
            learning_speed = self._calculate_learning_speed(game_metrics, 'game')
            generalization = self._calculate_generalization(game_metrics, 'game')
            
            evaluation_time = time.time() - start_time
            
            result = {
                'primary_score': game_score,
                'learning_speed': learning_speed,
                'generalization_ability': generalization,
                'evaluation_time': evaluation_time,
                'detailed_metrics': game_metrics
            }
            
            self.logger.debug(f"游戏性能评估完成: {game_score:.4f} (用时: {evaluation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"游戏性能评估失败: {e}")
            return {
                'primary_score': 0.0,
                'learning_speed': 0.0,
                'generalization_ability': 0.0,
                'evaluation_time': time.time() - start_time,
                'detailed_metrics': {'error': str(e)}
            }
    
    def evaluate_trading_performance(self, agent: Any,
                                   evaluation_config: Optional[Dict] = None) -> Dict[str, float]:
        """
        在村庄交易任务上测试智能体的交易技能
        
        Args:
            agent: 待评估的智能体
            evaluation_config: 评估配置参数
            
        Returns:
            包含交易性能指标的字典
        """
        start_time = time.time()
        
        try:
            # 获取交易评估器
            evaluator = self.evaluators.get('trading')
            if not evaluator:
                raise ValueError("交易评估器未初始化")
            
            # 执行交易性能评估
            trading_metrics = evaluator.evaluate(
                agent,
                config=evaluation_config or self.config.get('trading_evaluation', {})
            )
            
            # 计算衍生指标
            trading_score = self._calculate_trading_score(trading_metrics)
            learning_speed = self._calculate_learning_speed(trading_metrics, 'trading')
            generalization = self._calculate_generalization(trading_metrics, 'trading')
            
            evaluation_time = time.time() - start_time
            
            result = {
                'primary_score': trading_score,
                'learning_speed': learning_speed,
                'generalization_ability': generalization,
                'evaluation_time': evaluation_time,
                'detailed_metrics': trading_metrics
            }
            
            self.logger.debug(f"交易性能评估完成: {trading_score:.4f} (用时: {evaluation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"交易性能评估失败: {e}")
            return {
                'primary_score': 0.0,
                'learning_speed': 0.0,
                'generalization_ability': 0.0,
                'evaluation_time': time.time() - start_time,
                'detailed_metrics': {'error': str(e)}
            }
    
    def evaluate_real_world_performance(self, agent: Any,
                                      evaluation_config: Optional[Dict] = None) -> Dict[str, float]:
        """
        在真实堆叠任务上测试智能体的物理世界能力
        
        Args:
            agent: 待评估的智能体
            evaluation_config: 评估配置参数
            
        Returns:
            包含现实世界性能指标的字典
        """
        start_time = time.time()
        
        try:
            # 获取现实世界评估器
            evaluator = self.evaluators.get('real_world')
            if not evaluator:
                raise ValueError("现实世界评估器未初始化")
            
            # 执行现实世界性能评估
            real_world_metrics = evaluator.evaluate(
                agent,
                config=evaluation_config or self.config.get('real_world_evaluation', {})
            )
            
            # 计算衍生指标
            real_world_score = self._calculate_real_world_score(real_world_metrics)
            learning_speed = self._calculate_learning_speed(real_world_metrics, 'real_world')
            generalization = self._calculate_generalization(real_world_metrics, 'real_world')
            
            evaluation_time = time.time() - start_time
            
            result = {
                'primary_score': real_world_score,
                'learning_speed': learning_speed,
                'generalization_ability': generalization,
                'evaluation_time': evaluation_time,
                'detailed_metrics': real_world_metrics
            }
            
            self.logger.debug(f"现实世界性能评估完成: {real_world_score:.4f} (用时: {evaluation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"现实世界性能评估失败: {e}")
            return {
                'primary_score': 0.0,
                'learning_speed': 0.0,
                'generalization_ability': 0.0,
                'evaluation_time': time.time() - start_time,
                'detailed_metrics': {'error': str(e)}
            }
    
    def calculate_fitness_tuple(self, agent: Any,
                              evaluation_config: Optional[Dict] = None) -> FitnessResult:
        """
        综合计算三维适应度值
        
        该方法是核心方法，汇总三个任务域的性能，生成最终的适应度评估结果。
        使用多进程并行执行三个任务评估，显著提高评估效率。
        
        Args:
            agent: 待评估的智能体
            evaluation_config: 评估配置参数
            
        Returns:
            FitnessResult: 包含所有适应度维度的综合评估结果
        """
        start_time = time.time()
        
        # 并行执行三个任务评估
        if self.use_ray:
            result = self._parallel_evaluate_with_ray(agent, evaluation_config)
        else:
            result = self._parallel_evaluate_with_threads(agent, evaluation_config)
        
        evaluation_time = time.time() - start_time
        
        # 构建最终结果
        fitness_result = FitnessResult(
            game_score=result['game']['primary_score'],
            trading_score=result['trading']['primary_score'],
            real_world_score=result['real_world']['primary_score'],
            learning_speed=np.mean([
                result['game']['learning_speed'],
                result['trading']['learning_speed'],
                result['real_world']['learning_speed']
            ]),
            generalization_ability=np.mean([
                result['game']['generalization_ability'],
                result['trading']['generalization_ability'],
                result['real_world']['generalization_ability']
            ]),
            total_fitness=0.0,  # 将在__post_init__中计算
            evaluation_time=evaluation_time,
            detailed_metrics={
                'game_metrics': result['game']['detailed_metrics'],
                'trading_metrics': result['trading']['detailed_metrics'],
                'real_world_metrics': result['real_world']['detailed_metrics']
            },
            timestamp=time.time()
        )
        
        # 记录性能统计
        self._record_performance_stats(fitness_result)
        
        # 定期保存数据
        if len(self.performance_history) % self.config['save_interval'] == 0:
            self._save_performance_history()
        
        # 定期生成可视化
        if (self.config['visualization']['save_plots'] and 
            len(self.performance_history) % self.config['visualization']['plot_interval'] == 0):
            self._generate_fitness_landscape()
        
        self.logger.info(f"适应度评估完成 - 总分: {fitness_result.total_fitness:.4f}")
        return fitness_result
    
    def _parallel_evaluate_with_ray(self, agent: Any, config: Optional[Dict]) -> Dict:
        """使用Ray进行并行评估"""
        try:
            # 创建远程评估任务
            game_future = self._ray_evaluate_game.remote(agent, config)
            trading_future = self._ray_evaluate_trading.remote(agent, config)
            real_world_future = self._ray_evaluate_real_world.remote(agent, config)
            
            # 等待所有任务完成
            game_result = ray.get(game_future)
            trading_result = ray.get(trading_future)
            real_world_result = ray.get(real_world_future)
            
            return {
                'game': game_result,
                'trading': trading_result,
                'real_world': real_world_result
            }
            
        except Exception as e:
            self.logger.error(f"Ray并行评估失败，回退到多线程: {e}")
            return self._parallel_evaluate_with_threads(agent, config)
    
    def _ray_evaluate_game(self, agent: Any, config: Optional[Dict]):
        """Ray远程游戏评估函数"""
        return self.evaluate_game_performance(agent, config)
    
    def _ray_evaluate_trading(self, agent: Any, config: Optional[Dict]):
        """Ray远程交易评估函数"""
        return self.evaluate_trading_performance(agent, config)
    
    def _ray_evaluate_real_world(self, agent: Any, config: Optional[Dict]):
        """Ray远程现实世界评估函数"""
        return self.evaluate_real_world_performance(agent, config)
    
    def _parallel_evaluate_with_threads(self, agent: Any, config: Optional[Dict]) -> Dict:
        """使用ThreadPoolExecutor进行并行评估"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交任务
            futures = {
                executor.submit(self.evaluate_game_performance, agent, config): 'game',
                executor.submit(self.evaluate_trading_performance, agent, config): 'trading',
                executor.submit(self.evaluate_real_world_performance, agent, config): 'real_world'
            }
            
            # 收集结果
            results = {}
            for future in as_completed(futures, timeout=self.config['timeout_per_task']):
                task_type = futures[future]
                try:
                    results[task_type] = future.result()
                except Exception as e:
                    self.logger.error(f"任务 {task_type} 执行失败: {e}")
                    # 返回默认值
                    results[task_type] = {
                        'primary_score': 0.0,
                        'learning_speed': 0.0,
                        'generalization_ability': 0.0,
                        'evaluation_time': 0.0,
                        'detailed_metrics': {'error': str(e)}
                    }
            
            # 确保所有任务都已完成
            for task_type in ['game', 'trading', 'real_world']:
                if task_type not in results:
                    results[task_type] = {
                        'primary_score': 0.0,
                        'learning_speed': 0.0,
                        'generalization_ability': 0.0,
                        'evaluation_time': 0.0,
                        'detailed_metrics': {'timeout': '任务超时'}
                    }
            
            return results
    
    def _calculate_game_score(self, metrics: Dict[str, Any]) -> float:
        """计算游戏得分（基于CoinRun性能指标）"""
        if 'error' in metrics:
            return 0.0
        
        # 主要指标：完成率、得分、存活时间
        completion_rate = metrics.get('completion_rate', 0.0)
        avg_score = metrics.get('avg_score', 0.0)
        avg_survival_time = metrics.get('avg_survival_time', 0.0)
        
        # 标准化到0-1范围
        normalized_completion = min(1.0, completion_rate)
        normalized_score = min(1.0, avg_score / 1000.0)  # 假设1000为满分
        normalized_survival = min(1.0, avg_survival_time / 300.0)  # 假设300秒为满分
        
        # 加权计算
        game_score = 0.4 * normalized_completion + 0.4 * normalized_score + 0.2 * normalized_survival
        
        return max(0.0, min(1.0, game_score))
    
    def _calculate_trading_score(self, metrics: Dict[str, Any]) -> float:
        """计算交易得分（基于村庄交易性能指标）"""
        if 'error' in metrics:
            return 0.0
        
        # 主要指标：交易成功率、利润率、库存管理
        success_rate = metrics.get('success_rate', 0.0)
        profit_margin = metrics.get('profit_margin', 0.0)
        inventory_efficiency = metrics.get('inventory_efficiency', 0.0)
        
        # 标准化到0-1范围
        normalized_success = min(1.0, success_rate)
        normalized_profit = min(1.0, max(0.0, (profit_margin + 1.0) / 2.0))  # -1到1转换为0到1
        normalized_inventory = min(1.0, inventory_efficiency)
        
        # 加权计算
        trading_score = 0.4 * normalized_success + 0.3 * normalized_profit + 0.3 * normalized_inventory
        
        return max(0.0, min(1.0, trading_score))
    
    def _calculate_real_world_score(self, metrics: Dict[str, Any]) -> float:
        """计算现实世界得分（基于真实堆叠性能指标）"""
        if 'error' in metrics:
            return 0.0
        
        # 主要指标：堆叠成功率、稳定性、物理合理性
        stacking_success = metrics.get('stacking_success_rate', 0.0)
        stability_score = metrics.get('stability_score', 0.0)
        physics_accuracy = metrics.get('physics_accuracy', 0.0)
        
        # 标准化到0-1范围
        normalized_stacking = min(1.0, stacking_success)
        normalized_stability = min(1.0, stability_score)
        normalized_physics = min(1.0, physics_accuracy)
        
        # 加权计算
        real_world_score = 0.4 * normalized_stacking + 0.3 * normalized_stability + 0.3 * normalized_physics
        
        return max(0.0, min(1.0, real_world_score))
    
    def _calculate_learning_speed(self, metrics: Dict[str, Any], task_type: str) -> float:
        """计算学习速度指标"""
        if 'error' in metrics:
            return 0.0
        
        # 基于学习曲线斜率计算学习速度
        learning_curve = metrics.get('learning_curve', [])
        if len(learning_curve) < 2:
            return 0.0
        
        # 计算性能改善率
        initial_performance = learning_curve[0] if learning_curve else 0.0
        final_performance = learning_curve[-1] if learning_curve else 0.0
        
        if initial_performance <= 0:
            return 0.0
        
        improvement_rate = (final_performance - initial_performance) / initial_performance
        return max(0.0, min(1.0, improvement_rate))
    
    def _calculate_generalization(self, metrics: Dict[str, Any], task_type: str) -> float:
        """计算泛化能力指标"""
        if 'error' in metrics:
            return 0.0
        
        # 基于跨环境/任务表现计算泛化能力
        test_performance = metrics.get('test_performance', {})
        if not test_performance:
            return 0.0
        
        # 多个测试环境的表现方差（低方差表示高泛化）
        performances = list(test_performance.values())
        if len(performances) < 2:
            return 0.0
        
        variance = np.var(performances)
        mean_performance = np.mean(performances)
        
        # 方差越小，泛化能力越强
        generalization_score = max(0.0, 1.0 - variance / (mean_performance + 0.001))
        
        return max(0.0, min(1.0, generalization_score))
    
    def _record_performance_stats(self, result: FitnessResult):
        """记录性能统计信息"""
        with self.stats_lock:
            self.performance_history.append(asdict(result))
            
            # 记录详细统计
            self.performance_stats['total_fitness'].append(result.total_fitness)
            self.performance_stats['game_scores'].append(result.game_score)
            self.performance_stats['trading_scores'].append(result.trading_score)
            self.performance_stats['real_world_scores'].append(result.real_world_score)
            self.performance_stats['learning_speeds'].append(result.learning_speed)
            self.performance_stats['generalization_scores'].append(result.generalization_ability)
            self.performance_stats['evaluation_times'].append(result.evaluation_time)
    
    def _save_performance_history(self):
        """保存性能历史数据"""
        try:
            timestamp = int(time.time())
            filename = f"performance_history_{timestamp}.json"
            filepath = os.path.join(self.log_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(list(self.performance_history), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"性能历史数据已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存性能历史数据失败: {e}")
    
    def _generate_fitness_landscape(self):
        """生成适应度地形可视化"""
        if not self.config['visualization']['fitness_landscape']:
            return
        
        try:
            if len(self.performance_history) < 10:  # 数据不足
                return
            
            # 提取数据
            fitness_data = np.array([r['total_fitness'] for r in self.performance_history])
            game_data = np.array([r['game_score'] for r in self.performance_history])
            trading_data = np.array([r['trading_score'] for r in self.performance_history])
            real_world_data = np.array([r['real_world_score'] for r in self.performance_history])
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('适应度评估结果分析', fontsize=16, fontweight='bold')
            
            # 总适应度趋势
            axes[0, 0].plot(fitness_data, 'b-', linewidth=2, label='总适应度')
            axes[0, 0].set_title('总适应度变化趋势')
            axes[0, 0].set_xlabel('评估次数')
            axes[0, 0].set_ylabel('适应度得分')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # 各任务性能对比
            axes[0, 1].plot(game_data, 'r-', label='游戏性能', linewidth=2)
            axes[0, 1].plot(trading_data, 'g-', label='交易性能', linewidth=2)
            axes[0, 1].plot(real_world_data, 'orange', label='现实世界性能', linewidth=2)
            axes[0, 1].set_title('各任务域性能对比')
            axes[0, 1].set_xlabel('评估次数')
            axes[0, 1].set_ylabel('性能得分')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # 性能分布直方图
            axes[1, 0].hist(fitness_data, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('总适应度分布')
            axes[1, 0].set_xlabel('适应度得分')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 相关性热力图
            correlation_matrix = np.corrcoef([
                game_data, trading_data, real_world_data, fitness_data
            ])
            im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title('性能指标相关性矩阵')
            axes[1, 1].set_xticks(range(4))
            axes[1, 1].set_yticks(range(4))
            axes[1, 1].set_xticklabels(['游戏', '交易', '现实', '总适应'])
            axes[1, 1].set_yticklabels(['游戏', '交易', '现实', '总适应'])
            
            # 添加相关系数文本
            for i in range(4):
                for j in range(4):
                    axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                  ha='center', va='center', color='white', fontweight='bold')
            
            plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = int(time.time())
            filename = f"fitness_landscape_{timestamp}.png"
            filepath = os.path.join(self.log_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"适应度地形图已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"生成适应度地形图失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能评估总结报告"""
        with self.stats_lock:
            if not self.performance_stats['total_fitness']:
                return {"message": "暂无性能数据"}
            
            summary = {
                '评估总数': len(self.performance_stats['total_fitness']),
                '平均总适应度': np.mean(self.performance_stats['total_fitness']),
                '最佳总适应度': np.max(self.performance_stats['total_fitness']),
                '适应度标准差': np.std(self.performance_stats['total_fitness']),
                '平均游戏性能': np.mean(self.performance_stats['game_scores']),
                '平均交易性能': np.mean(self.performance_stats['trading_scores']),
                '平均现实世界性能': np.mean(self.performance_stats['real_world_scores']),
                '平均学习速度': np.mean(self.performance_stats['learning_speeds']),
                '平均泛化能力': np.mean(self.performance_stats['generalization_scores']),
                '平均评估时间': np.mean(self.performance_stats['evaluation_times'])
            }
            
            return summary
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.ray_initialized and ray.is_initialized():
                ray.shutdown()
                self.logger.info("Ray资源已清理")
        except Exception as e:
            self.logger.warning(f"清理Ray资源时出错: {e}")
        
        # 保存最终数据
        if self.performance_history:
            self._save_performance_history()
            self.logger.info("最终性能数据已保存")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.cleanup()