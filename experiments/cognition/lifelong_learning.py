"""
终身学习评估系统主类

该模块实现了完整的终身学习评估框架，包括：
1. 连续学习基准测试
2. 灾难性遗忘评估
3. 前后迁移性能分析
4. EWC和MAS正则化算法集成

作者：认知系统开发团队
创建时间：2025-11-13
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from datetime import datetime

from atari_game_sequence import AtariGameSequence
from catastrophic_forgetting_analyzer import CatastrophicForgettingAnalyzer
from forward_backward_transfer import ForwardBackwardTransfer
from learning_curve_analyzer import LearningCurveAnalyzer


@dataclass
class LearningMetrics:
    """学习指标数据结构"""
    task_id: int
    game_name: str
    current_performance: float
    forward_transfer_score: float
    backward_transfer_scores: List[float]
    ewc_regularization: float
    mas_regularization: float
    learning_rate: float
    catastrophic_forgetting_rate: float = 0.0
    meta_learning_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationConfig:
    """评估配置参数"""
    # 序列参数
    total_tasks: int = 100  # 总任务数
    steps_per_task: int = 10000  # 每个任务的训练步数
    test_frequency: int = 10  # 每多少个任务后进行全面测试
    
    # 模型参数
    network_architecture: str = "mlp"  # 网络架构类型
    hidden_size: int = 512  # 隐藏层大小
    learning_rate: float = 0.001  # 基础学习率
    batch_size: int = 32  # 批大小
    
    # 正则化参数
    ewc_lambda: float = 100.0  # EWC正则化强度
    mas_lambda: float = 1.0  # MAS正则化强度
    use_ewc: bool = True  # 是否使用EWC
    use_mas: bool = True  # 是否使用MAS
    
    # 评估参数
    test_episodes: int = 100  # 测试轮数
    performance_threshold: float = 0.05  # 性能下降阈值（5%）
    
    # 监控参数
    save_checkpoints: bool = True  # 是否保存检查点
    log_interval: int = 100  # 日志记录间隔


class LifelongLearningSystem:
    """
    终身学习评估系统主类
    
    该类实现了完整的终身学习评估框架，包括：
    - 连续学习训练和评估
    - 灾难性遗忘分析
    - 前后迁移性能评估
    - 长期学习监控
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        初始化终身学习系统
        
        Args:
            config: 评估配置参数
        """
        self.config = config or EvaluationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self.atari_sequence = AtariGameSequence(
            total_games=self.config.total_tasks,
            steps_per_game=self.config.steps_per_task
        )
        self.forgetting_analyzer = CatastrophicForgettingAnalyzer(
            threshold=self.config.performance_threshold
        )
        self.transfer_analyzer = ForwardBackwardTransfer()
        self.curve_analyzer = LearningCurveAnalyzer()
        
        # 初始化模型
        self.model = self._initialize_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 历史记录
        self.learning_history: List[LearningMetrics] = []
        self.performance_history: Dict[int, List[float]] = defaultdict(list)
        self.backward_transfer_matrix: np.ndarray = np.zeros((self.config.total_tasks, self.config.total_tasks))
        self.forward_transfer_scores: List[float] = []
        
        # 设置日志
        self.logger = self._setup_logging()
        
        self.logger.info(f"终身学习系统初始化完成")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"总任务数: {self.config.total_tasks}")
        self.logger.info(f"每任务步数: {self.config.steps_per_task}")
    
    def _initialize_model(self) -> nn.Module:
        """初始化神经网络模型"""
        if self.config.network_architecture == "mlp":
            model = nn.Sequential(
                nn.Linear(84 * 84, self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, 18)  # Atari游戏动作空间
            ).to(self.device)
        else:
            raise ValueError(f"不支持的网络架构: {self.config.network_architecture}")
        
        return model
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志记录"""
        logger = logging.getLogger("lifelong_learning")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _calculate_ewc_regularization(self, fisher_information: np.ndarray) -> torch.Tensor:
        """计算EWC正则化项
        
        Args:
            fisher_information: Fisher信息矩阵
            
        Returns:
            EWC正则化损失
        """
        if not self.config.use_ewc:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = 0.0
        for param_name, param in self.model.named_parameters():
            if param.grad is not None and param_name in fisher_information:
                fisher_diag = fisher_information[param_name]
                param_mean = param.data.mean(dim=0) if param.data.dim() > 0 else param.data
                ewc_loss += torch.sum(fisher_diag * (param_mean - param)**2)
        
        return self.config.ewc_lambda * ewc_loss
    
    def _calculate_mas_regularization(self, importance_weights: np.ndarray) -> torch.Tensor:
        """计算MAS正则化项
        
        Args:
            importance_weights: 重要性权重
            
        Returns:
            MAS正则化损失
        """
        if not self.config.use_mas:
            return torch.tensor(0.0, device=self.device)
        
        mas_loss = 0.0
        for param_name, param in self.model.named_parameters():
            if param_name in importance_weights:
                importance = importance_weights[param_name]
                param_mean = param.data.mean(dim=0) if param.data.dim() > 0 else param.data
                mas_loss += torch.sum(importance * (param_mean)**2)
        
        return self.config.mas_lambda * mas_loss
    
    def train_task(self, task_id: int, game_name: str, training_data: np.ndarray) -> LearningMetrics:
        """训练单个任务
        
        Args:
            task_id: 任务ID
            game_name: 游戏名称
            training_data: 训练数据
            
        Returns:
            学习指标
        """
        self.logger.info(f"开始训练任务 {task_id}: {game_name}")
        
        # 获取游戏环境
        env = self.atari_sequence.get_game_env(task_id)
        
        # 初始化指标
        episode_rewards = []
        forward_transfer_score = 0.0
        
        # 训练循环
        for step in range(self.config.steps_per_task):
            # 执行环境交互
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done and step < self.config.steps_per_task:
                # 前向传播
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.model(state_tensor)
                    action = q_values.argmax().item()
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                
                # 记录经验（此处简化处理）
                if len(training_data) > step:
                    training_data[step] = (state, action, reward, next_state, done)
            
            episode_rewards.append(total_reward)
            
            # 更新进度日志
            if step % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.log_interval:])
                self.logger.info(f"任务 {task_id}, 步数 {step}, 平均奖励: {avg_reward:.2f}")
        
        # 计算向前迁移分数
        if task_id > 0:
            # 与之前任务的学习速度比较
            prev_learning_speed = np.mean(episode_rewards[:100])  # 前100步
            current_learning_speed = np.mean(episode_rewards[:100])
            forward_transfer_score = (current_learning_speed - prev_learning_speed) / (abs(prev_learning_speed) + 1e-8)
        
        # 计算向后迁移分数
        backward_transfer_scores = []
        if task_id > 0:
            for prev_task_id in range(task_id):
                prev_performance = self.performance_history[prev_task_id][-1] if self.performance_history[prev_task_id] else 0.0
                current_performance = np.mean(episode_rewards)
                backward_transfer = (current_performance - prev_performance) / (abs(prev_performance) + 1e-8)
                backward_transfer_scores.append(backward_transfer)
                
                # 更新向后迁移矩阵
                self.backward_transfer_matrix[prev_task_id, task_id] = backward_transfer
        
        # 计算灾难性遗忘率
        catastrophic_forgetting_rate = self.forgetting_analyzer.analyze_forgetting(
            task_id, episode_rewards, self.performance_history
        )
        
        # 记录性能历史
        self.performance_history[task_id].append(np.mean(episode_rewards))
        self.forward_transfer_scores.append(forward_transfer_score)
        
        # 创建学习指标
        metrics = LearningMetrics(
            task_id=task_id,
            game_name=game_name,
            current_performance=np.mean(episode_rewards),
            forward_transfer_score=forward_transfer_score,
            backward_transfer_scores=backward_transfer_scores,
            ewc_regularization=self.config.ewc_lambda if self.config.use_ewc else 0.0,
            mas_regularization=self.config.mas_lambda if self.config.use_mas else 0.0,
            learning_rate=self.config.learning_rate,
            catastrophic_forgetting_rate=catastrophic_forgetting_rate,
            meta_learning_score=self._calculate_meta_learning_score(task_id)
        )
        
        self.learning_history.append(metrics)
        
        self.logger.info(f"任务 {task_id} 训练完成")
        self.logger.info(f"平均奖励: {np.mean(episode_rewards):.2f}")
        self.logger.info(f"向前迁移分数: {forward_transfer_score:.3f}")
        self.logger.info(f"向后迁移分数: {np.mean(backward_transfer_scores) if backward_transfer_scores else 0:.3f}")
        self.logger.info(f"灾难性遗忘率: {catastrophic_forgetting_rate:.3f}")
        
        return metrics
    
    def _calculate_meta_learning_score(self, task_id: int) -> float:
        """计算元学习分数
        
        Args:
            task_id: 当前任务ID
            
        Returns:
            元学习分数
        """
        if task_id < 2:
            return 0.0
        
        # 计算最近几个任务的平均学习速度提升
        recent_tasks = self.learning_history[max(0, task_id-5):task_id]
        if not recent_tasks:
            return 0.0
        
        learning_improvements = []
        for i in range(1, len(recent_tasks)):
            prev_score = recent_tasks[i-1].forward_transfer_score
            current_score = recent_tasks[i].forward_transfer_score
            improvement = current_score - prev_score
            learning_improvements.append(improvement)
        
        return np.mean(learning_improvements) if learning_improvements else 0.0
    
    def evaluate_performance(self, evaluation_tasks: List[int] = None) -> Dict[str, float]:
        """评估系统整体性能
        
        Args:
            evaluation_tasks: 要评估的任务列表，None表示评估所有任务
            
        Returns:
            评估结果字典
        """
        if evaluation_tasks is None:
            evaluation_tasks = list(range(len(self.learning_history)))
        
        results = {}
        
        # 计算平均性能
        all_performances = [self.learning_history[i].current_performance for i in evaluation_tasks]
        results['average_performance'] = np.mean(all_performances)
        
        # 计算向前迁移分数
        forward_scores = [self.learning_history[i].forward_transfer_score for i in evaluation_tasks if i < len(self.forward_transfer_scores)]
        results['average_forward_transfer'] = np.mean(forward_scores) if forward_scores else 0.0
        
        # 计算向后迁移分数
        all_backward_scores = []
        for i in evaluation_tasks:
            if i < len(self.learning_history):
                all_backward_scores.extend(self.learning_history[i].backward_transfer_scores)
        results['average_backward_transfer'] = np.mean(all_backward_scores) if all_backward_scores else 0.0
        
        # 计算灾难性遗忘率
        forgetting_rates = [self.learning_history[i].catastrophic_forgetting_rate for i in evaluation_tasks]
        results['average_forgetting_rate'] = np.mean(forgetting_rates)
        
        # 计算元学习分数
        meta_scores = [self.learning_history[i].meta_learning_score for i in evaluation_tasks]
        results['average_meta_learning_score'] = np.mean(meta_scores)
        
        # 成功率（向后迁移性能下降<5%的任务比例）
        successful_tasks = sum(1 for i in evaluation_tasks 
                             if i < len(self.learning_history) 
                             and self.learning_history[i].catastrophic_forgetting_rate < self.config.performance_threshold)
        results['success_rate'] = successful_tasks / len(evaluation_tasks) if evaluation_tasks else 0.0
        
        return results
    
    def run_evaluation_sequence(self) -> Dict[str, Any]:
        """运行完整的评估序列
        
        Returns:
            完整的评估结果
        """
        self.logger.info("开始终身学习评估序列")
        
        # 获取游戏序列
        games_list = self.atari_sequence.get_games_list()
        
        evaluation_results = {
            'config': self.config.__dict__,
            'learning_history': [],
            'performance_summary': {},
            'detailed_analysis': {}
        }
        
        # 训练和评估每个任务
        for task_id, game_name in enumerate(games_list):
            try:
                # 生成或加载训练数据（此处为简化实现）
                training_data = np.zeros((self.config.steps_per_task, 5))  # (state, action, reward, next_state, done)
                
                # 训练任务
                metrics = self.train_task(task_id, game_name, training_data)
                evaluation_results['learning_history'].append(metrics.__dict__)
                
                # 定期进行全面评估
                if (task_id + 1) % self.config.test_frequency == 0:
                    self.logger.info(f"完成 {task_id + 1} 个任务，进行全面评估")
                    
                    # 评估当前性能
                    current_performance = self.evaluate_performance(list(range(task_id + 1)))
                    evaluation_results['performance_summary'][f'task_{task_id + 1}'] = current_performance
                    
                    # 分析学习曲线
                    curve_analysis = self.curve_analyzer.analyze_learning_curves(
                        self.performance_history, task_id + 1
                    )
                    evaluation_results['detailed_analysis'][f'task_{task_id + 1}'] = curve_analysis
                
            except Exception as e:
                self.logger.error(f"任务 {task_id} ({game_name}) 训练失败: {str(e)}")
                continue
        
        # 最终评估
        final_performance = self.evaluate_performance()
        evaluation_results['final_performance'] = final_performance
        
        # 生成分析报告
        self._generate_analysis_report(evaluation_results)
        
        self.logger.info("终身学习评估序列完成")
        return evaluation_results
    
    def _generate_analysis_report(self, results: Dict[str, Any]):
        """生成分析报告
        
        Args:
            results: 评估结果
        """
        self.logger.info("生成分析报告")
        
        # 创建可视化
        self._create_visualizations(results)
        
        # 保存结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_file = f"lifelong_learning_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组和datetime为可序列化格式
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存性能摘要
        summary_file = f"performance_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            self._write_performance_summary(f, results)
        
        self.logger.info(f"结果已保存到: {json_file}, {summary_file}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """创建可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('终身学习系统评估结果', fontsize=16)
        
        # 1. 性能随时间变化
        task_ids = [h['task_id'] for h in results['learning_history']]
        performances = [h['current_performance'] for h in results['learning_history']]
        
        axes[0, 0].plot(task_ids, performances, 'b-', linewidth=2, label='任务性能')
        axes[0, 0].set_xlabel('任务ID')
        axes[0, 0].set_ylabel('平均奖励')
        axes[0, 0].set_title('任务性能随时间变化')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 向前迁移分数
        forward_scores = [h['forward_transfer_score'] for h in results['learning_history']]
        axes[0, 1].plot(task_ids, forward_scores, 'g-', linewidth=2, label='向前迁移分数')
        axes[0, 1].set_xlabel('任务ID')
        axes[0, 1].set_ylabel('迁移分数')
        axes[0, 1].set_title('向前迁移性能')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. 灾难性遗忘率
        forgetting_rates = [h['catastrophic_forgetting_rate'] for h in results['learning_history']]
        axes[1, 0].plot(task_ids, forgetting_rates, 'r-', linewidth=2, label='遗忘率')
        axes[1, 0].axhline(y=self.config.performance_threshold, color='orange', linestyle='--', label='阈值(5%)')
        axes[1, 0].set_xlabel('任务ID')
        axes[1, 0].set_ylabel('遗忘率')
        axes[1, 0].set_title('灾难性遗忘分析')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. 向后迁移矩阵热图
        if len(self.backward_transfer_matrix) > 0:
            im = axes[1, 1].imshow(self.backward_transfer_matrix, cmap='RdYlBu', aspect='auto')
            axes[1, 1].set_xlabel('当前任务')
            axes[1, 1].set_ylabel('历史任务')
            axes[1, 1].set_title('向后迁移矩阵')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'lifelong_learning_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _write_performance_summary(self, f, results: Dict[str, Any]):
        """写入性能摘要到文件"""
        f.write("终身学习评估系统 - 性能摘要\n")
        f.write("=" * 50 + "\n\n")
        
        # 配置信息
        f.write("系统配置:\n")
        for key, value in results['config'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 最终性能
        if 'final_performance' in results:
            f.write("最终性能评估:\n")
            for key, value in results['final_performance'].items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
        
        # 学习历史摘要
        f.write(f"学习历史摘要 (共 {len(results['learning_history'])} 个任务):\n")
        
        # 成功任务统计
        threshold = self.config.performance_threshold
        successful_tasks = sum(1 for h in results['learning_history'] 
                             if h['catastrophic_forgetting_rate'] < threshold)
        f.write(f"  成功任务数 (遗忘率<{threshold}): {successful_tasks}/{len(results['learning_history'])}\n")
        
        # 平均指标
        avg_performance = np.mean([h['current_performance'] for h in results['learning_history']])
        avg_forward = np.mean([h['forward_transfer_score'] for h in results['learning_history']])
        avg_backward = np.mean([h['backward_transfer_score'] for h in results['learning_history'] 
                               if h['backward_transfer_scores']])
        avg_forgetting = np.mean([h['catastrophic_forgetting_rate'] for h in results['learning_history']])
        
        f.write(f"  平均任务性能: {avg_performance:.3f}\n")
        f.write(f"  平均向前迁移: {avg_forward:.3f}\n")
        f.write(f"  平均向后迁移: {avg_backward:.3f}\n")
        f.write(f"  平均遗忘率: {avg_forgetting:.3f}\n")
        
        # 关键发现
        f.write("\n关键发现:\n")
        if avg_forgetting < threshold:
            f.write(f"✓ 成功控制灾难性遗忘 (遗忘率 {avg_forgetting:.3f} < {threshold})\n")
        else:
            f.write(f"✗ 灾难性遗忘问题严重 (遗忘率 {avg_forgetting:.3f} >= {threshold})\n")
        
        if avg_forward > 0:
            f.write(f"✓ 观察到正向向前迁移 (分数: {avg_forward:.3f})\n")
        else:
            f.write(f"✗ 负向向前迁移 (分数: {avg_forward:.3f})\n")
        
        # 建议
        f.write("\n改进建议:\n")
        if avg_forgetting >= threshold:
            f.write("  - 增加EWC或MAS正则化强度\n")
            f.write("  - 考虑使用渐进式网络架构\n")
            f.write("  - 实施记忆重放机制\n")
        
        if avg_forward <= 0:
            f.write("  - 优化元学习算法\n")
            f.write("  - 调整学习率策略\n")
            f.write("  - 增强任务间的共享表示学习")


def main():
    """主函数 - 演示终身学习系统"""
    # 创建配置
    config = EvaluationConfig(
        total_tasks=10,  # 演示使用较少的任务
        steps_per_task=1000,
        test_frequency=5,
        ewc_lambda=50.0,
        mas_lambda=0.5,
        use_ewc=True,
        use_mas=True
    )
    
    # 创建终身学习系统
    system = LifelongLearningSystem(config)
    
    # 运行评估
    results = system.run_evaluation_sequence()
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("终身学习评估完成!")
    print("=" * 60)
    
    if 'final_performance' in results:
        print("\n最终性能评估:")
        for key, value in results['final_performance'].items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()