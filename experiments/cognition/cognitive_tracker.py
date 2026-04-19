"""
六维认知能力跟踪器
==================

该模块实现了六维认知能力的实时跟踪和评估系统，包括：
- 记忆力 (Memory)
- 思维力 (Thinking) 
- 创造力 (Creativity)
- 观察力 (Observation)
- 注意力 (Attention)
- 想象力 (Imagination)

每个维度都通过多个指标进行量化评估，形成完整的认知能力画像。
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CognitiveMetrics:
    """认知指标数据类"""
    timestamp: datetime
    memory_score: float = 0.0      # 记忆力分数 (0-100)
    thinking_score: float = 0.0    # 思维力分数 (0-100)
    creativity_score: float = 0.0  # 创造力分数 (0-100)
    observation_score: float = 0.0 # 观察力分数 (0-100)
    attention_score: float = 0.0   # 注意力分数 (0-100)
    imagination_score: float = 0.0 # 想象力分数 (0-100)
    
    def overall_score(self) -> float:
        """计算六维综合分数"""
        scores = [self.memory_score, self.thinking_score, self.creativity_score,
                 self.observation_score, self.attention_score, self.imagination_score]
        return np.mean(scores)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['overall_score'] = self.overall_score()
        return result

class CognitiveTracker:
    """六维认知能力跟踪器"""
    
    def __init__(self, agent_id: str, baseline_correction: bool = True):
        """
        初始化认知跟踪器
        
        Args:
            agent_id: 智能体唯一标识
            baseline_correction: 是否启用基线校正
        """
        self.agent_id = agent_id
        self.baseline_correction = baseline_correction
        self.metrics_history: List[CognitiveMetrics] = []
        self.baseline_metrics: Optional[CognitiveMetrics] = None
        self.weights = {
            'memory': 1.0,
            'thinking': 1.0,
            'creativity': 1.0,
            'observation': 1.0,
            'attention': 1.0,
            'imagination': 1.0
        }
        
        # 认知评估参数
        self.evaluation_params = {
            'memory_window': 100,      # 记忆窗口大小
            'thinking_complexity': 0.7, # 思维复杂度系数
            'creativity_threshold': 0.5, # 创造力阈值
            'observation_sensitivity': 0.8, # 观察敏感性
            'attention_duration': 300, # 注意力持续时间(秒)
            'imagination_depth': 0.6   # 想象力深度系数
        }
        
        logger.info(f"认知跟踪器初始化完成 - 智能体ID: {agent_id}")
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        设置各维度的权重
        
        Args:
            weights: 权重字典 {'memory': 1.2, 'thinking': 0.8, ...}
        """
        for dim, weight in weights.items():
            if dim in self.weights:
                self.weights[dim] = weight
                logger.info(f"设置维度 {dim} 权重: {weight}")
    
    def _hash_environment(self, observation: Dict) -> str:
        """计算环境观察的哈希值，用于检测环境变化"""
        env_str = json.dumps(observation, sort_keys=True, default=str)
        return hashlib.md5(env_str.encode()).hexdigest()[:8]
    
    def _detect_patterns(self, sequence: List[float], window_size: int = 10) -> Dict:
        """
        检测序列中的模式
        
        Args:
            sequence: 数值序列
            window_size: 分析窗口大小
            
        Returns:
            模式分析结果字典
        """
        if len(sequence) < window_size:
            return {'trend': 0, 'variance': 0, 'patterns': []}
        
        # 计算趋势
        recent_window = sequence[-window_size:]
        x = np.arange(len(recent_window))
        slope, _ = np.polyfit(x, recent_window, 1)
        
        # 计算方差
        variance = np.var(recent_window)
        
        # 简单模式检测
        patterns = []
        for i in range(len(recent_window) - 2):
            if recent_window[i+1] > recent_window[i] and recent_window[i+1] > recent_window[i+2]:
                patterns.append('peak')
            elif recent_window[i+1] < recent_window[i] and recent_window[i+1] < recent_window[i+2]:
                patterns.append('valley')
        
        return {
            'trend': slope,
            'variance': variance,
            'patterns': patterns
        }
    
    def evaluate_memory(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估记忆力维度
        
        评估指标：
        - 环境记忆：回忆之前见过的环境状态
        - 经验整合：将新经验与已有知识整合
        - 记忆持久性：信息保持的时间长度
        - 检索效率：快速回忆信息的能力
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            记忆力分数 (0-100)
        """
        try:
            # 提取记忆相关指标
            memory_retention = agent_state.get('memory_retention', 0.5)  # 记忆保持率
            learning_speed = agent_state.get('learning_speed', 0.5)       # 学习速度
            recall_accuracy = agent_state.get('recall_accuracy', 0.5)     # 回忆准确率
            contextual_memory = agent_state.get('contextual_memory', 0.5) # 情境记忆
            
            # 环境变化检测
            env_hash = self._hash_environment(environment_state)
            
            # 计算基础记忆分数
            memory_score = (
                memory_retention * 30 +           # 记忆保持权重
                learning_speed * 25 +             # 学习速度权重  
                recall_accuracy * 25 +            # 回忆准确率权重
                contextual_memory * 20            # 情境记忆权重
            ) * 100
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_memory = self.baseline_metrics.memory_score
                memory_score = memory_score * 0.7 + baseline_memory * 0.3
            
            return min(100, max(0, memory_score))
            
        except Exception as e:
            logger.warning(f"记忆力评估出错: {e}")
            return 50.0  # 默认分数
    
    def evaluate_thinking(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估思维力维度
        
        评估指标：
        - 逻辑推理：基于已知信息进行逻辑推理
        - 问题分解：将复杂问题分解为子问题
        - 抽象思维：理解抽象概念和模式
        - 策略规划：制定长期行动计划
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            思维力分数 (0-100)
        """
        try:
            # 提取思维相关指标
            reasoning_accuracy = agent_state.get('reasoning_accuracy', 0.5)   # 推理准确率
            problem_decomposition = agent_state.get('problem_decomposition', 0.5) # 问题分解能力
            abstract_reasoning = agent_state.get('abstract_reasoning', 0.5)   # 抽象推理
            strategic_planning = agent_state.get('strategic_planning', 0.5)   # 策略规划
            
            # 计算思维复杂度
            complexity_factor = self.evaluation_params['thinking_complexity']
            environmental_complexity = len(str(environment_state)) / 1000  # 简单的复杂度估计
            
            # 计算基础思维分数
            thinking_score = (
                reasoning_accuracy * 30 +              # 推理准确率权重
                problem_decomposition * 25 +           # 问题分解权重
                abstract_reasoning * 25 +              # 抽象推理权重
                strategic_planning * 20                # 策略规划权重
            )
            
            # 复杂度调整
            thinking_score *= (1 + complexity_factor * environmental_complexity)
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_thinking = self.baseline_metrics.thinking_score
                thinking_score = thinking_score * 0.7 + baseline_thinking * 0.3
            
            return min(100, max(0, thinking_score * 100))
            
        except Exception as e:
            logger.warning(f"思维力评估出错: {e}")
            return 50.0  # 默认分数
    
    def evaluate_creativity(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估创造力维度
        
        评估指标：
        - 创新性行为：执行前所未见的行为模式
        - 问题解决新方法：尝试非传统的解决方案
        - 适应性创新：在新环境中快速适应并创新
        - 跨域联想：将不同领域的知识联系起来
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            创造力分数 (0-100)
        """
        try:
            # 提取创造力相关指标
            novel_behaviors = agent_state.get('novel_behaviors', 0.5)         # 新颖行为频率
            alternative_solutions = agent_state.get('alternative_solutions', 0.5) # 替代解决方案数量
            adaptation_speed = agent_state.get('adaptation_speed', 0.5)       # 适应速度
            cross_domain_transfer = agent_state.get('cross_domain_transfer', 0.5) # 跨域迁移
            
            # 计算创新性阈值
            creativity_threshold = self.evaluation_params['creativity_threshold']
            
            # 分析历史行为模式
            behavior_history = agent_state.get('behavior_history', [])
            recent_behaviors = behavior_history[-10:] if behavior_history else []
            
            # 计算新颖性分数
            novelty_score = 0
            if recent_behaviors:
                # 计算行为多样性
                unique_behaviors = len(set(recent_behaviors))
                behavior_diversity = unique_behaviors / len(recent_behaviors)
                novelty_score = behavior_diversity
            
            # 计算基础创造力分数
            creativity_score = (
                novel_behaviors * 25 +                # 新颖行为权重
                alternative_solutions * 25 +          # 替代解决方案权重
                adaptation_speed * 25 +               # 适应速度权重
                cross_domain_transfer * 25            # 跨域迁移权重
            )
            
            # 新颖性调整
            creativity_score = creativity_score * (0.7 + novelty_score * 0.3)
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_creativity = self.baseline_metrics.creativity_score
                creativity_score = creativity_score * 0.7 + baseline_creativity * 0.3
            
            return min(100, max(0, creativity_score * 100))
            
        except Exception as e:
            logger.warning(f"创造力评估出错: {e}")
            return 50.0  # 默认分数
    
    def evaluate_observation(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估观察力维度
        
        评估指标：
        - 环境感知：感知环境变化的敏感度
        - 细节识别：识别环境中的细微变化
        - 模式识别：识别环境中的模式和规律
        - 多感官整合：整合多种感知信息
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            观察力分数 (0-100)
        """
        try:
            # 提取观察相关指标
            environmental_awareness = agent_state.get('environmental_awareness', 0.5) # 环境感知
            detail_recognition = agent_state.get('detail_recognition', 0.5)             # 细节识别
            pattern_recognition = agent_state.get('pattern_recognition', 0.5)           # 模式识别
            sensory_integration = agent_state.get('sensory_integration', 0.5)           # 感官整合
            
            # 计算观察敏感性
            sensitivity = self.evaluation_params['observation_sensitivity']
            
            # 分析环境复杂度
            env_complexity = len(environment_state.get('objects', [])) / 50  # 简单的复杂度估计
            env_changes = agent_state.get('detected_changes', 0)
            
            # 计算观察效率
            observation_efficiency = min(1.0, env_changes / max(1, env_complexity))
            
            # 计算基础观察分数
            observation_score = (
                environmental_awareness * 25 +     # 环境感知权重
                detail_recognition * 25 +           # 细节识别权重
                pattern_recognition * 25 +          # 模式识别权重
                sensory_integration * 25            # 感官整合权重
            )
            
            # 敏感性和效率调整
            observation_score *= (1 + sensitivity * observation_efficiency)
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_observation = self.baseline_metrics.observation_score
                observation_score = observation_score * 0.7 + baseline_observation * 0.3
            
            return min(100, max(0, observation_score * 100))
            
        except Exception as e:
            logger.warning(f"观察力评估出错: {e}")
            return 50.0  # 默认分数
    
    def evaluate_attention(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估注意力维度
        
        评估指标：
        - 专注持续时间：在任务上的专注时间长度
        - 分心抗干扰：抵抗环境干扰的能力
        - 注意力转移：灵活转移注意力的能力
        - 注意焦点质量：注意焦点的精准度
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            注意力分数 (0-100)
        """
        try:
            # 提取注意力相关指标
            focus_duration = agent_state.get('focus_duration', 0.5)           # 专注持续时间
            distraction_resistance = agent_state.get('distraction_resistance', 0.5) # 抗干扰能力
            attention_shift = agent_state.get('attention_shift', 0.5)         # 注意力转移
            focus_quality = agent_state.get('focus_quality', 0.5)             # 焦点质量
            
            # 计算注意力持续时间
            duration = self.evaluation_params['attention_duration']
            current_duration = agent_state.get('current_focus_time', 0)
            duration_ratio = min(1.0, current_duration / duration)
            
            # 分析分心事件
            distraction_events = agent_state.get('distraction_events', 0)
            task_time = agent_state.get('total_task_time', 1)
            distraction_rate = distraction_events / task_time
            
            # 计算抗干扰分数
            interference_resistance = max(0, 1 - distraction_rate * 10)
            
            # 计算基础注意力分数
            attention_score = (
                focus_duration * 30 +                # 专注持续时间权重
                interference_resistance * 25 +       # 抗干扰能力权重
                attention_shift * 25 +               # 注意力转移权重
                focus_quality * 20                   # 焦点质量权重
            )
            
            # 持续性调整
            attention_score *= (0.5 + duration_ratio * 0.5)
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_attention = self.baseline_metrics.attention_score
                attention_score = attention_score * 0.7 + baseline_attention * 0.3
            
            return min(100, max(0, attention_score * 100))
            
        except Exception as e:
            logger.warning(f"注意力评估出错: {e}")
            return 50.0  # 默认分数
    
    def evaluate_imagination(self, agent_state: Dict, environment_state: Dict) -> float:
        """
        评估想象力维度
        
        评估指标：
        - 情景预演：在头脑中模拟未来情景
        - 创造性组合：将不同元素组合成新的构想
        - 假设推理：基于假设进行推理
        - 心理模拟：模拟其他智能体的行为
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            想象力分数 (0-100)
        """
        try:
            # 提取想象力相关指标
            scenario_previsualization = agent_state.get('scenario_previsualization', 0.5) # 情景预演
            creative_combination = agent_state.get('creative_combination', 0.5)           # 创造性组合
            hypothetical_reasoning = agent_state.get('hypothetical_reasoning', 0.5)       # 假设推理
            mental_simulation = agent_state.get('mental_simulation', 0.5)                 # 心理模拟
            
            # 计算想象力深度
            depth = self.evaluation_params['imagination_depth']
            
            # 分析想象力相关行为
            imagination_events = agent_state.get('imagination_events', [])
            recent_imagination = len(imagination_events[-5:]) if imagination_events else 0
            
            # 计算想象力活跃度
            imagination_activity = min(1.0, recent_imagination / 5)
            
            # 计算基础想象力分数
            imagination_score = (
                scenario_previsualization * 25 +     # 情景预演权重
                creative_combination * 25 +           # 创造性组合权重
                hypothetical_reasoning * 25 +         # 假设推理权重
                mental_simulation * 25                # 心理模拟权重
            )
            
            # 深度和活跃度调整
            imagination_score *= (1 + depth * imagination_activity)
            
            # 基线校正
            if self.baseline_correction and self.baseline_metrics:
                baseline_imagination = self.baseline_metrics.imagination_score
                imagination_score = imagination_score * 0.7 + baseline_imagination * 0.3
            
            return min(100, max(0, imagination_score * 100))
            
        except Exception as e:
            logger.warning(f"想象力评估出错: {e}")
            return 50.0  # 默认分数
    
    def track_cognitive_metrics(self, agent_state: Dict, environment_state: Dict) -> CognitiveMetrics:
        """
        跟踪所有六维认知指标
        
        Args:
            agent_state: 智能体当前状态
            environment_state: 环境状态
            
        Returns:
            认知指标对象
        """
        try:
            current_time = datetime.now()
            
            # 评估六个维度
            memory_score = self.evaluate_memory(agent_state, environment_state)
            thinking_score = self.evaluate_thinking(agent_state, environment_state)
            creativity_score = self.evaluate_creativity(agent_state, environment_state)
            observation_score = self.evaluate_observation(agent_state, environment_state)
            attention_score = self.evaluate_attention(agent_state, environment_state)
            imagination_score = self.evaluate_imagination(agent_state, environment_state)
            
            # 创建认知指标对象
            metrics = CognitiveMetrics(
                timestamp=current_time,
                memory_score=memory_score,
                thinking_score=thinking_score,
                creativity_score=creativity_score,
                observation_score=observation_score,
                attention_score=attention_score,
                imagination_score=imagination_score
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            # 限制历史记录长度（防止内存溢出）
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]
            
            logger.info(f"认知指标更新 - 综合分数: {metrics.overall_score():.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"认知指标跟踪失败: {e}")
            # 返回默认指标
            return CognitiveMetrics(timestamp=datetime.now())
    
    def set_baseline(self, baseline_metrics: CognitiveMetrics) -> None:
        """
        设置基线指标
        
        Args:
            baseline_metrics: 基线认知指标
        """
        self.baseline_metrics = baseline_metrics
        logger.info("基线指标已设置")
    
    def get_latest_metrics(self) -> Optional[CognitiveMetrics]:
        """获取最新的认知指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[CognitiveMetrics]:
        """
        获取指定时间范围内的历史指标
        
        Args:
            hours: 小时数
            
        Returns:
            历史指标列表
        """
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_dimension_trend(self, dimension: str, hours: int = 24) -> Dict:
        """
        获取指定维度的趋势分析
        
        Args:
            dimension: 维度名称 ('memory', 'thinking', etc.)
            hours: 分析时间范围
            
        Returns:
            趋势分析结果
        """
        history = self.get_metrics_history(hours)
        if len(history) < 2:
            return {'trend': 0, 'confidence': 0, 'recent_scores': []}
        
        # 提取指定维度的分数
        scores = []
        for metrics in history:
            score = getattr(metrics, f"{dimension}_score", 0)
            scores.append(score)
        
        # 计算趋势
        if len(scores) >= 2:
            trend, p_value = pearsonr(range(len(scores)), scores)
            
            # 计算最近分数
            recent_scores = scores[-10:] if len(scores) >= 10 else scores
            
            return {
                'trend': trend,
                'confidence': abs(trend),
                'recent_scores': recent_scores,
                'p_value': p_value,
                'total_points': len(scores)
            }
        
        return {'trend': 0, 'confidence': 0, 'recent_scores': scores}
    
    def save_metrics(self, filepath: str) -> None:
        """
        保存指标历史到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            data = [metric.to_dict() for metric in self.metrics_history]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"认知指标已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存认知指标失败: {e}")
    
    def load_metrics(self, filepath: str) -> None:
        """
        从文件加载指标历史
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metrics_history = []
            for item in data:
                timestamp = datetime.fromisoformat(item['timestamp'])
                metrics = CognitiveMetrics(
                    timestamp=timestamp,
                    memory_score=item['memory_score'],
                    thinking_score=item['thinking_score'],
                    creativity_score=item['creativity_score'],
                    observation_score=item['observation_score'],
                    attention_score=item['attention_score'],
                    imagination_score=item['imagination_score']
                )
                self.metrics_history.append(metrics)
            
            logger.info(f"认知指标已从文件加载: {filepath}")
        except Exception as e:
            logger.error(f"加载认知指标失败: {e}")

if __name__ == "__main__":
    # 测试认知跟踪器
    tracker = CognitiveTracker(agent_id="test_agent_001")
    
    # 模拟智能体状态
    test_state = {
        'memory_retention': 0.8,
        'learning_speed': 0.7,
        'recall_accuracy': 0.9,
        'contextual_memory': 0.6,
        'reasoning_accuracy': 0.8,
        'novel_behaviors': 0.7,
        'environmental_awareness': 0.9,
        'focus_duration': 0.8,
        'imagination_events': ['scenario1', 'scenario2']
    }
    
    test_env = {
        'objects': ['tree', 'stone', 'water'],
        'time': 'day',
        'weather': 'clear'
    }
    
    # 测试认知指标跟踪
    metrics = tracker.track_cognitive_metrics(test_state, test_env)
    print(f"认知指标测试结果: {metrics.to_dict()}")
    
    # 测试维度趋势
    trend = tracker.get_dimension_trend('memory', hours=1)
    print(f"记忆趋势: {trend}")
    
    print("认知跟踪器测试完成")