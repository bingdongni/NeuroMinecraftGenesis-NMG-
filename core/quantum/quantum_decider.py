"""
量子决策器主类

实现基于量子计算的智能决策系统，包括：
- 场景评估与量子态映射
- 价值函数计算与概率分布
- 量子仿真与决策权重生成
- 学习参数优化与性能监控
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import time
import json
from .decision_circuit import DecisionCircuit

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumDecider:
    """
    量子决策器主类
    
    该类整合了量子计算和机器学习的优势，实现智能决策系统：
    - 利用量子叠加态探索多种决策路径
    - 通过量子纠缠实现复杂的关联决策
    - 基于价值函数和概率分布做出最优决策
    - 支持参数调优和性能监控
    """
    
    def __init__(self, 
                 learning_params: Dict[str, float],
                 n_scenarios: int = 8,
                 n_decisions: int = 3,
                 simulator_backend: str = 'aer_simulator'):
        """
        初始化量子决策器
        
        Args:
            learning_params: 学习参数配置
                - quantum_exploration_rate: 量子探索概率 (0.01-0.5)
                - learning_rate: 学习率
                - discount_factor: 折扣因子
                - softmax_temperature: softmax温度参数
            n_scenarios: 场景数量，默认8个
            n_decisions: 决策维度，默认3个
            simulator_backend: 量子模拟器后端
        """
        self.learning_params = learning_params.copy()
        self.n_scenarios = n_scenarios
        self.n_decisions = n_decisions
        self.simulator_backend = simulator_backend
        
        # 初始化量子决策电路
        self.decision_circuit = DecisionCircuit(
            learning_params=learning_params,
            n_qubits=n_decisions,
            simulator_backend=simulator_backend
        )
        
        # 场景到量子态的映射表
        self.scenario_to_state = self._initialize_scenario_mapping()
        
        # 价值函数存储
        self.value_functions = np.zeros(n_scenarios)
        
        # 决策权重
        self.decision_weights = np.zeros(2**n_decisions)
        
        # 性能监控
        self.performance_history = []
        self.decision_history = []
        
        # 线程池用于并行计算
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"量子决策器初始化完成: {n_scenarios}场景, {n_decisions}决策维度")
    
    def _initialize_scenario_mapping(self) -> Dict[int, int]:
        """
        初始化8个想象场景到3量子比特状态的映射
        
        返回：
            Dict[int, int]: 场景索引到二进制状态的映射
        """
        mapping = {}
        
        # 定义8个想象场景，每个场景对应一个3量子比特状态
        scenarios = [
            "安全探索", "风险尝试", "保守策略", "激进策略",
            "合作模式", "竞争模式", "创新路径", "传统路径"
        ]
        
        # 将场景映射到0-7的二进制状态
        for i, scenario in enumerate(scenarios):
            mapping[i] = i  # 场景i映射到状态i
            
        logger.info(f"场景映射初始化完成: {len(scenarios)}个场景")
        return mapping
    
    def evaluate_scenarios(self, scenarios: List[Dict], 
                          context_info: Optional[Dict] = None) -> np.ndarray:
        """
        评估8个想象场景并映射到3量子比特状态
        
        该方法分析输入的场景，计算每个场景的期望值和可行性，
        然后将它们映射到量子系统的8个可能状态上。
        
        Args:
            scenarios: 场景描述列表，每个场景包含：
                      - 'description': 场景描述
                      - 'probability': 发生概率
                      - 'value': 潜在价值
                      - 'risk_level': 风险等级
            context_info: 上下文信息，如环境状态、目标等
            
        Returns:
            np.ndarray: 场景评估结果，形状为(n_scenarios,)
        """
        start_time = time.time()
        
        try:
            # 确保输入场景数量正确
            if len(scenarios) != self.n_scenarios:
                raise ValueError(f"期望{self.n_scenarios}个场景，实际得到{len(scenarios)}个")
            
            # 初始化评估结果
            scenario_scores = np.zeros(self.n_scenarios)
            
            # 并行计算场景评估
            futures = []
            for i, scenario in enumerate(scenarios):
                future = self.thread_pool.submit(
                    self._evaluate_single_scenario, 
                    i, scenario, context_info
                )
                futures.append(future)
            
            # 收集结果
            for i, future in enumerate(futures):
                scenario_scores[i] = future.result()
            
            # 应用量子探索调节
            exploration_rate = self.learning_params.get('quantum_exploration_rate', 0.1)
            scenario_scores = self._apply_quantum_exploration(scenario_scores, exploration_rate)
            
            # 更新价值函数
            self.value_functions = scenario_scores
            
            # 记录性能
            evaluation_time = time.time() - start_time
            logger.info(f"场景评估完成: 耗时{evaluation_time:.4f}秒, 平均分数: {np.mean(scenario_scores):.3f}")
            
            return scenario_scores
            
        except Exception as e:
            logger.error(f"场景评估失败: {e}")
            raise
    
    def _evaluate_single_scenario(self, scenario_idx: int, scenario: Dict, 
                                 context_info: Optional[Dict]) -> float:
        """
        评估单个场景
        
        Args:
            scenario_idx: 场景索引
            scenario: 场景描述字典
            context_info: 上下文信息
            
        Returns:
            float: 场景评分
        """
        # 提取场景特征
        description = scenario.get('description', '')
        probability = scenario.get('probability', 0.1)
        value = scenario.get('value', 0.0)
        risk_level = scenario.get('risk_level', 0.5)
        
        # 计算综合评分
        # 综合考虑价值、概率和风险
        base_score = value * probability * (1 - risk_level)
        
        # 添加上下文权重（如果有）
        context_weight = 1.0
        if context_info:
            goal_alignment = context_info.get('goal_alignment', 0.5)
            resource_availability = context_info.get('resource_availability', 0.5)
            context_weight = (goal_alignment + resource_availability) / 2
        
        final_score = base_score * context_weight
        
        # 标准化到[0, 1]范围
        normalized_score = max(0, min(1, final_score))
        
        return normalized_score
    
    def _apply_quantum_exploration(self, scores: np.ndarray, exploration_rate: float) -> np.ndarray:
        """
        应用量子探索机制调节场景评分
        
        Args:
            scores: 原始场景评分
            exploration_rate: 探索概率
            
        Returns:
            np.ndarray: 调节后的评分
        """
        # 量子探索增强低概率场景的权重
        # 这模拟了量子系统中不确定性带来的探索优势
        
        # 计算每个场景的探索增益
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            # 标准化分数
            normalized = (scores - min_score) / (max_score - min_score)
            # 反向加权：分数越低，获得的探索增益越大
            exploration_gain = exploration_rate * (1 - normalized)
            # 应用增益
            enhanced_scores = scores + exploration_gain
        else:
            enhanced_scores = scores + exploration_rate
        
        # 确保分数在合理范围内
        enhanced_scores = np.clip(enhanced_scores, 0, 1)
        
        return enhanced_scores
    
    def value_function_calculation(self, scenarios: np.ndarray, 
                                  temperature: Optional[float] = None) -> np.ndarray:
        """
        为每个场景计算价值函数V(s)，使用softmax转换为概率分布
        
        该方法将场景评分转换为概率分布，利用量子力学的概率特性
        来模拟决策过程中的不确定性。
        
        Args:
            scenarios: 场景评分数组
            temperature: softmax温度参数
            
        Returns:
            np.ndarray: 概率分布数组
        """
        try:
            # 使用配置的temperature或默认值
            if temperature is None:
                temperature = self.learning_params.get('softmax_temperature', 1.0)
            
            # 数值稳定性处理
            # 减去最大值防止指数溢出
            shifted_scenarios = scenarios - np.max(scenarios)
            
            # 计算softmax概率分布
            exp_scores = np.exp(shifted_scenarios / temperature)
            probabilities = exp_scores / np.sum(exp_scores)
            
            # 应用量子修正
            quantum_factor = self.learning_params.get('quantum_exploration_rate', 0.1)
            probabilities = self._apply_quantum_probability_correction(probabilities, quantum_factor)
            
            logger.info(f"价值函数计算完成: 熵={self._calculate_entropy(probabilities):.3f}")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"价值函数计算失败: {e}")
            raise
    
    def _apply_quantum_probability_correction(self, probabilities: np.ndarray, 
                                            quantum_factor: float) -> np.ndarray:
        """
        应用量子概率修正
        
        Args:
            probabilities: 原始概率分布
            quantum_factor: 量子因子
            
        Returns:
            np.ndarray: 修正后的概率分布
        """
        # 量子概率修正：增强低概率事件的权重
        # 这模拟了量子测量中的不确定性原理
        
        # 计算修正因子
        min_prob = np.min(probabilities[probabilities > 0])
        normalized_probs = probabilities / np.max(probabilities)
        
        # 应用量子修正：低概率事件获得额外权重
        correction_factor = 1 + quantum_factor * (1 - normalized_probs)
        corrected_probs = probabilities * correction_factor
        
        # 重新归一化
        corrected_probs = corrected_probs / np.sum(corrected_probs)
        
        return corrected_probs
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        计算概率分布的熵
        
        Args:
            probabilities: 概率分布
            
        Returns:
            float: 熵值
        """
        # 过滤零值避免log(0)
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
        return entropy
    
    def quantum_simulation(self, shots: int = 1000, 
                          use_optimization: bool = True) -> Dict[str, int]:
        """
        使用Qiskit Aer模拟器运行量子电路，获得测量次数分布
        
        Args:
            shots: 仿真次数
            use_optimization: 是否使用矩阵优化
            
        Returns:
            Dict[str, int]: 测量结果统计字典
        """
        try:
            logger.info(f"开始量子仿真: {shots}次测量")
            
            # 运行量子电路仿真
            measurement_results = self.decision_circuit.run_simulation(
                shots=shots,
                use_matrix_optimization=use_optimization
            )
            
            # 记录仿真结果
            self.performance_history.append({
                'timestamp': time.time(),
                'shots': shots,
                'results': measurement_results.copy(),
                'execution_time': self.decision_circuit.get_performance_stats()['execution_time']
            })
            
            logger.info(f"量子仿真完成: {len(measurement_results)}个不同结果")
            
            return measurement_results
            
        except Exception as e:
            logger.error(f"量子仿真失败: {e}")
            raise
    
    def decision_weight_generation(self, measurement_results: Dict[str, int]) -> np.ndarray:
        """
        将量子测量结果转换为决策权重
        
        Args:
            measurement_results: 量子电路测量结果
            
        Returns:
            np.ndarray: 决策权重数组
        """
        try:
            # 将测量计数转换为权重
            total_shots = sum(measurement_results.values())
            
            # 初始化权重数组
            weights = np.zeros(2**self.n_decisions)
            
            # 计算每个状态的权重
            for state_str, count in measurement_results.items():
                # 清理字符串，只保留前3个二进制字符（对应3量子比特）
                clean_state_str = ''.join(c for c in state_str if c in '01')[:self.n_decisions]
                
                # 如果清理后的字符串长度不足，补齐到n_decisions长度
                if len(clean_state_str) < self.n_decisions:
                    clean_state_str = clean_state_str.ljust(self.n_decisions, '0')
                
                # 确保不超过数组边界
                state_idx = min(int(clean_state_str, 2), len(weights) - 1)
                weights[state_idx] = count / total_shots
            
            # 应用量子优化
            quantum_factor = self.learning_params.get('quantum_exploration_rate', 0.1)
            weights = self._optimize_decision_weights(weights, quantum_factor)
            
            # 更新决策权重
            self.decision_weights = weights
            
            # 记录决策历史
            decision_record = {
                'timestamp': time.time(),
                'weights': weights.copy(),
                'max_weight_idx': np.argmax(weights),
                'weight_entropy': self._calculate_entropy(weights)
            }
            self.decision_history.append(decision_record)
            
            logger.info(f"决策权重生成完成: 最优决策={np.argmax(weights)}, 熵={decision_record['weight_entropy']:.3f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"决策权重生成失败: {e}")
            raise
    
    def _optimize_decision_weights(self, weights: np.ndarray, quantum_factor: float) -> np.ndarray:
        """
        优化决策权重以提高决策质量
        
        Args:
            weights: 原始决策权重
            quantum_factor: 量子探索因子
            
        Returns:
            np.ndarray: 优化后的决策权重
        """
        # 应用量子退火原理优化权重
        # 模拟量子系统寻找全局最优解的过程
        
        # 平滑权重分布
        smoothed_weights = self._smooth_weights(weights)
        
        # 应用量子隧穿效应：允许权重在局部最优间跳跃
        tunneling_weights = self._apply_quantum_tunneling(smoothed_weights, quantum_factor)
        
        # 最终归一化
        final_weights = tunneling_weights / np.sum(tunneling_weights)
        
        return final_weights
    
    def _smooth_weights(self, weights: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
        """
        平滑权重分布
        
        Args:
            weights: 原始权重
            smoothing_factor: 平滑因子
            
        Returns:
            np.ndarray: 平滑后的权重
        """
        # 计算邻域权重平均
        smoothed = weights.copy()
        n_states = len(weights)
        
        for i in range(n_states):
            neighbors = []
            # 考虑循环边界
            for offset in [-1, 1]:
                neighbor_idx = (i + offset) % n_states
                neighbors.append(weights[neighbor_idx])
            
            neighbor_avg = np.mean(neighbors)
            smoothed[i] = (1 - smoothing_factor) * weights[i] + smoothing_factor * neighbor_avg
        
        return smoothed
    
    def _apply_quantum_tunneling(self, weights: np.ndarray, quantum_factor: float) -> np.ndarray:
        """
        应用量子隧穿效应优化权重
        
        Args:
            weights: 平滑权重
            quantum_factor: 量子隧穿强度
            
        Returns:
            np.ndarray: 应用隧穿效应后的权重
        """
        # 量子隧穿允许系统越过能量壁垒，达到全局最优
        
        # 寻找局部最优
        local_optima = self._find_local_optima(weights)
        
        # 应用隧穿修正
        tunneled_weights = weights.copy()
        
        for opt_idx in local_optima:
            # 计算隧穿概率
            tunnel_prob = quantum_factor * (1 - weights[opt_idx])
            
            # 将部分权重隧穿到其他状态
            for i in range(len(weights)):
                if i != opt_idx and weights[i] < weights[opt_idx]:
                    tunneled_weights[opt_idx] -= tunnel_prob * weights[i] * 0.01
                    tunneled_weights[i] += tunnel_prob * weights[i] * 0.01
        
        return tunneled_weights
    
    def _find_local_optima(self, weights: np.ndarray) -> List[int]:
        """
        找到权重分布中的局部最优值
        
        Args:
            weights: 权重数组
            
        Returns:
            List[int]: 局部最优值的索引
        """
        local_optima = []
        n = len(weights)
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # 如果当前值大于或等于邻居值，则为局部最优
            if weights[i] >= weights[prev_idx] and weights[i] >= weights[next_idx]:
                local_optima.append(i)
        
        return local_optima
    
    def make_decision(self, scenarios: List[Dict], 
                     context_info: Optional[Dict] = None,
                     shots: int = 1000) -> Tuple[int, Dict]:
        """
        执行完整的决策过程
        
        Args:
            scenarios: 场景描述列表
            context_info: 上下文信息
            shots: 量子仿真次数
            
        Returns:
            Tuple[int, Dict]: (最优决策索引, 详细决策信息)
        """
        try:
            logger.info("开始量子决策过程")
            start_time = time.time()
            
            # 步骤1: 评估场景
            scenario_scores = self.evaluate_scenarios(scenarios, context_info)
            
            # 步骤2: 计算价值函数和概率分布
            probabilities = self.value_function_calculation(scenario_scores)
            
            # 步骤3: 运行量子仿真
            measurement_results = self.quantum_simulation(shots)
            
            # 步骤4: 生成决策权重
            decision_weights = self.decision_weight_generation(measurement_results)
            
            # 步骤5: 选择最优决策
            best_decision = np.argmax(decision_weights)
            
            # 构建决策结果
            decision_info = {
                'best_decision': best_decision,
                'decision_state': format(best_decision, f'0{self.n_decisions}b'),
                'scenario_scores': scenario_scores.tolist(),
                'probabilities': probabilities.tolist(),
                'decision_weights': decision_weights.tolist(),
                'measurement_results': measurement_results,
                'confidence': decision_weights[best_decision],
                'entropy': self._calculate_entropy(decision_weights),
                'execution_time': time.time() - start_time
            }
            
            logger.info(f"量子决策完成: 最优决策={best_decision}, 置信度={decision_info['confidence']:.3f}")
            
            return best_decision, decision_info
            
        except Exception as e:
            logger.error(f"决策过程失败: {e}")
            raise
    
    def get_performance_report(self) -> Dict:
        """
        获取性能报告
        
        Returns:
            Dict: 性能报告字典
        """
        circuit_stats = self.decision_circuit.get_performance_stats()
        
        report = {
            'quantum_circuit_stats': circuit_stats,
            'performance_history': self.performance_history[-10:],  # 最近10次
            'decision_history': self.decision_history[-10:],  # 最近10次
            'current_value_functions': self.value_functions.tolist(),
            'current_decision_weights': self.decision_weights.tolist(),
            'learning_params': self.learning_params
        }
        
        return report
    
    def save_performance_report(self, filepath: str):
        """
        保存性能报告到文件
        
        Args:
            filepath: 文件路径
        """
        report = self.get_performance_report()
        
        # 转换numpy数组为Python列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        clean_report = convert_numpy(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存到: {filepath}")
    
    def update_learning_params(self, new_params: Dict[str, float]):
        """
        更新学习参数
        
        Args:
            new_params: 新的学习参数
        """
        self.learning_params.update(new_params)
        
        # 更新决策电路参数
        self.decision_circuit.learning_params = self.learning_params.copy()
        self.decision_circuit.exploration_rate = max(0.01, min(0.5, new_params.get('quantum_exploration_rate', 0.1)))
        
        logger.info(f"学习参数已更新: {new_params}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
    def __repr__(self):
        return (f"QuantumDecider(scenarios={self.n_scenarios}, "
                f"decisions={self.n_decisions}, "
                f"exploration_rate={self.learning_params.get('quantum_exploration_rate', 0.1):.3f})")