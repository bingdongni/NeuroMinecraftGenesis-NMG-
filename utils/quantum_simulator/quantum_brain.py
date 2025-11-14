#!/usr/bin/env python3
"""
量子计算模拟器 - 量子大脑模拟模块
"""

import numpy as np
import json
import math
import random
from typing import Dict, List, Tuple, Optional
import logging

class QuantumBrain:
    """量子大脑模拟器"""
    
    def __init__(self, num_qubits: int = 4):
        self.logger = logging.getLogger(__name__)
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        
        # 量子态 |ψ⟩ = Σ αᵢ|⟨|i⟩⟨|i|ψ⟩|
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0  # |0000⟩ 初始状态
        
        # 量子门
        self.quantum_gates = {
            'H': self._hadamard_gate(),
            'X': self._pauli_x_gate(),
            'Y': self._pauli_y_gate(), 
            'Z': self._pauli_z_gate(),
            'CNOT': self._cnot_gate(),
            'Rx': self._rotation_x_gate,
            'Ry': self._rotation_y_gate,
            'Rz': self._rotation_z_gate
        }
        
        # 测量历史
        self.measurement_history = []
        self.entanglement_map = {}
        
    def _hadamard_gate(self) -> np.ndarray:
        """创建Hadamard门"""
        h = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        return h
        
    def _pauli_x_gate(self) -> np.ndarray:
        """创建Pauli-X门"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
        
    def _pauli_y_gate(self) -> np.ndarray:
        """创建Pauli-Y门"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
        
    def _pauli_z_gate(self) -> np.ndarray:
        """创建Pauli-Z门"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
        
    def _cnot_gate(self) -> np.ndarray:
        """创建CNOT门"""
        cnot = np.eye(4, dtype=complex)
        cnot[2:4, 2:4] = np.array([[0, 1], [1, 0]], dtype=complex)
        return cnot
        
    def _rotation_x_gate(self, theta: float) -> np.ndarray:
        """创建绕X轴旋转门"""
        return np.array([
            [math.cos(theta/2), -1j*math.sin(theta/2)],
            [-1j*math.sin(theta/2), math.cos(theta/2)]
        ], dtype=complex)
        
    def _rotation_y_gate(self, theta: float) -> np.ndarray:
        """创建绕Y轴旋转门"""
        return np.array([
            [math.cos(theta/2), -math.sin(theta/2)],
            [math.sin(theta/2), math.cos(theta/2)]
        ], dtype=complex)
        
    def _rotation_z_gate(self, theta: float) -> np.ndarray:
        """创建绕Z轴旋转门"""
        return np.array([
            [1, 0], [0, math.exp(1j*theta)]
        ], dtype=complex)
    
    def apply_gate(self, gate: np.ndarray, qubit: int = 0) -> bool:
        """在指定量子比特上应用量子门"""
        try:
            # 构建作用于整个量子系统的门
            if gate.shape == (2, 2):
                # 单量子比特门
                if qubit == 0:
                    # 作用于最低有效量子比特
                    full_gate = np.eye(self.num_states, dtype=complex)
                    for i in range(0, self.num_states, 2):
                        full_gate[i:i+2, i:i+2] = gate
                else:
                    # 作用于第qubit个量子比特
                    full_gate = np.eye(self.num_states, dtype=complex)
                    for i in range(self.num_states):
                        if (i >> qubit) & 1:
                            # 翻转该量子比特
                            j = i ^ (1 << qubit)
                            full_gate[j:j+1, i:i+1] = gate[1,1]
                            if i & 1:  # 如果翻转后的状态是|1⟩
                                full_gate[j:j+1, i:i+1] = gate[0,1]
            else:
                full_gate = gate
            
            # 应用门到量子态
            self.state_vector = full_gate @ self.state_vector
            
            # 归一化量子态
            norm = np.linalg.norm(self.state_vector)
            if norm > 0:
                self.state_vector /= norm
                
            return True
            
        except Exception as e:
            self.logger.error(f"应用量子门失败: {e}")
            return False
    
    def measure_qubit(self, qubit: int = 0) -> Tuple[int, float]:
        """测量指定量子比特"""
        try:
            # 计算该量子比特为0和1的概率
            prob_0 = 0.0
            prob_1 = 0.0
            
            for state_idx in range(self.num_states):
                amplitude = self.state_vector[state_idx]
                if ((state_idx >> qubit) & 1) == 0:
                    prob_0 += abs(amplitude) ** 2
                else:
                    prob_1 += abs(amplitude) ** 2
            
            # 根据概率测量
            result = 1 if random.random() < prob_1 else 0
            
            # 坍缩量子态
            if result == 1:
                for state_idx in range(self.num_states):
                    if ((state_idx >> qubit) & 1) == 0:
                        self.state_vector[state_idx] = 0
            else:
                for state_idx in range(self.num_states):
                    if ((state_idx >> qubit) & 1) == 1:
                        self.state_vector[state_idx] = 0
            
            # 重新归一化
            norm = np.linalg.norm(self.state_vector)
            if norm > 0:
                self.state_vector /= norm
            
            # 记录测量结果
            measurement = {
                'qubit': qubit,
                'result': result,
                'probability_0': prob_0,
                'probability_1': prob_1,
                'timestamp': np.datetime64('now')
            }
            self.measurement_history.append(measurement)
            
            return result, prob_0 if result == 0 else prob_1
            
        except Exception as e:
            self.logger.error(f"量子测量失败: {e}")
            return 0, 0.0
    
    def create_entanglement(self, qubit1: int, qubit2: int) -> bool:
        """在两个量子比特之间创建纠缠"""
        try:
            # 创建Bell态 |00⟩ + |11⟩
            self.state_vector = np.zeros(self.num_states, dtype=complex)
            # 找到对应|00⟩和|11⟩的状态
            state_00 = 0
            state_11 = (1 << qubit1) | (1 << qubit2)
            self.state_vector[state_00] = 1/np.sqrt(2)
            self.state_vector[state_11] = 1/np.sqrt(2)
            
            # 记录纠缠关系
            pair = tuple(sorted([qubit1, qubit2]))
            self.entanglement_map[pair] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"创建纠缠失败: {e}")
            return False
    
    def run_quantum_algorithm(self, algorithm_type: str) -> Dict:
        """运行量子算法"""
        try:
            if algorithm_type == "deutsch_josza":
                return self._deutsch_josza_algorithm()
            elif algorithm_type == "grover":
                return self._grover_algorithm()
            elif algorithm_type == "quantum_fourier":
                return self._quantum_fourier_transform()
            elif algorithm_type == "bell_test":
                return self._bell_inequality_test()
            else:
                return {'error': f'未知的量子算法: {algorithm_type}'}
                
        except Exception as e:
            self.logger.error(f"量子算法执行失败: {e}")
            return {'error': str(e)}
    
    def _deutsch_josza_algorithm(self) -> Dict:
        """Deutsch-Jozsa算法演示"""
        # 重置量子态
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0
        
        # 应用Hadamard门到所有量子比特
        for qubit in range(self.num_qubits):
            self.apply_gate(self.quantum_gates['H'], qubit)
        
        # 模拟Oracle操作
        oracle_result = random.choice(['constant', 'balanced'])
        if oracle_result == 'balanced':
            # 对于平衡函数，随机翻转一些量子比特
            for qubit in range(self.num_qubits):
                if random.random() < 0.5:
                    self.apply_gate(self.quantum_gates['X'], qubit)
        
        # 再次应用Hadamard门
        for qubit in range(self.num_qubits):
            self.apply_gate(self.quantum_gates['H'], qubit)
        
        # 测量第一个量子比特
        result, probability = self.measure_qubit(0)
        
        return {
            'algorithm': 'Deutsch-Jozsa',
            'oracle_type': oracle_result,
            'measurement_result': result,
            'probability': probability,
            'conclusion': 'constant' if result == 0 else 'balanced'
        }
    
    def _grover_algorithm(self) -> Dict:
        """Grover搜索算法演示"""
        # 重置到均匀叠加态
        for qubit in range(self.num_qubits):
            self.apply_gate(self.quantum_gates['H'], qubit)
        
        # Grover迭代（简化版本）
        iterations = min(4, self.num_qubits)
        for _ in range(iterations):
            # Oracle: 标记目标状态
            target_state = random.randint(0, self.num_states - 1)
            self.state_vector[target_state] = -self.state_vector[target_state]
            
            # 扩散操作
            avg_amplitude = np.mean(self.state_vector)
            for i in range(self.num_states):
                self.state_vector[i] = 2 * avg_amplitude - self.state_vector[i]
        
        # 测量
        result, probability = self.measure_qubit(0)
        
        return {
            'algorithm': 'Grover',
            'target_state': target_state,
            'iterations': iterations,
            'measurement_result': result,
            'probability': probability
        }
    
    def _quantum_fourier_transform(self) -> Dict:
        """量子傅里叶变换"""
        # 应用QFT序列
        for qubit in range(self.num_qubits):
            self.apply_gate(self.quantum_gates['H'], qubit)
            for j in range(qubit + 1, self.num_qubits):
                angle = np.pi / (2 ** (j - qubit))
                if random.random() < 0.5:  # 简化版本
                    self.apply_gate(self.quantum_gates['Rz'](angle), qubit)
        
        return {
            'algorithm': 'Quantum Fourier Transform',
            'qft_completed': True,
            'state_fidelity': np.linalg.norm(self.state_vector) ** 2
        }
    
    def _bell_inequality_test(self) -> Dict:
        """Bell不等式测试"""
        # 创建纠缠态
        self.create_entanglement(0, 1)
        
        # 在两个量子比特上执行Bell测量
        result_qubit0, prob0 = self.measure_qubit(0)
        result_qubit1, prob1 = self.measure_qubit(1)
        
        # 计算相关性
        correlation = result_qubit0 * result_qubit1 if result_qubit0 is not None and result_qubit1 is not None else 0
        
        return {
            'algorithm': 'Bell Inequality Test',
            'result_qubit_0': result_qubit0,
            'result_qubit_1': result_qubit1,
            'correlation': correlation,
            'entangled': correlation == 1  # 纠缠态的完美相关性
        }
    
    def get_quantum_state(self) -> Dict:
        """获取当前量子态信息"""
        probabilities = [abs(amp) ** 2 for amp in self.state_vector]
        max_amplitude_idx = np.argmax(abs(self.state_vector))
        
        return {
            'num_qubits': self.num_qubits,
            'state_vector': [complex(round(c.real, 3), round(c.imag, 3)) for c in self.state_vector],
            'probabilities': [round(p, 3) for p in probabilities],
            'most_probable_state': max_amplitude_idx,
            'entanglement_map': list(self.entanglement_map.keys()),
            'measurement_count': len(self.measurement_history)
        }
    
    def save_quantum_state(self, filepath: str) -> bool:
        """保存量子状态"""
        try:
            quantum_data = {
                'quantum_state': self.get_quantum_state(),
                'measurement_history': self.measurement_history,
                'gate_sequence': []  # 简化：未记录完整的门序列
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(quantum_data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"保存量子状态失败: {e}")
            return False

if __name__ == "__main__":
    # 测试代码
    quantum_brain = QuantumBrain(num_qubits=3)
    
    print("=== 量子大脑模拟器测试 ===")
    
    # 测试量子门
    print("\n1. 测试Hadamard门:")
    quantum_brain.apply_gate(quantum_brain.quantum_gates['H'])
    state = quantum_brain.get_quantum_state()
    print(f"最可能状态: {state['most_probable_state']}")
    
    # 测试量子算法
    print("\n2. 测试Deutsch-Jozsa算法:")
    result = quantum_brain.run_quantum_algorithm("deutsch_josza")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n3. 最终量子态:")
    final_state = quantum_brain.get_quantum_state()
    print(json.dumps(final_state, ensure_ascii=False, indent=2))