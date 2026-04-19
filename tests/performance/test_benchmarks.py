#!/usr/bin/env python3
"""
完整性能基准测试套件
"""

import pytest
import numpy as np
import time
import torch
import psutil
import threading
from typing import Dict, List, Any


class TestBrainPerformance:
    """大脑模块性能测试"""

    def test_hippocampus_throughput(self, benchmark):
        """测试海马体吞吐量"""
        from core.brain.hippocampus import Hippocampus

        memory = Hippocampus(max_capacity=10000, embedding_dim=128)

        # 准备批量数据
        batch_size = 100
        memories = [
            {
                'content': f'Memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            for i in range(batch_size)
        ]

        # 性能测试
        start_time = time.time()
        for mem in memories:
            memory.store_episodic(mem)
        elapsed = time.time() - start_time

        throughput = batch_size / elapsed

        print(f"\nHippocampus Storage Throughput: {throughput:.2f} memories/sec")

        assert throughput > 10, f"Throughput too low: {throughput:.2f}"

    def test_hippocampus_retrieval_latency(self, benchmark):
        """测试海马体检索延迟"""
        from core.brain.hippocampus import Hippocampus

        memory = Hippocampus(max_capacity=1000, embedding_dim=128)

        # 预填充数据
        for i in range(500):
            memory.store_episodic({
                'content': f'Memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            })

        # 测试检索延迟
        latencies = []
        for _ in range(100):
            query = np.random.randn(128)
            start = time.time()
            memory.retrieve(query, top_k=10)
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000  # ms

        print(f"\nHippocampus Retrieval Latency: {avg_latency:.2f} ms")

        assert avg_latency < 100, f"Latency too high: {avg_latency:.2f} ms"

    def test_attention_performance(self, benchmark):
        """测试注意力机制性能"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8)

        # 批量处理
        batch_sizes = [1, 8, 16, 32]
        results = {}

        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 100, 128)

            latencies = []
            for _ in range(50):
                start = time.time()
                gate.compute_attention(features)
                latencies.append(time.time() - start)

            results[batch_size] = np.mean(latencies) * 1000

        print(f"\nAttention Performance by Batch Size:")
        for bs, lat in results.items():
            print(f"  Batch {bs}: {lat:.2f} ms")

    def test_imagination_generation_speed(self, benchmark):
        """测试想象力生成速度"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=128, hidden_dim=256, spatial_dim=3)

        # 测试不同预测步数的性能
        steps_list = [1, 5, 10, 20]
        results = {}

        for steps in steps_list:
            times = []
            for _ in range(20):
                state = np.random.randn(128).astype(np.float32)
                spatial = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                temporal = np.array([0.0, 1.0, 0.5, 0.1], dtype=np.float32)

                start = time.time()
                engine.predict_future_state(state, steps=steps, spatial_info=spatial, temporal_info=temporal)
                times.append(time.time() - start)

            results[steps] = np.mean(times) * 1000

        print(f"\nImagination Generation Speed:")
        for steps, time_ms in results.items():
            print(f"  {steps} steps: {time_ms:.2f} ms")


class TestQuantumPerformance:
    """量子系统性能测试"""

    def test_quantum_state_operations(self, benchmark):
        """测试量子态操作性能"""
        from core.quantum_brain.fusion_system import QuantumState

        # 测试不同量子比特数
        qubit_counts = [4, 8, 12, 16]
        results = {}

        for n_qubits in qubit_counts:
            state = QuantumState(n_qubits=n_qubits)

            # 测量操作性能
            times = []
            for _ in range(100):
                state.set_superposition([i for i in range(min(4, 2**n_qubits))],
                                       [0.5]*min(4, 2**n_qubits))
                start = time.time()
                for _ in range(10):
                    state.measure()
                times.append(time.time() - start)

            results[n_qubits] = np.mean(times) * 1000

        print(f"\nQuantum State Operations:")
        for nq, t in results.items():
            print(f"  {nq} qubits: {t:.2f} ms (10 measurements)")

    def test_quantum_decision_latency(self, benchmark):
        """测试量子决策延迟"""
        from core.quantum_brain.fusion_system import QuantumDecisionCircuit

        circuit = QuantumDecisionCircuit(n_qubits=8, n_output_qubits=2)

        latencies = []
        for _ in range(100):
            input_signal = np.random.randn(8)
            start = time.time()
            circuit.quantum_decision(input_signal)
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000

        print(f"\nQuantum Decision Latency: {avg_latency:.2f} ms")

    def test_quantum_fusion_throughput(self, benchmark):
        """测试量子融合吞吐量"""
        from core.quantum_brain.fusion_system import QuantumBrainFusion

        fusion = QuantumBrainFusion(n_neurons=1000, n_qubits=5)
        fusion.initialize_system()

        # 测试吞吐量
        latencies = []
        for _ in range(50):
            input_signal = np.random.randn(8)
            start = time.time()
            fusion.process_input(input_signal)
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000
        throughput = 1000 / avg_latency

        print(f"\nQuantum Fusion: {throughput:.2f} req/sec, {avg_latency:.2f} ms latency")


class TestEvolutionPerformance:
    """进化系统性能测试"""

    def test_genetic_engine_speed(self, benchmark):
        """测试遗传引擎速度"""
        from core.evolution.genetic_engine import GeneticEngine

        population_sizes = [16, 32, 64]
        results = {}

        for pop_size in population_sizes:
            engine = GeneticEngine(population_size=pop_size, rule_dim=50)

            def fitness_func(individual):
                return (np.mean(individual), np.sum(individual > 0) / len(individual))

            engine.set_fitness_evaluator(fitness_func)

            times = []
            for _ in range(10):
                population = engine.initialize_population()
                start = time.time()
                engine.evaluate_fitness(population)
                times.append(time.time() - start)

            results[pop_size] = np.mean(times) * 1000

        print(f"\nGenetic Engine Evaluation Speed:")
        for ps, t in results.items():
            print(f"  Population {ps}: {t:.2f} ms")

    def test_nsga2_sorting_performance(self, benchmark):
        """测试NSGA-II排序性能"""
        from core.evolution.nsga_ii import NSGA2Selector

        selector = NSGA2Selector()

        sizes = [50, 100, 200, 500]
        results = {}

        for size in sizes:
            fitness_values = [(np.random.rand(), np.random.rand()) for _ in range(size)]

            times = []
            for _ in range(20):
                start = time.time()
                selector.fast_non_dominated_sort(fitness_values)
                times.append(time.time() - start)

            results[size] = np.mean(times) * 1000

        print(f"\nNSGA-II Sorting Performance:")
        for size, t in results.items():
            print(f"  {size} individuals: {t:.2f} ms")


class TestMemoryPerformance:
    """记忆系统性能测试"""

    def test_memory_capacity_scaling(self, benchmark):
        """测试记忆容量扩展"""
        from core.brain.hippocampus import Hippocampus

        capacities = [100, 1000, 5000, 10000]
        results = {}

        for capacity in capacities:
            memory = Hippocampus(max_capacity=capacity, embedding_dim=128)

            # 填充到容量
            for i in range(capacity):
                memory.store_episodic({
                    'content': f'Memory {i}',
                    'timestamp': time.time(),
                    'emotion': np.random.randn(5),
                    'sensory_data': np.random.randn(10),
                    'context': np.random.randn(20)
                })

            # 测试检索性能
            latencies = []
            for _ in range(50):
                query = np.random.randn(128)
                start = time.time()
                memory.retrieve(query, top_k=5)
                latencies.append(time.time() - start)

            results[capacity] = np.mean(latencies) * 1000

        print(f"\nMemory Capacity Scaling:")
        for cap, lat in results.items():
            print(f"  Capacity {cap}: {lat:.2f} ms latency")

    def test_memory_concurrent_access(self, benchmark):
        """测试记忆并发访问"""
        from core.brain.hippocampus import Hippocampus
        import threading

        memory = Hippocampus(max_capacity=5000, embedding_dim=128)

        # 预填充
        for i in range(1000):
            memory.store_episodic({
                'content': f'Memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            })

        # 并发访问测试
        def worker():
            for _ in range(50):
                query = np.random.randn(128)
                memory.retrieve(query, top_k=5)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        throughput = 200 / elapsed  # 4 threads * 50 operations

        print(f"\nConcurrent Memory Access: {throughput:.2f} ops/sec")


class TestSystemPerformance:
    """系统级性能测试"""

    def test_cpu_memory_usage(self, benchmark):
        """测试CPU和内存使用"""
        from core.brain.hippocampus import Hippocampus
        from core.brain.thalamic_gate import ThalamicGate
        from core.quantum_brain.fusion_system import QuantumBrainFusion

        process = psutil.Process()

        # 记录初始状态
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个模块
        modules = {
            'memory': Hippocampus(max_capacity=5000, embedding_dim=128),
            'attention': ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8),
            'quantum': QuantumBrainFusion(n_neurons=2000, n_qubits=6)
        }

        modules['quantum'].initialize_system()

        # 添加数据
        for i in range(2000):
            modules['memory'].store_episodic({
                'content': f'Memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            })

        # 记录最终状态
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nSystem Resource Usage:")
        print(f"  Initial Memory: {initial_memory:.2f} MB")
        print(f"  Final Memory: {final_memory:.2f} MB")
        print(f"  Memory Increase: {memory_increase:.2f} MB")

        modules['quantum'].shutdown()

    def test_full_pipeline_latency(self, benchmark):
        """测试完整流水线延迟"""
        from core.brain.perception_module import PerceptionModule
        from core.brain.thalamic_gate import ThalamicGate
        from core.brain.hippocampus import Hippocampus

        # 初始化模块
        perception = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=128)
        attention = ThalamicGate(input_dim=128, hidden_dim=256, num_attention_heads=8)
        memory = Hippocampus(max_capacity=1000, embedding_dim=128)

        # 测试流水线
        latencies = []
        for _ in range(50):
            start = time.time()

            # 感知
            image = torch.randn(1, 3, 224, 224)
            features = perception.process_visual(image)

            # 注意力
            attended = attention.compute_attention(features.unsqueeze(1))

            # 记忆
            memory.store_episodic({
                'content': 'Pipeline test',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            })

            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000

        print(f"\nFull Pipeline Latency: {avg_latency:.2f} ms")


class TestAgentPerformance:
    """智能体性能测试"""

    def test_action_execution_speed(self, benchmark):
        """测试动作执行速度"""
        from agents.single.action_executor import ActionExecutor, ActionType

        executor = ActionExecutor()

        latencies = []
        for _ in range(100):
            start = time.time()
            import asyncio
            asyncio.run(executor.execute_action(ActionType.MOVE_FORWARD, distance=3.0))
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000

        print(f"\nAction Execution Latency: {avg_latency:.2f} ms")

    def test_skill_execution_speed(self, benchmark):
        """测试技能执行速度"""
        from agents.single.skill_library import SkillLibrary
        from agents.single.action_executor import ActionExecutor

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)

        latencies = []
        for _ in range(20):
            start = time.time()
            import asyncio
            asyncio.run(skill_lib.execute_skill("tree_harvesting", tree_count=2))
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000

        print(f"\nSkill Execution Latency: {avg_latency:.2f} ms")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark'])
