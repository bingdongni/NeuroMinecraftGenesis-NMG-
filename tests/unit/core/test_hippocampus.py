#!/usr/bin/env python3
"""
海马体记忆系统完整单元测试
"""

import pytest
import numpy as np
import torch
import time
from typing import List, Dict, Any


class TestHippocampus:
    """海马体记忆系统测试类"""

    def test_initialization(self):
        """测试海马体初始化"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(
            max_capacity=1000,
            embedding_dim=128
        )

        assert hippocampus is not None
        assert hippocampus.max_capacity == 1000
        assert hippocampus.embedding_dim == 128
        assert len(hippocampus.episodic_memory) == 0
        assert len(hippocampus.semantic_memory) == 0

    def test_episodic_memory_storage(self):
        """测试情景记忆存储"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储多个记忆
        memories = []
        for i in range(5):
            memory = {
                'content': f'Memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            key = hippocampus.store_episodic(memory)
            memories.append(key)
            assert key is not None

        assert len(hippocampus.episodic_memory) == 5

    def test_semantic_memory_storage(self):
        """测试语义记忆存储"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储语义记忆
        semantic_data = {
            'concept': '狗',
            'definition': '一种哺乳动物',
            'features': np.random.randn(10),
            'category': '动物'
        }

        key = hippocampus.store_semantic('dog', semantic_data)
        assert key is not None
        assert len(hippocampus.semantic_memory) == 1

    def test_memory_retrieval(self):
        """测试记忆检索"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储记忆
        memory = {
            'content': 'Test memory',
            'timestamp': time.time(),
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        hippocampus.store_episodic(memory)

        # 检索
        query = np.random.randn(64)
        results = hippocampus.retrieve(query, top_k=3)

        assert results is not None
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_memory_consolidation(self):
        """测试记忆巩固"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储多次重复的记忆
        for _ in range(3):
            memory = {
                'content': 'Repeated memory',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            hippocampus.store_episodic(memory)

        # 执行巩固
        hippocampus.consolidate()

        # 检查工作记忆是否被处理
        assert hippocampus.working_memory is not None

    def test_memory_forgetting(self):
        """测试记忆遗忘"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=10, embedding_dim=64)

        # 存储超过容量的记忆
        for i in range(15):
            memory = {
                'content': f'Memory {i}',
                'timestamp': time.time() - i * 100,  # 旧记忆
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            hippocampus.store_episodic(memory)

        # 触发遗忘机制
        hippocampus.forget()

        # 检查记忆数量
        assert len(hippocampus.episodic_memory) <= hippocampus.max_capacity

    def test_spatial_memory(self):
        """测试空间记忆"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储空间位置
        position = np.array([10.0, 20.0, 5.0])
        location_id = hippocampus.store_spatial_memory('room_A', position)

        assert location_id is not None
        assert len(hippocampus.spatial_memory) == 1

        # 检索空间位置
        retrieved_pos = hippocampus.get_spatial_memory('room_A')
        assert retrieved_pos is not None
        np.testing.assert_array_almost_equal(retrieved_pos, position)

    def test_temporal_sequences(self):
        """测试时间序列"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储时间序列
        events = [
            {'event': f'Event {i}', 'timestamp': time.time() + i}
            for i in range(5)
        ]

        sequence_id = hippocampus.store_temporal_sequence('task_1', events)
        assert sequence_id is not None

        # 检索序列
        retrieved = hippocampus.get_temporal_sequence('task_1')
        assert retrieved is not None
        assert len(retrieved) == 5

    def test_memory_pattern_separation(self):
        """测试模式分离"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储相似但不同的记忆
        base_context = np.random.randn(20)
        for i in range(3):
            memory = {
                'content': f'Similar memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': base_context + np.random.randn(20) * 0.01  # 微小差异
            }
            hippocampus.store_episodic(memory)

        # 执行模式分离
        hippocampus.pattern_separation()

        # 验证记忆被分离
        assert len(hippocampus.episodic_memory) >= 3

    def test_memory_pattern_completion(self):
        """测试模式补全"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储完整记忆
        memory = {
            'content': 'Complete memory',
            'timestamp': time.time(),
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        hippocampus.store_episodic(memory)

        # 使用部分线索检索
        partial_cue = np.random.randn(64) * 0.5  # 部分线索
        completed = hippocampus.pattern_completion(partial_cue)

        assert completed is not None

    def test_performance_metrics(self):
        """测试性能指标"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=1000, embedding_dim=128)

        # 存储大量记忆
        start_time = time.time()
        for i in range(100):
            memory = {
                'content': f'Performance memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            hippocampus.store_episodic(memory)

        store_time = time.time() - start_time

        # 测试检索性能
        start_time = time.time()
        for _ in range(100):
            query = np.random.randn(128)
            hippocampus.retrieve(query, top_k=5)
        retrieve_time = time.time() - start_time

        metrics = hippocampus.get_performance_metrics()

        assert 'storage_count' in metrics
        assert 'retrieval_count' in metrics
        assert metrics['storage_count'] == 100
        assert metrics['retrieval_count'] == 100

    def test_memory_weights_update(self):
        """测试记忆权重更新"""
        from core.brain.hippocampus import Hippocampus

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储记忆
        memory = {
            'content': 'Memory for weight update',
            'timestamp': time.time(),
            'emotion': np.random.randn(5),
            'sensory_data': np.random.randn(10),
            'context': np.random.randn(20)
        }
        hippocampus.store_episodic(memory)

        # 更新权重
        key = list(hippocampus.episodic_memory.keys())[0]
        new_importance = 0.9
        hippocampus.update_memory_importance(key, new_importance)

        # 验证更新
        assert hippocampus.episodic_memory[key]['importance'] == new_importance

    def test_memory_export_import(self):
        """测试记忆导出导入"""
        from core.brain.hippocampus import Hippocampus
        import tempfile
        import os

        hippocampus = Hippocampus(max_capacity=100, embedding_dim=64)

        # 存储记忆
        for i in range(5):
            memory = {
                'content': f'Export memory {i}',
                'timestamp': time.time(),
                'emotion': np.random.randn(5),
                'sensory_data': np.random.randn(10),
                'context': np.random.randn(20)
            }
            hippocampus.store_episodic(memory)

        # 导出
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            hippocampus.export_memory(f.name)
            export_path = f.name

        # 创建新实例并导入
        hippocampus2 = Hippocampus(max_capacity=100, embedding_dim=64)
        hippocampus2.import_memory(export_path)

        assert len(hippocampus2.episodic_memory) == 5

        # 清理
        os.unlink(export_path)

    def test_concurrent_access(self):
        """测试并发访问"""
        from core.brain.hippocampus import Hippocampus
        import threading

        hippocampus = Hippocampus(max_capacity=1000, embedding_dim=128)
        errors = []

        def store_memories(count):
            try:
                for i in range(count):
                    memory = {
                        'content': f'Thread memory {i}',
                        'timestamp': time.time(),
                        'emotion': np.random.randn(5),
                        'sensory_data': np.random.randn(10),
                        'context': np.random.randn(20)
                    }
                    hippocampus.store_episodic(memory)
            except Exception as e:
                errors.append(e)

        # 并发存储
        threads = [
            threading.Thread(target=store_memories, args=(50,))
            for _ in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(hippocampus.episodic_memory) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
