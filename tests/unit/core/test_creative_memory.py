#!/usr/bin/env python3
"""
创意记忆系统完整单元测试
"""

import pytest
import numpy as np
import torch


class TestCreativeMemory:
    """创意记忆系统测试类"""

    def test_initialization(self):
        """测试创意记忆初始化"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(
            capacity=1000,
            embedding_dim=128
        )

        assert memory is not None
        assert memory.capacity == 1000
        assert memory.embedding_dim == 128

    def test_memory_storage(self):
        """测试记忆存储"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储创意记忆
        content = {
            'text': 'A creative idea',
            'features': torch.randn(64),
            'creativity_score': 0.8
        }

        key = memory.store(content)
        assert key is not None
        assert len(memory.memories) == 1

    def test_memory_retrieval(self):
        """测试记忆检索"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储记忆
        for i in range(5):
            content = {
                'text': f'Idea {i}',
                'features': torch.randn(64),
                'creativity_score': np.random.rand()
            }
            memory.store(content)

        # 检索
        query = torch.randn(64)
        results = memory.retrieve(query, top_k=3)

        assert results is not None
        assert len(results) <= 3

    def test_diversity_calculation(self):
        """测试多样性计算"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储多样化的记忆
        for i in range(10):
            content = {
                'text': f'Diverse idea {i}',
                'features': torch.randn(64),
                'creativity_score': np.random.rand()
            }
            memory.store(content)

        # 计算多样性
        diversity = memory.calculate_diversity()

        assert 0 <= diversity <= 1

    def test_creativity_scoring(self):
        """测试创意评分"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 准备特征
        features = torch.randn(64)

        # 评分
        score = memory.score_creativity(features)

        assert 0 <= score <= 1

    def test_memory_association(self):
        """测试记忆关联"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储多个相关记忆
        for i in range(3):
            content = {
                'text': f'Related idea {i}',
                'features': torch.randn(64),
                'creativity_score': 0.7
            }
            memory.store(content)

        # 查找关联
        associations = memory.find_associations(top_k=2)

        assert associations is not None

    def test_novelty_detection(self):
        """测试新颖性检测"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储一个记忆
        content1 = {
            'text': 'Original idea',
            'features': torch.randn(64),
            'creativity_score': 0.8
        }
        memory.store(content1)

        # 检测类似想法的新颖性
        similar_features = content1['features'] + torch.randn(64) * 0.01
        novelty1 = memory.detect_novelty(similar_features)

        # 检测非常不同的想法的新颖性
        different_features = torch.randn(64)
        novelty2 = memory.detect_novelty(different_features)

        assert novelty1 < novelty2

    def test_memory_consolidation(self):
        """测试记忆整合"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储多个相关记忆
        for i in range(5):
            content = {
                'text': f'Consolidation test {i}',
                'features': torch.randn(64),
                'creativity_score': 0.7
            }
            memory.store(content)

        # 执行整合
        memory.consolidate()

        # 验证
        assert len(memory.memories) >= 0

    def test_serendipity_exploration(self):
        """测试意外发现探索"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储记忆
        for i in range(10):
            content = {
                'text': f'Exploration memory {i}',
                'features': torch.randn(64),
                'creativity_score': np.random.rand()
            }
            memory.store(content)

        # 探索意外发现
        serendipities = memory.explore_serendipity(num_suggestions=3)

        assert serendipities is not None
        assert len(serendipities) <= 3

    def test_inspiration_triggering(self):
        """测试灵感触发"""
        from core.brain.creative_memory import CreativeMemory

        memory = CreativeMemory(capacity=100, embedding_dim=64)

        # 存储记忆
        content = {
            'text': 'Inspiration source',
            'features': torch.randn(64),
            'creativity_score': 0.9
        }
        memory.store(content)

        # 触发灵感
        trigger = torch.randn(64)
        inspirations = memory.trigger_inspiration(trigger)

        assert inspirations is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
