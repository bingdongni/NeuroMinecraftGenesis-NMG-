#!/usr/bin/env python3
"""
丘脑门控注意力系统完整单元测试
"""

import pytest
import numpy as np
import torch
import time


class TestThalamicGate:
    """丘脑门控系统测试类"""

    def test_initialization(self):
        """测试丘脑门控初始化"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(
            input_dim=128,
            hidden_dim=256,
            num_attention_heads=8
        )

        assert gate is not None
        assert gate.input_dim == 128
        assert gate.hidden_dim == 256

    def test_attention_weight_calculation(self):
        """测试注意力权重计算"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 输入特征
        features = torch.randn(4, 10, 64)

        # 计算注意力
        attention_weights = gate.compute_attention(features)

        assert attention_weights is not None
        assert attention_weights.shape[0] == 4  # batch size

    def test_attention_focus_switching(self):
        """测试注意力焦点切换"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 输入特征
        features = torch.randn(4, 10, 64)

        # 执行多次注意力焦点切换
        for _ in range(5):
            output = gate.focus_attention(features, focus_strength=0.8)
            assert output is not None

        # 验证焦点历史
        metrics = gate.get_attention_metrics()
        assert metrics['total_switches'] >= 5

    def test_metacognitive_monitoring(self):
        """测试元认知监控"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 执行注意力处理
        features = torch.randn(4, 10, 64)
        gate.compute_attention(features)

        # 获取元认知监控报告
        report = gate.metacognitive_monitor()

        assert report is not None
        assert 'attention_quality' in report
        assert 'focus_stability' in report
        assert 'cognitive_load' in report

    def test_multi_source_filtering(self):
        """测试多源信息过滤"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 模拟多源输入
        visual_input = torch.randn(4, 10, 64)
        audio_input = torch.randn(4, 10, 64)
        semantic_input = torch.randn(4, 10, 64)

        # 融合多源信息
        fused = gate.filter_multisource(
            visual=visual_input,
            audio=audio_input,
            semantic=semantic_input
        )

        assert fused is not None
        assert fused.shape[0] == 4  # batch size

    def test_dynamic_focus_adjustment(self):
        """测试动态焦点调整"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 执行处理
        features = torch.randn(4, 10, 64)
        initial_output = gate.compute_attention(features)

        # 调整焦点
        adjusted_output = gate.adjust_focus(
            initial_output,
            adjustment_type='enhance',
            target_region=0
        )

        assert adjusted_output is not None

    def test_attention_gate_mechanism(self):
        """测试门控机制"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 输入特征
        features = torch.randn(4, 10, 64)
        gate_values = torch.sigmoid(torch.randn(4, 10, 64))

        # 应用门控
        gated_output = gate.apply_gate(features, gate_values)

        assert gated_output is not None
        assert gated_output.shape == features.shape

    def test_attention_performance_metrics(self):
        """测试注意力性能指标"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 执行多次处理
        features = torch.randn(4, 10, 64)
        for _ in range(10):
            gate.compute_attention(features)

        # 获取性能指标
        metrics = gate.get_performance_metrics()

        assert 'processing_speed' in metrics
        assert 'attention_efficiency' in metrics
        assert 'focus_accuracy' in metrics

    def test_salience_computation(self):
        """测试显著性计算"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 输入特征
        features = torch.randn(4, 10, 64)

        # 计算显著性
        salience = gate.compute_salience(features)

        assert salience is not None
        assert salience.shape[0] == 4  # batch size

    def test_meta_learning_fast_adaptation(self):
        """测试元学习快速适应"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 准备支持集和查询集
        support_features = torch.randn(4, 5, 64)
        support_labels = torch.randn(4, 64)
        query_features = torch.randn(4, 5, 64)

        # 快速适应
        adapted_output = gate.fast_adaptation(
            support_features,
            support_labels,
            query_features
        )

        assert adapted_output is not None
        assert adapted_output.shape[0] == 4

    def test_attention_reset(self):
        """测试注意力重置"""
        from core.brain.thalamic_gate import ThalamicGate

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 执行处理
        features = torch.randn(4, 10, 64)
        gate.compute_attention(features)

        # 重置
        gate.reset()

        # 验证重置
        metrics = gate.get_attention_metrics()
        assert metrics['total_switches'] == 0

    def test_attention_serialization(self):
        """测试注意力序列化"""
        from core.brain.thalamic_gate import ThalamicGate
        import tempfile

        gate = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)

        # 执行一些处理
        features = torch.randn(4, 10, 64)
        gate.compute_attention(features)

        # 保存状态
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            state_path = f.name

        gate.save_state(state_path)

        # 创建新实例并加载
        gate2 = ThalamicGate(input_dim=64, hidden_dim=128, num_attention_heads=4)
        gate2.load_state(state_path)

        # 验证状态
        assert gate2.get_attention_metrics()['total_processed'] == \
               gate.get_attention_metrics()['total_processed']

        # 清理
        import os
        os.unlink(state_path)


class TestMetacognitiveModule:
    """元认知模块测试类"""

    def test_metacognitive_initialization(self):
        """测试元认知模块初始化"""
        from core.brain.thalamic_gate import MetacognitiveModule

        module = MetacognitiveModule(
            feature_dim=128,
            meta_dim=64
        )

        assert module is not None
        assert module.feature_dim == 128

    def test_task_similarity_calculation(self):
        """测试任务相似度计算"""
        from core.brain.thalamic_gate import MetacognitiveModule

        module = MetacognitiveModule(feature_dim=128, meta_dim=64)

        # 添加任务表示
        features1 = torch.randn(10, 128)
        features2 = torch.randn(10, 128)

        module.update_task_representation('task1', features1)
        module.update_task_representation('task2', features2)

        # 计算相似度
        similarity = module.get_task_similarity('task1', 'task2')

        assert 0 <= similarity <= 1

    def test_adaptation_rate_adjustment(self):
        """测试适应率调整"""
        from core.brain.thalamic_gate import MetacognitiveModule

        module = MetacognitiveModule(feature_dim=128, meta_dim=64)

        initial_rate = module.adaptation_rate

        # 调整适应率
        module.adjust_adaptation_rate(0.02)

        assert module.adaptation_rate != initial_rate


class TestSalienceComputation:
    """显著性计算测试类"""

    def test_salience_initialization(self):
        """测试显著性计算初始化"""
        from core.brain.thalamic_gate import SalienceComputation

        module = SalienceComputation(
            feature_dim=128,
            num_heads=8
        )

        assert module is not None
        assert module.feature_dim == 128

    def test_salience_map_generation(self):
        """测试显著性图生成"""
        from core.brain.thalamic_gate import SalienceComputation

        module = SalienceComputation(feature_dim=128, num_heads=8)

        # 输入特征
        features = torch.randn(4, 20, 128)

        # 生成显著性图
        salience_map = module.compute_salience_map(features)

        assert salience_map is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
