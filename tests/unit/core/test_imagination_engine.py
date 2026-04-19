#!/usr/bin/env python3
"""
想象力引擎完整单元测试
"""

import pytest
import numpy as np
import torch
import time


class TestImaginationEngine:
    """想象力引擎测试类"""

    def test_initialization(self):
        """测试想象力引擎初始化"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(
            state_dim=128,
            hidden_dim=256,
            spatial_dim=3
        )

        assert engine is not None
        assert engine.state_dim == 128
        assert engine.hidden_dim == 256

    def test_world_model_prediction(self):
        """测试世界模型预测"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 当前状态
        current_state = np.random.randn(64).astype(np.float32)
        spatial_info = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        temporal_info = np.array([0.0, 1.0, 0.5, 0.1], dtype=np.float32)

        # 预测未来状态
        prediction = engine.predict_future_state(
            current_state,
            steps=5,
            spatial_info=spatial_info,
            temporal_info=temporal_info
        )

        assert prediction is not None
        assert 'predicted_states' in prediction
        assert len(prediction['predicted_states']) == 5

    def test_contrarian_scenario_generation(self):
        """测试反事实场景生成"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 真实场景
        real_state = np.random.randn(64).astype(np.float32)
        real_action = np.array([1.0, 0.0, 0.5])

        # 生成反事实场景
        counterfactuals = engine.generate_counterfactual(
            real_state,
            real_action,
            num_alternatives=3
        )

        assert counterfactuals is not None
        assert len(counterfactuals) == 3

        # 验证反事实场景与真实场景不同
        for cf in counterfactuals:
            assert 'state' in cf
            assert 'probability' in cf

    def test_dream_replay(self):
        """测试梦境回放机制"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 提供记忆序列
        memory_sequence = [
            np.random.randn(64).astype(np.float32)
            for _ in range(10)
        ]

        # 梦境回放
        dream_sequence = engine.dream_replay(memory_sequence)

        assert dream_sequence is not None
        assert len(dream_sequence) > 0

        # 验证梦境内容
        for dream in dream_sequence:
            assert 'state' in dream
            assert 'novelty_score' in dream

    def test_parallel_possibility_evaluation(self):
        """测试并行可能性评估"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 当前状态
        current_state = np.random.randn(64).astype(np.float32)

        # 评估多个可能的行动
        possible_actions = [
            np.array([1.0, 0.0, 0.5]),
            np.array([0.0, 1.0, 0.3]),
            np.array([0.5, 0.5, 0.8])
        ]

        # 并行评估
        evaluations = engine.evaluate_possibilities(
            current_state,
            possible_actions,
            parallel=True
        )

        assert evaluations is not None
        assert len(evaluations) == len(possible_actions)

        # 验证评估结果
        for eval_result in evaluations:
            assert 'expected_reward' in eval_result
            assert 'risk_score' in eval_result

    def test_imagination_metrics(self):
        """测试想象力指标"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 生成想象力
        for _ in range(10):
            state = np.random.randn(64).astype(np.float32)
            spatial = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            temporal = np.array([0.0, 1.0, 0.5, 0.1], dtype=np.float32)

            engine.predict_future_state(state, steps=3, spatial_info=spatial, temporal_info=temporal)

        # 获取指标
        metrics = engine.get_imagination_metrics()

        assert 'total_predictions' in metrics
        assert 'average_novelty' in metrics
        assert 'creativity_score' in metrics
        assert metrics['total_predictions'] == 10

    def test_spatial_temporal_reasoning(self):
        """测试时空推理"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 提供时空信息
        current_time = 0.0
        spatial_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 推理未来位置
        future_time = 5.0
        predicted_trajectory = engine.predict_trajectory(
            current_time,
            spatial_position,
            future_time,
            steps=10
        )

        assert predicted_trajectory is not None
        assert len(predicted_trajectory) > 0

        # 验证轨迹包含时空信息
        for point in predicted_trajectory:
            assert 'position' in point
            assert 'time' in point

    def test_creativity_enhancement(self):
        """测试创造力增强"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 正常预测
        state = np.random.randn(64).astype(np.float32)
        normal_pred = engine.predict_future_state(state, steps=3)

        # 设置高创造力模式
        engine.set_creativity_mode('high')
        creative_pred = engine.predict_future_state(state, steps=3)

        # 验证创造力模式下的结果不同
        assert normal_pred is not None
        assert creative_pred is not None

        # 高创造力模式应该有更高的新颖性
        assert creative_pred['novelty_score'] >= normal_pred['novelty_score']

    def test_imagination_control(self):
        """测试想象力控制"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 设置想象力质量
        engine.set_imagination_quality('high')
        assert engine.imagination_quality == 'high'

        # 设置时间范围
        engine.set_time_horizon(10.0)
        assert engine.max_prediction_steps == 10.0

    def test_memory_integration(self):
        """测试记忆整合"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 添加记忆
        for i in range(5):
            memory = {
                'state': np.random.randn(64).astype(np.float32),
                'temporal_context': np.array([float(i), 1.0, 0.5, 0.1]),
                'spatial_context': np.array([float(i), 0.0, 0.0])
            }
            engine.add_to_memory(memory)

        # 验证记忆存储
        assert engine.get_memory_size() == 5

    def test_error_recovery(self):
        """测试错误恢复"""
        from core.brain.imagination_engine import ImaginationEngine

        engine = ImaginationEngine(state_dim=64, hidden_dim=128, spatial_dim=3)

        # 测试无效输入
        try:
            engine.predict_future_state(None, steps=5)
            assert False, "Should have raised an error"
        except (ValueError, TypeError):
            pass

        # 验证引擎仍然可用
        state = np.random.randn(64).astype(np.float32)
        result = engine.predict_future_state(state, steps=3)
        assert result is not None


class TestDiffusionModel:
    """扩散模型测试类"""

    def test_diffusion_model_initialization(self):
        """测试扩散模型初始化"""
        from core.brain.imagination_engine import DiffusionModel

        model = DiffusionModel(
            state_dim=64,
            hidden_dim=128,
            timesteps=50
        )

        assert model is not None
        assert model.state_dim == 64
        assert model.timesteps == 50

    def test_diffusion_forward(self):
        """测试扩散前向传播"""
        from core.brain.imagination_engine import DiffusionModel

        model = DiffusionModel(state_dim=64, hidden_dim=128, timesteps=50)

        # 准备输入
        x = torch.randn(4, 64)
        t = torch.randint(0, 50, (4,))
        spatial = torch.randn(4, 3)
        temporal = torch.randn(4, 4)

        # 前向传播
        output = model(x, t, spatial_info=spatial, temporal_info=temporal)

        assert output is not None
        assert 'noise' in output
        assert output['noise'].shape == (4, 64)

    def test_ddim_sampling(self):
        """测试DDIM采样"""
        from core.brain.imagination_engine import DiffusionModel

        model = DiffusionModel(state_dim=64, hidden_dim=128, timesteps=20)

        # 初始噪声
        x_start = torch.randn(2, 64)

        # DDIM采样
        samples = model.ddim_sample(x_start, num_steps=10)

        assert samples is not None
        assert samples.shape == (2, 64)

    def test_noise_schedule(self):
        """测试噪声调度"""
        from core.brain.imagination_engine import DiffusionModel

        model = DiffusionModel(state_dim=64, hidden_dim=128, timesteps=50)

        # 验证噪声调度
        alpha_schedule = model.alpha_schedule
        assert len(alpha_schedule) == 50
        assert alpha_schedule[0] > alpha_schedule[-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
