#!/usr/bin/env python3
"""
感知模块完整单元测试
"""

import pytest
import numpy as np
import torch


class TestPerceptionModule:
    """感知模块测试类"""

    def test_initialization(self):
        """测试感知模块初始化"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        )

        assert module is not None
        assert module.input_dim == 128

    def test_visual_processing(self):
        """测试视觉处理"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟图像输入
        image = torch.randn(4, 3, 224, 224)  # batch, channels, height, width

        # 处理视觉输入
        visual_features = module.process_visual(image)

        assert visual_features is not None
        assert visual_features.shape[0] == 4  # batch size

    def test_audio_processing(self):
        """测试音频处理"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟音频输入
        audio = torch.randn(4, 16000)  # batch, samples

        # 处理音频输入
        audio_features = module.process_audio(audio)

        assert audio_features is not None
        assert audio_features.shape[0] == 4

    def test_text_processing(self):
        """测试文本处理"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟文本输入
        text = ["Hello world", "This is a test", "Another sentence", "Final example"]

        # 处理文本输入
        text_features = module.process_text(text)

        assert text_features is not None
        assert text_features.shape[0] == 4

    def test_multimodal_fusion(self):
        """测试多模态融合"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 准备多模态输入
        visual = torch.randn(4, 128)
        audio = torch.randn(4, 128)
        text = torch.randn(4, 128)

        # 融合
        fused = module.fuse_multimodal(
            visual=visual,
            audio=audio,
            text=text
        )

        assert fused is not None
        assert fused.shape[0] == 4

    def test_attention_mechanism(self):
        """测试注意力机制"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 输入特征
        features = torch.randn(4, 20, 128)

        # 应用注意力
        attended = module.apply_attention(features)

        assert attended is not None
        assert attended.shape[0] == 4

    def test_object_detection(self):
        """测试对象检测"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟图像
        image = torch.randn(4, 3, 224, 224)

        # 检测对象
        detections = module.detect_objects(image)

        assert detections is not None
        assert isinstance(detections, list)

    def test_scene_understanding(self):
        """测试场景理解"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟图像
        image = torch.randn(4, 3, 224, 224)

        # 理解场景
        scene_info = module.understand_scene(image)

        assert scene_info is not None
        assert 'scene_type' in scene_info
        assert 'objects' in scene_info

    def test_depth_estimation(self):
        """测试深度估计"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 模拟图像
        image = torch.randn(4, 3, 224, 224)

        # 估计深度
        depth_map = module.estimate_depth(image)

        assert depth_map is not None
        assert depth_map.shape[2:] == (224, 224)

    def test_feature_extraction(self):
        """测试特征提取"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 输入
        features = torch.randn(4, 128)

        # 提取特征
        extracted = module.extract_features(features)

        assert extracted is not None
        assert extracted.shape[0] == 4

    def test_perception_metrics(self):
        """测试感知性能指标"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 执行处理
        for _ in range(10):
            image = torch.randn(4, 3, 224, 224)
            module.process_visual(image)

        # 获取指标
        metrics = module.get_performance_metrics()

        assert 'total_processed' in metrics
        assert 'average_latency' in metrics
        assert 'accuracy' in metrics

    def test_error_handling(self):
        """测试错误处理"""
        from core.brain.perception_module import PerceptionModule

        module = PerceptionModule(input_dim=128, hidden_dim=256, output_dim=64)

        # 测试无效输入
        try:
            module.process_visual(None)
            assert False, "Should have raised an error"
        except (ValueError, TypeError):
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
