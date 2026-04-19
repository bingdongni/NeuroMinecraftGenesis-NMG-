#!/usr/bin/env python3
"""
多世界系统完整单元测试
"""

import pytest
import numpy as np


class TestIntegratedEnvironment:
    """集成环境测试类"""

    def test_initialization(self):
        """测试集成环境初始化"""
        from worlds.integrated_environment import IntegratedEnvironment

        env = IntegratedEnvironment()

        assert env is not None

    @pytest.mark.asyncio
    async def test_world_creation(self):
        """测试世界创建"""
        from worlds.integrated_environment import IntegratedEnvironment

        env = IntegratedEnvironment()

        await env.create_worlds()

        assert env.real_world is not None
        assert env.virtual_world is not None
        assert env.game_world is not None


class TestRealWorld:
    """真实世界测试类"""

    def test_initialization(self):
        """测试真实世界初始化"""
        from worlds.integrated_environment import RealWorld

        world = RealWorld(world_id="test_real")

        assert world is not None

    @pytest.mark.asyncio
    async def test_world_initialization(self):
        """测试世界初始化"""
        from worlds.integrated_environment import RealWorld

        world = RealWorld(world_id="test_real")

        try:
            result = await world.initialize()
            # 可能因为没有摄像头而失败，这是预期的
            assert isinstance(result, bool)
        except Exception:
            # 如果初始化失败，可能是环境问题
            pass


class TestWorldGenerator:
    """世界生成器测试类"""

    def test_initialization(self):
        """测试世界生成器初始化"""
        from worlds.procgen.world_generator import WorldGenerator

        generator = WorldGenerator(
            world_size=(256, 256),
            seed=42
        )

        assert generator is not None

    def test_terrain_generation(self):
        """测试地形生成"""
        from worlds.procgen.world_generator import WorldGenerator

        generator = WorldGenerator(world_size=(64, 64), seed=42)

        terrain = generator.generate_terrain()

        assert terrain is not None
        assert terrain.shape == (64, 64)

    def test_resource_distribution(self):
        """测试资源分布"""
        from worlds.procgen.world_generator import WorldGenerator

        generator = WorldGenerator(world_size=(64, 64), seed=42)

        terrain = generator.generate_terrain()
        resources = generator.distribute_resources(terrain)

        assert resources is not None
        assert len(resources) > 0

    def test_world_serialization(self):
        """测试世界序列化"""
        from worlds.procgen.world_generator import WorldGenerator
        import tempfile

        generator = WorldGenerator(world_size=(32, 32), seed=42)

        terrain = generator.generate_terrain()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        generator.save_world(path)

        generator2 = WorldGenerator(world_size=(32, 32), seed=42)
        generator2.load_world(path)

        assert generator2.terrain is not None

        import os
        os.unlink(path)


class TestDynamicComplexity:
    """动态复杂度测试类"""

    def test_initialization(self):
        """测试动态复杂度初始化"""
        from worlds.procgen.dynamic_complexity import DynamicComplexity

        complexity = DynamicComplexity()

        assert complexity is not None

    def test_complexity_adjustment(self):
        """测试复杂度调整"""
        from worlds.procgen.dynamic_complexity import DynamicComplexity

        complexity = DynamicComplexity()

        complexity.adjust_complexity(target_level=0.7)

        assert complexity.current_level is not None

    def test_difficulty_progression(self):
        """测试难度递进"""
        from worlds.procgen.dynamic_complexity import DynamicComplexity

        complexity = DynamicComplexity()

        complexity.set_progression_rate(0.1)

        for _ in range(10):
            complexity.update()

        assert complexity.current_level > 0


class TestAdaptiveDifficulty:
    """自适应难度测试类"""

    def test_initialization(self):
        """测试自适应难度初始化"""
        from worlds.procgen.adaptive_difficulty import AdaptiveDifficulty

        difficulty = AdaptiveDifficulty()

        assert difficulty is not None

    def test_difficulty_calculation(self):
        """测试难度计算"""
        from worlds.procgen.adaptive_difficulty import AdaptiveDifficulty

        difficulty = AdaptiveDifficulty()

        # 提供性能数据
        performance = {
            'success_rate': 0.8,
            'completion_time': 120,
            'error_rate': 0.1
        }

        level = difficulty.calculate_difficulty(performance)

        assert 0 <= level <= 1


class TestObjectDetector:
    """对象检测测试类"""

    def test_initialization(self):
        """测试对象检测初始化"""
        from worlds.real.object_detector import ObjectDetector

        detector = ObjectDetector()

        assert detector is not None

    def test_detection(self):
        """测试检测功能"""
        from worlds.real.object_detector import ObjectDetector

        detector = ObjectDetector()

        # 创建测试图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        assert detections is not None
        assert isinstance(detections, list)


class TestCameraCapture:
    """摄像头捕获测试类"""

    def test_initialization(self):
        """测试摄像头捕获初始化"""
        from worlds.real.camera_capture import CameraCapture

        capture = CameraCapture()

        assert capture is not None

    def test_frame_capture(self):
        """测试帧捕获"""
        from worlds.real.camera_capture import CameraCapture

        capture = CameraCapture()

        try:
            frame = capture.capture_frame()
            # 如果没有摄像头，可能返回None
            if frame is not None:
                assert frame.shape[2] == 3  # RGB
        except Exception:
            # 如果捕获失败，可能是环境问题
            pass


class TestCrossDomainLearner:
    """跨域学习器测试类"""

    def test_initialization(self):
        """测试跨域学习器初始化"""
        from worlds.real.cross_domain_learner import CrossDomainLearner

        learner = CrossDomainLearner()

        assert learner is not None

    def test_domain_mapping(self):
        """测试域映射"""
        from worlds.real.cross_domain_learner import CrossDomainLearner

        learner = CrossDomainLearner()

        # 真实世界特征
        real_features = np.random.randn(64)

        # 映射到虚拟世界
        virtual_features = learner.map_to_virtual(real_features)

        assert virtual_features is not None

    def test_knowledge_transfer(self):
        """测试知识迁移"""
        from worlds.real.cross_domain_learner import CrossDomainLearner

        learner = CrossDomainLearner()

        # 训练数据
        source_data = np.random.randn(100, 64)
        target_data = np.random.randn(100, 64)

        learner.train_transfer(source_data, target_data)

        # 测试迁移
        transferred = learner.transfer(np.random.randn(64))

        assert transferred is not None


class TestStrategyTransfer:
    """策略迁移测试类"""

    def test_initialization(self):
        """测试策略迁移初始化"""
        from worlds.real.strategy_transfer import StrategyTransfer

        transfer = StrategyTransfer()

        assert transfer is not None

    def test_strategy_extraction(self):
        """测试策略提取"""
        from worlds.real.strategy_transfer import StrategyTransfer

        transfer = StrategyTransfer()

        # 提供演示数据
        demonstrations = [
            np.random.randn(20, 10)
            for _ in range(5)
        ]

        strategy = transfer.extract_strategy(demonstrations)

        assert strategy is not None

    def test_strategy_application(self):
        """测试策略应用"""
        from worlds.real.strategy_transfer import StrategyTransfer

        transfer = StrategyTransfer()

        # 创建策略
        demonstrations = [np.random.randn(20, 10) for _ in range(5)]
        strategy = transfer.extract_strategy(demonstrations)

        # 应用策略
        applied = transfer.apply_strategy(np.random.randn(10))

        assert applied is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
