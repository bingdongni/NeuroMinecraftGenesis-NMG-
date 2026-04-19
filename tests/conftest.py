"""
NeuroMinecraft Genesis 测试框架配置
"""

import sys
import os
import pytest
import numpy as np
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def project_root_path():
    """返回项目根目录路径"""
    return project_root


@pytest.fixture
def sample_state_vector():
    """返回示例状态向量"""
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def sample_batch_states():
    """返回批次状态数据"""
    return np.random.randn(16, 128).astype(np.float32)


@pytest.fixture
def sample_action():
    """返回示例动作"""
    return np.array([1.0, 0.0, 0.5])


@pytest.fixture
def torch_device():
    """返回PyTorch设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_config():
    """返回示例配置"""
    return {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'learning_rate': 0.001
    }


@pytest.fixture
def mock_environment():
    """返回模拟环境"""
    class MockEnvironment:
        def __init__(self):
            self.state = np.zeros(10)
            self.reward = 0.0
            self.done = False
            self.metadata = {}

        def reset(self):
            self.state = np.random.randn(10)
            self.reward = 0.0
            self.done = False
            return self.state

        def step(self, action):
            self.state = np.random.randn(10)
            self.reward = np.random.randn()
            self.done = np.random.rand() > 0.9
            return self.state, self.reward, self.done, self.metadata

    return MockEnvironment()


@pytest.fixture
def temp_model_dir(tmp_path):
    """返回临时模型目录"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def seed_random():
    """设置随机种子"""
    np.random.seed(42)
    torch.manual_seed(42)
    yield 42
    np.random.seed()
    torch.manual_seed()
