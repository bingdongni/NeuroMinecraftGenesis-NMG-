"""
测试配置模块
"""

__version__ = "1.0.0"

# 测试配置
TEST_CONFIG = {
    'verbose': True,
    'capture_output': True,
    'log_cli': True,
    'log_cli_level': 'INFO',
}

# 性能测试配置
PERFORMANCE_CONFIG = {
    'warmup_iterations': 10,
    'test_iterations': 100,
    'batch_sizes': [1, 8, 16, 32],
    'timeout_seconds': 300,
}

# 集成测试配置
INTEGRATION_CONFIG = {
    'min_success_rate': 0.8,
    'max_retry_count': 3,
    'cleanup_after_test': True,
}
