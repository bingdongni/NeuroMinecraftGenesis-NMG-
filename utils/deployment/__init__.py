"""
零成本部署系统包 - Zero Cost Deployment Package

提供完整的零成本AI部署解决方案，包括：
- CPU优化的PyTorch环境
- 轻量级量子模拟器
- 免费云资源管理
- Windows系统优化
- 性能调优和内存管理

主要类:
- ZeroCostOptimizer: 主要优化器类
- FreeResourceManager: 免费资源管理器
- MemoryOptimizer: 内存优化器
- WindowsOptimizer: Windows系统优化器
- ModelSubstitution: 模型替代方案管理器
"""

from .zero_cost_setup import (
    ZeroCostOptimizer,
    ZeroCostConfig,
    SystemInfo,
    FreeResourceManager,
    MemoryOptimizer,
    WindowsOptimizer,
    ModelSubstitution,
    QuantumSimulator,
    BatchProcessor,
    get_system_recommendations,
    quick_setup,
    create_minimal_setup
)

__version__ = "1.0.0"
__author__ = "ZeroCost AI Team"

# 导出的公共接口
__all__ = [
    'ZeroCostOptimizer',
    'ZeroCostConfig', 
    'SystemInfo',
    'FreeResourceManager',
    'MemoryOptimizer',
    'WindowsOptimizer',
    'ModelSubstitution',
    'QuantumSimulator',
    'BatchProcessor',
    'get_system_recommendations',
    'quick_setup',
    'create_minimal_setup'
]

# 快速使用函数
def create_zero_cost_env(output_dir: str = "zero_cost_env") -> str:
    """
    创建零成本AI环境的快速函数
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        创建结果消息
    """
    try:
        optimizer = ZeroCostOptimizer()
        files = optimizer.create_deployment_package(output_dir)
        return f"零成本环境创建成功！输出目录: {output_dir}"
    except Exception as e:
        return f"创建失败: {str(e)}"

def optimize_for_low_specs() -> dict:
    """
    针对低配置设备的优化建议
    
    Returns:
        优化建议字典
    """
    import psutil
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    if memory_gb < 4:
        return {
            "推荐模式": "极低资源模式",
            "批处理大小": 1,
            "混合精度": True,
            "模型选择": "tiny/small系列",
            "内存映射": True,
            "梯度检查点": True
        }
    elif memory_gb < 8:
        return {
            "推荐模式": "低资源模式", 
            "批处理大小": 2,
            "混合精度": True,
            "模型选择": "small/medium系列",
            "内存映射": True,
            "梯度检查点": False
        }
    else:
        return {
            "推荐模式": "标准模式",
            "批处理大小": 4,
            "混合精度": False,
            "模型选择": "medium/large系列",
            "内存映射": False,
            "梯度检查点": False
        }