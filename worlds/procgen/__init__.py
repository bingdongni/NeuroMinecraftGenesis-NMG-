"""
程序化世界生成包

该包提供完整的环境动态复杂度调节和程序化世界生成功能：

1. dynamic_complexity - 环境动态复杂度调节器
2. adaptive_difficulty - 自适应难度调节系统  
3. environment_evaluator - 环境评估器和监控系统
4. world_generator - 程序化世界生成器

主要功能：
- 自适应难度调节
- 程序化世界生成
- 实时环境监控
- 性能评估和优化
- 进化推动机制
"""

from .dynamic_complexity import (
    DynamicComplexityController,
    EnvironmentMetrics,
    AgentCapabilities,
    TerrainComplexity,
    ResourceScarcity,
    DangerLevel,
    create_complexity_controller
)

from .adaptive_difficulty import (
    AdaptiveDifficultyEngine,
    DifficultyParameters,
    PerformanceMetrics,
    DifficultyAdjustment,
    DifficultyStrategy,
    create_adaptive_difficulty_engine
)

from .environment_evaluator import (
    EnvironmentEvaluator,
    EnvironmentSnapshot,
    EvaluationResult,
    PerformanceMonitor,
    EvaluationType,
    MetricCategory,
    create_environment_evaluator
)

from .world_generator import (
    DynamicWorldGenerator,
    WorldConfig,
    TerrainCell,
    ResourceNode,
    EnvironmentEvent,
    TerrainType,
    ResourceType,
    EventType,
    create_world_generator
)

# 版本信息
__version__ = "1.0.0"
__author__ = "NeuroMinecraftGenesis Development Team"

# 导出的主要类和工厂函数
__all__ = [
    # 动态复杂度控制器
    'DynamicComplexityController',
    'EnvironmentMetrics', 
    'AgentCapabilities',
    'create_complexity_controller',
    
    # 自适应难度引擎
    'AdaptiveDifficultyEngine',
    'DifficultyParameters',
    'PerformanceMetrics',
    'create_adaptive_difficulty_engine',
    
    # 环境评估器
    'EnvironmentEvaluator',
    'EnvironmentSnapshot',
    'EvaluationResult',
    'PerformanceMonitor',
    'create_environment_evaluator',
    
    # 世界生成器
    'DynamicWorldGenerator',
    'WorldConfig',
    'TerrainCell',
    'ResourceNode',
    'EnvironmentEvent',
    'create_world_generator',
    
    # 枚举类型
    'TerrainComplexity',
    'ResourceScarcity',
    'DangerLevel',
    'DifficultyStrategy',
    'EvaluationType',
    'MetricCategory',
    'TerrainType',
    'ResourceType',
    'EventType'
]


def create_integrated_environment_system(config: dict):
    """
    创建集成环境系统的工厂函数
    
    该函数创建一个完整的环境系统，包含：
    - 动态复杂度控制器
    - 自适应难度引擎  
    - 环境评估器
    - 程序化世界生成器
    
    Args:
        config: 配置字典，包含所有子系统的配置
        
    Returns:
        dict: 包含所有子系统实例的字典
    """
    try:
        # 创建各个子系统
        complexity_controller = create_complexity_controller(config.get('complexity', {}))
        difficulty_engine = create_adaptive_difficulty_engine(config.get('difficulty', {}))
        environment_evaluator = create_environment_evaluator(config.get('evaluation', {}))
        world_generator = create_world_generator(config.get('world', {}))
        
        # 创建集成系统
        system = {
            'complexity_controller': complexity_controller,
            'difficulty_engine': difficulty_engine,
            'environment_evaluator': environment_evaluator,
            'world_generator': world_generator,
            'config': config,
            'created_at': __import__('time').time()
        }
        
        return system
        
    except Exception as e:
        raise RuntimeError(f"创建集成环境系统失败: {str(e)}")


def get_system_capabilities():
    """
    获取系统能力描述
    
    Returns:
        dict: 系统能力描述
    """
    return {
        'dynamic_complexity': {
            'description': '环境动态复杂度调节',
            'features': [
                '实时环境复杂度评估',
                '智能体能力综合评估', 
                '自适应难度调节算法',
                '挑战性维持机制',
                '性能监控和反馈'
            ]
        },
        'adaptive_difficulty': {
            'description': '自适应难度调节系统',
            'features': [
                '多种难度调节策略',
                '性能驱动的难度调整',
                '个性化适应算法',
                '多智能体难度平衡',
                '难度梯度控制'
            ]
        },
        'environment_evaluation': {
            'description': '环境评估和监控系统',
            'features': [
                '多维度环境指标计算',
                '实时性能监控',
                '环境变化趋势分析', 
                '进化效果评估',
                '智能建议系统'
            ]
        },
        'world_generation': {
            'description': '程序化世界生成',
            'features': [
                '基于复杂度的地形生成',
                '自适应资源分布',
                '动态生态系统生成',
                '气候和事件系统',
                '实时世界更新重构'
            ]
        },
        'integration': {
            'description': '系统集成能力',
            'features': [
                '模块化架构设计',
                '深度系统集成',
                '统一配置管理',
                '数据流协调',
                '性能优化'
            ]
        }
    }


def create_demo_config():
    """
    创建演示配置
    
    Returns:
        dict: 演示配置
    """
    return {
        'complexity': {
            'initial_complexity': 0.3,
            'adaptation_rate': 0.1,
            'complexity_range': [0.1, 1.0]
        },
        'difficulty': {
            'initial_difficulty': 0.3,
            'strategies': ['adaptive_balanced', 'performance_based'],
            'learning_window': 50,
            'min_adjustment_interval': 30.0
        },
        'evaluation': {
            'evaluation_interval': 30.0,
            'performance_window': 50,
            'enable_async': True
        },
        'world': {
            'world_size': [128, 128],
            'max_height': 64.0,
            'base_seed': 12345,
            'complexity_target': 0.3,
            'terrain_scale': 0.01,
            'terrain_octaves': 4,
            'cave_density': 0.3,
            'resource_density': 1.0,
            'resource_types': 5,
            'event_probability': 0.05
        }
    }


if __name__ == "__main__":
    # 演示集成系统
    print("=== NeuroMinecraftGenesis 程序化世界系统 ===")
    
    # 显示系统能力
    capabilities = get_system_capabilities()
    print("\n系统能力:")
    for system, info in capabilities.items():
        print(f"\n{system.upper()}:")
        print(f"  描述: {info['description']}")
        print("  功能:")
        for feature in info['features']:
            print(f"    - {feature}")
    
    # 创建演示系统
    print("\n=== 创建演示系统 ===")
    demo_config = create_demo_config()
    
    try:
        system = create_integrated_environment_system(demo_config)
        print("✓ 集成环境系统创建成功!")
        
        # 显示系统信息
        print("\n系统组件:")
        for name, component in system.items():
            if name not in ['config', 'created_at']:
                print(f"  - {name}: {type(component).__name__}")
        
        print(f"系统创建时间: {system['created_at']}")
        
    except Exception as e:
        print(f"✗ 系统创建失败: {str(e)}")
    
    print("\n=== 系统演示完成 ===")