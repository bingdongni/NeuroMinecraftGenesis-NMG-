"""
神经符号混合架构示例

该脚本展示了如何使用神经符号混合架构进行：
1. 架构初始化
2. 符号知识提取
3. 神经网络初始化
4. 混合推理
5. 性能分析

作者: NeuroMinecraftGenesis Team
日期: 2025-11-13
"""

import sys
import os
import json
import torch
import numpy as np
import logging
import time
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.symbolic import (
    NeuroSymbolicArchitecture,
    NeuralSymbolicBridge,
    SymbolExtraction, 
    NeuralInitialization,
    HybridReasoning
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_config() -> tuple:
    """创建示例配置"""
    
    # 神经网络配置
    network_config = {
        "input_dim": 128,
        "hidden_dims": [256, 128, 64],
        "output_dim": 32,
        "activation": "relu",
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32
    }
    
    # 符号推理配置
    symbolic_config = {
        "activation_threshold": 0.5,
        "concept_clustering": "dbscan",
        "relation_threshold": 0.7,
        "knowledge_confidence_threshold": 0.6,
        "inference_depth": 3,
        "confidence_threshold": 0.5,
        "reasoning_timeout": 10.0,
        "parallel_reasoning": True,
        "adaptive_mode_selection": True,
        "initialization_method": "knowledge_guided",
        "prior_strength": 0.8,
        "constraint_weight": 0.1,
        "regularization_factor": 0.001
    }
    
    return network_config, symbolic_config

def create_sample_knowledge_base() -> Dict[str, Any]:
    """创建示例知识库"""
    
    return {
        "concepts": {
            "object_recognition": {
                "attributes": {
                    "visual_features": ["shape", "color", "texture"],
                    "semantic_meaning": ["category", "function"],
                    "activation_level": "adaptive"
                },
                "neural_representation": {
                    "neuron_indices": [10, 11, 12, 13, 14],
                    "weights": [0.8, 0.9, 0.7, 0.6, 0.8]
                },
                "constraints": ["must_have_visual_features"],
                "properties": {
                    "complexity": "medium",
                    "domain": "computer_vision",
                    "learnable": True
                },
                "confidence": 0.9
            },
            "spatial_reasoning": {
                "attributes": {
                    "spatial_relations": ["distance", "direction", "position"],
                    "coordinate_system": "relative",
                    "dimensionality": "3D"
                },
                "neural_representation": {
                    "neuron_indices": [20, 21, 22, 23],
                    "weights": [0.7, 0.8, 0.9, 0.6]
                },
                "constraints": ["requires_spatial_context"],
                "properties": {
                    "complexity": "high",
                    "domain": "spatial_cognition",
                    "learnable": True
                },
                "confidence": 0.85
            },
            "temporal_understanding": {
                "attributes": {
                    "time_sequences": ["before", "after", "during"],
                    "event_ordering": "chronological",
                    "duration_modeling": "relative"
                },
                "neural_representation": {
                    "neuron_indices": [30, 31, 32, 33, 34],
                    "weights": [0.6, 0.7, 0.8, 0.9, 0.7]
                },
                "constraints": ["requires_temporal_context"],
                "properties": {
                    "complexity": "high",
                    "domain": "temporal_cognition",
                    "learnable": True
                },
                "confidence": 0.8
            }
        },
        "relations": {
            "spatial_object_relation": {
                "source_entities": ["object_recognition"],
                "target_entities": ["spatial_reasoning"],
                "relation_type": "depends_on",
                "strength": 0.9,
                "neural_mapping": {
                    "weight_pattern": {
                        "source_layer": 0,
                        "target_layer": 1
                    }
                },
                "constraints": ["spatial_context_required"]
            },
            "temporal_spatial_relation": {
                "source_entities": ["spatial_reasoning"],
                "target_entities": ["temporal_understanding"],
                "relation_type": "enables",
                "strength": 0.7,
                "neural_mapping": {
                    "weight_pattern": {
                        "source_layer": 1,
                        "target_layer": 2
                    }
                },
                "constraints": ["temporal_spatial_integration"]
            }
        },
        "rules": [
            {
                "id": "recognition_rule_1",
                "premise": "IF visual_features_detected AND spatial_context_available",
                "conclusion": "THEN object_recognition_with_spatial_info",
                "confidence": 0.85,
                "constraints": ["requires_spatial_context"]
            },
            {
                "id": "temporal_inference_rule",
                "premise": "IF spatial_reasoning_active AND temporal_sequence_detected",
                "conclusion": "THEN temporal_spatial_integration",
                "confidence": 0.75,
                "constraints": ["temporal_spatial_integration"]
            },
            {
                "id": "complex_reasoning_rule",
                "premise": "IF object_recognition_confidence > 0.8 AND spatial_reasoning_confidence > 0.7",
                "conclusion": "THEN advanced_scene_understanding",
                "confidence": 0.9,
                "constraints": ["high_confidence_threshold"]
            }
        ],
        "constraints": [
            {
                "id": "knowledge_consistency",
                "type": "consistency",
                "strength": 0.1,
                "target_parameters": ["weights", "biases"],
                "function": "l2"
            },
            {
                "id": "temporal_coherence",
                "type": "coherence",
                "strength": 0.05,
                "target_parameters": ["temporal_weights"],
                "function": "smoothing"
            }
        ]
    }

def create_sample_neural_data() -> torch.Tensor:
    """创建示例神经网络数据"""
    
    # 创建模拟的神经网络激活数据
    batch_size = 16
    input_dim = 128
    
    # 生成模拟激活数据
    activations = torch.randn(batch_size, input_dim) * 0.5 + 0.5
    
    # 模拟一些高激活区域
    activations[:, 10:15] = torch.randn(batch_size, 5) * 0.3 + 0.8  # 概念1
    activations[:, 20:25] = torch.randn(batch_size, 5) * 0.3 + 0.7  # 概念2
    activations[:, 30:35] = torch.randn(batch_size, 5) * 0.3 + 0.9  # 概念3
    
    return activations

def demonstrate_neuro_symbolic_architecture():
    """演示神经符号架构功能"""
    
    logger.info("=== 神经符号混合架构演示开始 ===")
    
    # 1. 创建配置
    logger.info("1. 创建配置...")
    network_config, symbolic_config = create_sample_config()
    logger.info(f"网络配置: 输入维度 {network_config['input_dim']}, "
                f"隐藏层 {network_config['hidden_dims']}, "
                f"输出维度 {network_config['output_dim']}")
    
    # 2. 创建知识库
    logger.info("2. 创建示例知识库...")
    knowledge_base = create_sample_knowledge_base()
    logger.info(f"知识库包含: {len(knowledge_base['concepts'])} 个概念, "
                f"{len(knowledge_base['relations'])} 个关系, "
                f"{len(knowledge_base['rules'])} 个规则")
    
    # 3. 初始化架构
    logger.info("3. 初始化神经符号架构...")
    architecture = NeuroSymbolicArchitecture(
        network_config, symbolic_config, inference_mode="hybrid"
    )
    
    # 4. 初始化架构组件
    logger.info("4. 初始化架构组件...")
    initialization_success = architecture.initialize_architecture(knowledge_base)
    
    if initialization_success:
        logger.info("架构初始化成功！")
    else:
        logger.error("架构初始化失败！")
        return
    
    # 5. 生成示例神经数据
    logger.info("5. 生成示例神经数据...")
    neural_data = create_sample_neural_data()
    logger.info(f"生成神经激活数据形状: {neural_data.shape}")
    
    # 6. 演示符号知识提取
    logger.info("6. 演示符号知识提取...")
    extraction_result = architecture.extract_symbolic_knowledge(
        neural_data, 
        context={"task": "object_recognition", "domain": "computer_vision"}
    )
    
    if "error" not in extraction_result:
        logger.info("符号知识提取成功！")
        logger.info(f"提取概念数量: {len(extraction_result.get('concepts', {}))}")
        logger.info(f"提取关系数量: {len(extraction_result.get('relations', {}))}")
        logger.info(f"生成规则数量: {len(extraction_result.get('rules', {}))}")
        logger.info(f"质量评估分数: {extraction_result.get('quality_assessment', {}).get('overall_quality', 'N/A')}")
    else:
        logger.error(f"符号知识提取失败: {extraction_result['error']}")
    
    # 7. 演示神经网络初始化
    logger.info("7. 演示神经网络初始化...")
    init_result = architecture.initialize_network(
        knowledge_base,
        optimization_config={"max_iterations": 50, "learning_rate": 0.01}
    )
    
    if "error" not in init_result:
        logger.info("神经网络初始化成功！")
        logger.info(f"初始化验证分数: {init_result.get('initialization_metadata', {}).get('validation_result', {}).get('overall_score', 'N/A')}")
        logger.info(f"初始化时间: {init_result.get('initialization_metadata', {}).get('initialization_time', 'N/A')}秒")
    else:
        logger.error(f"神经网络初始化失败: {init_result['error']}")
    
    # 8. 演示混合推理
    logger.info("8. 演示混合推理...")
    
    # 测试不同推理模式
    inference_modes = ["neural", "symbolic", "hybrid"]
    inference_results = {}
    
    for mode in inference_modes:
        logger.info(f"   测试推理模式: {mode}")
        start_time = time.time()
        
        result = architecture.hybrid_reasoning(
            neural_data[0],  # 使用单个样本
            reasoning_config={"mode": mode, "timeout": 5.0}
        )
        
        inference_time = time.time() - start_time
        inference_results[mode] = result
        
        if "error" not in result:
            logger.info(f"   {mode} 推理成功，耗时: {inference_time:.3f}秒")
            logger.info(f"   置信度: {result.get('confidence', 'N/A'):.3f}")
            logger.info(f"   推理方法: {result.get('reasoning_metadata', {}).get('strategy', 'N/A')}")
        else:
            logger.error(f"   {mode} 推理失败: {result['error']}")
    
    # 9. 生成性能报告
    logger.info("9. 生成性能报告...")
    performance_report = architecture.get_performance_report()
    logger.info("性能报告:")
    logger.info(f"   总推理次数: {performance_report['inference_statistics']['total_count']}")
    logger.info(f"   平均推理时间: {performance_report['inference_statistics']['average_time']:.3f}秒")
    logger.info(f"   符号一致性分数: {performance_report['consistency_scores']['symbolic_consistency']:.3f}")
    logger.info(f"   成功率: {performance_report['architecture_status']['initialized']}")
    
    # 10. 保存架构状态
    logger.info("10. 保存架构状态...")
    save_path = "/tmp/neuro_symbolic_architecture_state.json"
    save_success = architecture.save_architecture(save_path)
    
    if save_success:
        logger.info(f"架构状态已保存到: {save_path}")
    else:
        logger.warning("架构状态保存失败")
    
    # 11. 获取详细状态
    logger.info("11. 获取架构详细状态...")
    architecture_state = architecture.get_architecture_state()
    
    # 12. 演示桥接器功能
    logger.info("12. 演示神经符号桥接器功能...")
    bridge_stats = architecture.neural_bridge.get_bridge_statistics()
    logger.info(f"桥接器统计:")
    logger.info(f"   转换次数: 神经到符号 {bridge_stats['conversion_stats']['neural_to_symbolic_count']}, "
                f"符号到神经 {bridge_stats['conversion_stats']['symbolic_to_neural_count']}")
    logger.info(f"   平均转换时间: {bridge_stats['conversion_stats']['average_conversion_time']:.3f}秒")
    logger.info(f"   映射规则数量: {bridge_stats['mapping_rules_count']}")
    
    # 13. 演示提取器功能
    logger.info("13. 演示符号提取器功能...")
    extraction_stats = architecture.symbol_extractor.get_extraction_statistics()
    logger.info(f"提取器统计:")
    logger.info(f"   总提取次数: {extraction_stats['extraction_stats']['total_extractions']}")
    logger.info(f"   提取概念数: {extraction_stats['extraction_stats']['concepts_extracted']}")
    logger.info(f"   提取关系数: {extraction_stats['extraction_stats']['relations_extracted']}")
    logger.info(f"   生成规则数: {extraction_stats['extraction_stats']['rules_generated']}")
    
    # 14. 演示初始化器功能
    logger.info("14. 演示神经网络初始化器功能...")
    init_stats = architecture.neural_initializer.get_initialization_statistics()
    logger.info(f"初始化器统计:")
    logger.info(f"   总初始化次数: {init_stats['initialization_stats']['total_initializations']}")
    logger.info(f"   平均初始化质量: {init_stats['initialization_stats']['average_initialization_quality']:.3f}")
    logger.info(f"   约束满足率: {init_stats['initialization_stats']['constraint_satisfaction_rate']:.3f}")
    
    # 15. 演示推理引擎功能
    logger.info("15. 演示混合推理引擎功能...")
    reasoning_validation = architecture.hybrid_reasoner.validate_reasoning()
    logger.info(f"推理引擎验证:")
    logger.info(f"   引擎初始化: {reasoning_validation['engine_initialized']}")
    logger.info(f"   组件有效: {reasoning_validation['components_valid']}")
    logger.info(f"   缓存大小: {reasoning_validation['cache_status']['cache_size']}")
    
    # 总结
    logger.info("=== 神经符号混合架构演示完成 ===")
    
    # 返回演示结果
    demo_results = {
        "initialization_success": initialization_success,
        "extraction_results": extraction_result,
        "initialization_results": init_result,
        "inference_results": inference_results,
        "performance_report": performance_report,
        "architecture_state": architecture_state,
        "component_statistics": {
            "bridge_stats": bridge_stats,
            "extraction_stats": extraction_stats,
            "initialization_stats": init_stats,
            "reasoning_validation": reasoning_validation
        }
    }
    
    return demo_results

def run_performance_benchmark():
    """运行性能基准测试"""
    
    logger.info("=== 性能基准测试开始 ===")
    
    # 创建小规模配置进行测试
    network_config = {
        "input_dim": 64,
        "hidden_dims": [128, 64],
        "output_dim": 32,
        "activation": "relu"
    }
    
    symbolic_config = {
        "activation_threshold": 0.5,
        "inference_depth": 2,
        "confidence_threshold": 0.5,
        "initialization_method": "knowledge_guided"
    }
    
    # 创建架构
    architecture = NeuroSymbolicArchitecture(network_config, symbolic_config)
    
    # 简单知识库
    simple_knowledge = {
        "concepts": {
            "simple_concept": {
                "attributes": {"feature": "test"},
                "neural_representation": {"neuron_indices": [0, 1, 2], "weights": [0.8, 0.7, 0.9]},
                "confidence": 0.8
            }
        },
        "relations": {},
        "rules": []
    }
    
    # 初始化
    init_success = architecture.initialize_architecture(simple_knowledge)
    
    if not init_success:
        logger.error("架构初始化失败，基准测试终止")
        return
    
    # 生成测试数据
    test_data = torch.randn(10, 64)
    
    # 性能测试
    benchmark_results = {
        "initialization_time": 0.0,
        "inference_times": [],
        "extraction_times": [],
        "memory_usage": []
    }
    
    # 测试初始化时间
    start_time = time.time()
    # 重新初始化以测量时间
    architecture.initialize_architecture(simple_knowledge)
    benchmark_results["initialization_time"] = time.time() - start_time
    
    # 测试推理性能
    for i in range(10):
        start_time = time.time()
        result = architecture.hybrid_reasoning(test_data[i])
        benchmark_results["inference_times"].append(time.time() - start_time)
    
    # 测试提取性能
    for i in range(5):
        start_time = time.time()
        result = architecture.extract_symbolic_knowledge(test_data[i])
        benchmark_results["extraction_times"].append(time.time() - start_time)
    
    # 计算统计信息
    inference_times = benchmark_results["inference_times"]
    extraction_times = benchmark_results["extraction_times"]
    
    logger.info("基准测试结果:")
    logger.info(f"   初始化时间: {benchmark_results['initialization_time']:.3f}秒")
    logger.info(f"   推理时间 - 平均: {np.mean(inference_times):.3f}秒, "
                f"最大: {np.max(inference_times):.3f}秒, "
                f"最小: {np.min(inference_times):.3f}秒")
    logger.info(f"   提取时间 - 平均: {np.mean(extraction_times):.3f}秒, "
                f"最大: {np.max(extraction_times):.3f}秒, "
                f"最小: {np.min(extraction_times):.3f}秒")
    
    logger.info("=== 性能基准测试完成 ===")
    
    return benchmark_results

def main():
    """主函数"""
    
    print("神经符号混合架构演示")
    print("=" * 50)
    
    try:
        # 运行主要演示
        demo_results = demonstrate_neuro_symbolic_architecture()
        
        print("\n" + "=" * 50)
        
        # 运行性能基准测试
        benchmark_results = run_performance_benchmark()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
        # 保存演示结果
        results_file = "/tmp/neuro_symbolic_demo_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 处理无法序列化的对象
            serializable_results = {
                "demo_completed": True,
                "benchmark_results": benchmark_results,
                "timestamp": time.time()
            }
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"演示结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"演示执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()