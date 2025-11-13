#!/usr/bin/env python3
"""
零成本部署系统演示脚本
展示完整的使用流程和功能特性
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.deployment import (
    ZeroCostOptimizer, 
    ZeroCostConfig,
    get_system_recommendations,
    quick_setup,
    create_zero_cost_env,
    optimize_for_low_specs
)

def demo_system_analysis():
    """演示系统分析功能"""
    print("🔍 === 系统分析演示 ===")
    
    # 获取系统推荐
    recommendations = get_system_recommendations()
    
    print("📊 当前系统评估:")
    print(f"   推荐模式: {recommendations['推荐模式']}")
    print(f"   系统信息:")
    for key, value in recommendations['系统信息'].items():
        print(f"     • {key}: {value}")
    
    print("\n⚙️ 优化配置建议:")
    config = recommendations['当前配置']
    for key, value in config.items():
        print(f"   • {key}: {value}")
    
    return recommendations

def demo_quick_setup():
    """演示快速设置功能"""
    print("\n🚀 === 快速设置演示 ===")
    
    print("正在执行零成本环境快速设置...")
    success = quick_setup()
    
    if success:
        print("✅ 快速设置成功完成！")
        print("💡 接下来可以:")
        print("   1. 运行 setup_zero_cost_env.bat 安装依赖")
        print("   2. 运行 windows_optimization.bat 优化系统")
        print("   3. 查看生成的配置文件进行定制")
    else:
        print("❌ 快速设置遇到问题，请查看日志")
    
    return success

def demo_environment_creation():
    """演示环境创建功能"""
    print("\n📦 === 环境创建演示 ===")
    
    output_dir = "demo_zero_cost_env"
    result = create_zero_cost_env(output_dir)
    
    print(result)
    
    # 检查生成的文件
    output_path = Path(output_dir)
    if output_path.exists():
        print("\n📁 生成的文件结构:")
        for root, dirs, files in os.walk(output_path):
            level = root.replace(str(output_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    return output_dir

def demo_low_spec_optimization():
    """演示低配置优化功能"""
    print("\n🔧 === 低配置优化演示 ===")
    
    optimizations = optimize_for_low_specs()
    
    print("💡 专为您当前配置推荐的优化方案:")
    for key, value in optimizations.items():
        print(f"   • {key}: {value}")
    
    # 创建自定义优化配置
    custom_config = ZeroCostConfig(
        use_cpu_only=True,
        optimize_memory=True,
        batch_size=optimizations.get('批处理大小', 4),
        max_memory_usage=0.8
    )
    
    print(f"\n🎯 生成的优化配置:")
    print(f"   • CPU仅用模式: {custom_config.use_cpu_only}")
    print(f"   • 内存优化: {custom_config.optimize_memory}")
    print(f"   • 批次大小: {custom_config.batch_size}")
    print(f"   • 最大内存使用: {custom_config.max_memory_usage:.0%}")
    
    return custom_config

def demo_comprehensive_optimization():
    """演示全面优化功能"""
    print("\n🎯 === 全面优化演示 ===")
    
    # 创建自定义配置
    config = ZeroCostConfig(
        use_cpu_only=True,
        optimize_memory=True,
        use_free_clouds=True,
        use_lightweight_models=True,
        enable_windows_optimization=True,
        batch_size=4,
        max_memory_usage=0.75
    )
    
    optimizer = ZeroCostOptimizer(config)
    
    print("⚡ 开始全面系统优化...")
    
    # 1. 系统要求检测
    requirements = optimizer.detect_system_requirements()
    print(f"\n1️⃣ 系统评估完成 - {requirements['推荐模式']}")
    
    # 2. 性能优化
    optimizations = optimizer.optimize_system_performance()
    print(f"2️⃣ 性能优化完成 - 生成{len(optimizations)}项优化策略")
    
    # 3. 模型替代推荐
    test_models = ["GPT-3.5", "BERT-Large", "ResNet-50", "Whisper-Large"]
    print("3️⃣ 模型替代推荐:")
    for model in test_models:
        alt = optimizer.model_substitution.suggest_alternative(model)
        # 处理不同的返回格式
        recommendation = alt.get('推荐替代', alt.get('推荐', '无推荐'))
        print(f"   • {model} → {recommendation}")
    
    # 4. 免费资源
    print("4️⃣ 免费资源推荐:")
    resources = optimizer.free_resources
    print(f"   • 免费镜像源: {len(resources.free_mirrors)} 类")
    print(f"   • 轻量级模型: {len(resources.lightweight_models)} 类")
    print(f"   • 免费云平台: {len(resources.free_compute_platforms)} 个")
    
    # 5. 部署包创建
    print("5️⃣ 创建优化部署包...")
    deployment_files = optimizer.create_deployment_package("demo_optimized_deployment")
    print(f"   • 生成 {len(deployment_files)} 个文件")
    
    return optimizer

def demo_quantum_capabilities():
    """演示量子计算能力"""
    print("\n⚛️ === 量子计算演示 ===")
    
    from utils.deployment.zero_cost_setup import QuantumSimulator
    
    print("🔬 初始化量子模拟器...")
    simulator = QuantumSimulator(max_qubits=4)
    
    # 创建2量子比特系统
    simulator.initialize_state(2)
    print("✅ 2量子比特系统初始化完成")
    
    # 量子门操作演示
    print("\n🌀 量子门操作演示:")
    print("   应用Hadamard门到量子比特0...")
    simulator.apply_hadamard(0)
    
    print("   应用Hadamard门到量子比特1...")
    simulator.apply_hadamard(1)
    
    # 量子测量
    print("\n📊 量子测量结果:")
    for i in range(5):  # 进行5次测量
        result_q0 = simulator.measure(0)
        result_q1 = simulator.measure(1)
        print(f"   测量 {i+1}: |{result_q0}{result_q1}⟩")
    
    print("⚛️ 量子模拟演示完成！")

def demo_batch_processing():
    """演示批处理功能"""
    print("\n📦 === 批处理演示 ===")
    
    optimizer = ZeroCostOptimizer()
    
    # 创建简单的批处理脚本
    simple_task = '''echo ================================
echo    零成本AI系统批处理任务
echo ================================
echo.
echo 第1步: 系统检测
systeminfo | findstr /C:"Total Physical Memory"
echo.
echo 第2步: Python环境检查  
python --version
echo.
echo 第3步: 依赖安装状态
pip list | findstr torch
echo.
echo 第4步: 内存使用情况
wmic OS get TotalVisibleMemorySize^,FreePhysicalMemory /format:table
echo.
echo 批处理任务完成！'''
    
    batch_file = optimizer.batch_processor.create_batch_script(simple_task)
    print(f"✅ 批处理脚本创建: {batch_file}")
    
    # 创建多阶段流水线
    pipeline_stages = [
        "data_preprocessing",
        "model_training", 
        "model_evaluation",
        "model_deployment"
    ]
    
    pipeline_file = optimizer.batch_processor.create_multi_stage_pipeline(pipeline_stages)
    print(f"✅ 流水线脚本创建: {pipeline_file}")
    
    print("\n📋 流水线阶段:")
    for i, stage in enumerate(pipeline_stages, 1):
        print(f"   {i}. {stage}")
    
    return batch_file, pipeline_file

def demo_memory_management():
    """演示内存管理功能"""
    print("\n🧠 === 内存管理演示 ===")
    
    optimizer = ZeroCostOptimizer()
    memory_optimizer = optimizer.memory_optimizer
    
    # 获取当前内存信息
    memory_info = memory_optimizer.get_memory_info()
    print("💾 当前内存状态:")
    print(f"   • 总内存: {memory_info['total_gb']:.1f} GB")
    print(f"   • 可用内存: {memory_info['available_gb']:.1f} GB") 
    print(f"   • 已用内存: {memory_info['used_gb']:.1f} GB")
    print(f"   • 使用率: {memory_info['percent']:.1f}%")
    
    # 针对不同模型大小的优化建议
    model_sizes = [50, 100, 500, 1000]  # MB
    print(f"\n🎯 不同模型大小的优化建议:")
    
    for size_mb in model_sizes:
        suggestions = memory_optimizer.optimize_for_low_memory(size_mb)
        print(f"   模型大小: {size_mb}MB")
        for key, value in suggestions.items():
            print(f"     • {key}: {value}")
        print()
    
    # 应用优化设置
    print("🛠️ 应用优化设置...")
    if memory_info['percent'] > 80:
        print("⚠️ 检测到高内存使用率，启动紧急优化...")
        memory_optimizer.max_memory_ratio = 0.6  # 降低到60%
    else:
        print("✅ 内存使用正常")
    
    print("🧠 内存管理演示完成！")

def save_demo_results(results):
    """保存演示结果"""
    print("\n💾 保存演示结果...")
    
    demo_output = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "system_analysis": results.get('system_analysis', {}),
        "optimizations": results.get('optimizations', {}),
        "generated_files": results.get('generated_files', []),
        "recommendations": results.get('recommendations', {})
    }
    
    output_file = "zero_cost_demo_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 演示结果已保存到: {output_file}")
    return output_file

def main():
    """主演示函数"""
    print("🎭 零成本AI部署系统完整演示")
    print("=" * 60)
    print("本演示将展示系统的所有核心功能和使用方法")
    print("=" * 60)
    
    results = {}
    
    try:
        # 1. 系统分析
        results['system_analysis'] = demo_system_analysis()
        
        # 2. 低配置优化
        results['low_spec_config'] = demo_low_spec_optimization()
        
        # 3. 全面优化
        results['optimizer'] = demo_comprehensive_optimization()
        
        # 4. 量子计算演示
        demo_quantum_capabilities()
        
        # 5. 批处理演示
        batch_file, pipeline_file = demo_batch_processing()
        results['generated_files'] = [batch_file, pipeline_file]
        
        # 6. 内存管理演示
        demo_memory_management()
        
        # 7. 保存结果
        results['recommendations'] = results['system_analysis']
        output_file = save_demo_results(results)
        
        # 清理临时文件
        import tempfile
        temp_files = ['windows_optimization.bat', 'setup_zero_cost_env.bat']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("\n🎉 演示完成！")
        print("=" * 60)
        print("💡 使用建议:")
        print("   1. 根据系统分析结果选择合适的配置")
        print("   2. 使用生成的批处理脚本自动化任务")
        print("   3. 参考量子计算示例进行高级开发")
        print("   4. 查看详细文档了解所有功能")
        print("   5. 根据内存管理建议优化性能")
        print("=" * 60)
        print("📁 结果文件:")
        print(f"   • 演示结果: {output_file}")
        print(f"   • 批处理脚本: {batch_file}")
        print(f"   • 流水线脚本: {pipeline_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    success = main()
    sys.exit(0 if success else 1)