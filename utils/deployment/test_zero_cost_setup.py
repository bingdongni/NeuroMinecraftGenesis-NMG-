#!/usr/bin/env python3
"""
零成本部署系统测试脚本
用于验证所有功能模块是否正常工作
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.deployment.zero_cost_setup import (
    ZeroCostOptimizer,
    ZeroCostConfig,
    get_system_recommendations,
    quick_setup,
    create_minimal_setup
)

def test_system_info():
    """测试系统信息收集"""
    print("🔍 测试系统信息收集...")
    
    try:
        optimizer = ZeroCostOptimizer()
        system_info = optimizer._collect_system_info()
        
        print(f"✅ 系统平台: {system_info.platform}")
        print(f"✅ CPU核心数: {system_info.cpu_count}")
        print(f"✅ 内存大小: {system_info.memory_gb:.1f} GB")
        print(f"✅ Python版本: {system_info.python_version}")
        print(f"✅ GPU可用: {'是' if system_info.gpu_available else '否'}")
        
        return True
    except Exception as e:
        print(f"❌ 系统信息收集失败: {e}")
        return False

def test_requirements_detection():
    """测试系统要求检测"""
    print("\n🎯 测试系统要求检测...")
    
    try:
        optimizer = ZeroCostOptimizer()
        requirements = optimizer.detect_system_requirements()
        
        print(f"📊 推荐模式: {requirements['推荐模式']}")
        print(f"⚙️  当前配置: {json.dumps(requirements['当前配置'], indent=2, ensure_ascii=False)}")
        print(f"💻 系统信息: {json.dumps(requirements['系统信息'], indent=2, ensure_ascii=False)}")
        
        return True
    except Exception as e:
        print(f"❌ 系统要求检测失败: {e}")
        return False

def test_memory_optimizer():
    """测试内存优化器"""
    print("\n🧠 测试内存优化器...")
    
    try:
        optimizer = ZeroCostOptimizer()
        memory_info = optimizer.memory_optimizer.get_memory_info()
        
        print(f"💾 总内存: {memory_info['total_gb']:.1f} GB")
        print(f"💾 可用内存: {memory_info['available_gb']:.1f} GB")
        print(f"💾 已用内存: {memory_info['used_gb']:.1f} GB")
        print(f"💾 使用率: {memory_info['percent']:.1f}%")
        
        # 测试低内存优化建议
        suggestions = optimizer.memory_optimizer.optimize_for_low_memory(100)  # 假设模型100MB
        print(f"💡 优化建议: {json.dumps(suggestions, indent=2, ensure_ascii=False)}")
        
        return True
    except Exception as e:
        print(f"❌ 内存优化器测试失败: {e}")
        return False

def test_quantum_simulator():
    """测试量子模拟器"""
    print("\n⚛️ 测试量子模拟器...")
    
    try:
        from utils.deployment.zero_cost_setup import QuantumSimulator
        
        simulator = QuantumSimulator(max_qubits=3)
        simulator.initialize_state(2)
        
        print("✅ 量子态初始化成功")
        
        # 测试Hadamard门
        simulator.apply_hadamard(0)
        print("✅ Hadamard门应用成功")
        
        # 测试测量
        result = simulator.measure(0)
        print(f"✅ 测量结果: {result}")
        
        return True
    except Exception as e:
        print(f"❌ 量子模拟器测试失败: {e}")
        return False

def test_model_substitution():
    """测试模型替代方案"""
    print("\n🤖 测试模型替代方案...")
    
    try:
        optimizer = ZeroCostOptimizer()
        
        test_models = ['gpt3.5', 'bert-large', 'resnet50', 'whisper-large']
        
        for model in test_models:
            alternative = optimizer.model_substitution.suggest_alternative(model)
            print(f"📋 {model} -> {alternative['推荐替代']}")
        
        return True
    except Exception as e:
        print(f"❌ 模型替代方案测试失败: {e}")
        return False

def test_free_resources():
    """测试免费资源管理器"""
    print("\n🆓 测试免费资源管理器...")
    
    try:
        optimizer = ZeroCostOptimizer()
        resources = optimizer.free_resources
        
        print("🌐 免费镜像源:")
        for category, mirrors in resources.free_mirrors.items():
            print(f"  {category}: {len(mirrors)} 个镜像")
        
        print("🧠 轻量级模型:")
        for category, models in resources.lightweight_models.items():
            print(f"  {category}: {models[0]}")
        
        print("☁️ 免费云平台:")
        for platform in resources.free_compute_platforms:
            print(f"  {platform['name']}: {platform['specs']}")
        
        return True
    except Exception as e:
        print(f"❌ 免费资源管理器测试失败: {e}")
        return False

def test_batch_processor():
    """测试批处理器"""
    print("\n📦 测试批处理器...")
    
    try:
        optimizer = ZeroCostOptimizer()
        
        # 创建示例脚本
        script_content = '''echo 第一步：数据预处理
echo 第二步：模型训练  
echo 第三步：结果保存'''
        
        batch_script = optimizer.batch_processor.create_batch_script(script_content)
        print(f"✅ 批处理脚本创建: {batch_script}")
        
        # 创建多阶段流水线
        stages = ['preprocess', 'train', 'evaluate', 'deploy']
        pipeline_script = optimizer.batch_processor.create_multi_stage_pipeline(stages)
        print(f"✅ 流水线脚本创建: {pipeline_script}")
        
        return True
    except Exception as e:
        print(f"❌ 批处理器测试失败: {e}")
        return False

def test_deployment_package():
    """测试部署包创建"""
    print("\n📦 测试部署包创建...")
    
    try:
        optimizer = ZeroCostOptimizer()
        test_output = "test_deployment_package"
        
        # 清理测试目录
        if os.path.exists(test_output):
            import shutil
            shutil.rmtree(test_output)
        
        files = optimizer.create_deployment_package(test_output)
        print(f"✅ 部署包创建成功")
        print(f"📁 生成文件数量: {len(files)}")
        
        # 列出主要文件
        for name, path in files.items():
            print(f"  📄 {name}: {Path(path).name}")
        
        # 清理测试目录
        if os.path.exists(test_output):
            import shutil
            shutil.rmtree(test_output)
        
        return True
    except Exception as e:
        print(f"❌ 部署包创建测试失败: {e}")
        return False

def test_windows_optimization():
    """测试Windows优化功能"""
    print("\n🪟 测试Windows优化功能...")
    
    try:
        optimizer = ZeroCostOptimizer()
        
        # 创建Windows优化脚本
        opt_script = optimizer.windows_optimizer.create_optimization_script()
        print(f"✅ Windows优化脚本: {opt_script}")
        
        # 创建环境设置脚本
        env_script = optimizer.windows_optimizer.create_environment_setup_script()
        print(f"✅ 环境设置脚本: {env_script}")
        
        # 清理测试文件
        for script in [opt_script, env_script]:
            if os.path.exists(script):
                os.remove(script)
        
        return True
    except Exception as e:
        print(f"❌ Windows优化功能测试失败: {e}")
        return False

def run_performance_test():
    """运行性能测试"""
    print("\n⚡ 运行性能测试...")
    
    try:
        optimizer = ZeroCostOptimizer()
        
        # 测试系统检测速度
        import time
        start_time = time.time()
        
        requirements = optimizer.detect_system_requirements()
        detection_time = time.time() - start_time
        
        print(f"⏱️ 系统检测耗时: {detection_time:.3f} 秒")
        
        # 测试内存优化
        memory_optimizations = optimizer.optimize_system_performance()
        print(f"📊 生成优化项目: {len(memory_optimizations)} 个")
        
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始零成本部署系统测试...")
    print("=" * 60)
    
    tests = [
        ("系统信息收集", test_system_info),
        ("系统要求检测", test_requirements_detection), 
        ("内存优化器", test_memory_optimizer),
        ("量子模拟器", test_quantum_simulator),
        ("模型替代方案", test_model_substitution),
        ("免费资源管理", test_free_resources),
        ("批处理器", test_batch_processor),
        ("部署包创建", test_deployment_package),
        ("Windows优化", test_windows_optimization),
        ("性能测试", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！零成本部署系统工作正常。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)