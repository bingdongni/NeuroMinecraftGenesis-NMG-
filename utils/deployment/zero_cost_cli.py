#!/usr/bin/env python3
"""
零成本AI部署系统命令行工具
提供便捷的命令行接口来使用各种功能
"""

import argparse
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
    optimize_for_low_specs,
    create_minimal_setup
)

def cmd_check(args):
    """系统检查命令"""
    print("🔍 正在检查系统配置...")
    
    try:
        recommendations = get_system_recommendations()
        
        print("=" * 50)
        print("📊 系统评估结果")
        print("=" * 50)
        print(f"推荐模式: {recommendations['推荐模式']}")
        print(f"总内存: {recommendations['系统信息']['总内存']}")
        print(f"可用内存: {recommendations['系统信息']['可用内存']}")
        print(f"CPU核心: {recommendations['系统信息']['CPU核心数']}")
        print(f"GPU支持: {recommendations['系统信息']['GPU可用']}")
        
        print("\n⚙️ 建议配置:")
        config = recommendations['当前配置']
        for key, value in config.items():
            print(f"  • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统检查失败: {e}")
        return False

def cmd_quick_setup(args):
    """快速设置命令"""
    print("🚀 开始快速设置...")
    
    try:
        success = quick_setup()
        if success:
            print("✅ 快速设置成功完成！")
            print("📁 请查看生成的文件:")
            print("  • setup_zero_cost_env.bat - 环境设置脚本")
            print("  • windows_optimization.bat - Windows优化脚本")
            print("  • README.md - 详细文档")
        else:
            print("❌ 快速设置失败")
        return success
        
    except Exception as e:
        print(f"❌ 快速设置失败: {e}")
        return False

def cmd_create_env(args):
    """创建环境命令"""
    output_dir = args.output_dir
    print(f"📦 正在创建零成本环境到: {output_dir}")
    
    try:
        result = create_zero_cost_env(output_dir)
        print(result)
        
        if Path(output_dir).exists():
            print(f"\n📁 环境创建完成！")
            print(f"位置: {Path(output_dir).absolute()}")
            print("\n下一步:")
            print(f"1. cd {output_dir}")
            print("2. 运行 scripts/setup_zero_cost_env.bat")
            print("3. 查看 README.md 了解详情")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return False

def cmd_optimize(args):
    """优化命令"""
    print("⚡ 开始系统优化...")
    
    try:
        # 获取配置
        if args.memory_size == "low":
            config = ZeroCostConfig(
                use_cpu_only=True,
                optimize_memory=True,
                batch_size=1,
                max_memory_usage=0.6
            )
        elif args.memory_size == "medium":
            config = ZeroCostConfig(
                use_cpu_only=True,
                optimize_memory=True,
                batch_size=2,
                max_memory_usage=0.7
            )
        else:  # high
            config = ZeroCostConfig(
                use_cpu_only=True,
                optimize_memory=False,
                batch_size=4,
                max_memory_usage=0.8
            )
        
        optimizer = ZeroCostOptimizer(config)
        
        # 执行优化
        results = optimizer.run_comprehensive_setup()
        
        if results['status'] == 'success':
            print("✅ 系统优化完成！")
            
            # 显示主要结果
            print(f"\n📊 优化摘要:")
            if 'pytorch_setup' in results:
                print(f"  • PyTorch设置: {'成功' if results['pytorch_setup']['success'] else '失败'}")
            if 'quantum_setup' in results:
                print(f"  • 量子环境: {'成功' if results['quantum_setup']['success'] else '失败'}")
            if 'optimizations' in results:
                print(f"  • 优化项目: {len(results['optimizations'])} 项")
            if 'deployment' in results:
                print(f"  • 部署文件: {len(results['deployment']['files'])} 个")
        else:
            print(f"❌ 优化失败: {results['message']}")
        
        return results['status'] == 'success'
        
    except Exception as e:
        print(f"❌ 系统优化失败: {e}")
        return False

def cmd_quantum_demo(args):
    """量子演示命令"""
    print("⚛️ 启动量子计算演示...")
    
    try:
        from utils.deployment.zero_cost_setup import QuantumSimulator
        
        # 创建量子模拟器
        simulator = QuantumSimulator(max_qubits=args.qubits)
        
        print(f"🔬 初始化 {args.qubits} 量子比特系统...")
        simulator.initialize_state(args.qubits)
        
        print("🌀 执行量子门操作...")
        
        # 应用Hadamard门
        for i in range(args.qubits):
            simulator.apply_hadamard(i)
        
        print("📊 执行量子测量...")
        
        # 测量结果
        measurement_counts = {}
        for shot in range(args.shots):
            result = ""
            for qubit in range(args.qubits):
                measurement = simulator.measure(qubit)
                result += str(measurement)
            
            measurement_counts[result] = measurement_counts.get(result, 0) + 1
        
        print("\n📈 测量统计:")
        for state, count in sorted(measurement_counts.items()):
            probability = count / args.shots * 100
            print(f"  |{state}⟩: {count} 次 ({probability:.1f}%)")
        
        print("⚛️ 量子演示完成！")
        return True
        
    except Exception as e:
        print(f"❌ 量子演示失败: {e}")
        return False

def cmd_models(args):
    """模型替代命令"""
    print("🤖 查找模型替代方案...")
    
    try:
        optimizer = ZeroCostOptimizer()
        
        models = args.models if args.models else [
            "GPT-3.5", "BERT-Large", "ResNet-50", 
            "Whisper-Large", "Stable-Diffusion"
        ]
        
        print("=" * 60)
        print("📋 模型替代建议")
        print("=" * 60)
        
        for model in models:
            alt = optimizer.model_substitution.suggest_alternative(model)
            recommendation = alt.get('推荐替代', alt.get('推荐', '无推荐'))
            
            print(f"\n🔄 {model}")
            print(f"   推荐替代: {recommendation}")
            
            if '备选方案' in alt and alt['备选方案']:
                print(f"   备选方案: {', '.join(alt['备选方案'])}")
            if '优势' in alt:
                print(f"   主要优势: {', '.join(alt['优势'])}")
            if '资源需求' in alt:
                print(f"   资源需求: {alt['资源需求']}")
        
        print("\n" + "=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 模型替代查询失败: {e}")
        return False

def cmd_demo(args):
    """演示命令"""
    print("🎭 运行完整演示...")
    
    try:
        demo_script = Path(__file__).parent / "demo_zero_cost_setup.py"
        if demo_script.exists():
            import subprocess
            result = subprocess.run([sys.executable, str(demo_script)], 
                                  capture_output=False, text=True)
            return result.returncode == 0
        else:
            print("❌ 演示脚本不存在")
            return False
            
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        return False

def cmd_test(args):
    """测试命令"""
    print("🧪 运行系统测试...")
    
    try:
        test_script = Path(__file__).parent / "test_zero_cost_setup.py"
        if test_script.exists():
            import subprocess
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=False, text=True)
            return result.returncode == 0
        else:
            print("❌ 测试脚本不存在")
            return False
            
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="零成本AI部署系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s check                    # 检查系统配置
  %(prog)s quick-setup             # 快速设置
  %(prog)s create-env my_env       # 创建环境
  %(prog)s optimize --memory low   # 优化系统(低内存)
  %(prog)s quantum-demo --qubits 3 # 量子演示(3量子比特)
  %(prog)s models GPT-3.5 BERT     # 查询模型替代
  %(prog)s demo                    # 运行完整演示
  %(prog)s test                    # 运行系统测试
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 系统检查命令
    subparsers.add_parser('check', help='检查系统配置')
    
    # 快速设置命令
    subparsers.add_parser('quick-setup', help='快速设置零成本环境')
    
    # 创建环境命令
    create_parser = subparsers.add_parser('create-env', help='创建零成本环境')
    create_parser.add_argument('output_dir', nargs='?', default='zero_cost_env',
                              help='输出目录 (默认: zero_cost_env)')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='系统优化')
    optimize_parser.add_argument('--memory', choices=['low', 'medium', 'high'], 
                                default='medium', help='内存配置级别')
    
    # 量子演示命令
    quantum_parser = subparsers.add_parser('quantum-demo', help='量子计算演示')
    quantum_parser.add_argument('--qubits', type=int, default=2, 
                               help='量子比特数量 (默认: 2)')
    quantum_parser.add_argument('--shots', type=int, default=10, 
                               help='测量次数 (默认: 10)')
    
    # 模型替代命令
    models_parser = subparsers.add_parser('models', help='查询模型替代方案')
    models_parser.add_argument('models', nargs='*', help='要查询的模型名称')
    
    # 演示命令
    subparsers.add_parser('demo', help='运行完整演示')
    
    # 测试命令
    subparsers.add_parser('test', help='运行系统测试')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    # 命令映射
    commands = {
        'check': cmd_check,
        'quick-setup': cmd_quick_setup,
        'create-env': cmd_create_env,
        'optimize': cmd_optimize,
        'quantum-demo': cmd_quantum_demo,
        'models': cmd_models,
        'demo': cmd_demo,
        'test': cmd_test
    }
    
    # 执行命令
    if args.command in commands:
        success = commands[args.command](args)
        return 0 if success else 1
    else:
        print(f"❌ 未知命令: {args.command}")
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())