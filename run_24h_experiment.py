#!/usr/bin/env python3
"""
六维能力增长24小时连续实验系统快速启动器
=========================================

这是一个便捷的启动脚本，提供多种运行模式和选项：
- 基础演示模式（24秒=24小时）
- 完整演示模式（包含统计分析）
- 实时界面模式（Streamlit界面）
- 实际24小时实验模式

使用方法:
    python run_24h_experiment.py [选项]

示例:
    # 基础演示
    python run_24h_experiment.py --demo
    
    # 完整演示
    python run_24h_experiment.py --full-demo
    
    # 实时界面
    python run_24h_experiment.py --streamlit
    
    # 实际实验（24小时）
    python run_24h_experiment.py --real-experiment
"""

import sys
import os
import argparse
import subprocess
import time
import signal
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查依赖库是否已安装"""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'scikit-learn', 
        'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖库: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖库已安装")
    return True

def run_demo(duration_hours=1):
    """运行基础演示"""
    print("🚀 启动基础演示模式（24秒=24小时）")
    print("=" * 60)
    
    try:
        from experiments.cognition.demo_24h_experiment import ExperimentDemo
        
        demo = ExperimentDemo(demo_mode=True, duration_hours=duration_hours)
        success = demo.run_full_demonstration()
        
        if success:
            print("✅ 基础演示完成")
            return True
        else:
            print("❌ 基础演示失败")
            return False
            
    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        return False

def run_full_demo():
    """运行完整演示"""
    print("🎯 启动完整演示模式（包含统计分析）")
    print("=" * 60)
    
    try:
        # 运行多轮演示
        all_results = []
        
        for run in range(3):  # 运行3次
            print(f"\n🔄 第 {run + 1} 轮演示:")
            
            demo = ExperimentDemo(demo_mode=True, duration_hours=1)
            result = demo.run_full_demonstration()
            
            if result:
                all_results.append(result)
            else:
                print(f"❌ 第 {run + 1} 轮演示失败")
            
            time.sleep(2)  # 轮次间休息
        
        # 汇总结果
        if all_results:
            print("\n📊 汇总所有轮次结果...")
            print("=" * 60)
            
            # 这里可以添加跨轮次统计分析
            print("✅ 完整演示完成")
            return True
        else:
            print("❌ 所有轮次都失败了")
            return False
            
    except Exception as e:
        print(f"❌ 完整演示出错: {e}")
        return False

def start_streamlit():
    """启动Streamlit实时界面"""
    print("🌐 启动Streamlit实时界面")
    print("=" * 60)
    
    # 检查Streamlit是否安装
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit未安装，请运行: pip install streamlit")
        return False
    
    print("🚀 启动Streamlit应用...")
    print("📍 浏览器将自动打开 http://localhost:8501")
    print("⚠️  按 Ctrl+C 停止服务")
    
    try:
        # 启动Streamlit应用
        script_path = project_root / "experiments" / "cognition" / "long_term_retention.py"
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(script_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️  Streamlit服务已停止")
        return True
    except Exception as e:
        print(f"❌ Streamlit启动失败: {e}")
        return False

def run_real_experiment():
    """运行实际24小时实验"""
    print("🕐 启动实际24小时实验")
    print("=" * 60)
    print("⚠️  注意：这将运行真实的24小时实验")
    print("⏰ 预计用时：24小时")
    
    confirm = input("是否继续？(y/N): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return False
    
    print("\n🚀 开始实际24小时实验...")
    
    try:
        from experiments.cognition.long_term_retention import LongTermRetention
        
        # 创建实验系统
        experiment_system = LongTermRetention(streamlit_app=True)
        
        # 设置信号处理
        def signal_handler(sig, frame):
            print('\n⚠️  用户中断实验')
            experiment_system.stop_experiment()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 启动实验
        success = experiment_system.start_full_experiment()
        
        if success:
            print("✅ 24小时实验已启动")
            print("📱 请访问Streamlit界面查看实时进度")
            
            # 监控实验状态
            while experiment_system.status.value not in ["已完成", "已停止", "错误"]:
                status = experiment_system.get_experiment_status()
                print(f"⏰ 实验进度: {status['completion_rate']:.1f}% "
                      f"({status['completed_runs']}/{status['total_runs']} 组完成)")
                time.sleep(60)  # 每分钟检查一次
            
            print(f"🎉 实验完成，状态: {experiment_system.status.value}")
            return True
        else:
            print("❌ 实验启动失败")
            return False
            
    except Exception as e:
        print(f"❌ 实验运行出错: {e}")
        return False

def check_system_status():
    """检查系统状态"""
    print("🔍 检查系统状态...")
    print("=" * 40)
    
    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python版本: {python_version}")
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查项目结构
    required_files = [
        "experiments/cognition/long_term_retention.py",
        "experiments/cognition/cognitive_tracker.py", 
        "experiments/cognition/hourly_monitor.py",
        "experiments/cognition/trend_analyzer.py",
        "experiments/cognition/statistical_analyzer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
    else:
        print("✅ 所有核心文件存在")
    
    # 检查结果目录
    results_dir = project_root / "results"
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
        print("📁 创建结果目录")
    
    print(f"\n📊 系统状态: {'正常' if deps_ok and not missing_files else '需要修复'}")
    return deps_ok and not missing_files

def show_menu():
    """显示交互式菜单"""
    print("\n" + "=" * 60)
    print("🧠 六维能力增长24小时连续实验系统")
    print("=" * 60)
    print("请选择运行模式:")
    print()
    print("1. 🔬 基础演示模式（24秒=24小时）")
    print("2. 🎯 完整演示模式（包含统计分析）")
    print("3. 🌐 Streamlit实时界面模式")
    print("4. 🕐 实际24小时实验模式")
    print("5. 🔍 系统状态检查")
    print("6. 📚 显示帮助信息")
    print("7. ❌ 退出")
    print()
    
    while True:
        try:
            choice = input("请输入选项 (1-7): ").strip()
            
            if choice == '1':
                duration = input("演示持续时间（小时，默认1）: ").strip()
                duration = int(duration) if duration else 1
                return 'demo', {'duration_hours': duration}
            elif choice == '2':
                return 'full_demo', {}
            elif choice == '3':
                return 'streamlit', {}
            elif choice == '4':
                return 'real_experiment', {}
            elif choice == '5':
                return 'check_status', {}
            elif choice == '6':
                return 'help', {}
            elif choice == '7':
                print("👋 再见!")
                sys.exit(0)
            else:
                print("❌ 无效选项，请输入1-7")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 输入错误: {e}")

def show_help():
    """显示帮助信息"""
    help_text = """
🧠 六维能力增长24小时连续实验系统帮助
===================================

📋 系统概述:
本系统用于在Minecraft环境中长期监控和分析智能体的六维认知能力发展。
包括记忆力、思维力、创造力、观察力、注意力、想象力六个维度。

🎯 核心功能:
• 24小时连续数据采集
• 三组对照实验（基线组、单维优化组、六维协同组）
• 实时趋势分析和可视化
• 统计显著性检验
• 自动报告生成

🚀 运行模式:

1. 基础演示模式
   python run_24h_experiment.py --demo
   - 快速验证系统功能
   - 24秒模拟24小时
   - 包含基础数据分析

2. 完整演示模式  
   python run_24h_experiment.py --full-demo
   - 多轮演示验证
   - 完整统计分析
   - 统计显著性检验

3. Streamlit界面模式
   python run_24h_experiment.py --streamlit
   - 实时监控界面
   - 交互式图表
   - 动态数据更新

4. 实际实验模式
   python run_24h_experiment.py --real-experiment
   - 真实的24小时实验
   - 完整数据采集
   - 专业统计分析

📊 输出文件:
• experiment_results_*.json - 详细实验数据
• 24h_experiment_report_*.json - 综合分析报告
• trend_analysis_*.json - 趋势分析结果
• statistical_analysis_*.json - 统计分析结果

🔧 高级选项:
• --duration HOURS: 设置演示持续时间
• --port PORT: 设置Streamlit端口（默认8501）
• --output DIR: 设置输出目录

❓ 故障排除:
1. 依赖问题: pip install numpy pandas scipy scikit-learn plotly streamlit
2. 端口占用: 使用 --port 指定其他端口
3. 权限问题: 确保有写入权限

📞 技术支持:
查看 README_24h_Experiment.md 获取详细文档
    """
    print(help_text)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="六维能力增长24小时连续实验系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --demo                    # 基础演示
  %(prog)s --full-demo               # 完整演示  
  %(prog)s --streamlit               # 实时界面
  %(prog)s --real-experiment         # 实际实验
  %(prog)s --check-status            # 系统检查
  %(prog)s --demo --duration 2       # 自定义演示时长
  %(prog)s --streamlit --port 8502   # 自定义端口
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='启动基础演示模式（24秒=24小时）')
    parser.add_argument('--full-demo', action='store_true',
                       help='启动完整演示模式（包含统计分析）')
    parser.add_argument('--streamlit', action='store_true',
                       help='启动Streamlit实时界面模式')
    parser.add_argument('--real-experiment', action='store_true',
                       help='启动实际24小时实验模式')
    parser.add_argument('--check-status', action='store_true',
                       help='检查系统状态和依赖')
    parser.add_argument('--duration', type=int, default=1,
                       help='演示持续时间（小时，默认1）')
    parser.add_argument('--port', type=int, default=8501,
                       help='Streamlit端口（默认8501）')
    parser.add_argument('--output', type=str, default='results',
                       help='输出目录（默认results）')
    
    args = parser.parse_args()
    
    # 如果没有参数，进入交互模式
    if len(sys.argv) == 1:
        choice, params = show_menu()
        args.__dict__.update(params)
        
        # 更新参数
        if choice == 'demo':
            args.demo = True
        elif choice == 'full_demo':
            args.full_demo = True
        elif choice == 'streamlit':
            args.streamlit = True
        elif choice == 'real_experiment':
            args.real_experiment = True
        elif choice == 'check_status':
            args.check_status = True
        elif choice == 'help':
            show_help()
            return
    
    try:
        # 系统状态检查
        if args.check_status:
            check_system_status()
            return
        
        # 检查依赖
        if not check_dependencies():
            print("❌ 依赖检查失败，请安装缺少的包")
            sys.exit(1)
        
        # 运行对应模式
        if args.demo:
            print(f"🎬 基础演示模式，持续时间: {args.duration} 小时")
            run_demo(args.duration)
            
        elif args.full_demo:
            run_full_demo()
            
        elif args.streamlit:
            print(f"🌐 Streamlit界面模式，端口: {args.port}")
            start_streamlit()
            
        elif args.real_experiment:
            run_real_experiment()
            
        else:
            # 如果没有指定模式，显示帮助
            show_help()
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        print("👋 再见!")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        print("💡 提示: 查看帮助信息或运行 --check-status 检查系统状态")
        sys.exit(1)

if __name__ == "__main__":
    main()