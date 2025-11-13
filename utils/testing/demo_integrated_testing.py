#!/usr/bin/env python3
"""
集成测试系统使用示例
Integrated Testing System Usage Example

演示如何使用集成测试系统的各种功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.testing.integrated_testing import IntegratedTestingSystem, TestResult, PerformanceMetrics

def demo_basic_usage():
    """演示基本使用"""
    print("🚀 NeuroMinecraftGenesis 集成测试系统演示")
    print("=" * 50)
    
    # 1. 创建测试系统实例
    print("1. 创建测试系统实例...")
    testing_system = IntegratedTestingSystem("config/integrated_testing_config.json")
    print("✅ 测试系统创建成功")
    
    # 2. 运行功能测试
    print("\n2. 运行功能测试...")
    functional_results = testing_system.run_functional_tests()
    print(f"✅ 功能测试完成，共 {len(functional_results)} 个测试")
    
    # 3. 运行性能测试
    print("\n3. 运行性能测试...")
    performance_results = testing_system.run_performance_tests()
    print(f"✅ 性能测试完成，共 {len(performance_results)} 个基准测试")
    
    # 4. 运行兼容性测试
    print("\n4. 运行兼容性测试...")
    compatibility_results = testing_system.run_compatibility_tests()
    print(f"✅ 兼容性测试完成，共 {len(compatibility_results)} 个平台测试")
    
    # 5. 运行用户体验测试
    print("\n5. 运行用户体验测试...")
    ux_results = testing_system.run_ux_tests()
    print(f"✅ 用户体验测试完成，共 {len(ux_results)} 个测试")
    
    # 6. 准备GitHub发布
    print("\n6. 准备GitHub发布...")
    deployment_data = testing_system.prepare_github_deployment()
    print("✅ GitHub发布准备完成")
    
    # 7. 生成综合报告
    print("\n7. 生成综合测试报告...")
    comprehensive_results = testing_system.run_all_tests()
    
    # 8. 显示摘要结果
    print("\n📊 测试摘要:")
    print("=" * 50)
    
    summary = comprehensive_results.get("summary", {})
    overview = summary.get("test_overview", {})
    
    print(f"功能测试: {overview.get('functional_tests', {}).get('passed', 0)}/{overview.get('functional_tests', {}).get('total', 0)} 通过")
    print(f"性能测试: {len(performance_results)} 项基准测试")
    print(f"兼容性测试: {len(compatibility_results)} 个平台")
    print(f"用户体验测试: {overview.get('ux_tests', {}).get('passed', 0)}/{overview.get('ux_tests', {}).get('total', 0)} 通过")
    
    quality = summary.get("quality_metrics", {})
    deployment = quality.get("deployment_readiness", {})
    
    print(f"部署就绪性: {deployment.get('status', 'unknown')} (得分: {deployment.get('score', 0):.2f})")
    
    print("\n🎉 演示完成！")
    
    return comprehensive_results

def demo_individual_modules():
    """演示单独模块的使用"""
    print("\n🔧 单独模块演示")
    print("=" * 50)
    
    testing_system = IntegratedTestingSystem()
    
    # 仅功能测试
    print("执行功能测试...")
    func_results = testing_system.run_functional_tests()
    print(f"功能测试结果: {len(func_results)} 个测试")
    
    # 仅性能测试
    print("执行性能测试...")
    perf_results = testing_system.run_performance_tests()
    print(f"性能测试结果: {len(perf_results)} 个基准测试")
    
    # 仅兼容性测试
    print("执行兼容性测试...")
    comp_results = testing_system.run_compatibility_tests()
    print(f"兼容性测试结果: {len(comp_results)} 个平台测试")
    
    # 仅用户体验测试
    print("执行用户体验测试...")
    ux_results = testing_system.run_ux_tests()
    print(f"用户体验测试结果: {len(ux_results)} 个测试")
    
    # 仅GitHub发布准备
    print("准备GitHub发布...")
    deploy_results = testing_system.prepare_github_deployment()
    print("GitHub发布准备完成")

def demo_custom_configuration():
    """演示自定义配置"""
    print("\n⚙️ 自定义配置演示")
    print("=" * 50)
    
    # 创建自定义配置
    custom_config = {
        "functional_tests": {
            "enabled": True,
            "timeout": 60,
            "critical_modules": ["brain", "evolution"]
        },
        "performance_tests": {
            "enabled": True,
            "benchmark_duration": 30
        }
    }
    
    # 使用自定义配置运行测试
    testing_system = IntegratedTestingSystem()
    results = testing_system.run_all_tests(config_overrides=custom_config)
    
    print("使用自定义配置完成测试")
    return results

def demo_test_result_analysis():
    """演示测试结果分析"""
    print("\n📈 测试结果分析演示")
    print("=" * 50)
    
    testing_system = IntegratedTestingSystem()
    results = testing_system.run_all_tests()
    
    # 分析功能测试结果
    functional_tests = results.get("functional_tests", [])
    print("功能测试分析:")
    for test in functional_tests:
        if test.get("status") == "FAIL":
            print(f"  ❌ 失败: {test.get('test_name')} - {test.get('error_message')}")
        elif test.get("status") == "PASS":
            print(f"  ✅ 通过: {test.get('test_name')}")
    
    # 分析性能测试结果
    performance_tests = results.get("performance_tests", [])
    print("\\n性能测试分析:")
    for perf in performance_tests:
        efficiency = perf.get("resource_efficiency", 0)
        status = "🟢" if efficiency >= 80 else "🟡" if efficiency >= 60 else "🔴"
        print(f"  {status} 资源效率: {efficiency:.1f}% (CPU: {perf.get('cpu_usage', 0):.1f}%, Memory: {perf.get('memory_usage', 0):.1f}%)")
    
    # 分析部署就绪性
    summary = results.get("summary", {})
    quality_metrics = summary.get("quality_metrics", {})
    deployment_readiness = quality_metrics.get("deployment_readiness", {})
    
    print("\\n部署就绪性分析:")
    print(f"  状态: {deployment_readiness.get('status', 'unknown')}")
    print(f"  得分: {deployment_readiness.get('score', 0):.2f}")
    print(f"  消息: {deployment_readiness.get('message', '无信息')}")
    
    # 识别关键问题
    critical_issues = quality_metrics.get("critical_issues", [])
    if critical_issues:
        print("\\n🚨 关键问题:")
        for issue in critical_issues:
            print(f"  {issue.get('type', '未知')} - {issue.get('description', '无描述')}")
    else:
        print("\\n✅ 未发现关键问题")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="集成测试系统使用示例")
    parser.add_argument("--demo", choices=["basic", "individual", "custom", "analysis"], 
                       default="basic", help="演示类型")
    parser.add_argument("--config", help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        if args.demo == "basic":
            demo_basic_usage()
        elif args.demo == "individual":
            demo_individual_modules()
        elif args.demo == "custom":
            demo_custom_configuration()
        elif args.demo == "analysis":
            demo_test_result_analysis()
        else:
            print("未知的演示类型")
            
    except Exception as e:
        print(f"演示执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()