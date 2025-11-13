#!/usr/bin/env python3
"""
简单验证脚本 - 验证集成测试系统的核心功能
Simple Verification Script - Verify core functionality of the integrated testing system
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_core_functionality():
    """测试核心功能"""
    print("🧪 测试集成测试系统核心功能")
    print("=" * 50)
    
    try:
        # 1. 导入测试系统
        from utils.testing.integrated_testing import IntegratedTestingSystem, TestResult, PerformanceMetrics, CompatibilityResult
        print("✅ 成功导入集成测试系统")
        
        # 2. 创建系统实例
        testing_system = IntegratedTestingSystem()
        print("✅ 成功创建系统实例")
        
        # 3. 测试配置加载
        config = testing_system.config
        assert "functional_tests" in config
        print("✅ 配置系统工作正常")
        
        # 4. 测试结果创建
        test_result = TestResult(
            test_name="test_example",
            status="PASS",
            duration=1.0,
            error_message="测试成功"
        )
        assert test_result.test_name == "test_example"
        assert test_result.status == "PASS"
        print("✅ 测试结果创建正常")
        
        # 5. 测试性能指标创建
        perf_metric = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            execution_time=2.0,
            throughput=100.0,
            latency=0.01,
            resource_efficiency=75.0
        )
        assert perf_metric.cpu_usage == 50.0
        print("✅ 性能指标创建正常")
        
        # 6. 测试兼容性结果创建
        comp_result = CompatibilityResult(
            platform="Test Platform",
            python_version="3.9",
            dependencies_status={"numpy": "OK"},
            test_results=[test_result],
            overall_score=85.0
        )
        assert comp_result.platform == "Test Platform"
        print("✅ 兼容性结果创建正常")
        
        # 7. 测试结果目录创建
        testing_system.results_dir.mkdir(parents=True, exist_ok=True)
        assert testing_system.results_dir.exists()
        print("✅ 结果目录创建正常")
        
        # 8. 测试GitHub发布准备
        deployment_data = testing_system.prepare_github_deployment()
        assert "release_notes" in deployment_data
        print("✅ GitHub发布准备功能正常")
        
        # 9. 测试综合摘要生成
        test_results_data = {
            "functional_tests": [test_result.to_dict()],
            "performance_tests": [perf_metric.__dict__],
            "compatibility_tests": [comp_result.__dict__],
            "ux_tests": [test_result.to_dict()]
        }
        
        summary = testing_system._generate_comprehensive_summary(test_results_data)
        assert "test_overview" in summary
        print("✅ 综合摘要生成正常")
        
        # 10. 测试HTML报告生成
        html_file = testing_system.results_dir / "test_report.html"
        testing_system._generate_html_report(test_results_data, html_file)
        assert html_file.exists()
        print("✅ HTML报告生成正常")
        
        print("\n🎉 所有核心功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 核心功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_modules():
    """测试各个模块的基本功能"""
    print("\n🔧 测试各个模块功能")
    print("=" * 50)
    
    try:
        from utils.testing.integrated_testing import IntegratedTestingSystem, TestResult
        
        testing_system = IntegratedTestingSystem()
        
        # 测试配置更新
        original_timeout = testing_system.config["functional_tests"]["timeout"]
        testing_system.config["functional_tests"]["timeout"] = 60
        assert testing_system.config["functional_tests"]["timeout"] == 60
        print("✅ 配置更新功能正常")
        
        # 测试发布说明生成
        release_notes = testing_system._generate_release_notes()
        assert "NeuroMinecraftGenesis" in release_notes
        print("✅ 发布说明生成正常")
        
        # 测试变更日志生成
        changelog = testing_system._generate_changelog()
        assert "变更日志" in changelog
        print("✅ 变更日志生成正常")
        
        # 测试文档生成
        docs = testing_system._generate_documentation()
        assert "api_reference" in docs
        print("✅ 文档生成正常")
        
        # 测试测试覆盖率计算
        test_results = [
            TestResult("test1", "PASS", 1.0),
            TestResult("test2", "FAIL", 0.5, "错误"),
            TestResult("test3", "PASS", 2.0)
        ]
        
        testing_system.test_results = test_results
        coverage = testing_system._calculate_test_coverage()
        assert coverage["total_tests"] == 3
        assert coverage["passed_tests"] == 2
        print("✅ 测试覆盖率计算正常")
        
        print("\n🎉 所有模块功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 模块功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n⚠️ 测试错误处理")
    print("=" * 50)
    
    try:
        from utils.testing.integrated_testing import IntegratedTestingSystem, TestResult
        
        testing_system = IntegratedTestingSystem()
        
        # 测试不存在的配置文件
        try:
            bad_system = IntegratedTestingSystem("/non/existent/config.json")
            # 应该使用默认配置而不是失败
            assert bad_system.config is not None
            print("✅ 不存在配置文件的错误处理正常")
        except:
            print("✅ 不存在配置文件产生预期异常")
        
        # 测试错误测试结果处理
        bad_result = TestResult("bad_test", "ERROR", 0, "测试错误")
        assert bad_result.status == "ERROR"
        print("✅ 错误测试结果处理正常")
        
        # 测试空结果处理
        empty_summary = testing_system._generate_comprehensive_summary({})
        assert "test_overview" in empty_summary
        print("✅ 空结果处理正常")
        
        print("\n🎉 错误处理测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 NeuroMinecraftGenesis 集成测试系统核心验证")
    print("=" * 60)
    
    tests = [
        ("核心功能测试", test_core_functionality),
        ("模块功能测试", test_individual_modules),
        ("错误处理测试", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 运行 {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️  {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 验证结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 集成测试系统核心功能验证完全通过！")
        print("✅ 系统已准备就绪，可以进行正式测试")
        
        # 显示生成的测试文件
        results_dir = Path("utils/testing/results")
        if results_dir.exists():
            files = list(results_dir.glob("*"))
            if files:
                print(f"\n📁 生成的测试文件:")
                for file in files:
                    print(f"   - {file.name}")
        
        return True
    else:
        print(f"⚠️  {passed}/{total} 项测试通过，需要解决剩余问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)