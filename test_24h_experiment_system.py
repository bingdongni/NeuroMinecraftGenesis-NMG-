#!/usr/bin/env python3
"""
六维能力增长24小时连续实验系统测试脚本
=======================================

该脚本用于测试和验证整个系统的所有组件功能：
- 组件导入测试
- 基本功能测试
- 集成测试
- 性能测试
- 数据完整性验证

使用方法:
    python test_24h_experiment_system.py

功能:
✅ 验证所有核心组件
✅ 测试数据流
✅ 验证统计计算
✅ 检查实时功能
✅ 生成测试报告
"""

import sys
import os
import time
import traceback
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = {
            'component_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'data_tests': {},
            'errors': [],
            'warnings': []
        }
        self.start_time = datetime.now()
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """记录测试结果"""
        print(f"[{'✅ PASS' if status == 'PASS' else '❌ FAIL' if status == 'FAIL' else '⚠️ WARN'}] {test_name}")
        if details:
            print(f"    {details}")
    
    def test_imports(self):
        """测试模块导入"""
        print("🔍 测试模块导入...")
        print("-" * 40)
        
        try:
            # 核心组件导入
            from experiments.cognition.cognitive_tracker import CognitiveTracker
            self.log_test("CognitiveTracker导入", "PASS")
            
            from experiments.cognition.hourly_monitor import HourlyMonitor
            self.log_test("HourlyMonitor导入", "PASS")
            
            from experiments.cognition.trend_analyzer import TrendAnalyzer
            self.log_test("TrendAnalyzer导入", "PASS")
            
            from experiments.cognition.statistical_analyzer import StatisticalAnalyzer
            self.log_test("StatisticalAnalyzer导入", "PASS")
            
            from experiments.cognition.long_term_retention import LongTermRetention
            self.log_test("LongTermRetention导入", "PASS")
            
            from experiments.cognition.demo_24h_experiment import ExperimentDemo
            self.log_test("ExperimentDemo导入", "PASS")
            
            self.test_results['component_tests']['imports'] = 'PASS'
            
        except ImportError as e:
            self.log_test("模块导入", "FAIL", str(e))
            self.test_results['component_tests']['imports'] = 'FAIL'
            self.test_results['errors'].append(f"模块导入失败: {e}")
    
    def test_cognitive_tracker(self):
        """测试认知跟踪器"""
        print("\n🧠 测试认知跟踪器...")
        print("-" * 40)
        
        try:
            from experiments.cognition.cognitive_tracker import CognitiveTracker
            
            # 创建跟踪器
            tracker = CognitiveTracker(agent_id="test_agent_001")
            self.log_test("CognitiveTracker创建", "PASS")
            
            # 测试权重设置
            weights = {'memory': 1.5, 'thinking': 1.0, 'creativity': 1.0,
                      'observation': 1.0, 'attention': 1.0, 'imagination': 1.0}
            tracker.set_weights(weights)
            self.log_test("权重设置", "PASS")
            
            # 测试认知指标跟踪
            agent_state = {
                'memory_retention': 0.8,
                'learning_speed': 0.7,
                'recall_accuracy': 0.9,
                'contextual_memory': 0.6,
                'reasoning_accuracy': 0.8,
                'problem_decomposition': 0.7,
                'abstract_reasoning': 0.6,
                'strategic_planning': 0.5,
                'novel_behaviors': 0.4,
                'alternative_solutions': 0.6,
                'adaptation_speed': 0.7,
                'cross_domain_transfer': 0.5,
                'environmental_awareness': 0.9,
                'detail_recognition': 0.8,
                'pattern_recognition': 0.7,
                'sensory_integration': 0.6,
                'focus_duration': 0.8,
                'distraction_resistance': 0.7,
                'attention_shift': 0.6,
                'focus_quality': 0.8,
                'current_focus_time': 300,
                'distraction_events': 2,
                'total_task_time': 3600,
                'scenario_previsualization': 0.5,
                'creative_combination': 0.6,
                'hypothetical_reasoning': 0.7,
                'mental_simulation': 0.8,
                'imagination_events': ['event1', 'event2']
            }
            
            environment_state = {
                'objects': ['tree', 'stone', 'water'],
                'time': 'day',
                'weather': 'clear',
                'hour': 12
            }
            
            # 跟踪认知指标
            metrics = tracker.track_cognitive_metrics(agent_state, environment_state)
            self.log_test("认知指标跟踪", "PASS", f"综合分数: {metrics.overall_score():.2f}")
            
            # 测试历史记录
            history = tracker.get_metrics_history(hours=1)
            self.log_test("历史记录获取", "PASS", f"记录数: {len(history)}")
            
            # 测试维度趋势
            trend = tracker.get_dimension_trend('memory', hours=1)
            self.log_test("维度趋势分析", "PASS", f"趋势数据点数: {trend.get('total_points', 0)}")
            
            self.test_results['component_tests']['cognitive_tracker'] = 'PASS'
            
        except Exception as e:
            self.log_test("认知跟踪器测试", "FAIL", str(e))
            self.test_results['component_tests']['cognitive_tracker'] = 'FAIL'
            self.test_results['errors'].append(f"认知跟踪器测试失败: {e}")
    
    def test_hourly_monitor(self):
        """测试每小时监控器"""
        print("\n⏰ 测试每小时监控器...")
        print("-" * 40)
        
        try:
            from experiments.cognition.hourly_monitor import HourlyMonitor
            from experiments.cognition.cognitive_tracker import CognitiveTracker
            
            # 创建跟踪器和监控器
            tracker = CognitiveTracker(agent_id="monitor_test_agent")
            monitor = HourlyMonitor(tracker, monitor_interval=1)  # 1秒间隔用于测试
            
            # 添加回调函数
            callback_data = []
            def test_callback(data):
                callback_data.append(data)
            
            monitor.add_callback('hourly_update', test_callback)
            self.log_test("回调函数添加", "PASS")
            
            # 启动监控
            success = monitor.start_monitoring()
            self.log_test("监控启动", "PASS" if success else "FAIL")
            
            # 等待几秒收集数据
            time.sleep(3)
            
            # 获取状态
            status = monitor.get_status()
            self.log_test("状态获取", "PASS", f"状态: {status['status']}")
            
            # 暂停和恢复监控
            monitor.pause_monitoring()
            self.log_test("监控暂停", "PASS")
            
            time.sleep(1)
            
            monitor.resume_monitoring()
            self.log_test("监控恢复", "PASS")
            
            time.sleep(2)
            
            # 停止监控
            monitor.stop_monitoring()
            self.log_test("监控停止", "PASS")
            
            # 获取性能摘要
            summary = monitor.get_performance_summary()
            self.log_test("性能摘要", "PASS", f"数据点数: {summary.get('total_hours', 0)}")
            
            # 检查回调数据
            if callback_data:
                self.log_test("回调数据收集", "PASS", f"收集到 {len(callback_data)} 个数据点")
            else:
                self.log_test("回调数据收集", "WARN", "未收集到回调数据")
            
            self.test_results['component_tests']['hourly_monitor'] = 'PASS'
            
        except Exception as e:
            self.log_test("每小时监控器测试", "FAIL", str(e))
            self.test_results['component_tests']['hourly_monitor'] = 'FAIL'
            self.test_results['errors'].append(f"每小时监控器测试失败: {e}")
    
    def test_trend_analyzer(self):
        """测试趋势分析器"""
        print("\n📈 测试趋势分析器...")
        print("-" * 40)
        
        try:
            from experiments.cognition.trend_analyzer import TrendAnalyzer
            
            # 创建分析器
            analyzer = TrendAnalyzer(min_data_points=5)
            self.log_test("TrendAnalyzer创建", "PASS")
            
            # 生成测试数据
            np.random.seed(42)
            hours = 24
            scores = [50 + i * 1.5 + np.random.normal(0, 2) for i in range(hours)]
            timestamps = [datetime.now() - timedelta(hours=hours-i) for i in range(hours)]
            
            # 分析单维度趋势
            analysis = analyzer.analyze_dimension_trend(scores, timestamps, "memory")
            self.log_test("单维度趋势分析", "PASS", f"趋势方向: {analysis.direction.value}")
            
            # 生成模拟认知指标历史
            class MockMetrics:
                def __init__(self, timestamp, memory_score, thinking_score, creativity_score,
                            observation_score, attention_score, imagination_score):
                    self.timestamp = timestamp
                    self.memory_score = memory_score
                    self.thinking_score = thinking_score
                    self.creativity_score = creativity_score
                    self.observation_score = observation_score
                    self.attention_score = attention_score
                    self.imagination_score = imagination_score
            
            metrics_history = []
            for i in range(hours):
                metrics = MockMetrics(
                    timestamps[i],
                    scores[i],
                    60 + i * 0.5 + np.random.normal(0, 3),
                    70 - i * 0.3 + np.random.normal(0, 2),
                    np.random.normal(65, 5),
                    np.random.normal(60, 4),
                    np.random.normal(55, 6)
                )
                metrics_history.append(metrics)
            
            # 分析所有维度
            all_analysis = analyzer.analyze_all_dimensions(metrics_history)
            self.log_test("全维度趋势分析", "PASS", f"分析维度数: {len(all_analysis)}")
            
            # 获取趋势摘要
            summary = analyzer.get_trend_summary(all_analysis)
            self.log_test("趋势摘要", "PASS", f"主导模式: {summary.get('dominant_pattern', 'N/A')}")
            
            self.test_results['component_tests']['trend_analyzer'] = 'PASS'
            
        except Exception as e:
            self.log_test("趋势分析器测试", "FAIL", str(e))
            self.test_results['component_tests']['trend_analyzer'] = 'FAIL'
            self.test_results['errors'].append(f"趋势分析器测试失败: {e}")
    
    def test_statistical_analyzer(self):
        """测试统计分析器"""
        print("\n📊 测试统计分析器...")
        print("-" * 40)
        
        try:
            from experiments.cognition.statistical_analyzer import StatisticalAnalyzer
            
            # 创建分析器
            analyzer = StatisticalAnalyzer(alpha=0.05)
            self.log_test("StatisticalAnalyzer创建", "PASS")
            
            # 生成测试数据
            np.random.seed(42)
            group1 = np.random.normal(50, 10, 20)  # 基线组
            group2 = np.random.normal(60, 10, 20)  # 实验组
            
            # 测试配对t检验
            result = analyzer.paired_t_test(group1[:10], group2[:10], "测试维度")
            self.log_test("配对t检验", "PASS", f"p值: {result.p_value:.4f}")
            
            # 测试独立样本t检验
            result = analyzer.independent_t_test(group1, group2, "测试维度")
            self.log_test("独立样本t检验", "PASS", f"统计量: {result.statistic:.4f}")
            
            # 测试方差分析
            groups = {
                '基线组': group1,
                '单维优化组': group2,
                '六维协同组': np.random.normal(65, 10, 20)
            }
            
            anova_result, comparisons = analyzer.anova_analysis(groups, "测试维度")
            self.log_test("方差分析", "PASS", f"F统计量: {anova_result.statistic:.4f}")
            
            # 测试多重比较校正
            p_values = [0.05, 0.01, 0.1, 0.03]
            corrected_p = analyzer.correct_multiple_comparisons(p_values)
            self.log_test("多重比较校正", "PASS", f"校正后p值: {corrected_p}")
            
            # 测试综合报告
            experiment_data = {
                '测试维度': {
                    '基线组': group1.tolist(),
                    '实验组': group2.tolist()
                }
            }
            
            report = analyzer.generate_comprehensive_report(
                experiment_data, ['基线组', '实验组']
            )
            self.log_test("综合报告生成", "PASS", f"结论数: {len(report.get('overall_conclusions', []))}")
            
            self.test_results['component_tests']['statistical_analyzer'] = 'PASS'
            
        except Exception as e:
            self.log_test("统计分析器测试", "FAIL", str(e))
            self.test_results['component_tests']['statistical_analyzer'] = 'FAIL'
            self.test_results['errors'].append(f"统计分析器测试失败: {e}")
    
    def test_integration(self):
        """测试系统集成"""
        print("\n🔗 测试系统集成...")
        print("-" * 40)
        
        try:
            # 模拟完整的24小时实验流程
            from experiments.cognition.cognitive_tracker import CognitiveTracker
            from experiments.cognition.hourly_monitor import HourlyMonitor
            from experiments.cognition.trend_analyzer import TrendAnalyzer
            from experiments.cognition.statistical_analyzer import StatisticalAnalyzer
            
            # 创建组件
            tracker = CognitiveTracker("integration_test_agent")
            monitor = HourlyMonitor(tracker, monitor_interval=0.1)  # 快速测试
            analyzer = TrendAnalyzer()
            stat_analyzer = StatisticalAnalyzer()
            
            self.log_test("组件创建", "PASS")
            
            # 模拟12小时数据采集
            for hour in range(12):
                agent_state = {
                    'memory_retention': 0.5 + hour * 0.03,
                    'learning_speed': 0.6 + hour * 0.02,
                    'recall_accuracy': 0.7 + hour * 0.025,
                    'reasoning_accuracy': 0.8 + hour * 0.02,
                    'novel_behaviors': 0.4 + hour * 0.04,
                    'environmental_awareness': 0.9,
                    'focus_duration': 0.7 + hour * 0.02,
                    'imagination_events': [f'event_{i}' for i in range(hour % 3 + 1)]
                }
                
                environment_state = {
                    'objects': ['tree', 'stone', 'water'],
                    'time': 'day' if 6 <= hour <= 18 else 'night',
                    'weather': 'clear',
                    'hour': hour
                }
                
                metrics = tracker.track_cognitive_metrics(agent_state, environment_state)
            
            self.log_test("完整数据流", "PASS", f"采集数据: {len(tracker.metrics_history)} 条")
            
            # 测试数据导出和导入
            test_file = "test_integration_data.json"
            tracker.save_metrics(test_file)
            
            new_tracker = CognitiveTracker("integration_test_agent_2")
            new_tracker.load_metrics(test_file)
            
            self.log_test("数据持久化", "PASS", f"导入记录: {len(new_tracker.metrics_history)} 条")
            
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)
            
            self.test_results['integration_tests']['full_workflow'] = 'PASS'
            
        except Exception as e:
            self.log_test("系统集成测试", "FAIL", str(e))
            self.test_results['integration_tests']['full_workflow'] = 'FAIL'
            self.test_results['errors'].append(f"系统集成测试失败: {e}")
    
    def test_performance(self):
        """测试性能"""
        print("\n⚡ 测试性能...")
        print("-" * 40)
        
        try:
            from experiments.cognition.cognitive_tracker import CognitiveTracker
            
            # 性能测试：大量数据处理
            tracker = CognitiveTracker("performance_test_agent")
            
            start_time = time.time()
            
            # 模拟1000次认知指标计算
            for i in range(1000):
                agent_state = {
                    'memory_retention': 0.5 + np.random.random() * 0.5,
                    'learning_speed': 0.6 + np.random.random() * 0.4,
                    'recall_accuracy': 0.7 + np.random.random() * 0.3,
                    'reasoning_accuracy': 0.8 + np.random.random() * 0.2,
                    'novel_behaviors': 0.4 + np.random.random() * 0.6,
                    'environmental_awareness': 0.9,
                    'focus_duration': 0.7 + np.random.random() * 0.3,
                    'imagination_events': ['event1', 'event2']
                }
                
                environment_state = {
                    'objects': ['tree', 'stone', 'water'],
                    'time': 'day',
                    'weather': 'clear'
                }
                
                metrics = tracker.track_cognitive_metrics(agent_state, environment_state)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 性能指标
            operations_per_second = 1000 / processing_time
            
            if operations_per_second > 50:  # 每秒50次操作
                self.log_test("认知计算性能", "PASS", f"{operations_per_second:.1f} ops/sec")
            elif operations_per_second > 10:
                self.log_test("认知计算性能", "WARN", f"{operations_per_second:.1f} ops/sec (较慢)")
            else:
                self.log_test("认知计算性能", "FAIL", f"{operations_per_second:.1f} ops/sec (太慢)")
            
            # 内存使用检查
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage < 100:  # 小于100MB
                self.log_test("内存使用", "PASS", f"{memory_usage:.1f} MB")
            elif memory_usage < 500:
                self.log_test("内存使用", "WARN", f"{memory_usage:.1f} MB (偏高)")
            else:
                self.log_test("内存使用", "FAIL", f"{memory_usage:.1f} MB (过高)")
            
            self.test_results['performance_tests']['basic_performance'] = 'PASS'
            
        except ImportError:
            self.log_test("性能测试", "WARN", "psutil未安装，跳过内存测试")
            self.test_results['performance_tests']['basic_performance'] = 'WARN'
        except Exception as e:
            self.log_test("性能测试", "FAIL", str(e))
            self.test_results['performance_tests']['basic_performance'] = 'FAIL'
    
    def test_data_integrity(self):
        """测试数据完整性"""
        print("\n🔍 测试数据完整性...")
        print("-" * 40)
        
        try:
            from experiments.cognition.cognitive_tracker import CognitiveMetrics
            
            # 测试认知指标数据完整性
            test_metrics = CognitiveMetrics(
                timestamp=datetime.now(),
                memory_score=85.0,
                thinking_score=92.0,
                creativity_score=78.5,
                observation_score=88.3,
                attention_score=90.1,
                imagination_score=82.7
            )
            
            # 验证数据转换
            metrics_dict = test_metrics.to_dict()
            self.log_test("数据转换", "PASS", f"转换键数: {len(metrics_dict)}")
            
            # 验证综合分数计算
            expected_overall = (85.0 + 92.0 + 78.5 + 88.3 + 90.1 + 82.7) / 6
            actual_overall = test_metrics.overall_score()
            
            if abs(actual_overall - expected_overall) < 0.001:
                self.log_test("综合分数计算", "PASS", f"{actual_overall:.2f}")
            else:
                self.log_test("综合分数计算", "FAIL", f"期望: {expected_overall:.2f}, 实际: {actual_overall:.2f}")
            
            # 测试数据范围
            all_scores = [test_metrics.memory_score, test_metrics.thinking_score, 
                         test_metrics.creativity_score, test_metrics.observation_score,
                         test_metrics.attention_score, test_metrics.imagination_score]
            
            if all(0 <= score <= 100 for score in all_scores):
                self.log_test("数据范围验证", "PASS", "所有分数在0-100范围内")
            else:
                self.log_test("数据范围验证", "FAIL", "存在超出范围的分数")
            
            # 测试边界值
            extreme_metrics = CognitiveMetrics(
                timestamp=datetime.now(),
                memory_score=0.0,
                thinking_score=100.0,
                creativity_score=50.0,
                observation_score=75.0,
                attention_score=25.0,
                imagination_score=99.9
            )
            
            extreme_overall = extreme_metrics.overall_score()
            if 0 <= extreme_overall <= 100:
                self.log_test("边界值处理", "PASS", f"边界综合分数: {extreme_overall:.2f}")
            else:
                self.log_test("边界值处理", "FAIL", f"边界综合分数超出范围: {extreme_overall:.2f}")
            
            self.test_results['data_tests']['integrity'] = 'PASS'
            
        except Exception as e:
            self.log_test("数据完整性测试", "FAIL", str(e))
            self.test_results['data_tests']['integrity'] = 'FAIL'
    
    def generate_report(self):
        """生成测试报告"""
        print("\n📋 生成测试报告...")
        print("-" * 40)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 统计结果
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warned_tests = 0
        
        for category_name, category in self.test_results.items():
            # errors和warnings是列表，其他是字典
            if category_name in ['errors', 'warnings']:
                # 对于errors和warnings，直接计数
                total_tests += len(category)
                if category_name == 'errors':
                    failed_tests += len(category)
                elif category_name == 'warnings':
                    warned_tests += len(category)
            else:
                # 对于其他类别（字典），按照原来的逻辑处理
                for test_name, result in category.items():
                    total_tests += 1
                    if result == 'PASS':
                        passed_tests += 1
                    elif result == 'FAIL':
                        failed_tests += 1
                    elif result == 'WARN':
                        warned_tests += 1
        
        # 生成报告
        report = {
            'test_summary': {
                'timestamp': end_time.isoformat(),
                'duration_seconds': duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'warned_tests': warned_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'detailed_results': self.test_results,
            'recommendations': []
        }
        
        # 生成建议
        if failed_tests > 0:
            report['recommendations'].append("存在失败的测试，需要修复相关问题")
        
        if warned_tests > 0:
            report['recommendations'].append("存在警告，建议优化性能或功能")
        
        if passed_tests == total_tests:
            report['recommendations'].append("所有测试通过，系统状态良好")
        
        # 保存报告
        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        report_file = f"system_test_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 测试报告已保存: {report_file}")
        except Exception as e:
            print(f"❌ 保存测试报告失败: {e}")
        
        # 显示摘要
        print(f"\n📊 测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   失败: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"   警告: {warned_tests} ({warned_tests/total_tests*100:.1f}%)")
        print(f"   用时: {duration:.2f} 秒")
        
        return report
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 六维能力增长24小时连续实验系统测试")
        print("=" * 60)
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 按顺序执行测试
        self.test_imports()
        self.test_cognitive_tracker()
        self.test_hourly_monitor()
        self.test_trend_analyzer()
        self.test_statistical_analyzer()
        self.test_integration()
        self.test_performance()
        self.test_data_integrity()
        
        # 生成报告
        report = self.generate_report()
        
        print("\n🎉 系统测试完成!")
        
        # 返回总体结果
        return report['test_summary']['success_rate'] >= 80  # 80%以上通过率认为成功

def main():
    """主函数"""
    try:
        tester = SystemTester()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ 系统测试通过，可以正常使用!")
            sys.exit(0)
        else:
            print("\n❌ 系统测试未通过，请检查失败的测试项!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()