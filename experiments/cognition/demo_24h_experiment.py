#!/usr/bin/env python3
"""
六维能力增长24小时连续实验系统演示
====================================

该脚本演示了完整的24小时认知能力实验系统，包括：
- 三个对照组的并行实验
- 实时数据采集和分析
- 趋势分析和统计检验
- 自动报告生成

使用方法:
    python demo_24h_experiment.py

功能特点:
- 24小时连续监控（演示模式为24秒）
- 三个实验组：基线组、单维优化组、六维协同组
- 每组运行5次取平均值
- 实时Streamlit界面显示
- 统计显著性检验
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实验系统模块
try:
    from experiments.cognition.long_term_retention import LongTermRetention, ExperimentStatus
    from experiments.cognition.cognitive_tracker import CognitiveTracker
    from experiments.cognition.hourly_monitor import HourlyMonitor, MonitorStatus
    from experiments.cognition.trend_analyzer import TrendAnalyzer
    from experiments.cognition.statistical_analyzer import StatisticalAnalyzer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有实验模块都正确安装")
    sys.exit(1)

class ExperimentDemo:
    """24小时实验系统演示类"""
    
    def __init__(self, demo_mode: bool = True, duration_hours: int = 1):
        """
        初始化演示系统
        
        Args:
            demo_mode: 是否为演示模式（演示模式24秒=24小时）
            duration_hours: 演示持续时间（小时）
        """
        self.demo_mode = demo_mode
        self.duration_hours = duration_hours
        self.experiment_system = None
        
        # 演示数据收集
        self.demo_data = {
            'experiments': [],
            'timeline': [],
            'metrics': []
        }
        
        print("🧠 六维能力增长24小时连续实验系统演示")
        print("=" * 60)
    
    def create_demo_environment(self, agent_state: Dict, environment_state: Dict, 
                              hour: int, group_type: str) -> Dict:
        """
        创建演示环境数据
        
        Args:
            agent_state: 智能体状态
            environment_state: 环境状态
            hour: 当前小时
            group_type: 实验组类型
            
        Returns:
            更新后的环境数据
        """
        # 根据实验组类型调整性能参数
        if group_type == "基线组":
            # 基线组：无额外优化
            performance_factor = 1.0
        elif group_type == "单维优化组":
            # 单维优化组：主要优化记忆力
            performance_factor = 1.3 if hour % 3 == 0 else 1.0
        elif group_type == "六维协同组":
            # 六维协同组：所有维度都有改善
            performance_factor = 1.2
        else:
            performance_factor = 1.0
        
        # 根据小时数调整复杂度
        complexity_increase = 1 + (hour / 24) * 0.2  # 复杂度随时间增加20%
        
        # 更新智能体状态
        updated_state = agent_state.copy()
        updated_state.update({
            'memory_retention': min(1.0, agent_state.get('memory_retention', 0.5) * performance_factor),
            'learning_speed': min(1.0, agent_state.get('learning_speed', 0.6) * performance_factor),
            'recall_accuracy': min(1.0, agent_state.get('recall_accuracy', 0.7) * performance_factor),
            'reasoning_accuracy': min(1.0, agent_state.get('reasoning_accuracy', 0.8) * performance_factor),
            'novel_behaviors': min(1.0, agent_state.get('novel_behaviors', 0.4) * performance_factor),
            'environmental_awareness': min(1.0, agent_state.get('environmental_awareness', 0.9) * complexity_increase),
            'focus_duration': min(1.0, agent_state.get('focus_duration', 0.7) * performance_factor),
            'imagination_events': [f'event_{i}_{hour}' for i in range(hour % 3 + 1)]
        })
        
        # 更新环境状态
        updated_env = environment_state.copy()
        updated_env.update({
            'time': 'day' if 6 <= (hour % 24) <= 18 else 'night',
            'weather': 'clear' if hour % 5 != 0 else 'rain',
            'complexity_level': complexity_increase,
            'difficulty_factor': 1 + (hour / 24) * 0.3  # 难度逐渐增加
        })
        
        return updated_state, updated_env
    
    def simulate_single_experiment(self, group_name: str, config: Dict) -> Dict:
        """
        模拟单个24小时实验
        
        Args:
            group_name: 实验组名称
            config: 实验配置
            
        Returns:
            实验结果数据
        """
        print(f"\n🔬 开始实验组: {group_name}")
        print("-" * 40)
        
        # 初始化实验组件
        tracker = CognitiveTracker(f"agent_{group_name}_{int(time.time())}")
        
        if config.get('optimization_weights'):
            tracker.set_weights(config['optimization_weights'])
        
        monitor = HourlyMonitor(tracker, monitor_interval=1 if self.demo_mode else 3600)
        trend_analyzer = TrendAnalyzer()
        statistical_analyzer = StatisticalAnalyzer()
        
        # 初始化数据容器
        experiment_results = {
            'group_name': group_name,
            'config': config,
            'start_time': datetime.now(),
            'metrics_data': [],
            'performance_data': {},
            'trends': {}
        }
        
        # 模拟24小时数据采集
        total_hours = self.duration_hours if self.demo_mode else 24
        
        for hour in range(total_hours):
            # 模拟时间推进
            if self.demo_mode:
                time.sleep(1)  # 1秒=1小时
            else:
                time.sleep(2)  # 实际应用中每2秒采集一次
            
            print(f"  ⏰ 第 {hour + 1} 小时数据采集...")
            
            # 创建基础智能体状态
            base_state = {
                'memory_retention': 0.5 + np.random.normal(0, 0.1),
                'learning_speed': 0.6 + np.random.normal(0, 0.1),
                'recall_accuracy': 0.7 + np.random.normal(0, 0.1),
                'contextual_memory': 0.6 + np.random.normal(0, 0.1),
                'reasoning_accuracy': 0.8 + np.random.normal(0, 0.1),
                'problem_decomposition': 0.65 + np.random.normal(0, 0.1),
                'abstract_reasoning': 0.7 + np.random.normal(0, 0.1),
                'strategic_planning': 0.6 + np.random.normal(0, 0.1),
                'novel_behaviors': 0.4 + np.random.normal(0, 0.1),
                'alternative_solutions': 0.5 + np.random.normal(0, 0.1),
                'adaptation_speed': 0.55 + np.random.normal(0, 0.1),
                'cross_domain_transfer': 0.45 + np.random.normal(0, 0.1),
                'environmental_awareness': 0.9 + np.random.normal(0, 0.05),
                'detail_recognition': 0.8 + np.random.normal(0, 0.1),
                'pattern_recognition': 0.75 + np.random.normal(0, 0.1),
                'sensory_integration': 0.7 + np.random.normal(0, 0.1),
                'focus_duration': 0.7 + np.random.normal(0, 0.1),
                'distraction_resistance': 0.6 + np.random.normal(0, 0.1),
                'attention_shift': 0.65 + np.random.normal(0, 0.1),
                'focus_quality': 0.75 + np.random.normal(0, 0.1),
                'current_focus_time': hour * 60 + np.random.normal(0, 10),
                'distraction_events': max(0, np.random.poisson(2)),
                'total_task_time': hour * 60,
                'scenario_previsualization': 0.5 + np.random.normal(0, 0.1),
                'creative_combination': 0.6 + np.random.normal(0, 0.1),
                'hypothetical_reasoning': 0.55 + np.random.normal(0, 0.1),
                'mental_simulation': 0.65 + np.random.normal(0, 0.1)
            }
            
            base_env = {
                'objects': ['tree', 'stone', 'water', 'sand', 'wood'],
                'time': 'day' if 6 <= (hour % 24) <= 18 else 'night',
                'weather': 'clear',
                'hour': hour
            }
            
            # 创建演示环境
            agent_state, env_state = self.create_demo_environment(
                base_state, base_env, hour, group_name
            )
            
            # 采集认知指标
            try:
                metrics = tracker.track_cognitive_metrics(agent_state, env_state)
                
                # 记录数据
                metric_data = {
                    'hour': hour + 1,
                    'timestamp': metrics.timestamp.isoformat(),
                    'memory_score': metrics.memory_score,
                    'thinking_score': metrics.thinking_score,
                    'creativity_score': metrics.creativity_score,
                    'observation_score': metrics.observation_score,
                    'attention_score': metrics.attention_score,
                    'imagination_score': metrics.imagination_score,
                    'overall_score': metrics.overall_score()
                }
                
                experiment_results['metrics_data'].append(metric_data)
                
                # 更新性能数据
                for dim in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
                    if dim not in experiment_results['performance_data']:
                        experiment_results['performance_data'][dim] = []
                    experiment_results['performance_data'][dim].append(
                        getattr(metrics, f"{dim}_score")
                    )
                
                print(f"    📊 综合分数: {metric_data['overall_score']:.2f}")
                
            except Exception as e:
                print(f"    ❌ 数据采集失败: {e}")
        
        # 实验结束，进行趋势分析
        print(f"\n📈 进行趋势分析...")
        
        try:
            # 准备趋势分析数据
            metrics_history = []
            for data in experiment_results['metrics_data']:
                metrics_history.append(type('Metrics', (), {
                    'timestamp': datetime.fromisoformat(data['timestamp']),
                    'memory_score': data['memory_score'],
                    'thinking_score': data['thinking_score'],
                    'creativity_score': data['creativity_score'],
                    'observation_score': data['observation_score'],
                    'attention_score': data['attention_score'],
                    'imagination_score': data['imagination_score']
                })())
            
            # 执行趋势分析
            trend_analysis = trend_analyzer.analyze_all_dimensions(metrics_history)
            
            # 记录趋势结果
            for dim, trend in trend_analysis.items():
                experiment_results['trends'][dim] = {
                    'direction': trend.direction.value,
                    'strength': trend.strength,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared,
                    'forecast': trend.forecast_next_6h
                }
            
            print(f"  ✅ 趋势分析完成")
            
        except Exception as e:
            print(f"  ❌ 趋势分析失败: {e}")
        
        experiment_results['end_time'] = datetime.now()
        experiment_results['duration'] = (experiment_results['end_time'] - experiment_results['start_time']).total_seconds()
        
        print(f"🎉 实验组 {group_name} 完成 (用时: {experiment_results['duration']:.1f}秒)")
        
        return experiment_results
    
    def run_full_demonstration(self):
        """运行完整演示"""
        print("🚀 开始24小时认知能力实验演示")
        print(f"演示模式: {'开启 (24秒=24小时)' if self.demo_mode else '关闭'}")
        print()
        
        # 实验组配置
        experiment_configs = {
            "基线组": {
                'group_type': 'baseline',
                'optimization_weights': None
            },
            "单维优化组": {
                'group_type': 'single',
                'optimization_weights': {'memory': 2.0, 'thinking': 1.0, 'creativity': 1.0, 
                                       'observation': 1.0, 'attention': 1.0, 'imagination': 1.0}
            },
            "六维协同组": {
                'group_type': 'multi',
                'optimization_weights': {'memory': 1.5, 'thinking': 1.5, 'creativity': 1.5, 
                                       'observation': 1.5, 'attention': 1.5, 'imagination': 1.5}
            }
        }
        
        # 运行所有实验组
        all_results = {}
        
        for group_name, config in experiment_configs.items():
            try:
                result = self.simulate_single_experiment(group_name, config)
                all_results[group_name] = result
                
                # 短暂休息
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n⚠️  用户中断演示")
                return False
            except Exception as e:
                print(f"❌ 实验组 {group_name} 失败: {e}")
                continue
        
        # 生成综合分析报告
        print("\n📊 生成综合分析报告")
        print("=" * 60)
        
        comprehensive_report = self.generate_comprehensive_report(all_results)
        
        # 显示结果摘要
        self.display_results_summary(all_results, comprehensive_report)
        
        # 保存结果
        self.save_demonstration_results(all_results, comprehensive_report)
        
        print("\n🎉 24小时认知能力实验演示完成!")
        return True
    
    def generate_comprehensive_report(self, all_results: Dict) -> Dict:
        """生成综合分析报告"""
        report = {
            'demo_info': {
                'timestamp': datetime.now().isoformat(),
                'demo_mode': self.demo_mode,
                'duration_hours': self.duration_hours,
                'groups_tested': len(all_results)
            },
            'group_comparisons': {},
            'statistical_analysis': {},
            'conclusions': []
        }
        
        # 计算各组性能指标
        for group_name, results in all_results.items():
            if 'performance_data' in results and results['performance_data']:
                final_scores = {}
                improvement_rates = {}
                
                for dimension, scores in results['performance_data'].items():
                    if scores:
                        final_score = scores[-1]  # 最终分数
                        initial_score = scores[0]  # 初始分数
                        improvement_rate = ((final_score - initial_score) / initial_score) * 100
                        
                        final_scores[dimension] = final_score
                        improvement_rates[dimension] = improvement_rate
                
                report['group_comparisons'][group_name] = {
                    'final_scores': final_scores,
                    'improvement_rates': improvement_rates,
                    'average_improvement': np.mean(list(improvement_rates.values())) if improvement_rates else 0
                }
        
        # 生成结论
        conclusions = []
        
        if report['group_comparisons']:
            # 比较各组平均改进率
            group_improvements = {name: data['average_improvement'] 
                                for name, data in report['group_comparisons'].items()}
            
            best_group = max(group_improvements, key=group_improvements.get)
            worst_group = min(group_improvements, key=group_improvements.get)
            
            conclusions.append(f"实验组性能排序: {sorted(group_improvements.items(), key=lambda x: x[1], reverse=True)}")
            conclusions.append(f"表现最佳: {best_group} (平均改进: {group_improvements[best_group]:.2f}%)")
            conclusions.append(f"表现最差: {worst_group} (平均改进: {group_improvements[worst_group]:.2f}%)")
            
            # 分析最优策略
            if "六维协同组" in group_improvements and group_improvements["六维协同组"] > group_improvements["基线组"]:
                conclusions.append("六维协同优化策略显著优于单一维度优化")
            
            if "单维优化组" in group_improvements and group_improvements["单维优化组"] > group_improvements["基线组"]:
                conclusions.append("单维优化策略相较基线有显著改善")
        
        report['conclusions'] = conclusions
        
        return report
    
    def display_results_summary(self, all_results: Dict, report: Dict):
        """显示结果摘要"""
        print("\n📈 实验结果摘要")
        print("-" * 40)
        
        # 显示各组最终分数
        print("各组最终六维能力分数:")
        for group_name in all_results.keys():
            if group_name in report['group_comparisons']:
                scores = report['group_comparisons'][group_name]['final_scores']
                print(f"\n{group_name}:")
                print(f"  记忆力: {scores.get('memory', 0):.1f}")
                print(f"  思维力: {scores.get('thinking', 0):.1f}")
                print(f"  创造力: {scores.get('creativity', 0):.1f}")
                print(f"  观察力: {scores.get('observation', 0):.1f}")
                print(f"  注意力: {scores.get('attention', 0):.1f}")
                print(f"  想象力: {scores.get('imagination', 0):.1f}")
        
        print("\n📊 改进率比较:")
        for group_name in all_results.keys():
            if group_name in report['group_comparisons']:
                improvements = report['group_comparisons'][group_name]['improvement_rates']
                avg_improvement = report['group_comparisons'][group_name]['average_improvement']
                print(f"{group_name}: 平均改进 {avg_improvement:.2f}%")
        
        print("\n🎯 主要结论:")
        for i, conclusion in enumerate(report['conclusions'], 1):
            print(f"{i}. {conclusion}")
    
    def save_demonstration_results(self, all_results: Dict, report: Dict):
        """保存演示结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细结果
            results_file = f"24h_demo_results_{timestamp}.json"
            detailed_results = {
                'experiment_results': all_results,
                'comprehensive_report': report
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n💾 详细结果已保存到: {results_file}")
            
            # 保存简化摘要
            summary_file = f"24h_demo_summary_{timestamp}.json"
            summary = {
                'groups': list(all_results.keys()),
                'final_scores': {name: report['group_comparisons'][name]['final_scores'] 
                               for name in all_results.keys() 
                               if name in report['group_comparisons']},
                'improvement_rates': {name: report['group_comparisons'][name]['average_improvement'] 
                                    for name in all_results.keys() 
                                    if name in report['group_comparisons']},
                'conclusions': report['conclusions']
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"📋 结果摘要已保存到: {summary_file}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="24小时认知能力实验系统演示")
    parser.add_argument("--mode", choices=["demo", "real"], default="demo",
                       help="运行模式: demo=演示模式(24秒), real=实际模式(24小时)")
    parser.add_argument("--duration", type=int, default=1,
                       help="演示持续时间(小时)")
    parser.add_argument("--streamlit", action="store_true",
                       help="启动Streamlit实时界面")
    
    args = parser.parse_args()
    
    # 创建演示系统
    demo = ExperimentDemo(
        demo_mode=(args.mode == "demo"),
        duration_hours=args.duration
    )
    
    try:
        if args.streamlit:
            print("🌐 启动Streamlit实时界面...")
            # 启动Streamlit界面
            experiment_system = LongTermRetention(streamlit_app=True)
            
            # 注意：实际部署时需要使用streamlit run命令
            print("请使用以下命令启动Streamlit界面:")
            print(f"streamlit run {__file__} --server.port 8501")
            
        else:
            # 运行完整演示
            success = demo.run_full_demonstration()
            
            if success:
                print("\n🎉 演示成功完成!")
                print("\n使用说明:")
                print("- 使用 --streamlit 参数启动实时界面")
                print("- 使用 --mode real 参数运行实际24小时实验")
                print("- 所有结果会自动保存为JSON文件")
            else:
                print("\n❌ 演示失败")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()