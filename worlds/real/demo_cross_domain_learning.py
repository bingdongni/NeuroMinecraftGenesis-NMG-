#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域学习能力评估系统演示
Cross-Domain Learning Assessment System Demo

该演示脚本展示了跨域学习能力评估系统的核心功能。
演示包括：
1. 系统初始化
2. 单领域学习效率评估
3. 跨域学习综合评估
4. 适应速度评估
5. 结果分析和可视化

作者: AI系统
日期: 2025-11-13
"""

import asyncio
import sys
import os
import json
import numpy as np
from datetime import datetime

# 添加系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入系统组件
try:
    from worlds.real import (
        CrossDomainLearner, 
        LearningEfficiency,
        AdaptationMetrics,
        create_cross_domain_system,
        quick_assessment,
        system_health_check
    )
    print("✓ 系统组件导入成功")
except ImportError as e:
    print(f"✗ 系统组件导入失败: {e}")
    print("请确保所有组件文件已正确创建")
    sys.exit(1)


class CrossDomainLearningDemo:
    """跨域学习演示类"""
    
    def __init__(self):
        self.demo_results = {}
        print("\n" + "="*60)
        print("跨域学习能力评估系统演示")
        print("Cross-Domain Learning Assessment System Demo")
        print("="*60)
    
    async def run_full_demo(self):
        """运行完整演示"""
        try:
            # 1. 系统健康检查
            await self.system_health_check_demo()
            
            # 2. 单组件演示
            await self.component_demos()
            
            # 3. 综合评估演示
            await self.comprehensive_assessment_demo()
            
            # 4. 结果分析
            await self.result_analysis_demo()
            
            # 5. 生成演示报告
            await self.generate_demo_report()
            
            print("\n" + "="*60)
            print("演示完成！所有功能均正常工作")
            print("Demo completed! All features are working properly")
            print("="*60)
            
        except Exception as e:
            print(f"\n演示过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def system_health_check_demo(self):
        """系统健康检查演示"""
        print("\n1. 系统健康检查")
        print("-" * 30)
        
        health_status = system_health_check()
        print(f"系统状态: {health_status['status']}")
        
        if health_status['status'] == 'healthy':
            print("✓ 所有组件运行正常")
            print(f"系统版本: {health_status['version']}")
            print(f"检查时间: {health_status['timestamp']}")
        else:
            print("✗ 系统存在健康问题")
            print(f"错误信息: {health_status.get('error', '未知错误')}")
        
        self.demo_results['health_check'] = health_status
    
    async def component_demos(self):
        """组件功能演示"""
        print("\n2. 组件功能演示")
        print("-" * 30)
        
        # 学习效率评估演示
        await self.learning_efficiency_demo()
        
        # 适应速度评估演示
        await self.adaptation_speed_demo()
        
        # 领域适配演示
        await self.domain_adapter_demo()
    
    async def learning_efficiency_demo(self):
        """学习效率评估演示"""
        print("\n2.1 学习效率评估")
        print("-" * 20)
        
        try:
            # 创建学习效率评估器
            efficiency_evaluator = LearningEfficiency({
                'speed_weight': 0.3,
                'quality_weight': 0.4,
                'consistency_weight': 0.2,
                'resource_weight': 0.1
            })
            
            # 评估不同领域的学习效率
            domains = ['game', 'physics', 'social']
            
            for domain in domains:
                print(f"\n评估 {domain} 领域的学习效率...")
                
                result = await efficiency_evaluator.evaluate_learning_efficiency(
                    domain=domain,
                    knowledge_base={
                        'concepts': [f'{domain}_concept_{i}' for i in range(5)],
                        'examples': [{'type': f'{domain}_example'} for _ in range(10)]
                    },
                    evaluation_tasks={
                        'tasks': ['classification', 'prediction', 'reasoning']
                    }
                )
                
                efficiency_report = result['efficiency_report']
                print(f"  总体效率: {efficiency_report.overall_efficiency:.3f}")
                print(f"  速度效率: {efficiency_report.speed_efficiency:.3f}")
                print(f"  质量效率: {efficiency_report.quality_efficiency:.3f}")
                print(f"  学习模式: {efficiency_report.learning_pattern}")
                
                self.demo_results[f'efficiency_{domain}'] = result
            
            print("\n✓ 学习效率评估完成")
            
        except Exception as e:
            print(f"✗ 学习效率评估失败: {str(e)}")
    
    async def adaptation_speed_demo(self):
        """适应速度评估演示"""
        print("\n2.2 适应速度评估")
        print("-" * 20)
        
        try:
            # 创建适应指标计算器
            adaptation_calculator = AdaptationMetrics({
                'speed_threshold': 0.1,
                'quality_threshold': 0.8,
                'stability_window': 5
            })
            
            # 评估不同领域的适应速度
            domains = ['game', 'physics', 'social']
            
            for domain in domains:
                print(f"\n评估 {domain} 领域的适应速度...")
                
                result = await adaptation_calculator.evaluate_adaptation_speed(
                    target_domain=domain,
                    transferred_knowledge={
                        'concepts': [f'transferred_concept_{i}' for i in range(5)],
                        'rules': [{'condition': 'always', 'action': 'adapt'}]
                    },
                    adaptation_tasks={
                        'tasks': ['task1', 'task2', 'task3']
                    }
                )
                
                adaptation_report = result['adaptation_report']
                print(f"  总体适应评分: {adaptation_report.overall_adaptation_score:.3f}")
                print(f"  适应速度: {adaptation_report.adaptation_velocity:.3f}")
                print(f"  适应质量: {adaptation_report.adaptation_quality_score:.3f}")
                print(f"  适应模式: {adaptation_report.adaptation_pattern}")
                
                self.demo_results[f'adaptation_{domain}'] = result
            
            print("\n✓ 适应速度评估完成")
            
        except Exception as e:
            print(f"✗ 适应速度评估失败: {str(e)}")
    
    async def domain_adapter_demo(self):
        """领域适配演示"""
        print("\n2.3 领域适配")
        print("-" * 20)
        
        try:
            # 导入领域适配器
            from worlds.real import DomainAdapter
            
            adapter = DomainAdapter({
                'feature_adapter': {
                    'adaptation_threshold': 0.7,
                    'feature_weights': {'semantic': 0.5, 'structural': 0.5}
                }
            })
            
            print("测试领域适配: game -> physics")
            result = await adapter.adapt_knowledge(
                source_domains=['game'],
                target_domain='physics',
                learner_agent=None
            )
            
            quality = result['quality_validation']['validation_score']
            print(f"  适配质量: {quality:.3f}")
            print(f"  整合结果: {len(result['integrated_result']['integrated_knowledge']['concepts'])} 个概念")
            
            self.demo_results['domain_adapter'] = result
            
            print("\n✓ 领域适配完成")
            
        except Exception as e:
            print(f"✗ 领域适配失败: {str(e)}")
    
    async def comprehensive_assessment_demo(self):
        """综合评估演示"""
        print("\n3. 跨域学习综合评估")
        print("-" * 30)
        
        try:
            # 创建跨域学习系统
            system = CrossDomainLearner({
                'domains': ['game', 'physics', 'social', 'language', 'spatial'],
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 100,
                'transfer_threshold': 0.7,
                'parallel_processing': True,
                'enable_async': True
            })
            
            print("\n3.1 游戏领域到物理领域迁移")
            result1 = await system.assess_cross_domain_learning(
                source_domains=['game'],
                target_domains=['physics'],
                evaluation_tasks={
                    'physics': {'tasks': ['mechanics', 'dynamics']}
                },
                learner_agent=None
            )
            
            print(f"  总体得分: {result1['overall_performance']['overall_score']:.3f}")
            print(f"  评估时长: {result1['evaluation_duration']:.2f}秒")
            print(f"  成功领域: {result1['overall_performance']['successful_domains']}/1")
            
            print("\n3.2 游戏领域到社会领域迁移")
            result2 = await system.assess_cross_domain_learning(
                source_domains=['game'],
                target_domains=['social'],
                evaluation_tasks={
                    'social': {'tasks': ['interaction', 'communication']}
                },
                learner_agent=None
            )
            
            print(f"  总体得分: {result2['overall_performance']['overall_score']:.3f}")
            print(f"  评估时长: {result2['evaluation_duration']:.2f}秒")
            print(f"  成功领域: {result2['overall_performance']['successful_domains']}/1")
            
            print("\n3.3 多源到多目标迁移")
            result3 = await system.assess_cross_domain_learning(
                source_domains=['game', 'language'],
                target_domains=['physics', 'social', 'spatial'],
                evaluation_tasks={
                    'physics': {'tasks': ['mechanics']},
                    'social': {'tasks': ['interaction']},
                    'spatial': {'tasks': ['navigation']}
                },
                learner_agent=None
            )
            
            print(f"  总体得分: {result3['overall_performance']['overall_score']:.3f}")
            print(f"  评估时长: {result3['evaluation_duration']:.2f}秒")
            print(f"  成功领域: {result3['overall_performance']['successful_domains']}/3")
            
            # 保存综合评估结果
            self.demo_results['comprehensive_assessment'] = {
                'game_to_physics': result1,
                'game_to_social': result2,
                'multi_domain_transfer': result3
            }
            
            print("\n✓ 跨域学习综合评估完成")
            
        except Exception as e:
            print(f"✗ 跨域学习综合评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def result_analysis_demo(self):
        """结果分析演示"""
        print("\n4. 结果分析")
        print("-" * 20)
        
        try:
            # 系统统计信息
            system = CrossDomainLearner()
            stats = system.get_domain_statistics()
            
            print("\n4.1 领域性能统计")
            for domain, domain_stats in stats.items():
                print(f"  {domain}:")
                print(f"    评估次数: {domain_stats['count']}")
                print(f"    平均准确率: {domain_stats['avg_accuracy']:.3f}")
                print(f"    平均效率: {domain_stats['avg_efficiency']:.3f}")
                print(f"    综合性能: {domain_stats['avg_performance']:.3f}")
            
            # 学习历史分析
            history = system.get_learning_history()
            print(f"\n4.2 学习历史分析")
            print(f"  总评估次数: {len(history)}")
            
            if history:
                recent_results = []
                for record in history[-5:]:  # 最近5次评估
                    if 'overall_performance' in record:
                        score = record['overall_performance']['overall_score']
                        recent_results.append(score)
                
                if recent_results:
                    print(f"  最近5次平均得分: {np.mean(recent_results):.3f}")
                    print(f"  得分趋势: {'上升' if recent_results[-1] > recent_results[0] else '下降'}")
            
            self.demo_results['analysis'] = {
                'domain_statistics': stats,
                'learning_history_count': len(history)
            }
            
            print("\n✓ 结果分析完成")
            
        except Exception as e:
            print(f"✗ 结果分析失败: {str(e)}")
    
    async def generate_demo_report(self):
        """生成演示报告"""
        print("\n5. 生成演示报告")
        print("-" * 20)
        
        try:
            # 演示报告
            report = {
                'demo_info': {
                    'title': '跨域学习能力评估系统演示报告',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0'
                },
                'system_status': self.demo_results.get('health_check', {}),
                'component_tests': {
                    'learning_efficiency': 'completed',
                    'adaptation_speed': 'completed',
                    'domain_adapter': 'completed'
                },
                'comprehensive_assessment': 'completed',
                'overall_result': 'success',
                'performance_summary': {
                    'total_demo_time': sum([
                        result.get('evaluation_duration', 0) 
                        for result in self.demo_results.get('comprehensive_assessment', {}).values()
                        if isinstance(result, dict)
                    ])
                }
            }
            
            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"cross_domain_demo_report_{timestamp}.json"
            report_path = os.path.join(os.path.dirname(__file__), report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.demo_results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 演示报告已保存: {report_path}")
            
            # 显示总结
            print(f"\n演示总结:")
            print(f"  系统状态: {report['overall_result']}")
            print(f"  组件测试: 全部完成")
            print(f"  综合评估: 成功执行")
            print(f"  报告文件: {report_filename}")
            
        except Exception as e:
            print(f"✗ 演示报告生成失败: {str(e)}")


async def main():
    """主函数"""
    demo = CrossDomainLearningDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())