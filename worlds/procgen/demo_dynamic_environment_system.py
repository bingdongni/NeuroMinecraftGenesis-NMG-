#!/usr/bin/env python3
"""
环境动态复杂度调节器演示程序

该程序演示如何使用环境动态复杂度调节系统的完整功能：
1. 创建集成环境系统
2. 执行动态复杂度调节
3. 自适应难度调节
4. 环境评估和监控
5. 程序化世界生成

运行方式:
    python demo_dynamic_environment_system.py
"""

import sys
import os
import time
import json
import random
import logging
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from procgen import (
    create_integrated_environment_system,
    create_demo_config,
    get_system_capabilities,
    PerformanceMetrics,
    EnvironmentSnapshot
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicEnvironmentDemo:
    """动态环境系统演示"""
    
    def __init__(self):
        """初始化演示系统"""
        self.system = None
        self.demo_data = {
            'world_states': [],
            'performance_data': [],
            'evaluation_results': [],
            'adaptation_history': []
        }
        logger.info("动态环境系统演示器初始化")
    
    def run_full_demo(self):
        """运行完整演示"""
        try:
            logger.info("开始完整环境系统演示")
            
            # 1. 创建系统
            self._create_system()
            
            # 2. 演示系统能力
            self._demonstrate_capabilities()
            
            # 3. 模拟环境演进
            self._simulate_environment_evolution()
            
            # 4. 演示评估和监控
            self._demonstrate_monitoring()
            
            # 5. 生成性能报告
            self._generate_performance_report()
            
            logger.info("演示完成!")
            return True
            
        except Exception as e:
            logger.error(f"演示失败: {str(e)}")
            return False
    
    def _create_system(self):
        """创建集成环境系统"""
        logger.info("创建集成环境系统...")
        
        config = create_demo_config()
        
        # 自定义一些演示参数
        config['world']['world_size'] = [64, 64]  # 减小尺寸以便快速演示
        config['evaluation']['evaluation_interval'] = 5.0  # 加快评估频率
        
        self.system = create_integrated_environment_system(config)
        
        logger.info("✓ 系统创建成功")
        logger.info(f"  - 复杂度控制器: {type(self.system['complexity_controller']).__name__}")
        logger.info(f"  - 难度引擎: {type(self.system['difficulty_engine']).__name__}")
        logger.info(f"  - 环境评估器: {type(self.system['environment_evaluator']).__name__}")
        logger.info(f"  - 世界生成器: {type(self.system['world_generator']).__name__}")
    
    def _demonstrate_capabilities(self):
        """演示系统能力"""
        logger.info("演示系统能力...")
        
        capabilities = get_system_capabilities()
        
        print("\n" + "="*60)
        print("  环境动态复杂度调节系统能力展示")
        print("="*60)
        
        for system_name, info in capabilities.items():
            print(f"\n【{system_name.upper()}】")
            print(f"  描述: {info['description']}")
            print("  核心功能:")
            for feature in info['features']:
                print(f"    ✓ {feature}")
        
        print("\n" + "="*60)
    
    def _simulate_environment_evolution(self):
        """模拟环境演进过程"""
        logger.info("开始模拟环境演进...")
        
        initial_complexity = 0.3
        target_complexity = 0.8
        steps = 8
        
        complexity_step = (target_complexity - initial_complexity) / steps
        
        print(f"\n{'='*60}")
        print(f"  环境演进模拟 (复杂度: {initial_complexity:.1f} → {target_complexity:.1f})")
        print(f"{'='*60}")
        
        for step in range(steps + 1):
            current_complexity = initial_complexity + step * complexity_step
            
            print(f"\n--- 步骤 {step + 1}/{steps + 1}: 复杂度 {current_complexity:.3f} ---")
            
            # 生成世界
            world_state = self._generate_world(current_complexity)
            
            # 模拟智能体性能
            performance_data = self._simulate_agent_performance(current_complexity, step)
            
            # 执行复杂度自适应
            adaptation_result = self._execute_complexity_adaptation(world_state, performance_data)
            
            # 执行难度调整
            difficulty_result = self._execute_difficulty_adjustment(world_state, performance_data)
            
            # 保存演示数据
            self.demo_data['world_states'].append(world_state)
            self.demo_data['performance_data'].append(performance_data)
            self.demo_data['adaptation_history'].append({
                'step': step,
                'complexity': current_complexity,
                'adaptation': adaptation_result,
                'difficulty': difficulty_result
            })
            
            # 显示关键指标
            metrics = world_state.get('metrics', {})
            print(f"  地形复杂度: {metrics.get('terrain_complexity', 0):.3f}")
            print(f"  资源稀缺度: {metrics.get('resource_scarcity', 0):.3f}")
            print(f"  危险系数: {metrics.get('danger_level', 0):.3f}")
            print(f"  可访问性: {metrics.get('accessibility', 0):.3f}")
            
            if adaptation_result.get('adaptation_applied'):
                print(f"  复杂度调整: {adaptation_result['complexity_change_pct']:+.1f}%")
            
            if difficulty_result.get('adjustment_made'):
                print(f"  难度调整: {difficulty_result['strategy']} "
                      f"({difficulty_result['adjustment_amount']:+.3f})")
            
            time.sleep(0.5)  # 短暂延迟以便观察
        
        print(f"\n{'='*60}")
        print("  环境演进模拟完成")
        print(f"{'='*60}")
    
    def _generate_world(self, complexity: float) -> Dict[str, Any]:
        """生成世界"""
        try:
            world_generator = self.system['world_generator']
            
            # 简单的世界生成参数
            world_config = {
                'world_size': [32, 32],  # 小尺寸快速生成
                'complexity_target': complexity,
                'cave_density': complexity * 0.8,  # 洞穴密度随复杂度增加
                'resource_density': max(0.3, 1.0 - complexity * 0.7),  # 资源密度随复杂度减少
                'base_danger_level': complexity * 0.5  # 危险系数随复杂度增加
            }
            
            # 创建临时世界配置
            from procgen.world_generator import WorldConfig
            temp_config = WorldConfig(**world_config)
            
            # 临时替换生成器的配置
            original_config = world_generator.config
            world_generator.config = temp_config
            
            # 生成世界
            world_state = world_generator.generate_world(complexity, preserve_progress=True)
            
            # 恢复原配置
            world_generator.config = original_config
            
            return world_state
            
        except Exception as e:
            logger.error(f"世界生成失败: {str(e)}")
            # 返回模拟数据
            return self._create_mock_world_state(complexity)
    
    def _create_mock_world_state(self, complexity: float) -> Dict[str, Any]:
        """创建模拟世界状态"""
        return {
            'terrain': [[None] * 32 for _ in range(32)],  # 简化地形
            'resource_nodes': [],
            'active_events': [],
            'complexity': complexity,
            'size': [32, 32],
            'metrics': {
                'terrain_complexity': min(1.0, complexity * 0.8),
                'resource_scarcity': max(0.3, 1.0 - complexity * 0.7),
                'danger_level': complexity * 0.6,
                'accessibility': max(0.2, 1.0 - complexity * 0.4),
                'temporal_stability': max(0.3, 1.0 - complexity * 0.3)
            },
            'statistics': {
                'terrain': {
                    'total_cells': 1024,
                    'average_height': complexity * 0.5,
                    'average_hardness': complexity * 0.6
                },
                'resources': {
                    'total_nodes': max(10, 100 - int(complexity * 50)),
                    'total_resource_value': max(100, 1000 - int(complexity * 500))
                }
            }
        }
    
    def _simulate_agent_performance(self, complexity: float, step: int) -> Dict[str, PerformanceMetrics]:
        """模拟智能体性能"""
        # 模拟智能体在当前复杂度下的表现
        agents = {}
        
        for i in range(3):  # 3个智能体
            agent_id = f"agent_{i+1}"
            
            # 基础性能受复杂度影响
            base_success = max(0.2, 0.8 - complexity * 0.3)
            base_learning = max(0.1, 0.6 - complexity * 0.2)
            base_stress = min(0.9, complexity * 0.7)
            
            # 添加随机变化和逐步改善
            improvement_factor = step * 0.02  # 逐步改善
            random_factor = random.uniform(-0.1, 0.1)
            
            success_rate = max(0.1, min(0.95, base_success + improvement_factor + random_factor))
            learning_rate = max(0.05, min(0.8, base_learning + improvement_factor + random_factor))
            stress_level = max(0.0, min(0.9, base_stress + random_factor * 0.5))
            
            performance = PerformanceMetrics(
                agent_id=agent_id,
                timestamp=time.time(),
                success_rate=success_rate,
                task_completion_time=random.uniform(20, 60) * (1 + complexity * 0.5),
                resource_efficiency=max(0.2, min(0.9, 0.7 + random_factor)),
                survival_score=max(0.1, min(0.9, success_rate * 0.8 + 0.1)),
                learning_rate=learning_rate,
                challenge_level=complexity,
                stress_level=stress_level
            )
            
            agents[agent_id] = performance
        
        return agents
    
    def _execute_complexity_adaptation(self, world_state: Dict, performance_data: Dict):
        """执行复杂度自适应"""
        try:
            controller = self.system['complexity_controller']
            
            # 转换性能数据格式
            agent_performance = {}
            for agent_id, performance in performance_data.items():
                agent_performance[agent_id] = {
                    'navigation_success_rate': performance.success_rate,
                    'resource_collection_rate': performance.resource_efficiency,
                    'danger_avoidance_rate': performance.survival_score,
                    'average_survival_time': performance.task_completion_time,
                    'learning_rate': performance.learning_rate,
                    'death_avoidance_rate': performance.survival_score
                }
            
            result = controller.adapt_complexity(world_state, agent_performance)
            return result
            
        except Exception as e:
            logger.error(f"复杂度自适应失败: {str(e)}")
            return {'error': str(e)}
    
    def _execute_difficulty_adjustment(self, world_state: Dict, performance_data: Dict):
        """执行难度调整"""
        try:
            engine = self.system['difficulty_engine']
            
            # 转换性能数据格式
            performance_dict = {
                agent_id: {
                    'success_rate': perf.success_rate,
                    'task_completion_time': perf.task_completion_time,
                    'resource_efficiency': perf.resource_efficiency,
                    'survival_score': perf.survival_score,
                    'learning_rate': perf.learning_rate,
                    'stress_level': perf.stress_level
                }
                for agent_id, perf in performance_data.items()
            }
            
            # 模拟环境状态
            environment_state = {
                'terrain_complexity': world_state.get('metrics', {}).get('terrain_complexity', 0.5),
                'resource_availability': 1.0 - world_state.get('metrics', {}).get('resource_scarcity', 0.5),
                'danger_level': world_state.get('metrics', {}).get('danger_level', 0.3)
            }
            
            result = engine.evaluate_and_adjust(performance_dict, environment_state)
            return result
            
        except Exception as e:
            logger.error(f"难度调整失败: {str(e)}")
            return {'error': str(e)}
    
    def _demonstrate_monitoring(self):
        """演示环境监控"""
        logger.info("演示环境监控功能...")
        
        try:
            evaluator = self.system['environment_evaluator']
            
            print(f"\n{'='*60}")
            print("  环境评估和监控系统演示")
            print(f"{'='*60}")
            
            # 创建快照
            if self.demo_data['world_states']:
                latest_world = self.demo_data['world_states'][-1]
                snapshot = evaluator.create_snapshot(latest_world)
                print(f"✓ 创建环境快照: {snapshot.timestamp}")
            
            # 执行评估
            if self.demo_data['world_states'] and self.demo_data['performance_data']:
                world_state = self.demo_data['world_states'][-1]
                agent_states = {agent_id: {'performance': perf.__dict__} 
                              for agent_id, perf in self.demo_data['performance_data'][-1].items()}
                
                evaluation_result = evaluator.evaluate_environment(
                    world_state, agent_states, 
                    evaluation_type='comprehensive'
                )
                
                print(f"✓ 环境评估完成: {evaluation_result.evaluation_id}")
                print(f"  - 整体评分: {evaluation_result.overall_score:.3f}")
                print(f"  - 评估时长: {evaluation_result.duration:.3f}秒")
                
                print("  - 分类评分:")
                for category, score in evaluation_result.category_scores.items():
                    print(f"    * {category.value}: {score:.3f}")
                
                if evaluation_result.recommendations:
                    print("  - 智能建议:")
                    for i, recommendation in enumerate(evaluation_result.recommendations[:3], 1):
                        print(f"    {i}. {recommendation}")
                
                # 保存评估结果
                self.demo_data['evaluation_results'].append(evaluation_result)
            
            # 获取监控统计
            stats = evaluator.get_evaluation_statistics()
            print(f"\n✓ 监控统计:")
            print(f"  - 评估总数: {stats['evaluation_stats']['total_evaluations']}")
            print(f"  - 平均评估时间: {stats['evaluation_stats']['avg_evaluation_time']:.3f}秒")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            logger.error(f"监控演示失败: {str(e)}")
    
    def _generate_performance_report(self):
        """生成性能报告"""
        logger.info("生成性能报告...")
        
        print(f"\n{'='*60}")
        print("  动态环境系统性能报告")
        print(f"{'='*60}")
        
        # 系统概览
        if self.system:
            print("\n【系统组件】")
            for name, component in self.system.items():
                if name not in ['config', 'created_at']:
                    print(f"  ✓ {name}: {type(component).__name__}")
        
        # 演进统计
        if self.demo_data['adaptation_history']:
            print(f"\n【环境演进统计】")
            adaptations = self.demo_data['adaptation_history']
            
            print(f"  - 演进步骤: {len(adaptations)}步")
            print(f"  - 复杂度范围: {adaptations[0]['complexity']:.3f} - {adaptations[-1]['complexity']:.3f}")
            
            # 复杂度调整统计
            complexity_adjustments = [a['adaptation'] for a in adaptations 
                                    if a['adaptation'].get('adaptation_applied')]
            if complexity_adjustments:
                adjustment_amounts = [a['complexity_change_pct'] for a in complexity_adjustments]
                print(f"  - 复杂度调整次数: {len(complexity_adjustments)}")
                print(f"  - 平均调整幅度: {sum(adjustment_amounts)/len(adjustment_amounts):.1f}%")
            
            # 难度调整统计
            difficulty_adjustments = [a['difficulty'] for a in adaptations 
                                    if a['difficulty'].get('adjustment_made')]
            if difficulty_adjustments:
                print(f"  - 难度调整次数: {len(difficulty_adjustments)}")
                strategies_used = {}
                for adjustment in difficulty_adjustments:
                    strategy = adjustment.get('strategy', 'unknown')
                    strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
                print(f"  - 策略使用: {strategies_used}")
        
        # 评估统计
        if self.demo_data['evaluation_results']:
            print(f"\n【评估统计】")
            evaluations = self.demo_data['evaluation_results']
            
            scores = [e.overall_score for e in evaluations]
            print(f"  - 评估次数: {len(evaluations)}")
            print(f"  - 平均评分: {sum(scores)/len(scores):.3f}")
            print(f"  - 评分范围: {min(scores):.3f} - {max(scores):.3f}")
            
            # 推荐统计
            all_recommendations = []
            for e in evaluations:
                all_recommendations.extend(e.recommendations)
            
            if all_recommendations:
                print(f"  - 建议数量: {len(all_recommendations)}")
        
        # 世界生成统计
        if self.demo_data['world_states']:
            print(f"\n【世界生成统计】")
            worlds = self.demo_data['world_states']
            
            complexities = [w.get('complexity', 0) for w in worlds]
            resource_counts = [len(w.get('resource_nodes', [])) for w in worlds]
            event_counts = [len(w.get('active_events', [])) for w in worlds]
            
            print(f"  - 世界生成次数: {len(worlds)}")
            print(f"  - 平均资源节点: {sum(resource_counts)/len(resource_counts):.1f}")
            print(f"  - 平均事件数: {sum(event_counts)/len(event_counts):.1f}")
        
        # 系统性能
        if self.system:
            print(f"\n【系统性能】")
            try:
                # 复杂度控制器统计
                complexity_stats = self.system['complexity_controller'].get_performance_statistics()
                if complexity_stats:
                    print(f"  - 复杂度评估次数: {complexity_stats['evaluation_count']}")
                
                # 世界生成器统计
                world_stats = self.system['world_generator'].get_world_statistics()
                if world_stats:
                    gen_stats = world_stats.get('generation_stats', {})
                    print(f"  - 世界生成次数: {gen_stats.get('total_generations', 0)}")
                    print(f"  - 平均生成时间: {gen_stats.get('avg_generation_time', 0):.3f}秒")
                
            except Exception as e:
                print(f"  - 性能数据获取失败: {str(e)}")
        
        print(f"\n{'='*60}")
        print("  性能报告生成完成")
        print(f"{'='*60}")
        
        # 保存详细报告
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """保存详细报告"""
        try:
            report_data = {
                'report_timestamp': time.time(),
                'system_info': {
                    'version': '1.0.0',
                    'components': {name: type(comp).__name__ 
                                 for name, comp in self.system.items() 
                                 if name not in ['config', 'created_at']}
                },
                'demo_summary': {
                    'total_steps': len(self.demo_data['adaptation_history']),
                    'worlds_generated': len(self.demo_data['world_states']),
                    'evaluations_performed': len(self.demo_data['evaluation_results']),
                    'adaptations_made': len([a for a in self.demo_data['adaptation_history'] 
                                           if a['adaptation'].get('adaptation_applied')])
                },
                'detailed_data': {
                    'adaptation_history': [
                        {
                            'step': a['step'],
                            'complexity': a['complexity'],
                            'adaptation_applied': a['adaptation'].get('adaptation_applied', False),
                            'difficulty_adjusted': a['difficulty'].get('adjustment_made', False)
                        }
                        for a in self.demo_data['adaptation_history']
                    ]
                }
            }
            
            report_file = f"demo_performance_report_{int(time.time())}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ 详细报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存详细报告失败: {str(e)}")


def main():
    """主函数"""
    print("环境动态复杂度调节器演示程序")
    print("=" * 50)
    
    # 创建演示器
    demo = DynamicEnvironmentDemo()
    
    # 运行演示
    success = demo.run_full_demo()
    
    if success:
        print("\n🎉 演示成功完成!")
        print("\n关键成果:")
        print("  ✓ 成功创建集成环境系统")
        print("  ✓ 完成动态复杂度调节演示")
        print("  ✓ 实现自适应难度控制")
        print("  ✓ 演示环境评估监控")
        print("  ✓ 生成程序化世界")
        print("  ✓ 输出性能分析报告")
        
        print(f"\n📊 查看详细报告文件获取更多信息")
    else:
        print("\n❌ 演示失败，请检查错误日志")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())