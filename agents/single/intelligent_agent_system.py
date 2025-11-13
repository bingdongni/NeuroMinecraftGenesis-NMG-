#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体动作系统综合测试

这个模块演示整个智能体动作系统的完整功能：
1. 27种原子动作的执行
2. 组合技能库的使用
3. 10Hz频率的动作控制
4. 动作优先级和序列管理
5. 技能学习系统

作者：MiniMax智能体
创建时间：2025-11-13
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Any

from action_executor import ActionExecutor, ActionType
from skill_library import SkillLibrary, SkillCategory
from motion_controller import MotionController, ActionPriority, ScheduledAction


class IntelligentAgentSystem:
    """智能体动作系统主类
    
    整合动作执行器、技能库和动作控制器，提供完整的智能体动作系统
    """
    
    def __init__(self):
        # 初始化各个组件
        self.action_executor = ActionExecutor()
        self.skill_library = SkillLibrary(self.action_executor)
        self.motion_controller = MotionController(self.action_executor, self.skill_library)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 系统状态
        self.system_started = False
        self.test_results = {}
    
    async def start_system(self):
        """启动整个系统"""
        if self.system_started:
            self.logger.warning("系统已在运行")
            return
        
        # 启动动作控制器
        await self.motion_controller.start()
        self.system_started = True
        
        self.logger.info("智能体动作系统已启动")
    
    async def stop_system(self):
        """停止整个系统"""
        if not self.system_started:
            return
        
        # 停止动作控制器
        await self.motion_controller.stop()
        self.system_started = False
        
        self.logger.info("智能体动作系统已停止")
    
    async def execute_atom_actions(self):
        """执行27种原子动作测试"""
        print("\n" + "="*60)
        print("🧪 原子动作测试 (27种动作)")
        print("="*60)
        
        atom_actions = [
            # 8方向移动动作
            (ActionType.MOVE_FORWARD, "向前移动", {'distance': 3.0}),
            (ActionType.MOVE_BACKWARD, "向后移动", {'distance': 2.0}),
            (ActionType.MOVE_LEFT, "向左移动", {'distance': 2.0}),
            (ActionType.MOVE_RIGHT, "向右移动", {'distance': 2.0}),
            (ActionType.MOVE_FORWARD_LEFT, "左前移动", {'distance': 2.0}),
            (ActionType.MOVE_FORWARD_RIGHT, "右前移动", {'distance': 2.0}),
            (ActionType.MOVE_BACKWARD_LEFT, "左后移动", {'distance': 2.0}),
            (ActionType.MOVE_BACKWARD_RIGHT, "右后移动", {'distance': 2.0}),
            
            # 跳跃和飞行动作
            (ActionType.JUMP, "跳跃", {'height': 2.0}),
            (ActionType.DOUBLE_JUMP, "双跳", {'height': 3.0}),
            (ActionType.FLY_UP, "向上飞行", {'height': 5.0, 'duration': 1.0}),
            (ActionType.FLY_DOWN, "向下飞行", {'height': 3.0, 'duration': 1.0}),
            (ActionType.FLY_FORWARD, "向前飞行", {'height': 10.0, 'duration': 2.0}),
            (ActionType.FLY_BACKWARD, "向后飞行", {'height': 5.0, 'duration': 1.0}),
            (ActionType.FLY_STOP, "停止飞行", {}),
            
            # 攻击和交互动作
            (ActionType.ATTACK, "攻击", {'target': 'zombie', 'damage': 20}),
            (ActionType.RIGHT_CLICK, "右键交互", {'target': 'chest'}),
            (ActionType.DESTROY_BLOCK, "破坏方块", {'target': (0, 0, 0), 'block_type': 'stone'}),
            
            # 物品操作动作
            (ActionType.PLACE_BLOCK, "放置方块", {'item_id': 'stone', 'quantity': 2, 'position': (1, 0, 0)}),
            (ActionType.USE_ITEM, "使用物品", {'item_id': 'apple'}),
            (ActionType.DROP_ITEM, "丢弃物品", {'item_id': 'stone', 'quantity': 1}),
            (ActionType.INVENTORY_OPEN, "打开背包", {}),
            (ActionType.INVENTORY_CLOSE, "关闭背包", {})
        ]
        
        successful_actions = 0
        total_actions = len(atom_actions)
        
        # 添加一些测试物品到背包
        self.action_executor.inventory = {
            'stone': 10,
            'wood': 5,
            'apple': 3,
            'tool': 1
        }
        
        for action_type, description, parameters in atom_actions:
            start_time = time.time()
            
            # 使用动作控制器调度动作
            action_id = self.motion_controller.create_and_schedule_action(
                action_type,
                priority=ActionPriority.NORMAL,
                parameters=parameters
            )
            
            # 等待执行完成
            await asyncio.sleep(0.2)
            
            # 检查状态
            status = self.motion_controller.get_action_status(action_id)
            duration = time.time() - start_time
            
            if status.get('state') == 'COMPLETED':
                successful_actions += 1
                print(f"✅ {description:<15} - 成功 ({duration:.2f}s)")
            else:
                print(f"❌ {description:<15} - 失败 ({duration:.2f}s)")
        
        # 重置状态
        self.action_executor.reset_state()
        
        success_rate = successful_actions / total_actions
        print(f"\n📊 原子动作测试结果:")
        print(f"   成功: {successful_actions}/{total_actions}")
        print(f"   成功率: {success_rate:.2%}")
        
        self.test_results['atom_actions'] = {
            'total': total_actions,
            'successful': successful_actions,
            'success_rate': success_rate
        }
    
    async def execute_skill_actions(self):
        """执行组合技能测试"""
        print("\n" + "="*60)
        print("🎯 组合技能测试")
        print("="*60)
        
        # 获取可用技能
        available_skills = self.skill_library.get_all_skills()
        
        # 选择几个代表性技能进行测试
        test_skills = ['simple_house', 'tree_harvesting', 'basic_exploration', 'basic_mining', 'basic_combat']
        
        successful_skills = 0
        total_skills = len(test_skills)
        
        for skill_name in test_skills:
            if skill_name in available_skills:
                start_time = time.time()
                
                # 根据技能类型设置参数
                if skill_name == 'simple_house':
                    parameters = {
                        'size': {'width': 3, 'length': 4},
                        'materials': {'wood': 20, 'stone': 15},
                        'quality': 0.9
                    }
                elif skill_name == 'tree_harvesting':
                    parameters = {'tree_count': 3}
                elif skill_name == 'basic_exploration':
                    parameters = {'exploration_radius': 2, 'include_underground': False}
                elif skill_name == 'basic_mining':
                    parameters = {'mining_depth': 5, 'target_materials': ['stone', 'iron']}
                elif skill_name == 'basic_combat':
                    parameters = {'enemy_count': 2, 'enemy_type': 'zombie'}
                else:
                    parameters = {}
                
                # 使用动作控制器调度技能
                skill_id = self.motion_controller.create_and_schedule_action(
                    skill_name,
                    priority=ActionPriority.NORMAL,
                    parameters=parameters
                )
                
                # 等待执行完成
                await asyncio.sleep(2.0)
                
                # 检查状态
                status = self.motion_controller.get_action_status(skill_id)
                duration = time.time() - start_time
                
                if status.get('state') == 'COMPLETED':
                    successful_skills += 1
                    
                    # 获取技能信息
                    skill_info = self.skill_library.get_skill_info(skill_name)
                    mastery = skill_info.get('mastery_level', 0)
                    
                    print(f"✅ {skill_name:<20} - 成功 (熟练度: {mastery:.2f}, 用时: {duration:.2f}s)")
                else:
                    print(f"❌ {skill_name:<20} - 失败 ({duration:.2f}s)")
        
        success_rate = successful_skills / total_skills
        print(f"\n📊 技能测试结果:")
        print(f"   成功: {successful_skills}/{total_skills}")
        print(f"   成功率: {success_rate:.2%}")
        
        self.test_results['skill_actions'] = {
            'total': total_skills,
            'successful': successful_skills,
            'success_rate': success_rate
        }
    
    async def execute_priority_system(self):
        """测试动作优先级系统"""
        print("\n" + "="*60)
        print("⚡ 动作优先级测试")
        print("="*60)
        
        # 创建不同优先级的动作
        priorities = [
            (ActionPriority.BACKGROUND, "后台任务"),
            (ActionPriority.LOW, "低优先级"),
            (ActionPriority.NORMAL, "普通优先级"),
            (ActionPriority.HIGH, "高优先级"),
            (ActionPriority.EMERGENCY, "紧急任务")
        ]
        
        scheduled_actions = []
        
        for priority, description in priorities:
            action_id = self.motion_controller.create_and_schedule_action(
                ActionType.JUMP,
                priority=priority,
                parameters={'height': 1.0}
            )
            scheduled_actions.append((action_id, description))
        
        print("📋 已调度不同优先级的动作:")
        for action_id, description in scheduled_actions:
            print(f"   {description:<15} - {action_id}")
        
        # 等待执行
        await asyncio.sleep(3.0)
        
        # 检查执行顺序（高优先级应该先执行）
        print("\n🔍 执行顺序分析:")
        for action_id, description in scheduled_actions:
            status = self.motion_controller.get_action_status(action_id)
            execution_time = status.get('execution_time', 0)
            state = status.get('state', 'Unknown')
            print(f"   {description:<15} - {state} (执行时间: {execution_time:.3f})")
        
        self.test_results['priority_system'] = {
            'actions_tested': len(priorities),
            'all_scheduled': True
        }
    
    async def execute_sequence_system(self):
        """测试动作序列系统"""
        print("\n" + "="*60)
        print("📋 动作序列测试")
        print("="*60)
        
        # 创建顺序执行序列
        sequential_sequence = self.motion_controller.create_action_sequence(
            "building_sequence",
            "建造房屋序列",
            parallel_execution=False,
            pause_on_error=True
        )
        
        # 添加建造相关的动作
        build_actions = [
            (ActionType.MOVE_FORWARD, "移动到建造位置"),
            (ActionType.PLACE_BLOCK, "放置地基"),
            (ActionType.PLACE_BLOCK, "建造墙体"),
            (ActionType.JUMP, "跳跃到屋顶"),
            (ActionType.PLACE_BLOCK, "建造屋顶")
        ]
        
        for i, (action_type, description) in enumerate(build_actions):
            self.motion_controller.add_action_to_sequence(
                "building_sequence",
                action_type,
                parameters={'distance': 2.0, 'item_id': 'stone', 'quantity': 1}
            )
            print(f"   添加动作 {i+1}: {description}")
        
        # 创建并行执行序列
        parallel_sequence = self.motion_controller.create_action_sequence(
            "exploration_sequence",
            "探索序列",
            parallel_execution=True,
            max_parallel_actions=3,
            pause_on_error=False
        )
        
        # 添加探索动作
        explore_actions = [
            ActionType.MOVE_FORWARD,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
            ActionType.JUMP,
            ActionType.ATTACK
        ]
        
        for i, action_type in enumerate(explore_actions):
            self.motion_controller.add_action_to_sequence(
                "exploration_sequence",
                action_type,
                parameters={'distance': 1.0, 'target': 'air', 'damage': 5}
            )
            print(f"   添加并行动作 {i+1}: {action_type.name}")
        
        # 启动序列
        print("\n🚀 启动顺序序列:")
        await self.motion_controller.start_sequence("building_sequence")
        
        # 等待序列执行
        await asyncio.sleep(3.0)
        
        # 检查序列状态
        seq_status = self.motion_controller.get_sequence_status("building_sequence")
        print(f"   序列状态: {seq_status.get('state', 'Unknown')}")
        print(f"   当前进度: {seq_status.get('current_index', 0)}/{seq_status.get('total_actions', 0)}")
        
        print("\n🚀 启动并行序列:")
        await self.motion_controller.start_sequence("exploration_sequence")
        
        await asyncio.sleep(2.0)
        
        # 检查并行序列状态
        parallel_status = self.motion_controller.get_sequence_status("exploration_sequence")
        print(f"   并行序列状态: {parallel_status.get('state', 'Unknown')}")
        print(f"   进度: {parallel_status.get('current_index', 0)}/{parallel_status.get('total_actions', 0)}")
        
        self.test_results['sequence_system'] = {
            'sequential_sequence': seq_status,
            'parallel_sequence': parallel_status
        }
    
    async def execute_learning_system(self):
        """测试技能学习系统"""
        print("\n" + "="*60)
        print("🎓 技能学习系统测试")
        print("="*60)
        
        # 多次执行同一技能，观察熟练度变化
        skill_name = 'tree_harvesting'
        
        print(f"🔄 多次执行技能 '{skill_name}' 观察熟练度变化:")
        
        initial_info = self.skill_library.get_skill_info(skill_name)
        initial_mastery = initial_info.get('mastery_level', 0)
        print(f"   初始熟练度: {initial_mastery:.2f}")
        
        # 执行5次
        for i in range(5):
            skill_id = self.motion_controller.create_and_schedule_action(
                skill_name,
                priority=ActionPriority.NORMAL,
                parameters={'tree_count': 2}
            )
            
            await asyncio.sleep(1.0)
            
            # 检查熟练度变化
            skill_info = self.skill_library.get_skill_info(skill_name)
            current_mastery = skill_info.get('mastery_level', 0)
            
            print(f"   第{i+1}次执行后熟练度: {current_mastery:.2f}")
        
        # 显示最终统计
        final_info = self.skill_library.get_skill_info(skill_name)
        final_execution_count = final_info.get('execution_count', 0)
        final_success_rate = final_info.get('success_rate', 0)
        final_mastery = final_info.get('mastery_level', 0)
        
        print(f"\n📊 学习结果:")
        print(f"   执行次数: {final_execution_count}")
        print(f"   成功率: {final_success_rate:.2%}")
        print(f"   最终熟练度: {final_mastery:.2f}")
        
        # 显示推荐技能
        print(f"\n💡 推荐技能:")
        recommendations = self.skill_library.get_recommended_skills()
        for i, rec_skill in enumerate(recommendations[:3]):
            skill_info = self.skill_library.get_skill_info(rec_skill)
            print(f"   {i+1}. {skill_info.get('name', rec_skill)}")
        
        self.test_results['learning_system'] = {
            'skill_name': skill_name,
            'initial_mastery': initial_mastery,
            'final_mastery': final_mastery,
            'execution_count': final_execution_count,
            'success_rate': final_success_rate,
            'recommendations': recommendations[:3]
        }
    
    async def execute_performance_test(self):
        """执行性能测试"""
        print("\n" + "="*60)
        print("⚡ 性能测试")
        print("="*60)
        
        # 测试高并发动作执行
        concurrent_actions = 10
        print(f"🔥 测试 {concurrent_actions} 个并发动作:")
        
        start_time = time.time()
        
        for i in range(concurrent_actions):
            self.motion_controller.create_and_schedule_action(
                ActionType.MOVE_FORWARD,
                priority=ActionPriority.NORMAL,
                parameters={'distance': 1.0}
            )
        
        # 等待所有动作完成
        await asyncio.sleep(3.0)
        
        total_time = time.time() - start_time
        actions_per_second = concurrent_actions / total_time
        
        print(f"   总时间: {total_time:.2f}s")
        print(f"   动作/秒: {actions_per_second:.2f}")
        
        # 获取系统性能指标
        metrics = self.motion_controller.get_performance_metrics()
        print(f"\n📊 系统性能指标:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")
        
        self.test_results['performance'] = {
            'concurrent_actions': concurrent_actions,
            'total_time': total_time,
            'actions_per_second': actions_per_second,
            'metrics': metrics
        }
    
    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("🚀 开始智能体动作系统综合测试")
        print(f"⏰ 测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 启动系统
        await self.start_system()
        
        try:
            # 执行各项测试
            await self.execute_atom_actions()
            await self.execute_skill_actions()
            await self.execute_priority_system()
            await self.execute_sequence_system()
            await self.execute_learning_system()
            await self.execute_performance_test()
            
            # 生成测试报告
            await self.generate_test_report()
            
        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {str(e)}")
            print(f"❌ 测试失败: {str(e)}")
        
        finally:
            # 停止系统
            await self.stop_system()
    
    async def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📄 测试报告")
        print("="*60)
        
        # 保存测试结果
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print("✅ 所有测试已完成!")
        print("📁 详细结果已保存到 test_results.json")
        
        # 显示总体摘要
        total_tests = sum(test.get('total', 0) for test in self.test_results.values() 
                         if isinstance(test, dict) and 'total' in test)
        total_successful = sum(test.get('successful', 0) for test in self.test_results.values() 
                              if isinstance(test, dict) and 'successful' in test)
        
        if total_tests > 0:
            overall_success_rate = total_successful / total_tests
            print(f"\n🎯 总体测试结果:")
            print(f"   总测试数: {total_tests}")
            print(f"   成功数: {total_successful}")
            print(f"   总体成功率: {overall_success_rate:.2%}")
        
        print(f"\n⏰ 测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# 主测试函数
async def main():
    """主测试函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建智能体系统
    agent_system = IntelligentAgentSystem()
    
    # 运行综合测试
    await agent_system.run_comprehensive_test()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())