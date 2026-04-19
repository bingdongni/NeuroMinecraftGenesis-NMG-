#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体动作系统简化测试

用于验证核心功能：
1. 27种原子动作
2. 技能系统
3. 动作优先级
4. 基本性能

作者：MiniMax智能体
创建时间：2025-11-13
"""

import asyncio
import time
import logging

from action_executor import ActionExecutor, ActionType
from skill_library import SkillLibrary
from motion_controller import MotionController, ActionPriority


async def test_basic_functionality():
    """测试基本功能"""
    print("🚀 开始基本功能测试")
    
    # 创建组件
    executor = ActionExecutor()
    skill_lib = SkillLibrary(executor)
    controller = MotionController(executor, skill_lib)
    
    print(f"✅ 系统组件创建完成")
    print(f"   技能数量: {len(skill_lib.skills)}")
    print(f"   原子动作: {len(list(ActionType))} 种")
    
    # 测试原子动作
    print("\n🧪 测试原子动作执行:")
    test_actions = [
        (ActionType.MOVE_FORWARD, "向前移动"),
        (ActionType.JUMP, "跳跃"),
        (ActionType.ATTACK, "攻击"),
        (ActionType.PLACE_BLOCK, "放置方块")
    ]
    
    # 添加测试物品
    executor.inventory = {"stone": 10, "apple": 3}
    
    for action_type, description in test_actions:
        start_time = time.time()
        
        result = await executor.execute_action(
            action_type,
            distance=2.0 if "移动" in description else None,
            height=2.0 if "跳跃" in description else None,
            target="test" if "攻击" in description else None,
            item_id="stone" if "放置" in description else None,
            quantity=1 if "放置" in description else None
        )
        
        duration = time.time() - start_time
        print(f"   {description}: {'✅ 成功' if result.success else '❌ 失败'} ({duration:.3f}s)")
    
    # 测试技能系统
    print("\n🎯 测试技能系统:")
    skill_result = await skill_lib.execute_skill("tree_harvesting", tree_count=2)
    print(f"   树木采伐: {'✅ 成功' if skill_result.success else '❌ 失败'}")
    if skill_result.success:
        print(f"   性能得分: {skill_result.performance_score:.2f}")
    
    # 测试动作优先级
    print("\n⚡ 测试动作优先级:")
    await controller.start()
    
    # 调度不同优先级的动作
    high_prio = controller.create_and_schedule_action(
        ActionType.JUMP, priority=ActionPriority.HIGH
    )
    low_prio = controller.create_and_schedule_action(
        ActionType.MOVE_LEFT, priority=ActionPriority.LOW
    )
    
    await asyncio.sleep(1.0)
    
    high_status = controller.get_action_status(high_prio)
    low_status = controller.get_action_status(low_prio)
    
    print(f"   高优先级动作: {high_status.get('state', 'Unknown')}")
    print(f"   低优先级动作: {low_status.get('state', 'Unknown')}")
    
    # 获取性能指标
    metrics = controller.get_performance_metrics()
    print(f"\n📊 性能指标:")
    print(f"   队列大小: {metrics['queue_size']:.0f}")
    print(f"   成功率: {metrics['success_rate']:.2%}")
    
    await controller.stop()
    print("✅ 基本功能测试完成")


async def test_skill_learning():
    """测试技能学习系统"""
    print("\n🎓 测试技能学习系统")
    
    executor = ActionExecutor()
    skill_lib = SkillLibrary(executor)
    
    # 多次执行同一技能
    skill_name = "basic_exploration"
    initial_info = skill_lib.get_skill_info(skill_name)
    
    print(f"初始熟练度: {initial_info.get('mastery_level', 0):.2f}")
    
    # 执行5次
    for i in range(5):
        result = await skill_lib.execute_skill(
            skill_name,
            exploration_radius=2,
            include_underground=False
        )
        
        if result.success:
            # 获取更新后的信息
            info = skill_lib.get_skill_info(skill_name)
            mastery = info.get('mastery_level', 0)
            print(f"第{i+1}次执行后熟练度: {mastery:.2f}")
        
        await asyncio.sleep(0.1)
    
    # 显示最终统计
    final_info = skill_lib.get_skill_info(skill_name)
    print(f"\n📊 学习结果:")
    print(f"   执行次数: {final_info.get('execution_count', 0)}")
    print(f"   成功率: {final_info.get('success_rate', 0):.2%}")
    print(f"   最终熟练度: {final_info.get('mastery_level', 0):.2f}")
    
    print("✅ 技能学习测试完成")


async def test_motion_sequences():
    """测试动作序列"""
    print("\n📋 测试动作序列")
    
    executor = ActionExecutor()
    skill_lib = SkillLibrary(executor)
    controller = MotionController(executor, skill_lib)
    
    await controller.start()
    
    # 创建序列
    sequence = controller.create_action_sequence(
        "test_sequence",
        "测试序列",
        parallel_execution=False
    )
    
    # 添加动作
    controller.add_action_to_sequence("test_sequence", ActionType.MOVE_FORWARD)
    controller.add_action_to_sequence("test_sequence", ActionType.JUMP)
    controller.add_action_to_sequence("test_sequence", ActionType.MOVE_LEFT)
    
    print(f"序列创建完成，包含 {len(sequence.actions)} 个动作")
    
    # 启动序列
    await controller.start_sequence("test_sequence")
    
    # 等待执行
    await asyncio.sleep(3.0)
    
    # 检查状态
    seq_status = controller.get_sequence_status("test_sequence")
    print(f"序列状态: {seq_status.get('state', 'Unknown')}")
    print(f"进度: {seq_status.get('current_index', 0)}/{seq_status.get('total_actions', 0)}")
    
    await controller.stop()
    print("✅ 动作序列测试完成")


async def main():
    """主测试函数"""
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)  # 只显示警告和错误
    
    print("=" * 60)
    print("🎮 智能体动作系统简化测试")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        await test_basic_functionality()
        await test_skill_learning()
        await test_motion_sequences()
        
        total_time = time.time() - start_time
        print(f"\n🎉 所有测试完成!")
        print(f"总用时: {total_time:.2f}秒")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())