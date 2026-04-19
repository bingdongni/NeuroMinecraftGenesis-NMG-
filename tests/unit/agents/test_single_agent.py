#!/usr/bin/env python3
"""
单智能体系统完整单元测试
"""

import pytest
import numpy as np
import asyncio


class TestActionExecutor:
    """动作执行器测试类"""

    def test_initialization(self):
        """测试动作执行器初始化"""
        from agents.single.action_executor import ActionExecutor

        executor = ActionExecutor()

        assert executor is not None
        assert executor.action_queue is not None

    @pytest.mark.asyncio
    async def test_action_execution(self):
        """测试动作执行"""
        from agents.single.action_executor import ActionExecutor, ActionType

        executor = ActionExecutor()

        # 执行动作
        result = await executor.execute_action(
            ActionType.MOVE_FORWARD,
            distance=3.0
        )

        assert result is not None
        assert hasattr(result, 'success')

    @pytest.mark.asyncio
    async def test_atomic_actions(self):
        """测试原子动作"""
        from agents.single.action_executor import ActionExecutor, ActionType

        executor = ActionExecutor()

        # 测试所有动作类型
        action_types = [
            ActionType.MOVE_FORWARD,
            ActionType.MOVE_BACKWARD,
            ActionType.JUMP,
            ActionType.ATTACK
        ]

        for action_type in action_types:
            result = await executor.execute_action(action_type)
            assert result is not None


class TestMotionController:
    """运动控制器测试类"""

    def test_initialization(self):
        """测试运动控制器初始化"""
        from agents.single.motion_controller import MotionController
        from agents.single.action_executor import ActionExecutor
        from agents.single.skill_library import SkillLibrary

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)
        controller = MotionController(executor, skill_lib)

        assert controller is not None
        assert controller.is_running == False

    @pytest.mark.asyncio
    async def test_controller_start_stop(self):
        """测试控制器启动停止"""
        from agents.single.motion_controller import MotionController
        from agents.single.action_executor import ActionExecutor
        from agents.single.skill_library import SkillLibrary

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)
        controller = MotionController(executor, skill_lib)

        await controller.start()
        assert controller.is_running == True

        await controller.stop()
        assert controller.is_running == False

    @pytest.mark.asyncio
    async def test_action_scheduling(self):
        """测试动作调度"""
        from agents.single.motion_controller import MotionController, ActionPriority
        from agents.single.action_executor import ActionExecutor, ActionType
        from agents.single.skill_library import SkillLibrary

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)
        controller = MotionController(executor, skill_lib)

        await controller.start()

        # 调度动作
        action_id = controller.create_and_schedule_action(
            ActionType.MOVE_FORWARD,
            priority=ActionPriority.NORMAL
        )

        assert action_id is not None

        await controller.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """测试优先级排序"""
        from agents.single.motion_controller import MotionController, ActionPriority
        from agents.single.action_executor import ActionExecutor, ActionType
        from agents.single.skill_library import SkillLibrary

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)
        controller = MotionController(executor, skill_lib)

        await controller.start()

        # 调度不同优先级的动作
        low_action = controller.create_and_schedule_action(
            ActionType.MOVE_LEFT,
            priority=ActionPriority.LOW
        )

        high_action = controller.create_and_schedule_action(
            ActionType.JUMP,
            priority=ActionPriority.HIGH
        )

        # 验证高优先级先执行
        await asyncio.sleep(0.1)

        high_status = controller.get_action_status(high_action)
        low_status = controller.get_action_status(low_action)

        await controller.stop()

        assert high_status is not None


class TestSkillLibrary:
    """技能库测试类"""

    def test_initialization(self):
        """测试技能库初始化"""
        from agents.single.skill_library import SkillLibrary
        from agents.single.action_executor import ActionExecutor

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)

        assert skill_lib is not None
        assert len(skill_lib.skills) > 0

    @pytest.mark.asyncio
    async def test_skill_execution(self):
        """测试技能执行"""
        from agents.single.skill_library import SkillLibrary
        from agents.single.action_executor import ActionExecutor

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)

        # 执行技能
        result = await skill_lib.execute_skill("tree_harvesting", tree_count=2)

        assert result is not None
        assert hasattr(result, 'success')

    def test_skill_creation(self):
        """测试技能创建"""
        from agents.single.skill_library import SkillLibrary, Skill
        from agents.single.action_executor import ActionExecutor

        executor = ActionExecutor()
        skill_lib = SkillLibrary(executor)

        # 创建新技能
        new_skill = Skill(
            name="test_skill",
            category="test",
            required_actions=[],
            parameters={}
        )

        skill_lib.register_skill(new_skill)

        assert "test_skill" in skill_lib.skills


class TestIntelligentAgent:
    """智能体测试类"""

    def test_initialization(self):
        """测试智能体初始化"""
        from agents.single.intelligent_agent_system import IntelligentAgentSystem

        agent = IntelligentAgentSystem()

        assert agent is not None
        assert hasattr(agent, 'action_executor')
        assert hasattr(agent, 'skill_library')
        assert hasattr(agent, 'motion_controller')

    @pytest.mark.asyncio
    async def test_agent_start_stop(self):
        """测试智能体启动停止"""
        from agents.single.intelligent_agent_system import IntelligentAgentSystem

        agent = IntelligentAgentSystem()

        await agent.start_system()
        assert agent.system_started == True

        await agent.stop_system()
        assert agent.system_started == False

    @pytest.mark.asyncio
    async def test_atom_actions(self):
        """测试原子动作"""
        from agents.single.intelligent_agent_system import IntelligentAgentSystem

        agent = IntelligentAgentSystem()

        await agent.start_system()
        await agent.execute_atom_actions()

        assert 'atom_actions' in agent.test_results

        await agent.stop_system()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
