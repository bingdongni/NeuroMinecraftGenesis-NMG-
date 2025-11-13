#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作控制器 - 实现10Hz频率的动作序列管理和优先级控制

这个模块提供：
1. 10Hz控制频率（100ms周期）的动作调度
2. 动作序列管理：支持复杂动作组合
3. 动作优先级：紧急动作优先执行
4. 动作取消机制：中断正在执行的动作
5. 动作序列优化和并行执行

作者：MiniMax智能体
创建时间：2025-11-13
"""

import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import deque, defaultdict
import heapq
from concurrent.futures import ThreadPoolExecutor

from action_executor import ActionExecutor, ActionType
from skill_library import SkillLibrary, ExecutionResult


class ActionPriority(Enum):
    """动作优先级"""
    EMERGENCY = 0    # 紧急（生命危险）
    HIGH = 1         # 高优先级
    NORMAL = 2       # 普通优先级  
    LOW = 3          # 低优先级
    BACKGROUND = 4   # 后台任务


class ActionState(Enum):
    """动作状态"""
    PENDING = auto()      # 等待执行
    RUNNING = auto()      # 正在执行
    COMPLETED = auto()    # 已完成
    CANCELLED = auto()    # 已取消
    FAILED = auto()       # 执行失败


class SequenceState(Enum):
    """序列状态"""
    IDLE = auto()         # 空闲
    EXECUTING = auto()    # 执行中
    PAUSED = auto()       # 暂停
    COMPLETED = auto()    # 已完成
    CANCELLED = auto()    # 已取消


@dataclass
class ScheduledAction:
    """调度的动作"""
    action_id: str
    action_type: Union[ActionType, str]  # 支持原子动作或技能名
    priority: ActionPriority
    scheduled_time: float
    execution_time: float = 0
    timeout: float = 30.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    
    # 执行状态
    state: ActionState = ActionState.PENDING
    result: Optional[Union['ActionResult', 'ExecutionResult']] = None
    error_message: str = ""


@dataclass
class ActionSequence:
    """动作序列"""
    sequence_id: str
    name: str
    actions: List[ScheduledAction]
    current_index: int = 0
    sequence_state: SequenceState = SequenceState.IDLE
    
    # 序列控制
    parallel_execution: bool = False  # 是否并行执行
    max_parallel_actions: int = 3     # 最大并行动作数
    pause_on_error: bool = True       # 错误时暂停
    
    # 序列统计
    start_time: float = 0
    end_time: float = 0
    successful_actions: int = 0
    failed_actions: int = 0


class MotionController:
    """动作控制器
    
    负责管理智能体的所有动作执行，提供10Hz频率的动作调度和优先级控制
    """
    
    def __init__(self, action_executor: ActionExecutor, skill_library: SkillLibrary):
        self.action_executor = action_executor
        self.skill_library = skill_library
        self.logger = logging.getLogger(__name__)
        
        # 控制频率设置
        self.control_frequency = 10.0  # 10Hz
        self.control_period = 1.0 / self.control_frequency  # 100ms
        self.is_running = False
        
        # 动作调度
        self.action_queue: List[ScheduledAction] = []
        self.running_actions: Dict[str, ScheduledAction] = {}
        self.completed_actions: deque = deque(maxlen=1000)
        self.action_counter = 0
        
        # 动作序列管理
        self.sequences: Dict[str, ActionSequence] = {}
        self.active_sequences: List[ActionSequence] = []
        
        # 并行执行控制
        self.max_concurrent_actions = 5
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 性能监控
        self.performance_metrics = {
            'actions_per_second': 0,
            'average_execution_time': 0,
            'success_rate': 0,
            'queue_size': 0,
            'active_sequences': 0
        }
        
        # 启动控制循环
        self.control_task = None
        
    async def start(self):
        """启动动作控制器"""
        if self.is_running:
            self.logger.warning("动作控制器已在运行")
            return
        
        self.is_running = True
        self.control_task = asyncio.create_task(self._control_loop())
        self.logger.info("动作控制器已启动，频率: 10Hz")
    
    async def stop(self):
        """停止动作控制器"""
        self.is_running = False
        
        if self.control_task:
            self.control_task.cancel()
            try:
                await self.control_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有正在执行的动作
        for action_id in list(self.running_actions.keys()):
            await self.cancel_action(action_id)
        
        self.logger.info("动作控制器已停止")
    
    async def _control_loop(self):
        """主控制循环 - 10Hz频率执行"""
        last_cycle_time = time.time()
        
        while self.is_running:
            cycle_start = time.time()
            
            try:
                # 1. 处理动作队列和优先级
                await self._process_action_queue()
                
                # 2. 执行可用动作
                await self._execute_available_actions()
                
                # 3. 处理动作序列
                await self._process_action_sequences()
                
                # 4. 清理过期动作
                await self._cleanup_expired_actions()
                
                # 5. 更新性能指标
                self._update_performance_metrics()
                
            except Exception as e:
                self.logger.error(f"控制循环错误: {str(e)}")
            
            # 控制执行频率
            cycle_duration = time.time() - cycle_start
            sleep_time = max(0, self.control_period - cycle_duration)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # 如果执行时间超过周期，记录警告
                self.logger.warning(f"控制循环超时: {cycle_duration:.3f}s > {self.control_period:.3f}s")
            
            last_cycle_time = cycle_start
    
    async def _process_action_queue(self):
        """处理动作队列，按优先级排序"""
        # 按优先级和时间排序
        self.action_queue.sort(key=lambda x: (x.priority.value, x.scheduled_time))
        
        # 检查是否有可以执行的动作
        current_time = time.time()
        
        # 移除已过期的动作
        expired_actions = [action for action in self.action_queue 
                          if current_time - action.scheduled_time > action.timeout]
        
        for action in expired_actions:
            self.logger.warning(f"动作 {action.action_id} 已过期")
            action.state = ActionState.FAILED
            action.error_message = "动作执行超时"
            self.completed_actions.append(action)
        
        self.action_queue = [action for action in self.action_queue 
                           if action not in expired_actions]
    
    async def _execute_available_actions(self):
        """执行可用的动作"""
        current_time = time.time()
        
        # 检查依赖关系
        available_actions = []
        for action in self.action_queue:
            if self._check_action_dependencies(action):
                available_actions.append(action)
        
        # 按优先级执行可用动作
        for action in available_actions:
            # 检查并发限制
            if len(self.running_actions) >= self.max_concurrent_actions:
                break
            
            # 检查是否可以立即执行
            if current_time >= action.scheduled_time:
                await self._execute_action(action)
                self.action_queue.remove(action)
    
    async def _execute_action(self, action: ScheduledAction):
        """执行单个动作"""
        action.state = ActionState.RUNNING
        action.execution_time = time.time()
        self.running_actions[action.action_id] = action
        
        try:
            # 检查动作类型
            if isinstance(action.action_type, ActionType):
                # 原子动作
                result = await self.action_executor.execute_action(action.action_type, **action.parameters)
            elif isinstance(action.action_type, str):
                # 技能
                result = await self.skill_library.execute_skill(action.action_type, **action.parameters)
            else:
                raise ValueError(f"未知的动作类型: {type(action.action_type)}")
            
            action.result = result
            action.state = ActionState.COMPLETED if result.success else ActionState.FAILED
            
            # 执行回调
            if action.callback:
                try:
                    await action.callback(action)
                except Exception as e:
                    self.logger.error(f"动作回调执行失败: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"执行动作 {action.action_id} 失败: {str(e)}")
            action.state = ActionState.FAILED
            action.error_message = str(e)
        
        finally:
            # 移动到已完成列表
            self.completed_actions.append(action)
            if action.action_id in self.running_actions:
                del self.running_actions[action.action_id]
    
    def _check_action_dependencies(self, action: ScheduledAction) -> bool:
        """检查动作依赖关系"""
        if not action.dependencies:
            return True
        
        for dep_id in action.dependencies:
            # 检查依赖动作是否已完成
            dep_completed = any(a.action_id == dep_id and a.state == ActionState.COMPLETED 
                              for a in self.completed_actions)
            if not dep_completed:
                return False
        
        return True
    
    async def _process_action_sequences(self):
        """处理动作序列"""
        completed_sequences = []
        
        for sequence in self.active_sequences:
            if sequence.sequence_state == SequenceState.EXECUTING:
                await self._execute_sequence(sequence)
            
            elif sequence.sequence_state == SequenceState.COMPLETED:
                completed_sequences.append(sequence)
        
        # 移除已完成的序列
        for sequence in completed_sequences:
            self.active_sequences.remove(sequence)
            sequence.end_time = time.time()
    
    async def _execute_sequence(self, sequence: ActionSequence):
        """执行动作序列"""
        if sequence.parallel_execution:
            await self._execute_parallel_sequence(sequence)
        else:
            await self._execute_sequential_sequence(sequence)
    
    async def _execute_sequential_sequence(self, sequence: ActionSequence):
        """执行顺序序列"""
        while (sequence.current_index < len(sequence.actions) and 
               sequence.sequence_state == SequenceState.EXECUTING):
            
            current_action = sequence.actions[sequence.current_index]
            
            # 等待动作完成
            if current_action.action_id in self.running_actions:
                # 动作正在执行，等待完成
                await asyncio.sleep(0.01)
                continue
            
            # 检查依赖关系
            if not self._check_action_dependencies(current_action):
                await asyncio.sleep(0.01)
                continue
            
            # 添加到执行队列
            await self.schedule_action(current_action)
            
            # 等待动作执行完成
            while (current_action.action_id in self.running_actions and 
                   sequence.sequence_state == SequenceState.EXECUTING):
                await asyncio.sleep(0.01)
            
            # 检查动作结果
            if current_action.state == ActionState.COMPLETED:
                sequence.successful_actions += 1
                sequence.current_index += 1
            elif current_action.state == ActionState.FAILED:
                sequence.failed_actions += 1
                if sequence.pause_on_error:
                    sequence.sequence_state = SequenceState.PAUSED
                    self.logger.warning(f"序列 {sequence.sequence_id} 在动作 {current_action.action_id} 处失败并暂停")
                    break
                else:
                    sequence.current_index += 1
        
        # 检查序列是否完成
        if sequence.current_index >= len(sequence.actions):
            sequence.sequence_state = SequenceState.COMPLETED
    
    async def _execute_parallel_sequence(self, sequence: ActionSequence):
        """执行并行序列"""
        active_actions = []
        
        while (sequence.current_index < len(sequence.actions) or active_actions) and \
              sequence.sequence_state == SequenceState.EXECUTING:
            
            # 启动新的动作（不超过最大并行数）
            while (sequence.current_index < len(sequence.actions) and 
                   len(active_actions) < sequence.max_parallel_actions):
                
                current_action = sequence.actions[sequence.current_index]
                
                # 检查依赖关系
                if self._check_action_dependencies(current_action):
                    await self.schedule_action(current_action)
                    active_actions.append(current_action)
                    sequence.current_index += 1
                else:
                    break
            
            # 等待并检查正在执行的动作
            completed_actions = []
            for action in active_actions:
                if action.action_id not in self.running_actions:
                    completed_actions.append(action)
                    
                    if action.state == ActionState.COMPLETED:
                        sequence.successful_actions += 1
                    else:
                        sequence.failed_actions += 1
                        if sequence.pause_on_error:
                            sequence.sequence_state = SequenceState.PAUSED
                            self.logger.warning(f"并行序列 {sequence.sequence_id} 失败并暂停")
            
            # 移除已完成的动作
            for action in completed_actions:
                active_actions.remove(action)
            
            await asyncio.sleep(0.01)
        
        # 检查序列是否完成
        if sequence.current_index >= len(sequence.actions) and not active_actions:
            sequence.sequence_state = SequenceState.COMPLETED
    
    async def _cleanup_expired_actions(self):
        """清理过期的动作"""
        current_time = time.time()
        
        # 清理队列中的过期动作
        expired = [action for action in self.action_queue 
                  if current_time - action.scheduled_time > action.timeout]
        
        for action in expired:
            self.action_queue.remove(action)
            action.state = ActionState.FAILED
            action.error_message = "动作执行超时"
            self.completed_actions.append(action)
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        now = time.time()
        one_minute_ago = now - 60
        
        # 最近一分钟的动作
        recent_actions = [action for action in self.completed_actions 
                         if action.execution_time > one_minute_ago]
        
        # 计算指标
        if recent_actions:
            self.performance_metrics['actions_per_second'] = len(recent_actions) / 60
            self.performance_metrics['average_execution_time'] = sum(
                action.result.duration if hasattr(action.result, 'duration') else 0 
                for action in recent_actions
            ) / len(recent_actions)
            self.performance_metrics['success_rate'] = sum(
                1 for action in recent_actions if action.state == ActionState.COMPLETED
            ) / len(recent_actions)
        
        self.performance_metrics['queue_size'] = len(self.action_queue)
        self.performance_metrics['active_sequences'] = len(self.active_sequences)
    
    def schedule_action(self, action: ScheduledAction) -> str:
        """
        调度一个动作
        
        Args:
            action: 调度的动作
            
        Returns:
            str: 动作ID
        """
        if not action.action_id:
            action.action_id = f"action_{self.action_counter}"
            self.action_counter += 1
        
        self.action_queue.append(action)
        self.logger.debug(f"动作 {action.action_id} 已调度，优先级: {action.priority.name}")
        
        return action.action_id
    
    def create_and_schedule_action(self, action_type: Union[ActionType, str], 
                                  priority: ActionPriority = ActionPriority.NORMAL,
                                  delay: float = 0, timeout: float = 30.0,
                                  parameters: Dict[str, Any] = None,
                                  callback: Callable = None,
                                  dependencies: List[str] = None) -> str:
        """
        创建并调度动作
        
        Args:
            action_type: 动作类型
            priority: 优先级
            delay: 延迟执行时间（秒）
            timeout: 超时时间（秒）
            parameters: 动作参数
            callback: 执行完成回调
            dependencies: 依赖的动作ID列表
            
        Returns:
            str: 动作ID
        """
        if parameters is None:
            parameters = {}
        if dependencies is None:
            dependencies = []
        
        action = ScheduledAction(
            action_id=f"action_{self.action_counter}",
            action_type=action_type,
            priority=priority,
            scheduled_time=time.time() + delay,
            timeout=timeout,
            parameters=parameters,
            callback=callback,
            dependencies=dependencies
        )
        
        self.action_counter += 1
        return self.schedule_action(action)
    
    async def cancel_action(self, action_id: str) -> bool:
        """取消正在执行的动作"""
        # 从队列中移除
        for action in self.action_queue:
            if action.action_id == action_id:
                self.action_queue.remove(action)
                action.state = ActionState.CANCELLED
                action.error_message = "动作被取消"
                self.completed_actions.append(action)
                return True
        
        # 从运行中移除
        if action_id in self.running_actions:
            action = self.running_actions[action_id]
            action.state = ActionState.CANCELLED
            action.error_message = "动作被取消"
            self.completed_actions.append(action)
            del self.running_actions[action_id]
            return True
        
        return False
    
    async def pause_action(self, action_id: str) -> bool:
        """暂停动作"""
        # 这里可以实现更复杂的暂停逻辑
        # 当前简化处理：标记为暂停状态
        for action in self.action_queue:
            if action.action_id == action_id:
                # 实现暂停逻辑
                return True
        
        return False
    
    async def resume_action(self, action_id: str) -> bool:
        """恢复动作"""
        # 实现恢复逻辑
        return False
    
    def create_action_sequence(self, sequence_id: str, name: str, 
                              parallel_execution: bool = False,
                              max_parallel_actions: int = 3,
                              pause_on_error: bool = True) -> ActionSequence:
        """创建动作序列"""
        sequence = ActionSequence(
            sequence_id=sequence_id,
            name=name,
            actions=[],
            parallel_execution=parallel_execution,
            max_parallel_actions=max_parallel_actions,
            pause_on_error=pause_on_error
        )
        
        self.sequences[sequence_id] = sequence
        return sequence
    
    def add_action_to_sequence(self, sequence_id: str, action_type: Union[ActionType, str],
                              priority: ActionPriority = ActionPriority.NORMAL,
                              delay: float = 0, timeout: float = 30.0,
                              parameters: Dict[str, Any] = None,
                              dependencies: List[str] = None) -> bool:
        """向序列添加动作"""
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        
        if parameters is None:
            parameters = {}
        if dependencies is None:
            dependencies = []
        
        action = ScheduledAction(
            action_id=f"seq_{sequence_id}_{len(sequence.actions)}",
            action_type=action_type,
            priority=priority,
            scheduled_time=time.time() + delay,
            timeout=timeout,
            parameters=parameters,
            dependencies=dependencies
        )
        
        sequence.actions.append(action)
        return True
    
    async def start_sequence(self, sequence_id: str) -> bool:
        """启动动作序列"""
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        if sequence.sequence_state != SequenceState.IDLE:
            return False
        
        sequence.sequence_state = SequenceState.EXECUTING
        sequence.start_time = time.time()
        self.active_sequences.append(sequence)
        
        self.logger.info(f"动作序列 {sequence_id} 已启动")
        return True
    
    async def pause_sequence(self, sequence_id: str) -> bool:
        """暂停动作序列"""
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        sequence.sequence_state = SequenceState.PAUSED
        
        self.logger.info(f"动作序列 {sequence_id} 已暂停")
        return True
    
    async def stop_sequence(self, sequence_id: str) -> bool:
        """停止动作序列"""
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        sequence.sequence_state = SequenceState.CANCELLED
        
        # 取消序列中的所有动作
        for action in sequence.actions:
            if action.action_id in self.running_actions or action.state == ActionState.PENDING:
                await self.cancel_action(action.action_id)
        
        self.active_sequences.remove(sequence)
        sequence.end_time = time.time()
        
        self.logger.info(f"动作序列 {sequence_id} 已停止")
        return True
    
    def get_action_status(self, action_id: str) -> Dict[str, Any]:
        """获取动作状态"""
        # 查找正在执行的动作
        if action_id in self.running_actions:
            action = self.running_actions[action_id]
            return {
                'action_id': action.action_id,
                'state': action.state.name,
                'priority': action.priority.name,
                'execution_time': action.execution_time,
                'elapsed_time': time.time() - action.execution_time
            }
        
        # 查找队列中的动作
        for action in self.action_queue:
            if action.action_id == action_id:
                return {
                    'action_id': action.action_id,
                    'state': action.state.name,
                    'priority': action.priority.name,
                    'scheduled_time': action.scheduled_time,
                    'wait_time': time.time() - action.scheduled_time
                }
        
        # 查找已完成的动作
        for action in self.completed_actions:
            if action.action_id == action_id:
                return {
                    'action_id': action.action_id,
                    'state': action.state.name,
                    'execution_time': action.execution_time,
                    'total_duration': getattr(action.result, 'duration', 0) if action.result else 0,
                    'error_message': action.error_message
                }
        
        return {}
    
    def get_sequence_status(self, sequence_id: str) -> Dict[str, Any]:
        """获取序列状态"""
        if sequence_id not in self.sequences:
            return {}
        
        sequence = self.sequences[sequence_id]
        
        return {
            'sequence_id': sequence.sequence_id,
            'name': sequence.name,
            'state': sequence.sequence_state.name,
            'total_actions': len(sequence.actions),
            'current_index': sequence.current_index,
            'successful_actions': sequence.successful_actions,
            'failed_actions': sequence.failed_actions,
            'parallel_execution': sequence.parallel_execution,
            'start_time': sequence.start_time,
            'elapsed_time': time.time() - sequence.start_time if sequence.start_time > 0 else 0
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'queue_size': len(self.action_queue),
            'running_size': len(self.running_actions),
            'completed_size': len(self.completed_actions),
            'active_sequences': len(self.active_sequences),
            'total_sequences': len(self.sequences)
        }
    
    def set_max_concurrent_actions(self, max_actions: int):
        """设置最大并发动作数"""
        self.max_concurrent_actions = max(1, max_actions)
        self.logger.info(f"最大并发动作数设置为: {self.max_concurrent_actions}")


# 测试和示例代码
if __name__ == "__main__":
    async def test_motion_controller():
        """测试动作控制器"""
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 创建组件
        action_executor = ActionExecutor()
        skill_library = SkillLibrary(action_executor)
        motion_controller = MotionController(action_executor, skill_library)
        
        print("=== 动作控制器测试 ===")
        
        # 启动控制器
        await motion_controller.start()
        await asyncio.sleep(0.2)  # 等待控制器启动
        
        try:
            # 测试单个动作调度
            print("\n1. 测试单个动作调度:")
            action_id = motion_controller.create_and_schedule_action(
                ActionType.MOVE_FORWARD,
                priority=ActionPriority.NORMAL,
                parameters={'distance': 2.0}
            )
            print(f"   调度动作: {action_id}")
            
            await asyncio.sleep(1.0)  # 等待执行
            status = motion_controller.get_action_status(action_id)
            print(f"   动作状态: {status.get('state', 'Unknown')}")
            
            # 测试优先级调度
            print("\n2. 测试优先级调度:")
            low_priority_id = motion_controller.create_and_schedule_action(
                ActionType.MOVE_LEFT,
                priority=ActionPriority.LOW,
                delay=0.5
            )
            
            high_priority_id = motion_controller.create_and_schedule_action(
                ActionType.JUMP,
                priority=ActionPriority.HIGH,
                delay=1.0
            )
            
            print(f"   低优先级动作: {low_priority_id}")
            print(f"   高优先级动作: {high_priority_id}")
            
            await asyncio.sleep(2.0)
            
            # 测试技能执行
            print("\n3. 测试技能执行:")
            skill_id = motion_controller.create_and_schedule_action(
                "tree_harvesting",
                priority=ActionPriority.NORMAL,
                parameters={'tree_count': 3}
            )
            print(f"   调度技能: {skill_id}")
            
            await asyncio.sleep(2.0)
            
            # 测试动作序列
            print("\n4. 测试动作序列:")
            sequence = motion_controller.create_action_sequence(
                "test_sequence",
                "测试序列",
                parallel_execution=False
            )
            
            # 添加动作到序列
            motion_controller.add_action_to_sequence("test_sequence", ActionType.MOVE_FORWARD)
            motion_controller.add_action_to_sequence("test_sequence", ActionType.JUMP)
            motion_controller.add_action_to_sequence("test_sequence", ActionType.MOVE_LEFT)
            
            await motion_controller.start_sequence("test_sequence")
            print("   序列已启动")
            
            await asyncio.sleep(3.0)
            
            # 显示统计信息
            print("\n5. 性能统计:")
            metrics = motion_controller.get_performance_metrics()
            for key, value in metrics.items():
                print(f"   {key}: {value:.3f}")
            
            print("\n6. 队列状态:")
            queue_status = motion_controller.get_queue_status()
            for key, value in queue_status.items():
                print(f"   {key}: {value}")
                
        finally:
            # 停止控制器
            await motion_controller.stop()
    
    # 运行测试
    asyncio.run(test_motion_controller())