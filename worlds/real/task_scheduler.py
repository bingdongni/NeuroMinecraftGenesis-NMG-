#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务调度器模块
=============

这个模块实现了智能体测试任务的高效调度系统。
负责管理任务队列、分配资源、处理并发执行，
确保测试任务的准时和有序执行。

核心功能：
- 智能任务调度和队列管理
- 资源分配和负载均衡
- 并发任务执行控制
- 任务优先级管理
- 异常处理和恢复机制

作者：AI研究团队
日期：2025-11-13
"""

import asyncio
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import uuid


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """任务数据类"""
    task_id: str
    name: str
    task_type: str
    priority: TaskPriority
    created_at: datetime
    scheduled_at: datetime
    timeout: int = 300  # 默认超时5分钟
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """初始化后的处理"""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat(),
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'result': str(self.result) if self.result else None,
            'error': self.error
        }


class TaskScheduler:
    """
    任务调度器类
    
    负责管理所有测试任务的调度、执行和监控。
    支持并发执行、优先级队列、依赖管理等功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化任务调度器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 任务管理
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # 执行控制
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self.running_tasks: Dict[str, Any] = {}
        
        # 任务类型处理器
        self.task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # 调度状态
        self.is_running = False
        self.scheduler_thread = None
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        self.logger.info("任务调度器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_concurrent_tasks': 4,
            'default_timeout': 300,  # 5分钟
            'max_retries': 3,
            'retry_delay': 5,  # 重试延迟5秒
            'queue_check_interval': 1,  # 队列检查间隔1秒
            'task_cleanup_interval': 3600,  # 任务清理间隔1小时
            'enable_monitoring': True,
            'enable_statistics': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TaskScheduler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 创建日志目录
            log_dir = Path('/workspace/worlds/real/logs')
            log_dir.mkdir(exist_ok=True)
            
            # 设置文件处理器
            file_handler = logging.FileHandler(
                log_dir / f'task_scheduler_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            # 设置控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _register_default_handlers(self):
        """注册默认的任务处理器"""
        self.task_handlers = {
            'image_classification': self._handle_image_classification,
            'object_detection': self._handle_object_detection,
            'scene_analysis': self._handle_scene_analysis,
            'cross_domain_transfer': self._handle_cross_domain_transfer,
            'adaptation_test': self._handle_adaptation_test,
            'performance_test': self._handle_performance_test,
            'stress_test': self._handle_stress_test
        }
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self.task_handlers[task_type] = handler
        self.logger.info(f"已注册任务处理器: {task_type}")
    
    def create_task(self, name: str, task_type: str, priority: TaskPriority = TaskPriority.NORMAL,
                   scheduled_at: Optional[datetime] = None, timeout: Optional[int] = None,
                   dependencies: Optional[List[str]] = None,
                   parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        创建新任务
        
        Args:
            name: 任务名称
            task_type: 任务类型
            priority: 任务优先级
            scheduled_at: 计划执行时间
            timeout: 超时时间（秒）
            dependencies: 依赖任务ID列表
            parameters: 任务参数
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        created_at = datetime.now()
        scheduled_time = scheduled_at or created_at
        
        task = Task(
            task_id=task_id,
            name=name,
            task_type=task_type,
            priority=priority,
            created_at=created_at,
            scheduled_at=scheduled_time,
            timeout=timeout or self.config['default_timeout'],
            dependencies=dependencies or [],
            parameters=parameters or {}
        )
        
        # 添加到任务字典
        self.tasks[task_id] = task
        
        # 添加到队列（如果可以立即执行）
        if self._can_execute_task(task):
            self._add_to_queue(task)
        
        self.stats['total_tasks'] += 1
        self.logger.info(f"创建任务: {name} ({task_id})")
        
        return task_id
    
    def _can_execute_task(self, task: Task) -> bool:
        """
        检查任务是否可以执行
        
        Args:
            task: 任务对象
            
        Returns:
            是否可以执行
        """
        # 检查依赖任务是否完成
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        
        # 检查是否已到执行时间
        if datetime.now() < task.scheduled_at:
            return False
        
        return True
    
    def _add_to_queue(self, task: Task):
        """将任务添加到队列"""
        # 使用负的优先级值（因为priority_queue是最小堆）
        priority_value = -task.priority.value
        self.task_queue.put((priority_value, task.created_at.timestamp(), task.task_id))
    
    def start_scheduler(self):
        """启动调度器"""
        if self.is_running:
            self.logger.warning("调度器已在运行中")
            return
        
        self.is_running = True
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # 启动监控线程
        if self.config['enable_monitoring']:
            self._start_monitoring()
        
        self.logger.info("任务调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        
        # 取消所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            self.cancel_task(task_id)
        
        # 等待调度线程结束
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # 关闭执行器
        self.executor.shutdown(wait=True)
        
        self.logger.info("任务调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                self._process_task_queue()
                self._check_timeout_tasks()
                self._retry_failed_tasks()
                time.sleep(self.config['queue_check_interval'])
            except Exception as e:
                self.logger.error(f"调度器循环错误: {e}")
                time.sleep(5)
    
    def _process_task_queue(self):
        """处理任务队列"""
        # 处理队列中的任务
        processed_count = 0
        max_process_per_cycle = self.max_concurrent_tasks - len(self.running_tasks)
        
        while not self.task_queue.empty() and processed_count < max_process_per_cycle:
            try:
                priority_value, created_time, task_id = self.task_queue.get_nowait()
                
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    # 再次检查是否可以执行
                    if self._can_execute_task(task):
                        self._execute_task(task)
                        processed_count += 1
                    else:
                        # 重新放回队列
                        self._add_to_queue(task)
                        
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"处理任务队列错误: {e}")
    
    def _execute_task(self, task: Task):
        """
        执行任务
        
        Args:
            task: 任务对象
        """
        try:
            task.status = TaskStatus.RUNNING
            start_time = time.time()
            
            self.logger.info(f"开始执行任务: {task.name} ({task.task_id})")
            
            # 提交到线程池执行
            future = self.executor.submit(self._run_task_logic, task)
            self.running_tasks[task.task_id] = future
            
        except Exception as e:
            self.logger.error(f"提交任务执行失败: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
    
    def _run_task_logic(self, task: Task) -> Any:
        """
        运行任务逻辑
        
        Args:
            task: 任务对象
            
        Returns:
            任务结果
        """
        try:
            start_time = time.time()
            
            # 查找处理器
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"未找到任务类型 {task.task_type} 的处理器")
            
            # 执行任务逻辑
            result = handler(task.parameters)
            
            # 更新任务状态
            execution_time = time.time() - start_time
            task.execution_time = execution_time
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            # 移动到完成列表
            self.completed_tasks.append(task)
            
            # 更新统计
            self._update_statistics(execution_time, True)
            
            self.logger.info(f"任务执行完成: {task.name} ({execution_time:.2f}s)")
            
            # 检查依赖此任务的其他任务
            self._check_dependent_tasks(task.task_id)
            
            return result
            
        except Exception as e:
            # 处理异常
            execution_time = time.time() - start_time
            task.execution_time = execution_time
            task.error = str(e)
            task.status = TaskStatus.FAILED
            
            self.failed_tasks.append(task)
            self._update_statistics(execution_time, False)
            
            self.logger.error(f"任务执行失败: {task.name} - {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.logger.info(f"任务将在 {self.config['retry_delay']} 秒后重试 (第 {task.retry_count} 次)")
                
                # 延迟重试
                task.scheduled_at = datetime.now() + timedelta(seconds=self.config['retry_delay'])
                self._add_to_queue(task)
            else:
                self.logger.error(f"任务已达到最大重试次数: {task.name}")
            
            raise
    
    def _check_dependent_tasks(self, completed_task_id: str):
        """检查依赖任务"""
        for task in self.tasks.values():
            if completed_task_id in task.dependencies and task.status == TaskStatus.PENDING:
                if self._can_execute_task(task):
                    self._add_to_queue(task)
                    self.logger.info(f"依赖任务 {completed_task_id} 已完成，添加任务 {task.name} 到队列")
    
    def _check_timeout_tasks(self):
        """检查超时任务"""
        current_time = datetime.now()
        
        for task_id, future in list(self.running_tasks.items()):
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # 检查是否超时
                if (current_time - task.created_at).total_seconds() > task.timeout:
                    if not future.done():
                        future.cancel()
                    
                    task.status = TaskStatus.TIMEOUT
                    task.error = "任务执行超时"
                    
                    # 从运行列表中移除
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
                    
                    self.logger.warning(f"任务超时: {task.name} ({task_id})")
    
    def _retry_failed_tasks(self):
        """重试失败任务"""
        for task in self.failed_tasks[:]:  # 使用切片避免修改列表时出错
            if task.retry_count < task.max_retries:
                # 重置任务状态
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = None
                task.scheduled_at = datetime.now() + timedelta(seconds=self.config['retry_delay'])
                
                # 从失败列表移除
                self.failed_tasks.remove(task)
                
                # 添加到队列
                self._add_to_queue(task)
                
                self.logger.info(f"重试失败任务: {task.name} (第 {task.retry_count} 次)")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.COMPLETED:
            return False
        
        # 如果任务正在运行，尝试取消
        if task_id in self.running_tasks:
            future = self.running_tasks[task_id]
            if not future.done():
                future.cancel()
            del self.running_tasks[task_id]
        
        task.status = TaskStatus.CANCELLED
        self.logger.info(f"任务已取消: {task.name} ({task_id})")
        
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典
        """
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def get_all_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """
        获取所有任务
        
        Args:
            status: 状态过滤条件
            
        Returns:
            任务列表
        """
        tasks = []
        for task in self.tasks.values():
            if status is None or task.status == status:
                tasks.append(task.to_dict())
        return tasks
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            **self.stats,
            'running_tasks': len(self.running_tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'queue_size': self.task_queue.qsize(),
            'success_rate': self.stats['completed_tasks'] / max(self.stats['total_tasks'], 1)
        }
    
    def _start_monitoring(self):
        """启动监控"""
        def monitor_loop():
            while self.is_running:
                try:
                    stats = self.get_statistics()
                    self.logger.info(f"调度器状态: {json.dumps(stats, ensure_ascii=False)}")
                    
                    # 清理旧任务
                    self._cleanup_old_tasks()
                    
                    time.sleep(300)  # 5分钟监控一次
                    
                except Exception as e:
                    self.logger.error(f"监控循环错误: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _cleanup_old_tasks(self):
        """清理旧任务"""
        cutoff_time = datetime.now() - timedelta(days=7)  # 保留7天
        
        # 清理已完成的任务
        self.completed_tasks = [
            task for task in self.completed_tasks 
            if task.created_at > cutoff_time
        ]
        
        # 清理失败的任务
        self.failed_tasks = [
            task for task in self.failed_tasks
            if task.created_at > cutoff_time
        ]
    
    def _update_statistics(self, execution_time: float, success: bool):
        """更新统计信息"""
        if success:
            self.stats['completed_tasks'] += 1
        else:
            self.stats['failed_tasks'] += 1
        
        # 更新平均执行时间
        total_completed = self.stats['completed_tasks']
        if total_completed > 0:
            current_avg = self.stats['average_execution_time']
            self.stats['average_execution_time'] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
            )
        
        self.stats['total_execution_time'] += execution_time
    
    # 默认任务处理器实现
    def _handle_image_classification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像分类任务"""
        time.sleep(2)  # 模拟处理时间
        
        import random
        return {
            'accuracy': random.uniform(0.85, 0.95),
            'precision': random.uniform(0.80, 0.92),
            'recall': random.uniform(0.82, 0.94),
            'f1_score': random.uniform(0.81, 0.93)
        }
    
    def _handle_object_detection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理目标检测任务"""
        time.sleep(3)  # 模拟处理时间
        
        import random
        return {
            'mAP': random.uniform(0.75, 0.88),
            'precision': random.uniform(0.78, 0.90),
            'recall': random.uniform(0.76, 0.89),
            'detection_speed': random.uniform(15, 30)
        }
    
    def _handle_scene_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理场景分析任务"""
        time.sleep(4)  # 模拟处理时间
        
        import random
        return {
            'scene_understanding_score': random.uniform(0.70, 0.85),
            'complexity_handling': random.uniform(0.75, 0.88),
            'context_awareness': random.uniform(0.72, 0.86)
        }
    
    def _handle_cross_domain_transfer(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理跨域迁移任务"""
        time.sleep(5)  # 模拟处理时间
        
        import random
        return {
            'transfer_efficiency': random.uniform(0.65, 0.82),
            'domain_adaptation_speed': random.uniform(0.6, 0.8),
            'performance_retention': random.uniform(0.70, 0.85)
        }
    
    def _handle_adaptation_test(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理适应测试任务"""
        time.sleep(3)  # 模拟处理时间
        
        import random
        return {
            'adaptation_time': random.uniform(10, 30),
            'adaptation_accuracy': random.uniform(0.68, 0.84),
            'stability_score': random.uniform(0.72, 0.87)
        }
    
    def _handle_performance_test(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理性能测试任务"""
        time.sleep(6)  # 模拟处理时间
        
        import random
        return {
            'throughput': random.uniform(100, 500),
            'latency': random.uniform(10, 50),
            'cpu_usage': random.uniform(30, 80),
            'memory_usage': random.uniform(20, 70)
        }
    
    def _handle_stress_test(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理压力测试任务"""
        time.sleep(10)  # 模拟处理时间
        
        import random
        return {
            'max_load': random.uniform(1000, 5000),
            'stability_under_load': random.uniform(0.7, 0.9),
            'recovery_time': random.uniform(5, 20),
            'error_rate': random.uniform(0.01, 0.1)
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop_scheduler()


if __name__ == "__main__":
    # 示例用法
    scheduler = TaskScheduler()
    
    # 启动调度器
    scheduler.start_scheduler()
    
    # 创建任务
    task1_id = scheduler.create_task(
        name="图像分类测试",
        task_type="image_classification",
        priority=TaskPriority.HIGH,
        parameters={'dataset_size': 1000}
    )
    
    task2_id = scheduler.create_task(
        name="目标检测测试", 
        task_type="object_detection",
        priority=TaskPriority.NORMAL,
        parameters={'model_type': 'yolo'}
    )
    
    # 等待任务执行
    time.sleep(10)
    
    # 获取统计信息
    stats = scheduler.get_statistics()
    print(f"调度器统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 停止调度器
    scheduler.stop_scheduler()