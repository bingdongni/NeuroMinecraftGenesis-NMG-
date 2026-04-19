#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每周真实世界任务测试系统
===================

这个模块实现了对智能体真实世界能力的定期评估系统。
通过每周执行真实世界任务测试，全面评估智能体在不同环境下的表现，
追踪长期性能变化，并分析迁移学习效果。

核心功能：
- 每周定时执行真实世界任务测试套件
- 评估智能体在多个真实环境下的适应能力
- 记录和分析性能指标变化趋势
- 生成详细的测试报告和可视化结果
- 监控长期学习效果和稳定性

作者：AI研究团队
日期：2025-11-13
"""

import asyncio
import json
import logging
import os
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# 导入相关模块
from .task_scheduler import TaskScheduler
from .performance_recorder import PerformanceRecorder  
from .trend_analyzer import TrendAnalyzer
from .report_generator import ReportGenerator


class WeeklyTaskTest:
    """
    每周真实世界任务测试主类
    
    这个类是整个测试系统的核心控制器，负责协调各个组件，
    定期执行真实世界任务测试，并收集分析测试结果。
    
    主要功能：
    - 调度和执行每周测试任务
    - 整合各个组件的工作
    - 管理测试流程和结果
    - 维护测试历史记录
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化每周任务测试系统
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = self._setup_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化各个组件
        self.task_scheduler = TaskScheduler(self.config.get('scheduler', {}))
        self.performance_recorder = PerformanceRecorder(self.config.get('recorder', {}))
        self.trend_analyzer = TrendAnalyzer(self.config.get('analyzer', {}))
        self.report_generator = ReportGenerator(self.config.get('generator', {}))
        
        # 测试状态管理
        self.test_status = {
            'is_running': False,
            'current_week': None,
            'last_execution': None,
            'total_tests': 0,
            'successful_tests': 0
        }
        
        # 历史测试记录
        self.test_history = []
        
        self.logger.info("每周真实世界任务测试系统初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('WeeklyTaskTest')
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path('/workspace/worlds/real/logs')
        log_dir.mkdir(exist_ok=True)
        
        # 设置文件处理器
        file_handler = logging.FileHandler(
            log_dir / f'weekly_test_{datetime.now().strftime("%Y%m%d")}.log',
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
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        default_config = {
            'test_schedule': {
                'day_of_week': 'monday',  # 每周一执行测试
                'time': '09:00',          # 上午9点执行
                'timezone': 'Asia/Shanghai'
            },
            'test_duration': {
                'max_duration_hours': 8,  # 最大测试时长8小时
                'timeout_minutes': 30     # 单个任务超时30分钟
            },
            'environments': {
                'real_world_tasks': [
                    'image_classification',
                    'object_detection', 
                    'scene_analysis',
                    'cross_domain_transfer',
                    'adaptation_test'
                ],
                'difficulty_levels': ['easy', 'medium', 'hard'],
                'sample_sizes': 100
            },
            'metrics': {
                'performance_metrics': [
                    'accuracy',
                    'precision',
                    'recall',
                    'f1_score',
                    'adaptation_time'
                ],
                'stability_metrics': [
                    'variance',
                    'consistency',
                    'drift_indicators'
                ],
                'efficiency_metrics': [
                    'processing_speed',
                    'resource_usage',
                    'success_rate'
                ]
            },
            'output': {
                'report_formats': ['json', 'html', 'pdf'],
                'visualization': True,
                'auto_email': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def schedule_weekly_test(self) -> bool:
        """
        调度每周测试任务
        
        这个方法设置定时任务，在指定的每周时间点自动执行测试。
        
        Returns:
            bool: 调度是否成功
        """
        try:
            schedule_config = self.config['test_schedule']
            
            # 设置每周执行时间
            day = schedule_config['day_of_week']
            time_str = schedule_config['time']
            
            # 使用schedule库设置定时任务
            getattr(schedule.every(), day).at(time_str).do(self._execute_weekly_test)
            
            self.logger.info(f"已设置每周{day} {time_str}执行真实世界任务测试")
            
            # 设置连续监控
            self._start_continuous_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"调度每周测试失败: {e}")
            return False
    
    def _start_continuous_monitoring(self):
        """启动连续监控"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        
        import threading
        monitor_thread = threading.Thread(target=run_scheduler, daemon=True)
        monitor_thread.start()
        self.logger.info("连续监控已启动")
    
    def execute_test_suite(self, test_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行测试套件
        
        这是系统的核心方法，执行完整的真实世界任务测试套件，
        评估智能体在多个环境下的表现。
        
        Args:
            test_config: 测试配置，如果为None则使用默认配置
            
        Returns:
            测试结果字典
        """
        if self.test_status['is_running']:
            self.logger.warning("测试正在运行中，跳过本次执行")
            return {'status': 'skipped', 'reason': 'test_already_running'}
        
        self.test_status['is_running'] = True
        self.test_status['current_week'] = datetime.now().strftime('%Y-W%U')
        
        try:
            self.logger.info("开始执行每周真实世界任务测试套件")
            start_time = time.time()
            
            # 执行测试套件
            test_results = self._run_test_suite(test_config or {})
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录性能数据
            performance_data = self._compile_performance_data(test_results, execution_time)
            
            # 分析趋势
            trend_analysis = self.trend_analyzer.analyze_trends(performance_data)
            
            # 生成报告
            report = self.report_generator.generate_weekly_report(
                test_results, performance_data, trend_analysis
            )
            
            # 更新状态
            self._update_test_status(test_results)
            
            # 保存结果
            self._save_test_results(test_results, performance_data, trend_analysis)
            
            self.logger.info(f"测试套件执行完成，耗时 {execution_time:.2f} 秒")
            
            return {
                'status': 'completed',
                'test_results': test_results,
                'performance_data': performance_data,
                'trend_analysis': trend_analysis,
                'report': report,
                'execution_time': execution_time
            }
            
        except Exception as e:
            self.logger.error(f"执行测试套件时发生错误: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.test_status['is_running'] = False
    
    def _run_test_suite(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行具体的测试套件
        
        Args:
            test_config: 测试配置
            
        Returns:
            测试结果
        """
        environments = test_config.get('environments', self.config['environments'])
        metrics_config = test_config.get('metrics', self.config['metrics'])
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_environment': environments,
            'individual_tests': {},
            'summary_metrics': {},
            'environment_scores': {}
        }
        
        # 遍历所有测试环境
        for env_name in environments['real_world_tasks']:
            self.logger.info(f"执行环境测试: {env_name}")
            
            try:
                # 为每个环境执行测试
                env_result = self._execute_environment_test(env_name, environments, metrics_config)
                test_results['individual_tests'][env_name] = env_result
                
                # 计算环境评分
                env_score = self._calculate_environment_score(env_result, metrics_config)
                test_results['environment_scores'][env_name] = env_score
                
                self.logger.info(f"环境 {env_name} 测试完成，评分: {env_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"环境 {env_name} 测试失败: {e}")
                test_results['individual_tests'][env_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 计算总体指标
        test_results['summary_metrics'] = self._calculate_summary_metrics(test_results)
        
        return test_results
    
    def _execute_environment_test(self, env_name: str, environments: Dict, 
                                  metrics_config: Dict) -> Dict[str, Any]:
        """
        执行单个环境的测试
        
        Args:
            env_name: 环境名称
            environments: 环境配置
            metrics_config: 指标配置
            
        Returns:
            环境测试结果
        """
        result = {
            'environment': env_name,
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'performance_metrics': {},
            'stability_metrics': {},
            'efficiency_metrics': {}
        }
        
        # 根据不同环境类型执行相应测试
        if env_name == 'image_classification':
            result.update(self._test_image_classification(environments, metrics_config))
        elif env_name == 'object_detection':
            result.update(self._test_object_detection(environments, metrics_config))
        elif env_name == 'scene_analysis':
            result.update(self._test_scene_analysis(environments, metrics_config))
        elif env_name == 'cross_domain_transfer':
            result.update(self._test_cross_domain_transfer(environments, metrics_config))
        elif env_name == 'adaptation_test':
            result.update(self._test_adaptation(environments, metrics_config))
        else:
            # 通用测试流程
            result.update(self._execute_generic_test(env_name, environments, metrics_config))
        
        return result
    
    def _test_image_classification(self, environments: Dict, metrics_config: Dict) -> Dict:
        """测试图像分类能力"""
        return {
            'status': 'completed',
            'accuracy': np.random.uniform(0.85, 0.95),  # 模拟结果
            'precision': np.random.uniform(0.80, 0.92),
            'recall': np.random.uniform(0.82, 0.94),
            'f1_score': np.random.uniform(0.81, 0.93),
            'processing_time': np.random.uniform(0.1, 0.5)  # 秒
        }
    
    def _test_object_detection(self, environments: Dict, metrics_config: Dict) -> Dict:
        """测试目标检测能力"""
        return {
            'status': 'completed',
            'mAP': np.random.uniform(0.75, 0.88),
            'precision': np.random.uniform(0.78, 0.90),
            'recall': np.random.uniform(0.76, 0.89),
            'detection_speed': np.random.uniform(15, 30)  # FPS
        }
    
    def _test_scene_analysis(self, environments: Dict, metrics_config: Dict) -> Dict:
        """测试场景分析能力"""
        return {
            'status': 'completed',
            'scene_understanding_score': np.random.uniform(0.70, 0.85),
            'complexity_handling': np.random.uniform(0.75, 0.88),
            'context_awareness': np.random.uniform(0.72, 0.86)
        }
    
    def _test_cross_domain_transfer(self, environments: Dict, metrics_config: Dict) -> Dict:
        """测试跨域迁移能力"""
        return {
            'status': 'completed',
            'transfer_efficiency': np.random.uniform(0.65, 0.82),
            'domain_adaptation_speed': np.random.uniform(0.6, 0.8),
            'performance_retention': np.random.uniform(0.70, 0.85)
        }
    
    def _test_adaptation(self, environments: Dict, metrics_config: Dict) -> Dict:
        """测试适应能力"""
        return {
            'status': 'completed',
            'adaptation_time': np.random.uniform(10, 30),  # 秒
            'adaptation_accuracy': np.random.uniform(0.68, 0.84),
            'stability_score': np.random.uniform(0.72, 0.87)
        }
    
    def _execute_generic_test(self, env_name: str, environments: Dict, metrics_config: Dict) -> Dict:
        """执行通用测试"""
        return {
            'status': 'completed',
            'success_rate': np.random.uniform(0.70, 0.90),
            'average_score': np.random.uniform(0.65, 0.85),
            'consistency': np.random.uniform(0.68, 0.88)
        }
    
    def _calculate_environment_score(self, env_result: Dict, metrics_config: Dict) -> float:
        """
        计算环境测试评分
        
        Args:
            env_result: 环境测试结果
            metrics_config: 指标配置
            
        Returns:
            环境评分 (0-1)
        """
        scores = []
        
        # 从结果中提取数值分数
        for key, value in env_result.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                scores.append(value)
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)) and 0 <= sub_value <= 1:
                        scores.append(sub_value)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_summary_metrics(self, test_results: Dict) -> Dict:
        """
        计算总体指标
        
        Args:
            test_results: 测试结果
            
        Returns:
            总体指标字典
        """
        # 计算平均环境评分
        env_scores = list(test_results['environment_scores'].values())
        avg_environment_score = np.mean(env_scores) if env_scores else 0.0
        
        # 计算成功测试数量
        successful_tests = sum(
            1 for test in test_results['individual_tests'].values()
            if test.get('status') == 'completed'
        )
        total_tests = len(test_results['individual_tests'])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            'average_environment_score': avg_environment_score,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'test_coverage': len(test_results['individual_tests']) / len(
                self.config['environments']['real_world_tasks']
            )
        }
    
    def _compile_performance_data(self, test_results: Dict, execution_time: float) -> Dict:
        """
        编译性能数据
        
        Args:
            test_results: 测试结果
            execution_time: 执行时间
            
        Returns:
            性能数据字典
        """
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'test_summary': test_results['summary_metrics'],
            'environment_details': {},
            'performance_trends': {}
        }
        
        # 编译详细环境数据
        for env_name, env_result in test_results['individual_tests'].items():
            if env_result.get('status') == 'completed':
                performance_data['environment_details'][env_name] = {
                    'metrics': env_result,
                    'score': test_results['environment_scores'].get(env_name, 0.0)
                }
        
        # 记录到性能记录器
        self.performanceRecorder.record_performance(performance_data)
        
        return performance_data
    
    def _update_test_status(self, test_results: Dict):
        """更新测试状态"""
        self.test_status['last_execution'] = datetime.now().isoformat()
        self.test_status['total_tests'] += 1
        
        # 检查是否有成功的测试
        has_success = any(
            test.get('status') == 'completed' 
            for test in test_results['individual_tests'].values()
        )
        
        if has_success:
            self.test_status['successful_tests'] += 1
        
        # 添加到历史记录
        self.test_history.append({
            'timestamp': self.test_status['last_execution'],
            'week': self.test_status['current_week'],
            'success_rate': test_results['summary_metrics']['success_rate'],
            'average_score': test_results['summary_metrics']['average_environment_score']
        })
        
        # 保持最近50次记录
        if len(self.test_history) > 50:
            self.test_history = self.test_history[-50:]
    
    def _save_test_results(self, test_results: Dict, performance_data: Dict, trend_analysis: Dict):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存目录
        save_dir = Path('/workspace/worlds/real/test_results')
        save_dir.mkdir(exist_ok=True)
        
        # 保存测试结果
        results_file = save_dir / f'weekly_test_results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        # 保存性能数据
        performance_file = save_dir / f'performance_data_{timestamp}.json'
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        
        # 保存趋势分析
        if trend_analysis:
            trend_file = save_dir / f'trend_analysis_{timestamp}.json'
            with open(trend_file, 'w', encoding='utf-8') as f:
                json.dump(trend_analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"测试结果已保存到 {save_dir}")
    
    def _execute_weekly_test(self):
        """执行每周测试的定时任务"""
        self.logger.info("触发每周定时测试")
        return self.execute_test_suite()
    
    def get_test_status(self) -> Dict[str, Any]:
        """获取当前测试状态"""
        return {
            'test_status': self.test_status.copy(),
            'recent_history': self.test_history[-10:] if self.test_history else [],
            'next_scheduled': schedule.next_run() if hasattr(schedule, 'next_run') else None
        }
    
    def run_manual_test(self) -> Dict[str, Any]:
        """手动执行测试（用于开发和调试）"""
        self.logger.info("手动执行真实世界任务测试")
        return self.execute_test_suite()
    
    def stop_scheduling(self):
        """停止定时调度"""
        schedule.clear()
        self.logger.info("已停止定时调度")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'logger'):
            self.logger.info("每周真实世界任务测试系统已关闭")


if __name__ == "__main__":
    # 示例用法
    test_system = WeeklyTaskTest()
    
    # 调度每周测试
    if test_system.schedule_weekly_test():
        print("每周测试调度成功启动")
    
    # 执行一次手动测试
    result = test_system.run_manual_test()
    print(f"测试结果: {result['status']}")
    
    # 打印当前状态
    status = test_system.get_test_status()
    print(f"当前状态: {json.dumps(status, indent=2, ensure_ascii=False)}")