"""
每小时监控器
===========

该模块实现24小时连续实验的每小时监控功能，包括：
- 定时数据采集
- 实时状态监控
- 异常检测与处理
- 数据缓存与同步
- 监控报告生成
"""

import time
import threading
import queue
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import json
import logging
from enum import Enum

from .cognitive_tracker import CognitiveTracker, CognitiveMetrics

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitorStatus(Enum):
    """监控状态枚举"""
    IDLE = "空闲"
    RUNNING = "运行中"
    PAUSED = "暂停"
    ERROR = "错误"
    STOPPED = "已停止"

class HourlyMonitor:
    """24小时连续监控器"""
    
    def __init__(self, 
                 tracker: CognitiveTracker,
                 monitor_interval: int = 3600,  # 1小时间隔
                 data_retention_hours: int = 48):
        """
        初始化每小时监控器
        
        Args:
            tracker: 认知能力跟踪器实例
            monitor_interval: 监控间隔（秒），默认1小时
            data_retention_hours: 数据保留时间（小时）
        """
        self.tracker = tracker
        self.monitor_interval = monitor_interval
        self.data_retention_hours = data_retention_hours
        
        # 监控状态
        self.status = MonitorStatus.IDLE
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_hour = 0
        self.total_hours = 24
        
        # 数据存储
        self.hourly_data: Dict[int, Dict] = {}
        self.alerts: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # 线程控制
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # 回调函数
        self.callbacks = {
            'hourly_update': [],
            'alert': [],
            'status_change': [],
            'completion': []
        }
        
        # 异常检测参数
        self.anomaly_thresholds = {
            'score_drop': 20,      # 分数下降阈值
            'no_progress': 5,      # 无进展阈值（小时）
            'system_error': 3      # 系统错误阈值
        }
        
        # 数据缓存
        self.recent_data = deque(maxlen=10)  # 保留最近10次记录
        
        logger.info(f"每小时监控器初始化完成 - 监控间隔: {monitor_interval}秒")
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """
        添加监控回调函数
        
        Args:
            event: 事件类型 ('hourly_update', 'alert', 'status_change', 'completion')
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.info(f"添加回调函数: {event}")
        else:
            logger.warning(f"未知事件类型: {event}")
    
    def _execute_callbacks(self, event: str, data: Any = None) -> None:
        """
        执行指定事件的回调函数
        
        Args:
            event: 事件类型
            data: 回调数据
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    def _check_anomalies(self, current_metrics: CognitiveMetrics) -> List[Dict]:
        """
        检测数据异常
        
        Args:
            current_metrics: 当前认知指标
            
        Returns:
            异常列表
        """
        anomalies = []
        
        # 检查分数下降
        if len(self.recent_data) >= 2:
            prev_data = self.recent_data[-2]
            prev_overall = prev_data.get('overall_score', 50)
            current_overall = current_metrics.overall_score()
            
            if prev_overall - current_overall > self.anomaly_thresholds['score_drop']:
                anomalies.append({
                    'type': 'score_drop',
                    'severity': 'high',
                    'message': f'认知分数下降 {prev_overall - current_overall:.1f}分',
                    'timestamp': current_metrics.timestamp,
                    'details': {
                        'previous_score': prev_overall,
                        'current_score': current_overall,
                        'drop': prev_overall - current_overall
                    }
                })
        
        # 检查系统错误
        error_count = sum(1 for alert in self.alerts if alert.get('type') == 'system_error')
        if error_count >= self.anomaly_thresholds['system_error']:
            anomalies.append({
                'type': 'system_error',
                'severity': 'critical',
                'message': f'系统错误次数达到阈值 ({error_count})',
                'timestamp': current_metrics.timestamp,
                'details': {'error_count': error_count}
            })
        
        return anomalies
    
    def _generate_hourly_report(self, hour: int, metrics: CognitiveMetrics) -> Dict:
        """
        生成每小时监控报告
        
        Args:
            hour: 当前小时数
            metrics: 认知指标
            
        Returns:
            监控报告字典
        """
        report = {
            'hour': hour,
            'timestamp': metrics.timestamp.isoformat(),
            'overall_score': metrics.overall_score(),
            'dimension_scores': {
                'memory': metrics.memory_score,
                'thinking': metrics.thinking_score,
                'creativity': metrics.creativity_score,
                'observation': metrics.observation_score,
                'attention': metrics.attention_score,
                'imagination': metrics.imagination_score
            },
            'performance_metrics': {
                'uptime_hours': hour + 1,
                'data_points': len(self.recent_data),
                'alerts_count': len([a for a in self.alerts if a.get('timestamp', datetime.min).hour == hour])
            }
        }
        
        # 计算趋势
        if len(self.recent_data) >= 2:
            prev_data = self.recent_data[-2]
            score_change = metrics.overall_score() - prev_data.get('overall_score', 50)
            report['trend'] = {
                'direction': 'up' if score_change > 0 else 'down' if score_change < 0 else 'stable',
                'change': score_change,
                'magnitude': abs(score_change)
            }
        else:
            report['trend'] = {
                'direction': 'baseline',
                'change': 0,
                'magnitude': 0
            }
        
        # 计算各维度变化
        if len(self.recent_data) >= 2:
            prev_dims = prev_data.get('dimension_scores', {})
            current_dims = report['dimension_scores']
            
            dimension_changes = {}
            for dim in ['memory', 'thinking', 'creativity', 'observation', 'attention', 'imagination']:
                change = current_dims[dim] - prev_dims.get(dim, 50)
                dimension_changes[dim] = {
                    'change': change,
                    'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                }
            
            report['dimension_trends'] = dimension_changes
        
        return report
    
    def _cleanup_old_data(self) -> None:
        """清理过期的数据"""
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)
        
        # 清理过期的每小时数据
        expired_hours = [hour for hour, data in self.hourly_data.items() 
                        if datetime.fromisoformat(data['timestamp']) < cutoff_time]
        for hour in expired_hours:
            del self.hourly_data[hour]
        
        # 清理过期的警报
        self.alerts = [alert for alert in self.alerts 
                      if alert.get('timestamp', datetime.min) >= cutoff_time]
        
        logger.info(f"清理过期数据完成，保留 {len(self.hourly_data)} 小时数据")
    
    def _collect_hourly_metrics(self, agent_state: Dict, environment_state: Dict) -> Optional[CognitiveMetrics]:
        """
        收集每小时的认知指标
        
        Args:
            agent_state: 智能体状态
            environment_state: 环境状态
            
        Returns:
            认知指标对象或None
        """
        try:
            # 使用认知跟踪器收集指标
            metrics = self.tracker.track_cognitive_metrics(agent_state, environment_state)
            
            # 异常检测
            anomalies = self._check_anomalies(metrics)
            for anomaly in anomalies:
                self.alerts.append(anomaly)
                self._execute_callbacks('alert', anomaly)
                logger.warning(f"检测到异常: {anomaly['message']}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"收集每小时指标失败: {e}"
            logger.error(error_msg)
            
            # 添加系统错误警报
            self.alerts.append({
                'type': 'system_error',
                'severity': 'high',
                'message': error_msg,
                'timestamp': datetime.now()
            })
            self._execute_callbacks('alert', self.alerts[-1])
            
            return None
    
    def _monitor_loop(self) -> None:
        """监控主循环"""
        logger.info("监控线程启动")
        
        try:
            while not self.stop_event.is_set():
                # 检查是否需要暂停
                if self.pause_event.is_set():
                    logger.info("监控已暂停")
                    self.pause_event.wait()
                    logger.info("监控恢复")
                
                # 记录开始时间
                if self.start_time is None:
                    self.start_time = datetime.now()
                
                # 计算当前应该采集的小时数
                elapsed_time = datetime.now() - self.start_time
                current_hour = int(elapsed_time.total_seconds() // 3600)
                
                # 检查是否到达24小时
                if current_hour >= self.total_hours:
                    logger.info("24小时监控完成")
                    self.status = MonitorStatus.STOPPED
                    self._execute_callbacks('completion', self.hourly_data)
                    break
                
                # 如果是新小时，记录数据
                if current_hour != self.current_hour:
                    self.current_hour = current_hour
                    
                    # 模拟智能体状态（实际应用中从Minecraft获取）
                    agent_state = {
                        'memory_retention': 0.5 + 0.1 * current_hour,  # 模拟改进
                        'learning_speed': 0.6 + 0.05 * current_hour,
                        'recall_accuracy': 0.7 + 0.03 * current_hour,
                        'reasoning_accuracy': 0.8 + 0.02 * current_hour,
                        'novel_behaviors': 0.4 + 0.08 * current_hour,
                        'environmental_awareness': 0.9,
                        'focus_duration': 0.7 + 0.03 * current_hour,
                        'imagination_events': [f'event_{i}' for i in range(current_hour % 5)]
                    }
                    
                    environment_state = {
                        'objects': ['tree', 'stone', 'water'],
                        'time': 'day' if current_hour % 24 < 18 else 'night',
                        'weather': 'clear',
                        'hour': current_hour
                    }
                    
                    # 收集认知指标
                    metrics = self._collect_hourly_metrics(agent_state, environment_state)
                    
                    if metrics:
                        # 生成每小时报告
                        hourly_report = self._generate_hourly_report(current_hour, metrics)
                        self.hourly_data[current_hour] = hourly_report
                        
                        # 添加到最近数据缓存
                        self.recent_data.append(hourly_report)
                        
                        # 执行每小时更新回调
                        self._execute_callbacks('hourly_update', hourly_report)
                        
                        logger.info(f"第 {current_hour} 小时数据采集完成")
                
                # 等待下次采集
                time.sleep(60)  # 每分钟检查一次
                
        except Exception as e:
            logger.error(f"监控循环出错: {e}")
            self.status = MonitorStatus.ERROR
            self._execute_callbacks('alert', {
                'type': 'monitor_error',
                'severity': 'critical',
                'message': f'监控循环异常: {e}',
                'timestamp': datetime.now()
            })
        
        finally:
            self.end_time = datetime.now()
            logger.info("监控线程结束")
    
    def start_monitoring(self) -> bool:
        """
        开始24小时监控
        
        Returns:
            是否成功启动
        """
        if self.status == MonitorStatus.RUNNING:
            logger.warning("监控已在运行中")
            return False
        
        # 重置状态
        self.status = MonitorStatus.RUNNING
        self.stop_event.clear()
        self.pause_event.clear()
        self.start_time = None
        self.end_time = None
        self.current_hour = 0
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self._execute_callbacks('status_change', self.status)
        logger.info("24小时监控已启动")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        停止监控
        
        Returns:
            是否成功停止
        """
        if self.status not in [MonitorStatus.RUNNING, MonitorStatus.PAUSED]:
            logger.warning("监控未在运行")
            return False
        
        # 停止监控
        self.stop_event.set()
        if self.status == MonitorStatus.PAUSED:
            self.pause_event.set()  # 恢复暂停的线程以允许其退出
        
        # 等待线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.status = MonitorStatus.STOPPED
        self._execute_callbacks('status_change', self.status)
        logger.info("监控已停止")
        return True
    
    def pause_monitoring(self) -> bool:
        """
        暂停监控
        
        Returns:
            是否成功暂停
        """
        if self.status != MonitorStatus.RUNNING:
            logger.warning("监控未在运行，无法暂停")
            return False
        
        self.pause_event.set()
        self.status = MonitorStatus.PAUSED
        self._execute_callbacks('status_change', self.status)
        logger.info("监控已暂停")
        return True
    
    def resume_monitoring(self) -> bool:
        """
        恢复监控
        
        Returns:
            是否成功恢复
        """
        if self.status != MonitorStatus.PAUSED:
            logger.warning("监控未暂停，无法恢复")
            return False
        
        self.pause_event.clear()
        self.status = MonitorStatus.RUNNING
        self._execute_callbacks('status_change', self.status)
        logger.info("监控已恢复")
        return True
    
    def get_status(self) -> Dict:
        """
        获取当前监控状态
        
        Returns:
            状态信息字典
        """
        runtime = 0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'runtime_hours': runtime,
            'current_hour': self.current_hour,
            'total_hours': self.total_hours,
            'completion_rate': min(100, (runtime / self.total_hours) * 100),
            'data_points': len(self.hourly_data),
            'alerts_count': len(self.alerts)
        }
    
    def get_hourly_data(self, hour: Optional[int] = None) -> Dict:
        """
        获取每小时数据
        
        Args:
            hour: 指定小时，如果为None则返回所有数据
            
        Returns:
            小时数据字典
        """
        if hour is not None:
            return self.hourly_data.get(hour, {})
        return self.hourly_data.copy()
    
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            性能摘要字典
        """
        if not self.hourly_data:
            return {}
        
        scores = [data['overall_score'] for data in self.hourly_data.values()]
        
        return {
            'total_hours': len(self.hourly_data),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'score_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0,
            'alerts_count': len(self.alerts),
            'data_retention_hours': self.data_retention_hours
        }
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """
        获取最近的警报
        
        Args:
            hours: 过去多少小时的警报
            
        Returns:
            警报列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts 
                if alert.get('timestamp', datetime.min) >= cutoff_time]
    
    def export_data(self, filepath: str) -> bool:
        """
        导出监控数据
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                'monitor_info': {
                    'agent_id': self.tracker.agent_id,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None,
                    'total_hours': self.total_hours
                },
                'hourly_data': self.hourly_data,
                'alerts': [alert for alert in self.alerts],
                'performance_summary': self.get_performance_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"监控数据已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出监控数据失败: {e}")
            return False
    
    def clear_data(self) -> None:
        """清除所有监控数据"""
        self.hourly_data.clear()
        self.alerts.clear()
        self.recent_data.clear()
        self.current_hour = 0
        logger.info("监控数据已清除")

if __name__ == "__main__":
    # 测试每小时监控器
    from .cognitive_tracker import CognitiveTracker
    
    # 创建认知跟踪器
    tracker = CognitiveTracker(agent_id="test_agent_002")
    
    # 创建监控器
    monitor = HourlyMonitor(tracker, monitor_interval=10)  # 10秒间隔用于测试
    
    # 添加回调函数
    def on_hourly_update(data):
        print(f"每小时更新: 第{data['hour']}小时, 分数: {data['overall_score']:.2f}")
    
    def on_alert(alert):
        print(f"警报: {alert['message']}")
    
    monitor.add_callback('hourly_update', on_hourly_update)
    monitor.add_callback('alert', on_alert)
    
    # 启动监控（测试模式 - 24小时改为24秒）
    monitor.total_hours = 1  # 测试用，改为1小时
    success = monitor.start_monitoring()
    
    if success:
        print("监控已启动，等待数据...")
        
        try:
            # 运行一段时间
            time.sleep(30)  # 运行30秒
            
            # 获取状态
            status = monitor.get_status()
            print(f"当前状态: {status}")
            
            # 暂停监控
            monitor.pause_monitoring()
            print("监控已暂停")
            
            time.sleep(5)
            
            # 恢复监控
            monitor.resume_monitoring()
            print("监控已恢复")
            
            # 等待完成
            while monitor.status == MonitorStatus.RUNNING:
                time.sleep(1)
            
            # 获取性能摘要
            summary = monitor.get_performance_summary()
            print(f"性能摘要: {summary}")
            
            # 导出数据
            monitor.export_data("test_monitoring_data.json")
            
        except KeyboardInterrupt:
            print("用户中断，停止监控")
        
        finally:
            monitor.stop_monitoring()
            print("监控已停止")
    else:
        print("监控启动失败")