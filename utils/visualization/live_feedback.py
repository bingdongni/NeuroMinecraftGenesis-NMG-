"""
实时反馈系统
提供参数变更的即时反馈和智能体行为影响分析
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math


@dataclass
class ParameterChange:
    """参数变更记录"""
    parameter_name: str
    old_value: float
    new_value: float
    timestamp: datetime
    user_action: bool = True
    applied_to_agent: bool = False
    agent_response: Optional[Dict[str, Any]] = None


@dataclass
class BehaviorChange:
    """行为变化记录"""
    behavior_type: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    intensity: float  # 变化强度 0-1
    timestamp: datetime
    related_parameters: List[str]


@dataclass
class FeedbackMessage:
    """反馈消息"""
    type: str  # 'info', 'success', 'warning', 'error', 'behavior_change'
    title: str
    message: str
    timestamp: datetime
    parameters: List[str] = None
    importance: int = 1  # 1-5, 5最高
    auto_hide: bool = True


class LiveFeedbackSystem:
    """实时反馈系统类
    
    负责监控参数变更并提供即时反馈，包括：
    - 参数变更的实时监控
    - 智能体行为变化的分析和报告
    - 参数组合效果评估
    - 异常情况检测和警告
    - 性能指标分析
    - 用户反馈收集和处理
    """
    
    def __init__(self, buffer_size: int = 1000):
        """初始化实时反馈系统
        
        参数:
            buffer_size: 历史记录缓冲区大小
        """
        self.buffer_size = buffer_size
        
        # 变更记录
        self.parameter_changes: deque = deque(maxlen=buffer_size)
        self.behavior_changes: deque = deque(maxlen=buffer_size)
        self.feedback_messages: deque = deque(maxlen=buffer_size)
        
        # 统计信息
        self.parameter_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'change_count': 0,
            'total_change_amount': 0.0,
            'last_change_time': None,
            'average_change_interval': 0.0
        })
        
        # 监听器
        self.change_listeners: List[Callable] = []
        self.feedback_listeners: List[Callable] = []
        self.behavior_listeners: List[Callable] = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_lock = threading.Lock()
        
        # 行为分析器
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # 智能体状态缓存
        self.agent_state_cache: Dict[str, Any] = {}
        self.last_agent_state: Dict[str, Any] = {}
        
        # 性能指标
        self.performance_metrics = {
            'average_response_time': 0.0,
            'parameter_stability': 0.0,
            'behavior_consistency': 0.0,
            'system_load': 0.0
        }
        
        # 初始化系统
        self._initialize_feedback_thresholds()
        
        print("实时反馈系统初始化完成")
    
    def _initialize_feedback_thresholds(self):
        """初始化反馈阈值配置"""
        self.feedback_thresholds = {
            # 参数变更阈值
            'large_parameter_change': 0.5,  # 大幅参数变更阈值
            'frequent_changes': {
                'time_window': 60,  # 时间窗口(秒)
                'change_count': 5   # 变更次数阈值
            },
            
            # 行为变化阈值
            'behavioral_impact_threshold': 0.3,  # 行为影响强度阈值
            'multiple_parameter_impact': 3,      # 多参数影响阈值
            
            # 性能阈值
            'response_time_warning': 2.0,  # 响应时间警告阈值(秒)
            'stability_threshold': 0.7,    # 稳定性阈值
        }
        
        # 参数影响的映射关系
        self.parameter_impacts = {
            'curiosity_weight': {
                'exploration_frequency': 0.8,
                'novelty_seeking': 0.9,
                'risk_taking': 0.3
            },
            'exploration_rate': {
                'random_action_frequency': 0.7,
                'environment_coverage': 0.6,
                'learning_opportunities': 0.5
            },
            'learning_rate': {
                'adaptation_speed': 0.9,
                'convergence_rate': 0.8,
                'stability': -0.4  # 负相关
            },
            'attention_span': {
                'focus_duration': 0.9,
                'task_completion_rate': 0.6,
                'distraction_resistance': 0.7
            },
            'decision_threshold': {
                'action_latency': 0.8,
                'confidence_level': 0.9,
                'hesitation_time': 0.7
            }
        }
    
    def start_monitoring(self):
        """启动实时监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("实时反馈监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        print("实时反馈监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 分析当前状态
                self._analyze_current_state()
                
                # 检查异常情况
                self._check_anomalies()
                
                # 更新性能指标
                self._update_performance_metrics()
                
                # 发送定期状态报告
                self._send_periodic_feedback()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(5)  # 出错后等待较长时间
    
    def notify_parameter_change(self, parameter_name: str, old_value: float, new_value: float, 
                               user_action: bool = True) -> List[FeedbackMessage]:
        """通知参数变更并生成反馈
        
        参数:
            parameter_name: 参数名称
            old_value: 旧值
            new_value: 新值
            user_action: 是否为用户操作
            
        返回:
            生成的反馈消息列表
        """
        with self.monitor_lock:
            # 创建变更记录
            change = ParameterChange(
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                timestamp=datetime.now(),
                user_action=user_action
            )
            
            self.parameter_changes.append(change)
            
            # 更新统计信息
            self._update_parameter_stats(parameter_name)
            
            # 分析参数变更的影响
            impact_analysis = self._analyze_parameter_impact(parameter_name, old_value, new_value)
            
            # 生成反馈消息
            feedback_messages = self._generate_parameter_feedback(parameter_name, old_value, new_value, impact_analysis)
            
            # 添加到消息队列
            for msg in feedback_messages:
                self.feedback_messages.append(msg)
            
            # 通知监听器
            self._notify_change_listeners(change)
            
            return feedback_messages
    
    def notify_behavior_change(self, behavior_type: str, before_state: Dict[str, Any], 
                              after_state: Dict[str, Any], related_parameters: List[str] = None) -> bool:
        """通知行为变化
        
        参数:
            behavior_type: 行为类型
            before_state: 变化前状态
            after_state: 变化后状态
            related_parameters: 相关参数列表
            
        返回:
            是否成功处理
        """
        try:
            # 计算变化强度
            intensity = self._calculate_behavior_intensity(before_state, after_state)
            
            # 创建行为变化记录
            behavior_change = BehaviorChange(
                behavior_type=behavior_type,
                before_state=before_state,
                after_state=after_state,
                intensity=intensity,
                timestamp=datetime.now(),
                related_parameters=related_parameters or []
            )
            
            self.behavior_changes.append(behavior_change)
            
            # 生成行为反馈消息
            feedback_msg = self._generate_behavior_feedback(behavior_change)
            if feedback_msg:
                self.feedback_messages.append(feedback_msg)
            
            # 通知监听器
            self._notify_behavior_listeners(behavior_change)
            
            return True
            
        except Exception as e:
            print(f"处理行为变化失败: {e}")
            return False
    
    def get_recent_changes(self, time_window: timedelta = None) -> Dict[str, List]:
        """获取最近的变更记录
        
        参数:
            time_window: 时间窗口，默认获取最近10分钟
            
        返回:
            变更记录字典
        """
        if time_window is None:
            time_window = timedelta(minutes=10)
        
        cutoff_time = datetime.now() - time_window
        
        recent_parameter_changes = [
            change for change in self.parameter_changes 
            if change.timestamp >= cutoff_time
        ]
        
        recent_behavior_changes = [
            change for change in self.behavior_changes 
            if change.timestamp >= cutoff_time
        ]
        
        recent_feedback_messages = [
            msg for msg in self.feedback_messages 
            if msg.timestamp >= cutoff_time
        ]
        
        return {
            'parameter_changes': recent_parameter_changes,
            'behavior_changes': recent_behavior_changes,
            'feedback_messages': recent_feedback_messages
        }
    
    def get_parameter_statistics(self, parameter_name: str = None) -> Dict[str, Any]:
        """获取参数统计信息
        
        参数:
            parameter_name: 参数名称，为None时获取所有参数的统计
            
        返回:
            统计信息字典
        """
        if parameter_name:
            return dict(self.parameter_stats.get(parameter_name, {}))
        
        # 整体统计
        total_changes = sum(stats['change_count'] for stats in self.parameter_stats.values())
        most_changed_param = max(self.parameter_stats.keys(), 
                               key=lambda k: self.parameter_stats[k]['change_count']) if self.parameter_stats else None
        
        # 变更频率分析
        change_frequency = {}
        current_time = datetime.now()
        
        for param_name, stats in self.parameter_stats.items():
            if stats['last_change_time']:
                time_since_last = (current_time - stats['last_change_time']).total_seconds()
                change_frequency[param_name] = {
                    'time_since_last_change': time_since_last,
                    'average_interval': stats['average_change_interval'],
                    'activity_level': 'high' if time_since_last < 300 else 'normal'
                }
        
        return {
            'total_parameter_changes': total_changes,
            'most_frequently_changed': most_changed_param,
            'parameter_activity': change_frequency,
            'change_distribution': dict(self.parameter_stats)
        }
    
    def get_behavior_insights(self, time_window: timedelta = None) -> Dict[str, Any]:
        """获取行为洞察分析
        
        参数:
            time_window: 分析时间窗口，默认最近1小时
            
        返回:
            行为洞察字典
        """
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        
        # 分析行为变化趋势
        behavior_trends = defaultdict(list)
        for change in self.behavior_changes:
            if change.timestamp >= cutoff_time:
                behavior_trends[change.behavior_type].append(change.intensity)
        
        # 计算趋势统计
        behavior_insights = {}
        for behavior_type, intensities in behavior_trends.items():
            if intensities:
                behavior_insights[behavior_type] = {
                    'change_count': len(intensities),
                    'average_intensity': sum(intensities) / len(intensities),
                    'max_intensity': max(intensities),
                    'intensity_variance': self._calculate_variance(intensities),
                    'trend': 'increasing' if len(intensities) > 1 and intensities[-1] > intensities[0] else 'stable'
                }
        
        # 参数-行为关联分析
        parameter_behavior_correlation = self._analyze_parameter_behavior_correlation(cutoff_time)
        
        return {
            'behavior_trends': behavior_insights,
            'parameter_behavior_correlation': parameter_behavior_correlation,
            'overall_behavioral_stability': self._calculate_behavioral_stability(cutoff_time),
            'behavior_change_frequency': len([c for c in self.behavior_changes if c.timestamp >= cutoff_time])
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告
        
        返回:
            性能报告字典
        """
        recent_changes = self.get_recent_changes(timedelta(hours=1))
        
        # 响应时间分析
        response_times = []
        for change in recent_changes['parameter_changes']:
            if change.applied_to_agent and change.agent_response:
                # 这里应该从agent_response中提取实际响应时间
                response_times.append(1.0)  # 模拟数据
        
        # 稳定性分析
        stability_score = self._calculate_stability_score()
        
        # 一致性分析
        consistency_score = self._calculate_consistency_score()
        
        return {
            'response_time': {
                'average': sum(response_times) / max(len(response_times), 1),
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'count': len(response_times)
            },
            'stability': stability_score,
            'consistency': consistency_score,
            'system_performance': self.performance_metrics,
            'recent_feedback_count': len(recent_changes['feedback_messages']),
            'parameter_change_frequency': len(recent_changes['parameter_changes']),
            'behavior_change_frequency': len(recent_changes['behavior_changes'])
        }
    
    def add_change_listener(self, listener: Callable[[ParameterChange], None]):
        """添加参数变更监听器
        
        参数:
            listener: 监听器函数
        """
        self.change_listeners.append(listener)
    
    def add_feedback_listener(self, listener: Callable[[FeedbackMessage], None]):
        """添加反馈消息监听器
        
        参数:
            listener: 监听器函数
        """
        self.feedback_listeners.append(listener)
    
    def add_behavior_listener(self, listener: Callable[[BehaviorChange], None]):
        """添加行为变化监听器
        
        参数:
            listener: 监听器函数
        """
        self.behavior_listeners.append(listener)
    
    def export_feedback_data(self, file_path: str, time_window: timedelta = None) -> bool:
        """导出反馈数据
        
        参数:
            file_path: 导出文件路径
            time_window: 时间窗口
            
        返回:
            导出是否成功
        """
        try:
            import json
            
            recent_data = self.get_recent_changes(time_window)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_window': str(time_window) if time_window else 'all',
                'parameter_changes': [asdict(change) for change in recent_data['parameter_changes']],
                'behavior_changes': [asdict(change) for change in recent_data['behavior_changes']],
                'feedback_messages': [asdict(msg) for msg in recent_data['feedback_messages']],
                'statistics': self.get_parameter_statistics(),
                'performance_report': self.generate_performance_report()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"导出反馈数据失败: {e}")
            return False
    
    def _update_parameter_stats(self, parameter_name: str):
        """更新参数统计信息"""
        stats = self.parameter_stats[parameter_name]
        stats['change_count'] += 1
        stats['last_change_time'] = datetime.now()
        
        # 计算平均变更间隔
        recent_changes = [
            change for change in self.parameter_changes 
            if change.parameter_name == parameter_name
        ][-10:]  # 最近10次变更
        
        if len(recent_changes) > 1:
            intervals = []
            for i in range(1, len(recent_changes)):
                interval = (recent_changes[i].timestamp - recent_changes[i-1].timestamp).total_seconds()
                intervals.append(interval)
            stats['average_change_interval'] = sum(intervals) / len(intervals)
    
    def _analyze_parameter_impact(self, parameter_name: str, old_value: float, new_value: float) -> Dict[str, Any]:
        """分析参数变更的影响"""
        impact_data = {
            'direct_impact': {},
            'indirect_impact': {},
            'overall_intensity': 0.0,
            'affected_behaviors': []
        }
        
        # 计算变化量
        change_amount = abs(new_value - old_value)
        change_percent = change_amount / max(abs(old_value), 1e-6) * 100
        
        # 获取参数影响映射
        param_impacts = self.parameter_impacts.get(parameter_name, {})
        
        # 分析直接影响
        for behavior, intensity in param_impacts.items():
            direct_impact = intensity * change_percent / 100
            impact_data['direct_impact'][behavior] = {
                'impact_strength': direct_impact,
                'impact_direction': 'positive' if intensity > 0 else 'negative'
            }
            impact_data['affected_behaviors'].append(behavior)
        
        # 计算总体影响强度
        if param_impacts:
            impact_data['overall_intensity'] = sum(
                abs(impact['impact_strength']) for impact in impact_data['direct_impact'].values()
            ) / len(param_impacts)
        
        return impact_data
    
    def _generate_parameter_feedback(self, parameter_name: str, old_value: float, new_value: float, 
                                   impact_analysis: Dict[str, Any]) -> List[FeedbackMessage]:
        """生成参数变更的反馈消息"""
        feedback_messages = []
        
        # 基本变更信息消息
        change_amount = abs(new_value - old_value)
        change_percent = (change_amount / max(abs(old_value), 1e-6)) * 100 if old_value != 0 else float('inf')
        
        if change_amount > self.feedback_thresholds['large_parameter_change']:
            feedback_messages.append(FeedbackMessage(
                type='warning',
                title='大幅参数调整',
                message=f"参数 '{parameter_name}' 发生了大幅调整 ({old_value:.3f} → {new_value:.3f}, {change_percent:.1f}%)",
                timestamp=datetime.now(),
                parameters=[parameter_name],
                importance=3
            ))
        
        # 行为影响反馈
        if impact_analysis['overall_intensity'] > self.feedback_thresholds['behavioral_impact_threshold']:
            affected_behaviors = impact_analysis['affected_behaviors']
            feedback_messages.append(FeedbackMessage(
                type='behavior_change',
                title='行为模式变化',
                message=f"参数调整可能影响以下行为: {', '.join(affected_behaviors)}",
                timestamp=datetime.now(),
                parameters=[parameter_name],
                importance=4
            ))
        
        # 频繁变更警告
        recent_changes = self._get_recent_parameter_changes(parameter_name)
        if len(recent_changes) > self.feedback_thresholds['frequent_changes']['change_count']:
            feedback_messages.append(FeedbackMessage(
                type='warning',
                title='频繁参数调整',
                message=f"参数 '{parameter_name}' 在最近1分钟内调整了 {len(recent_changes)} 次，建议检查稳定性",
                timestamp=datetime.now(),
                parameters=[parameter_name],
                importance=2
            ))
        
        return feedback_messages
    
    def _generate_behavior_feedback(self, behavior_change: BehaviorChange) -> Optional[FeedbackMessage]:
        """生成行为变化的反馈消息"""
        if behavior_change.intensity > self.feedback_thresholds['behavioral_impact_threshold']:
            return FeedbackMessage(
                type='info',
                title=f'行为模式变化: {behavior_change.behavior_type}',
                message=f"检测到 {behavior_change.behavior_type} 行为发生变化，强度: {behavior_change.intensity:.2f}",
                timestamp=behavior_change.timestamp,
                parameters=behavior_change.related_parameters,
                importance=2
            )
        return None
    
    def _calculate_behavior_intensity(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> float:
        """计算行为变化强度"""
        if not before_state or not after_state:
            return 0.0
        
        # 简化的变化强度计算
        total_difference = 0.0
        compared_keys = 0
        
        for key in set(before_state.keys()) | set(after_state.keys()):
            if key in before_state and key in after_state:
                try:
                    before_val = float(before_state[key])
                    after_val = float(after_state[key])
                    if before_val != 0:
                        diff = abs(after_val - before_val) / abs(before_val)
                        total_difference += diff
                        compared_keys += 1
                except (ValueError, TypeError):
                    continue
        
        return min(total_difference / max(compared_keys, 1), 1.0)
    
    def _analyze_current_state(self):
        """分析当前系统状态"""
        # 这里可以实现更复杂的状态分析逻辑
        pass
    
    def _check_anomalies(self):
        """检查异常情况"""
        current_time = datetime.now()
        
        # 检查参数变更频率异常
        for param_name, stats in self.parameter_stats.items():
            if stats['last_change_time']:
                time_since_last = (current_time - stats['last_change_time']).total_seconds()
                if time_since_last < 10 and stats['change_count'] > 10:
                    # 短时间内频繁变更
                    self._create_anomaly_feedback(param_name, 'rapid_changes')
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        # 模拟性能指标更新
        self.performance_metrics.update({
            'average_response_time': max(0.1, min(2.0, self.performance_metrics['average_response_time'] + 0.01)),
            'parameter_stability': max(0.0, min(1.0, 0.8 + (hash(str(datetime.now().second)) % 100 - 50) / 500)),
            'behavior_consistency': max(0.0, min(1.0, 0.7 + (hash(str(datetime.now().minute)) % 100 - 50) / 500)),
            'system_load': max(0.0, min(1.0, (len(self.parameter_changes) + len(self.behavior_changes)) / 1000))
        })
    
    def _send_periodic_feedback(self):
        """发送定期状态反馈"""
        current_time = datetime.now()
        
        # 每10分钟发送一次状态报告
        if hasattr(self, '_last_status_report'):
            if (current_time - self._last_status_report).total_seconds() < 600:
                return
        
        self._last_status_report = current_time
        
        # 生成状态报告
        status_report = FeedbackMessage(
            type='info',
            title='系统状态报告',
            message=f"参数变更: {len(self.parameter_changes)}, 行为变化: {len(self.behavior_changes)}",
            timestamp=current_time,
            importance=1
        )
        
        self.feedback_messages.append(status_report)
    
    def _create_anomaly_feedback(self, parameter_name: str, anomaly_type: str):
        """创建异常反馈消息"""
        anomaly_messages = {
            'rapid_changes': f"参数 '{parameter_name}' 变更过于频繁",
            'extreme_values': f"参数 '{parameter_name}' 接近极限值",
            'conflicting_changes': f"参数 '{parameter_name}' 的变更与其他参数冲突"
        }
        
        message = anomaly_messages.get(anomaly_type, f"参数 '{parameter_name}' 出现异常")
        
        anomaly_feedback = FeedbackMessage(
            type='error',
            title='参数异常',
            message=message,
            timestamp=datetime.now(),
            parameters=[parameter_name],
            importance=4
        )
        
        self.feedback_messages.append(anomaly_feedback)
    
    def _get_recent_parameter_changes(self, parameter_name: str, time_window: int = 60) -> List[ParameterChange]:
        """获取最近的参数变更记录"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        return [
            change for change in self.parameter_changes
            if change.parameter_name == parameter_name and change.timestamp >= cutoff_time
        ]
    
    def _analyze_parameter_behavior_correlation(self, cutoff_time: datetime) -> Dict[str, Any]:
        """分析参数-行为关联性"""
        # 简化的关联性分析
        correlations = {}
        
        for param_name in self.parameter_impacts.keys():
            param_changes = [
                change for change in self.parameter_changes
                if change.parameter_name == param_name and change.timestamp >= cutoff_time
            ]
            
            behavior_changes = [
                change for change in self.behavior_changes
                if change.timestamp >= cutoff_time
            ]
            
            # 简单的相关性计算（时间重叠）
            if param_changes and behavior_changes:
                correlation_score = len(behavior_changes) / max(len(param_changes), 1)
                correlations[param_name] = {
                    'correlation_strength': min(correlation_score, 1.0),
                    'param_change_count': len(param_changes),
                    'behavior_change_count': len(behavior_changes)
                }
        
        return correlations
    
    def _calculate_behavioral_stability(self, cutoff_time: datetime) -> float:
        """计算行为稳定性"""
        recent_changes = [
            change for change in self.behavior_changes
            if change.timestamp >= cutoff_time
        ]
        
        if not recent_changes:
            return 1.0
        
        # 基于变化强度和频率计算稳定性
        total_intensity = sum(change.intensity for change in recent_changes)
        change_frequency = len(recent_changes)
        
        stability = max(0.0, 1.0 - (total_intensity + change_frequency * 0.1) / 10)
        return min(stability, 1.0)
    
    def _calculate_stability_score(self) -> float:
        """计算稳定性得分"""
        if not self.parameter_changes:
            return 1.0
        
        recent_changes = list(self.parameter_changes)[-20:]  # 最近20次变更
        
        # 计算变更的方差
        if len(recent_changes) < 2:
            return 1.0
        
        change_amounts = [abs(c.new_value - c.old_value) for c in recent_changes]
        variance = self._calculate_variance(change_amounts)
        
        # 稳定性与方差成反比
        stability = max(0.0, 1.0 - variance / 10)
        return min(stability, 1.0)
    
    def _calculate_consistency_score(self) -> float:
        """计算一致性得分"""
        if len(self.parameter_stats) < 2:
            return 1.0
        
        # 基于参数变更的一致性计算
        change_counts = [stats['change_count'] for stats in self.parameter_stats.values()]
        
        if not change_counts:
            return 1.0
        
        mean_changes = sum(change_counts) / len(change_counts)
        variance = self._calculate_variance(change_counts)
        
        # 一致性与方差成反比
        consistency = max(0.0, 1.0 - variance / mean_changes if mean_changes > 0 else 0)
        return min(consistency, 1.0)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _notify_change_listeners(self, change: ParameterChange):
        """通知参数变更监听器"""
        for listener in self.change_listeners:
            try:
                listener(change)
            except Exception as e:
                print(f"参数变更监听器错误: {e}")
    
    def _notify_behavior_listeners(self, behavior_change: BehaviorChange):
        """通知行为变化监听器"""
        for listener in self.behavior_listeners:
            try:
                listener(behavior_change)
            except Exception as e:
                print(f"行为变化监听器错误: {e}")


class BehaviorAnalyzer:
    """行为分析器类"""
    
    def __init__(self):
        """初始化行为分析器"""
        self.behavior_patterns = {}
        self.baseline_behaviors = {}
    
    def analyze_behavior_change(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析行为变化"""
        analysis = {
            'change_detected': False,
            'change_types': [],
            'intensity': 0.0,
            'significance': 'low',
            'recommendations': []
        }
        
        # 这里可以实现更复杂的行为分析逻辑
        # 简化实现
        analysis['change_detected'] = len(before_state) > 0 and len(after_state) > 0
        analysis['intensity'] = 0.5  # 默认中等强度
        
        return analysis


# 使用示例
if __name__ == "__main__":
    # 创建实时反馈系统
    feedback_system = LiveFeedbackSystem()
    
    # 启动监控
    feedback_system.start_monitoring()
    
    # 模拟参数变更
    try:
        time.sleep(2)  # 等待系统启动
        
        # 发送参数变更通知
        messages = feedback_system.notify_parameter_change(
            "curiosity_weight", 1.0, 1.5
        )
        
        for msg in messages:
            print(f"反馈消息: {msg.title} - {msg.message}")
        
        # 获取统计数据
        stats = feedback_system.get_parameter_statistics()
        print("参数统计:", stats)
        
        # 生成性能报告
        report = feedback_system.generate_performance_report()
        print("性能报告:", report)
        
        # 导出反馈数据
        feedback_system.export_feedback_data("feedback_data.json")
        
    finally:
        # 停止监控
        feedback_system.stop_monitoring()