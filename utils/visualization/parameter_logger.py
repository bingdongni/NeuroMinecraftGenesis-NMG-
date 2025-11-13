"""
参数日志记录器
记录和追踪智能体参数的所有变更历史
"""

import json
import os
import csv
import gzip
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path


@dataclass
class ParameterLogEntry:
    """参数日志条目"""
    timestamp: datetime
    parameter_name: str
    old_value: float
    new_value: float
    change_type: str  # 'set', 'increase', 'decrease', 'reset'
    change_amount: float
    change_percent: float
    user_id: str = "system"
    session_id: str = "default"
    context: Dict[str, Any] = None
    application_status: str = "pending"  # 'pending', 'success', 'failed'
    agent_response: Dict[str, Any] = None
    tags: List[str] = None


@dataclass
class ParameterSession:
    """参数调节会话"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    user_id: str
    total_changes: int
    parameters_modified: List[str]
    session_summary: Dict[str, Any]
    tags: List[str] = None


@dataclass
class LogStatistics:
    """日志统计信息"""
    total_entries: int
    date_range: tuple
    most_changed_parameter: str
    change_frequency: Dict[str, float]
    session_count: int
    average_session_duration: float
    parameter_usage_stats: Dict[str, Dict[str, Any]]


class ParameterLogger:
    """参数日志记录器类
    
    负责记录智能体参数的所有变更历史，包括：
    - 参数变更的详细记录
    - 会话管理和追踪
    - 日志数据的压缩和归档
    - 统计分析和报告生成
    - 日志数据的查询和过滤
    - 日志导出和备份功能
    """
    
    def __init__(self, log_directory: str = "parameter_logs", max_memory_entries: int = 10000):
        """初始化参数日志记录器
        
        参数:
            log_directory: 日志文件存储目录
            max_memory_entries: 内存中保存的日志条目数量
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.memory_buffer: deque = deque(maxlen=max_memory_entries)
        
        # 日志统计
        self.log_statistics = {
            'total_entries': 0,
            'total_sessions': 0,
            'current_session': None,
            'parameter_stats': defaultdict(lambda: {
                'change_count': 0,
                'first_change': None,
                'last_change': None,
                'total_change_amount': 0.0,
                'max_single_change': 0.0
            })
        }
        
        # 监听器
        self.log_listeners: List[Callable] = []
        self.session_listeners: List[Callable] = []
        
        # 线程锁
        self.log_lock = threading.Lock()
        
        # 会话管理
        self.active_sessions: Dict[str, ParameterSession] = {}
        self.current_session_id: Optional[str] = None
        
        # 日志文件管理
        self.current_log_file: Optional[Path] = None
        self.log_rotation_size = 100 * 1024 * 1024  # 100MB
        self.auto_compress_after_days = 7
        
        # 初始化系统
        self._initialize_logging()
        
        print("参数日志记录器初始化完成")
    
    def _initialize_logging(self):
        """初始化日志系统"""
        # 创建必要的子目录
        (self.log_directory / "daily").mkdir(exist_ok=True)
        (self.log_directory / "sessions").mkdir(exist_ok=True)
        (self.log_directory / "compressed").mkdir(exist_ok=True)
        (self.log_directory / "exports").mkdir(exist_ok=True)
        
        # 初始化当日日志文件
        self._rotate_log_file()
        
        # 加载历史会话数据
        self._load_session_data()
        
        print(f"日志目录: {self.log_directory}")
    
    def start_session(self, user_id: str = "system", session_tags: List[str] = None) -> str:
        """开始新的参数调节会话
        
        参数:
            user_id: 用户ID
            session_tags: 会话标签
            
        返回:
            会话ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        session = ParameterSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            user_id=user_id,
            total_changes=0,
            parameters_modified=[],
            session_summary={},
            tags=session_tags or []
        )
        
        with self.log_lock:
            self.active_sessions[session_id] = session
            self.current_session_id = session_id
        
        # 通知监听器
        self._notify_session_listeners('start', session)
        
        print(f"开始新会话: {session_id}")
        return session_id
    
    def end_session(self, session_id: str = None) -> bool:
        """结束参数调节会话
        
        参数:
            session_id: 会话ID，为None时结束当前会话
            
        返回:
            是否成功结束
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.active_sessions:
            print(f"错误：未找到会话 {session_id}")
            return False
        
        with self.log_lock:
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            
            # 计算会话统计
            self._calculate_session_statistics(session)
            
            # 保存会话数据
            self._save_session_data(session)
            
            # 从活动会话中移除
            del self.active_sessions[session_id]
            
            # 清理当前会话
            if self.current_session_id == session_id:
                self.current_session_id = None
        
        # 通知监听器
        self._notify_session_listeners('end', session)
        
        print(f"会话结束: {session_id}, 总变更: {session.total_changes}")
        return True
    
    def log_parameter_change(self, parameter_name: str, old_value: float, new_value: float,
                           user_id: str = "system", context: Dict[str, Any] = None,
                           session_id: str = None) -> str:
        """记录参数变更
        
        参数:
            parameter_name: 参数名称
            old_value: 旧值
            new_value: 新值
            user_id: 用户ID
            context: 上下文信息
            session_id: 会话ID
            
        返回:
            日志条目ID
        """
        if session_id is None:
            session_id = self.current_session_id or "default"
        
        # 确定变更类型
        if new_value == old_value:
            change_type = "set"
        elif new_value > old_value:
            change_type = "increase"
        else:
            change_type = "decrease"
        
        # 计算变更量
        change_amount = abs(new_value - old_value)
        change_percent = (change_amount / max(abs(old_value), 1e-6)) * 100 if old_value != 0 else float('inf')
        
        # 创建日志条目
        log_entry = ParameterLogEntry(
            timestamp=datetime.now(),
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type,
            change_amount=change_amount,
            change_percent=change_percent,
            user_id=user_id,
            session_id=session_id,
            context=context or {},
            application_status="pending",
            agent_response=None,
            tags=self._generate_log_tags(parameter_name, change_type, old_value, new_value)
        )
        
        # 记录日志
        entry_id = self._record_log_entry(log_entry)
        
        # 更新会话统计
        self._update_session_statistics(session_id, parameter_name)
        
        # 通知监听器
        self._notify_log_listeners(log_entry)
        
        return entry_id
    
    def log_parameter_application(self, parameter_name: str, value: float, 
                                 application_result: Dict[str, Any]) -> bool:
        """记录参数应用到智能体的结果
        
        参数:
            parameter_name: 参数名称
            value: 应用的值
            application_result: 应用结果
            
        返回:
            记录是否成功
        """
        try:
            # 查找对应的日志条目
            latest_entry = self._find_latest_pending_entry(parameter_name, value)
            
            if latest_entry:
                latest_entry.application_status = application_result.get('success', False)
                latest_entry.agent_response = application_result
                
                # 更新内存缓冲区
                self._update_memory_buffer(latest_entry)
                
                print(f"参数应用结果已记录: {parameter_name}")
                return True
            else:
                print(f"警告：未找到对应的参数变更记录: {parameter_name}")
                return False
                
        except Exception as e:
            print(f"记录参数应用结果失败: {e}")
            return False
    
    def get_parameter_history(self, parameter_name: str, time_range: timedelta = None,
                            session_id: str = None) -> List[ParameterLogEntry]:
        """获取参数历史记录
        
        参数:
            parameter_name: 参数名称
            time_range: 时间范围
            session_id: 会话ID
            
        返回:
            历史记录列表
        """
        results = []
        cutoff_time = datetime.now() - time_range if time_range else datetime.min
        
        # 从内存缓冲区查找
        for entry in reversed(self.memory_buffer):
            if (entry.parameter_name == parameter_name and 
                entry.timestamp >= cutoff_time and
                (session_id is None or entry.session_id == session_id)):
                results.append(entry)
        
        # 从持久化存储查找
        if time_range is None or time_range.days > 1:
            # 查找持久化文件
            persistent_entries = self._search_persistent_logs(parameter_name, cutoff_time, session_id)
            results.extend(persistent_entries)
        
        # 按时间排序
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def get_session_history(self, user_id: str = None, time_range: timedelta = None) -> List[ParameterSession]:
        """获取会话历史
        
        参数:
            user_id: 用户ID
            time_range: 时间范围
            
        返回:
            会话列表
        """
        results = []
        cutoff_time = datetime.now() - time_range if time_range else datetime.min
        
        # 从活动会话中查找
        for session in self.active_sessions.values():
            if (session.start_time >= cutoff_time and 
                (user_id is None or session.user_id == user_id)):
                results.append(session)
        
        # 从文件加载历史会话
        session_files = list((self.log_directory / "sessions").glob("session_*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session = ParameterSession(**session_data)
                
                if (session.start_time >= cutoff_time and 
                    (user_id is None or session.user_id == user_id)):
                    results.append(session)
                    
            except Exception as e:
                print(f"加载会话文件失败: {session_file}, 错误: {e}")
        
        # 按开始时间排序
        results.sort(key=lambda x: x.start_time, reverse=True)
        
        return results
    
    def generate_statistics_report(self, time_range: timedelta = None) -> LogStatistics:
        """生成统计报告
        
        参数:
            time_range: 统计时间范围
            
        返回:
            统计报告对象
        """
        cutoff_time = datetime.now() - time_range if time_range else datetime.min
        
        # 获取所有相关日志条目
        relevant_entries = [
            entry for entry in self.memory_buffer
            if entry.timestamp >= cutoff_time
        ]
        
        if not relevant_entries:
            return LogStatistics(
                total_entries=0,
                date_range=(datetime.now(), datetime.now()),
                most_changed_parameter="",
                change_frequency={},
                session_count=0,
                average_session_duration=0.0,
                parameter_usage_stats={}
            )
        
        # 计算统计数据
        total_entries = len(relevant_entries)
        date_range = (
            min(entry.timestamp for entry in relevant_entries),
            max(entry.timestamp for entry in relevant_entries)
        )
        
        # 统计参数变更频率
        parameter_counts = defaultdict(int)
        for entry in relevant_entries:
            parameter_counts[entry.parameter_name] += 1
        
        most_changed_parameter = max(parameter_counts.keys(), key=lambda k: parameter_counts[k]) if parameter_counts else ""
        
        # 计算变更频率
        time_span_days = max(1, (date_range[1] - date_range[0]).days)
        change_frequency = {
            param: count / time_span_days 
            for param, count in parameter_counts.items()
        }
        
        # 统计会话信息
        sessions = self.get_session_history(time_range=time_range)
        session_count = len(sessions)
        
        if sessions:
            total_duration = sum(
                (session.end_time or datetime.now()) - session.start_time 
                for session in sessions
            ).total_seconds()
            average_session_duration = total_duration / session_count / 60  # 分钟
        else:
            average_session_duration = 0.0
        
        # 计算参数使用统计
        parameter_usage_stats = {}
        for param_name in parameter_counts.keys():
            param_entries = [entry for entry in relevant_entries if entry.parameter_name == param_name]
            
            if param_entries:
                change_amounts = [entry.change_amount for entry in param_entries]
                change_percents = [entry.change_percent for entry in param_entries]
                
                parameter_usage_stats[param_name] = {
                    'change_count': len(param_entries),
                    'average_change_amount': sum(change_amounts) / len(change_amounts),
                    'max_change_amount': max(change_amounts),
                    'average_change_percent': sum(change_percents) / len(change_percents),
                    'change_types': {
                        change_type: len([e for e in param_entries if e.change_type == change_type])
                        for change_type in set(entry.change_type for entry in param_entries)
                    }
                }
        
        return LogStatistics(
            total_entries=total_entries,
            date_range=date_range,
            most_changed_parameter=most_changed_parameter,
            change_frequency=change_frequency,
            session_count=session_count,
            average_session_duration=average_session_duration,
            parameter_usage_stats=parameter_usage_stats
        )
    
    def export_logs(self, file_path: str, format_type: str = "json", 
                   time_range: timedelta = None, parameters: List[str] = None,
                   compression: bool = False) -> bool:
        """导出日志数据
        
        参数:
            file_path: 导出文件路径
            format_type: 导出格式 ("json", "csv", "txt")
            time_range: 时间范围
            parameters: 要导出的参数列表
            compression: 是否压缩
            
        返回:
            导出是否成功
        """
        try:
            # 准备导出数据
            cutoff_time = datetime.now() - time_range if time_range else datetime.min
            
            export_entries = []
            for entry in self.memory_buffer:
                if (entry.timestamp >= cutoff_time and 
                    (parameters is None or entry.parameter_name in parameters)):
                    export_entries.append(entry)
            
            if not export_entries:
                print("没有找到符合条件的数据")
                return False
            
            # 根据格式导出
            if format_type.lower() == "json":
                success = self._export_json(export_entries, file_path, compression)
            elif format_type.lower() == "csv":
                success = self._export_csv(export_entries, file_path, compression)
            elif format_type.lower() == "txt":
                success = self._export_txt(export_entries, file_path, compression)
            else:
                print(f"不支持的导出格式: {format_type}")
                return False
            
            if success:
                print(f"成功导出 {len(export_entries)} 条记录到: {file_path}")
            
            return success
            
        except Exception as e:
            print(f"导出日志失败: {e}")
            return False
    
    def analyze_parameter_trends(self, parameter_name: str, time_range: timedelta = None) -> Dict[str, Any]:
        """分析参数趋势
        
        参数:
            parameter_name: 参数名称
            time_range: 分析时间范围
            
        返回:
            趋势分析结果
        """
        history = self.get_parameter_history(parameter_name, time_range)
        
        if not history:
            return {'error': '没有找到相关历史数据'}
        
        # 分析趋势
        values = [entry.new_value for entry in history]
        timestamps = [entry.timestamp for entry in history]
        
        # 计算趋势
        if len(values) > 1:
            # 简单线性趋势
            trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
            
            # 计算变化幅度
            total_change = values[-1] - values[0]
            max_change = max(values) - min(values)
            
            # 计算稳定性（标准差）
            import statistics
            stability = 1.0 - min(statistics.stdev(values) / max(abs(min(values)), 1e-6), 1.0)
            
        else:
            trend_direction = "stable"
            total_change = 0
            max_change = 0
            stability = 1.0
        
        # 变更频率分析
        if len(history) > 1:
            time_spans = []
            for i in range(1, len(history)):
                span = (history[i-1].timestamp - history[i].timestamp).total_seconds()
                time_spans.append(abs(span))
            
            average_change_interval = sum(time_spans) / len(time_spans)
            change_frequency = len(history) / max(1, (timestamps[0] - timestamps[-1]).days)
        else:
            average_change_interval = 0
            change_frequency = 0
        
        return {
            'parameter': parameter_name,
            'data_points': len(values),
            'trend_direction': trend_direction,
            'total_change': total_change,
            'max_change': max_change,
            'stability_score': stability,
            'change_frequency': change_frequency,
            'average_change_interval': average_change_interval,
            'current_value': values[-1] if values else 0,
            'value_range': {
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'mean': sum(values) / len(values) if values else 0
            }
        }
    
    def add_log_listener(self, listener: Callable[[ParameterLogEntry], None]):
        """添加日志监听器
        
        参数:
            listener: 监听器函数
        """
        self.log_listeners.append(listener)
    
    def add_session_listener(self, listener: Callable[[str, ParameterSession], None]):
        """添加会话监听器
        
        参数:
            listener: 监听器函数，接受事件类型和会话对象
        """
        self.session_listeners.append(listener)
    
    def cleanup_old_logs(self, retention_days: int = 30) -> int:
        """清理旧的日志文件
        
        参数:
            retention_days: 保留天数
            
        返回:
            清理的文件数量
        """
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        try:
            # 清理旧文件
            for file_path in self.log_directory.rglob("*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception as e:
                        print(f"删除文件失败: {file_path}, 错误: {e}")
            
            # 压缩旧日志
            self._compress_old_logs(cutoff_time)
            
            print(f"清理完成，删除了 {cleaned_count} 个文件")
            return cleaned_count
            
        except Exception as e:
            print(f"清理日志失败: {e}")
            return cleaned_count
    
    def _record_log_entry(self, log_entry: ParameterLogEntry) -> str:
        """记录日志条目"""
        with self.log_lock:
            # 生成唯一ID
            entry_id = f"log_{log_entry.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            
            # 添加到内存缓冲区
            self.memory_buffer.append(log_entry)
            
            # 更新统计信息
            self._update_log_statistics(log_entry)
            
            # 写入文件
            self._write_to_log_file(log_entry)
            
            return entry_id
    
    def _update_log_statistics(self, log_entry: ParameterLogEntry):
        """更新日志统计信息"""
        self.log_statistics['total_entries'] += 1
        
        param_stats = self.log_statistics['parameter_stats'][log_entry.parameter_name]
        param_stats['change_count'] += 1
        param_stats['total_change_amount'] += log_entry.change_amount
        
        if param_stats['first_change'] is None:
            param_stats['first_change'] = log_entry.timestamp
        
        param_stats['last_change'] = log_entry.timestamp
        
        if log_entry.change_amount > param_stats['max_single_change']:
            param_stats['max_single_change'] = log_entry.change_amount
    
    def _rotate_log_file(self):
        """轮转日志文件"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        new_log_file = self.log_directory / "daily" / f"parameter_log_{current_date}.json"
        
        # 检查是否需要轮转
        if (self.current_log_file and 
            self.current_log_file.exists() and 
            self.current_log_file.stat().st_size > self.log_rotation_size):
            self._compress_log_file(self.current_log_file)
        
        self.current_log_file = new_log_file
    
    def _write_to_log_file(self, log_entry: ParameterLogEntry):
        """写入日志文件"""
        if not self.current_log_file:
            self._rotate_log_file()
        
        try:
            # 追加模式写入JSON行
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(asdict(log_entry), default=str, ensure_ascii=False)
                f.write(json_line + '\n')
                
        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def _update_session_statistics(self, session_id: str, parameter_name: str):
        """更新会话统计"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.total_changes += 1
            
            if parameter_name not in session.parameters_modified:
                session.parameters_modified.append(parameter_name)
    
    def _calculate_session_statistics(self, session: ParameterSession):
        """计算会话统计"""
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
            session.session_summary = {
                'duration_seconds': duration,
                'average_change_interval': duration / max(session.total_changes, 1),
                'parameters_per_minute': (session.total_changes / max(duration, 1)) * 60
            }
    
    def _generate_log_tags(self, parameter_name: str, change_type: str, 
                          old_value: float, new_value: float) -> List[str]:
        """生成日志标签"""
        tags = [change_type]
        
        # 添加参数分类标签
        if parameter_name in ['curiosity_weight', 'exploration_rate', 'novelty_threshold']:
            tags.append('curiosity')
        elif parameter_name in ['learning_rate', 'memory_capacity', 'forgetting_rate']:
            tags.append('learning')
        elif parameter_name in ['attention_span', 'focus_intensity', 'distraction_filter']:
            tags.append('attention')
        elif parameter_name in ['decision_threshold', 'risk_tolerance', 'patience_level']:
            tags.append('decision')
        
        # 添加变更幅度标签
        change_percent = abs(new_value - old_value) / max(abs(old_value), 1e-6) * 100
        if change_percent > 50:
            tags.append('major_change')
        elif change_percent > 20:
            tags.append('moderate_change')
        
        return tags
    
    def _find_latest_pending_entry(self, parameter_name: str, value: float) -> Optional[ParameterLogEntry]:
        """查找最新的待处理条目"""
        for entry in reversed(self.memory_buffer):
            if (entry.parameter_name == parameter_name and 
                entry.new_value == value and 
                entry.application_status == "pending"):
                return entry
        return None
    
    def _update_memory_buffer(self, log_entry: ParameterLogEntry):
        """更新内存缓冲区"""
        # 在实际实现中，可能需要更新现有的条目
        pass
    
    def _search_persistent_logs(self, parameter_name: str, cutoff_time: datetime, 
                              session_id: str = None) -> List[ParameterLogEntry]:
        """搜索持久化日志"""
        results = []
        
        # 搜索相关日志文件
        log_files = list((self.log_directory / "daily").glob("parameter_log_*.json"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry_data = json.loads(line)
                            entry = ParameterLogEntry(**entry_data)
                            
                            if (entry.parameter_name == parameter_name and 
                                entry.timestamp >= cutoff_time and
                                (session_id is None or entry.session_id == session_id)):
                                results.append(entry)
                                
            except Exception as e:
                print(f"读取日志文件失败: {log_file}, 错误: {e}")
        
        return results
    
    def _export_json(self, entries: List[ParameterLogEntry], file_path: str, 
                    compression: bool = False) -> bool:
        """导出JSON格式"""
        try:
            data = [asdict(entry) for entry in entries]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            if compression and not file_path.endswith('.gz'):
                self._compress_file(file_path)
            
            return True
            
        except Exception as e:
            print(f"导出JSON失败: {e}")
            return False
    
    def _export_csv(self, entries: List[ParameterLogEntry], file_path: str, 
                   compression: bool = False) -> bool:
        """导出CSV格式"""
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                if entries:
                    fieldnames = asdict(entries[0]).keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in entries:
                        entry_dict = asdict(entry)
                        # 转换datetime对象为字符串
                        for key, value in entry_dict.items():
                            if isinstance(value, datetime):
                                entry_dict[key] = value.isoformat()
                        writer.writerow(entry_dict)
            
            if compression and not file_path.endswith('.gz'):
                self._compress_file(file_path)
            
            return True
            
        except Exception as e:
            print(f"导出CSV失败: {e}")
            return False
    
    def _export_txt(self, entries: List[ParameterLogEntry], file_path: str, 
                   compression: bool = False) -> bool:
        """导出TXT格式"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("参数变更日志报告\n")
                f.write("=" * 50 + "\n\n")
                
                for entry in entries:
                    f.write(f"时间: {entry.timestamp}\n")
                    f.write(f"参数: {entry.parameter_name}\n")
                    f.write(f"变更: {entry.old_value} → {entry.new_value}\n")
                    f.write(f"类型: {entry.change_type}\n")
                    f.write(f"幅度: {entry.change_amount:.4f} ({entry.change_percent:.2f}%)\n")
                    f.write(f"用户: {entry.user_id}\n")
                    f.write(f"会话: {entry.session_id}\n")
                    if entry.context:
                        f.write(f"上下文: {entry.context}\n")
                    f.write("-" * 30 + "\n\n")
            
            if compression and not file_path.endswith('.gz'):
                self._compress_file(file_path)
            
            return True
            
        except Exception as e:
            print(f"导出TXT失败: {e}")
            return False
    
    def _compress_file(self, file_path: str):
        """压缩文件"""
        try:
            compressed_path = f"{file_path}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(file_path)
        except Exception as e:
            print(f"压缩文件失败: {e}")
    
    def _compress_log_file(self, log_file: Path):
        """压缩日志文件"""
        if log_file.exists() and log_file.stat().st_size > 1024:  # 至少1KB才压缩
            self._compress_file(str(log_file))
    
    def _compress_old_logs(self, cutoff_time: datetime):
        """压缩旧日志"""
        pass  # 简化实现
    
    def _save_session_data(self, session: ParameterSession):
        """保存会话数据"""
        session_file = self.log_directory / "sessions" / f"{session.session_id}.json"
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"保存会话数据失败: {e}")
    
    def _load_session_data(self):
        """加载会话数据"""
        session_files = list((self.log_directory / "sessions").glob("session_*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session = ParameterSession(**session_data)
                self.log_statistics['total_sessions'] += 1
                
            except Exception as e:
                print(f"加载会话文件失败: {session_file}, 错误: {e}")
    
    def _notify_log_listeners(self, log_entry: ParameterLogEntry):
        """通知日志监听器"""
        for listener in self.log_listeners:
            try:
                listener(log_entry)
            except Exception as e:
                print(f"日志监听器错误: {e}")
    
    def _notify_session_listeners(self, event_type: str, session: ParameterSession):
        """通知会话监听器"""
        for listener in self.session_listeners:
            try:
                listener(event_type, session)
            except Exception as e:
                print(f"会话监听器错误: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建参数日志记录器
    logger = ParameterLogger()
    
    # 开始会话
    session_id = logger.start_session("test_user", ["test", "demo"])
    
    try:
        # 记录参数变更
        logger.log_parameter_change("curiosity_weight", 1.0, 1.5, "test_user")
        logger.log_parameter_change("learning_rate", 0.001, 0.002, "test_user")
        logger.log_parameter_change("attention_span", 1.0, 1.5, "test_user")
        
        # 记录参数应用结果
        logger.log_parameter_application("curiosity_weight", 1.5, {"success": True})
        
        # 获取历史记录
        history = logger.get_parameter_history("curiosity_weight")
        print(f"好奇心权重变更历史: {len(history)} 条记录")
        
        # 生成统计报告
        stats = logger.generate_statistics_report()
        print(f"统计报告: {stats.total_entries} 条记录")
        
        # 分析趋势
        trends = logger.analyze_parameter_trends("curiosity_weight")
        print(f"趋势分析: {trends}")
        
        # 导出日志
        logger.export_logs("parameter_logs.json", "json")
        
    finally:
        # 结束会话
        logger.end_session(session_id)
        
        # 清理旧日志
        logger.cleanup_old_logs(1)  # 保留1天