#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能记录器模块
=============

这个模块负责全面记录、存储和管理智能体性能数据。
支持多种数据类型、性能指标、时间序列分析，
为趋势分析和报告生成提供数据基础。

核心功能：
- 多维度性能数据记录
- 时间序列数据存储和管理
- 数据压缩和归档
- 性能指标计算和统计
- 数据导出和查询接口

作者：AI研究团队
日期：2025-11-13
"""

import gzip
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict, deque


class PerformanceRecorder:
    """
    性能记录器类
    
    负责记录和管理所有与智能体性能相关的数据。
    支持多种数据存储格式、高效的查询分析、
    自动数据清理和归档功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能记录器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 数据存储配置
        self.data_dir = Path('/workspace/worlds/real/performance_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # 内存缓存
        self.memory_cache: deque = deque(maxlen=self.config.get('cache_size', 10000))
        self.cache_lock = threading.Lock()
        
        # 数据库配置
        self.db_path = self.data_dir / 'performance.db'
        self._init_database()
        
        # 性能指标配置
        self.metric_config = {
            'accuracy': {'min': 0.0, 'max': 1.0, 'higher_better': True},
            'precision': {'min': 0.0, 'max': 1.0, 'higher_better': True},
            'recall': {'min': 0.0, 'max': 1.0, 'higher_better': True},
            'f1_score': {'min': 0.0, 'max': 1.0, 'higher_better': True},
            'processing_time': {'min': 0.0, 'max': float('inf'), 'higher_better': False},
            'adaptation_time': {'min': 0.0, 'max': float('inf'), 'higher_better': False},
            'memory_usage': {'min': 0.0, 'max': float('inf'), 'higher_better': False},
            'cpu_usage': {'min': 0.0, 'max': 100.0, 'higher_better': False},
            'success_rate': {'min': 0.0, 'max': 1.0, 'higher_better': True},
            'stability_score': {'min': 0.0, 'max': 1.0, 'higher_better': True}
        }
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'db_writes': 0,
            'compression_ratio': 0.0
        }
        
        # 启动后台任务
        self._start_background_tasks()
        
        self.logger.info("性能记录器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'cache_size': 10000,
            'batch_size': 100,
            'auto_flush_interval': 300,  # 5分钟自动flush
            'compression_enabled': True,
            'auto_archive': True,
            'archive_interval': 24,  # 24小时归档一次
            'retention_days': 30,  # 保留30天数据
            'database_pool_size': 5,
            'enable_statistics': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('PerformanceRecorder')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_dir = Path('/workspace/worlds/real/logs')
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f'performance_recorder_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 创建性能记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        test_type TEXT NOT NULL,
                        environment TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        additional_data TEXT,
                        file_path TEXT,
                        compressed BOOLEAN DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建环境性能汇总表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS environment_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        environment TEXT NOT NULL,
                        avg_accuracy REAL,
                        avg_precision REAL,
                        avg_recall REAL,
                        avg_f1_score REAL,
                        avg_processing_time REAL,
                        total_tests INTEGER,
                        success_rate REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_records(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_type ON performance_records(test_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_environment ON performance_records(environment)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON environment_summary(date)')
                
                conn.commit()
                self.logger.info("数据库初始化完成")
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def record_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        记录性能数据
        
        这是性能记录器的核心方法，用于记录所有性能相关数据。
        
        Args:
            performance_data: 性能数据字典
            
        Returns:
            是否记录成功
        """
        try:
            # 验证数据格式
            if not self._validate_performance_data(performance_data):
                self.logger.warning("性能数据格式验证失败")
                return False
            
            # 标准化数据格式
            standardized_data = self._standardize_performance_data(performance_data)
            
            # 添加到内存缓存
            with self.cache_lock:
                self.memory_cache.append(standardized_data)
            
            # 立即写入数据库（重要数据）
            if standardized_data.get('priority', False):
                self._flush_to_database([standardized_data])
            
            self.stats['total_records'] += 1
            
            self.logger.debug(f"性能数据已记录: {standardized_data['test_type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"记录性能数据失败: {e}")
            return False
    
    def _validate_performance_data(self, data: Dict[str, Any]) -> bool:
        """验证性能数据格式"""
        required_fields = ['timestamp', 'test_type', 'environment']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"缺少必需字段: {field}")
                return False
        
        # 验证时间格式
        try:
            datetime.fromisoformat(data['timestamp'])
        except ValueError:
            self.logger.error("时间戳格式无效")
            return False
        
        return True
    
    def _standardize_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化性能数据格式"""
        standardized = {
            'timestamp': data['timestamp'],
            'test_type': data['test_type'],
            'environment': data['environment'],
            'metrics': {},
            'additional_data': {},
            'priority': data.get('priority', False)
        }
        
        # 处理性能指标
        if 'metrics' in data:
            metrics = data['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    # 验证指标范围
                    if self._validate_metric_value(metric_name, metric_value):
                        standardized['metrics'][metric_name] = float(metric_value)
        
        # 处理额外数据
        if 'additional_data' in data:
            standardized['additional_data'] = data['additional_data']
        
        # 添加执行时间
        if 'execution_time' in data:
            standardized['additional_data']['execution_time'] = data['execution_time']
        
        return standardized
    
    def _validate_metric_value(self, metric_name: str, value: float) -> bool:
        """验证指标值范围"""
        if metric_name not in self.metric_config:
            return True  # 未知指标，允许记录
        
        config = self.metric_config[metric_name]
        min_val = config['min']
        max_val = config['max']
        
        if not (min_val <= value <= max_val):
            self.logger.warning(f"指标 {metric_name} 值 {value} 超出范围 [{min_val}, {max_val}]")
            return False
        
        return True
    
    def batch_record_performance(self, performance_list: List[Dict[str, Any]]) -> int:
        """
        批量记录性能数据
        
        Args:
            performance_list: 性能数据列表
            
        Returns:
            成功记录的条数
        """
        successful_count = 0
        
        for data in performance_list:
            if self.record_performance(data):
                successful_count += 1
        
        self.logger.info(f"批量记录完成: {successful_count}/{len(performance_list)} 条成功")
        return successful_count
    
    def get_performance_records(self, start_time: Optional[str] = None,
                               end_time: Optional[str] = None,
                               test_type: Optional[str] = None,
                               environment: Optional[str] = None,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        查询性能记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            test_type: 测试类型过滤
            environment: 环境过滤
            limit: 返回记录数限制
            
        Returns:
            性能记录列表
        """
        try:
            query_conditions = []
            query_params = []
            
            # 构建查询条件
            if start_time:
                query_conditions.append("timestamp >= ?")
                query_params.append(start_time)
            
            if end_time:
                query_conditions.append("timestamp <= ?")
                query_params.append(end_time)
            
            if test_type:
                query_conditions.append("test_type = ?")
                query_params.append(test_type)
            
            if environment:
                query_conditions.append("environment = ?")
                query_params.append(environment)
            
            # 构建SQL查询
            where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"
            sql = f"SELECT * FROM performance_records WHERE {where_clause} ORDER BY timestamp DESC"
            
            if limit:
                sql += f" LIMIT {limit}"
            
            # 执行查询
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, query_params)
                
                records = []
                for row in cursor.fetchall():
                    record = dict(row)
                    # 解析JSON字段
                    if record['metrics']:
                        record['metrics'] = json.loads(record['metrics'])
                    if record['additional_data']:
                        record['additional_data'] = json.loads(record['additional_data'])
                    
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"查询性能记录失败: {e}")
            return []
    
    def get_environment_performance_summary(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        获取环境性能汇总
        
        Args:
            days: 天数
            
        Returns:
            环境性能汇总字典
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 查询指定时间范围内的数据
                cursor.execute('''
                    SELECT environment, test_type, metrics, additional_data
                    FROM performance_records 
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                # 按环境分组统计
                env_stats = defaultdict(lambda: {
                    'test_count': 0,
                    'total_execution_time': 0,
                    'metrics_sum': defaultdict(float),
                    'metrics_count': defaultdict(int)
                })
                
                for row in cursor.fetchall():
                    env = row['environment']
                    metrics = json.loads(row['metrics'])
                    additional_data = json.loads(row['additional_data']) if row['additional_data'] else {}
                    
                    env_stats[env]['test_count'] += 1
                    
                    if 'execution_time' in additional_data:
                        env_stats[env]['total_execution_time'] += additional_data['execution_time']
                    
                    # 累加指标
                    for metric_name, value in metrics.items():
                        env_stats[env]['metrics_sum'][metric_name] += value
                        env_stats[env]['metrics_count'][metric_name] += 1
                
                # 计算平均值
                summary = {}
                for env, stats in env_stats.items():
                    env_summary = {
                        'test_count': stats['test_count'],
                        'average_execution_time': stats['total_execution_time'] / max(stats['test_count'], 1)
                    }
                    
                    # 计算各指标平均值
                    for metric_name, sum_value in stats['metrics_sum'].items():
                        count = stats['metrics_count'][metric_name]
                        env_summary[f'avg_{metric_name}'] = sum_value / max(count, 1)
                    
                    summary[env] = env_summary
                
                return summary
                
        except Exception as e:
            self.logger.error(f"获取环境性能汇总失败: {e}")
            return {}
    
    def calculate_performance_trends(self, metric_name: str, 
                                   days: int = 30) -> Dict[str, Any]:
        """
        计算性能趋势
        
        Args:
            metric_name: 指标名称
            days: 计算天数
            
        Returns:
            趋势分析结果
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 获取数据
            records = self.get_performance_records(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat()
            )
            
            # 提取指标数据
            metric_data = []
            timestamps = []
            
            for record in records:
                if metric_name in record['metrics']:
                    metric_data.append(record['metrics'][metric_name])
                    timestamps.append(datetime.fromisoformat(record['timestamp']))
            
            if len(metric_data) < 2:
                return {'error': '数据点不足，无法计算趋势'}
            
            # 计算统计信息
            data_array = np.array(metric_data)
            timestamps_array = np.array([t.timestamp() for t in timestamps])
            
            # 基本统计
            trend_stats = {
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'std': float(np.std(data_array)),
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'trend': 'stable'
            }
            
            # 计算趋势
            if len(data_array) >= 2:
                # 线性回归
                coeffs = np.polyfit(timestamps_array, data_array, 1)
                slope = coeffs[0]
                
                # 计算相关系数
                correlation = np.corrcoef(timestamps_array, data_array)[0, 1]
                
                # 判断趋势方向
                if abs(slope) > 1e-6:  # 斜率显著
                    trend_stats['trend'] = 'increasing' if slope > 0 else 'decreasing'
                
                trend_stats['slope'] = float(slope)
                trend_stats['correlation'] = float(correlation)
            
            # 计算变化率
            if len(data_array) > 1:
                first_half = data_array[:len(data_array)//2]
                second_half = data_array[len(data_array)//2:]
                
                first_mean = np.mean(first_half)
                second_mean = np.mean(second_half)
                
                if first_mean > 0:
                    change_rate = (second_mean - first_mean) / first_mean
                    trend_stats['change_rate'] = float(change_rate)
                    trend_stats['improvement'] = change_rate > 0
            
            trend_stats['data_points'] = len(metric_data)
            trend_stats['time_range'] = {
                'start': timestamps[0].isoformat(),
                'end': timestamps[-1].isoformat()
            }
            
            return trend_stats
            
        except Exception as e:
            self.logger.error(f"计算性能趋势失败: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, format: str = 'json',
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None,
                              output_path: Optional[str] = None) -> str:
        """
        导出性能数据
        
        Args:
            format: 导出格式 ('json', 'csv', 'excel')
            start_time: 开始时间
            end_time: 结束时间
            output_path: 输出文件路径
            
        Returns:
            导出文件路径
        """
        try:
            # 获取数据
            records = self.get_performance_records(start_time, end_time)
            
            if not records:
                self.logger.warning("没有找到符合条件的数据")
                return ""
            
            # 生成输出路径
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.data_dir / f'performance_export_{timestamp}.{format}'
            
            output_path = Path(output_path)
            
            # 根据格式导出
            if format.lower() == 'json':
                self._export_to_json(records, output_path)
            elif format.lower() == 'csv':
                self._export_to_csv(records, output_path)
            elif format.lower() == 'excel':
                self._export_to_excel(records, output_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"性能数据已导出到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"导出性能数据失败: {e}")
            return ""
    
    def _export_to_json(self, records: List[Dict], output_path: Path):
        """导出为JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_to_csv(self, records: List[Dict], output_path: Path):
        """导出为CSV格式"""
        # 将嵌套字典展开为平面结构
        flat_records = []
        for record in records:
            flat_record = {
                'timestamp': record['timestamp'],
                'test_type': record['test_type'],
                'environment': record['environment']
            }
            
            # 添加指标
            for metric_name, value in record['metrics'].items():
                flat_record[f'metric_{metric_name}'] = value
            
            # 添加额外数据
            if record['additional_data']:
                for key, value in record['additional_data'].items():
                    flat_record[f'extra_{key}'] = value
            
            flat_records.append(flat_record)
        
        df = pd.DataFrame(flat_records)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _export_to_excel(self, records: List[Dict], output_path: Path):
        """导出为Excel格式"""
        # 转换数据
        flat_records = []
        for record in records:
            flat_record = {
                'timestamp': record['timestamp'],
                'test_type': record['test_type'],
                'environment': record['environment']
            }
            
            # 添加指标
            for metric_name, value in record['metrics'].items():
                flat_record[f'metric_{metric_name}'] = value
            
            flat_records.append(flat_record)
        
        df = pd.DataFrame(flat_records)
        df.to_excel(output_path, index=False, engine='openpyxl')
    
    def _flush_to_database(self, data_list: List[Dict[str, Any]]):
        """将数据写入数据库"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                for data in data_list:
                    cursor.execute('''
                        INSERT INTO performance_records 
                        (timestamp, test_type, environment, metrics, additional_data, priority)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        data['timestamp'],
                        data['test_type'],
                        data['environment'],
                        json.dumps(data['metrics']),
                        json.dumps(data['additional_data']),
                        data.get('priority', False)
                    ))
                
                conn.commit()
                self.stats['db_writes'] += len(data_list)
                
        except Exception as e:
            self.logger.error(f"写入数据库失败: {e}")
            raise
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def background_flush():
            while True:
                try:
                    self._flush_cache()
                    time.sleep(self.config['auto_flush_interval'])
                except Exception as e:
                    self.logger.error(f"后台flush任务错误: {e}")
                    time.sleep(60)
        
        def background_archive():
            while True:
                try:
                    self._archive_old_data()
                    time.sleep(self.config['archive_interval'] * 3600)  # 转换为秒
                except Exception as e:
                    self.logger.error(f"后台archive任务错误: {e}")
                    time.sleep(3600)
        
        # 启动flush线程
        flush_thread = threading.Thread(target=background_flush, daemon=True)
        flush_thread.start()
        
        # 启动archive线程
        if self.config['auto_archive']:
            archive_thread = threading.Thread(target=background_archive, daemon=True)
            archive_thread.start()
    
    def _flush_cache(self):
        """刷新缓存到数据库"""
        with self.cache_lock:
            if not self.memory_cache:
                return
            
            # 获取所有缓存数据
            cache_data = list(self.memory_cache)
            self.memory_cache.clear()
        
        if cache_data:
            try:
                self._flush_to_database(cache_data)
                self.logger.debug(f"已刷新 {len(cache_data)} 条缓存记录到数据库")
            except Exception as e:
                self.logger.error(f"刷新缓存失败: {e}")
                # 将数据重新放入缓存
                with self.cache_lock:
                    self.memory_cache.extend(cache_data)
    
    def _archive_old_data(self):  # noqa: F841
        """归档旧数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 查询需要归档的记录
                cursor.execute('''
                    SELECT * FROM performance_records 
                    WHERE timestamp < ? 
                    ORDER BY timestamp
                ''', (cutoff_date.isoformat(),))
                
                records = cursor.fetchall()
                
                if not records:
                    return
                
                # 创建归档文件
                archive_dir = self.data_dir / 'archive'
                archive_dir.mkdir(exist_ok=True)
                
                archive_file = archive_dir / f'performance_archive_{datetime.now().strftime("%Y%m%d")}.json.gz'
                
                # 导出到压缩文件
                archive_data = [dict(row) for row in records]
                
                with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
                    json.dump(archive_data, f, indent=2, ensure_ascii=False, default=str)
                
                # 删除已归档的记录
                cursor.execute('''
                    DELETE FROM performance_records 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                conn.commit()
                
                self.logger.info(f"已归档 {len(records)} 条记录到 {archive_file}")
                
                # 计算压缩比
                original_size = len(str(archive_data))
                compressed_size = archive_file.stat().st_size
                self.stats['compression_ratio'] = compressed_size / original_size
                
        except Exception as e:
            self.logger.error(f"归档旧数据失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_size': len(self.memory_cache),
            'database_records': self._get_database_record_count(),
            'disk_usage_mb': self._calculate_disk_usage() / (1024 * 1024)
        }
    
    def _get_database_record_count(self) -> int:
        """获取数据库记录数"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM performance_records')
                return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def _calculate_disk_usage(self) -> int:
        """计算磁盘使用量"""
        total_size = 0
        try:
            # 计算数据库文件大小
            if self.db_path.exists():
                total_size += self.db_path.stat().st_size
            
            # 计算归档文件大小
            archive_dir = self.data_dir / 'archive'
            if archive_dir.exists():
                for file_path in archive_dir.glob('*.gz'):
                    total_size += file_path.stat().st_size
                    
        except Exception as e:
            self.logger.error(f"计算磁盘使用量失败: {e}")
        
        return total_size
    
    def cleanup(self):
        """清理资源"""
        try:
            # 刷新所有缓存
            self._flush_cache()
            
            # 关闭数据库连接池
            self.logger.info("性能记录器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'cleanup'):
            self.cleanup()


if __name__ == "__main__":
    # 示例用法
    recorder = PerformanceRecorder()
    
    # 记录测试数据
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'image_classification',
        'environment': 'real_world_task',
        'metrics': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.91,
            'f1_score': 0.90
        },
        'execution_time': 2.5,
        'additional_data': {
            'model_version': 'v1.0',
            'dataset_size': 1000
        },
        'priority': True
    }
    
    # 记录性能
    if recorder.record_performance(test_data):
        print("性能数据记录成功")
    
    # 获取汇总统计
    summary = recorder.get_environment_performance_summary()
    print(f"环境性能汇总: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 计算趋势
    trends = recorder.calculate_performance_trends('accuracy')
    print(f"准确性趋势: {json.dumps(trends, indent=2, ensure_ascii=False)}")
    
    # 获取统计信息
    stats = recorder.get_statistics()
    print(f"记录器统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 清理资源
    recorder.cleanup()