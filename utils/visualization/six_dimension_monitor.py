"""
六维认知能力监控主类
================

六维认知能力监控的核心管理类，负责统一管理六种认知能力：记忆力、思维力、创造力、观察力、注意力、想象力。

主要功能：
- 统一的六维能力管理接口
- 实时数据聚合和分析
- 性能指标计算
- 能力发展趋势分析
- 历史数据存储和查询

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import numpy as np
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os


class SixDimensionMonitor:
    """
    六维认知能力监控主类
    
    管理六种核心认知能力：
    1. 记忆力 (Memory) - 信息的存储、保持和检索能力
    2. 思维力 (Thinking) - 逻辑推理、分析和解决问题能力
    3. 创造力 (Creativity) - 创新思维和原创性思维能力
    4. 观察力 (Observation) - 细节捕捉和感知能力
    5. 注意力 (Attention) - 专注和认知资源分配能力
    6. 想象力 (Imagination) - 概念构建和虚拟场景能力
    """
    
    def __init__(self, data_path: str = "/workspace/data/monitoring_data"):
        """
        初始化六维能力监控器
        
        Args:
            data_path: 数据存储路径
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # 六维能力名称和描述
        self.dimensions = {
            "memory": {
                "name": "记忆力",
                "description": "信息的存储、保持和检索能力",
                "weight": 1.0
            },
            "thinking": {
                "name": "思维力", 
                "description": "逻辑推理、分析和解决问题能力",
                "weight": 1.0
            },
            "creativity": {
                "name": "创造力",
                "description": "创新思维和原创性思维能力", 
                "weight": 1.0
            },
            "observation": {
                "name": "观察力",
                "description": "细节捕捉和感知能力",
                "weight": 1.0
            },
            "attention": {
                "name": "注意力", 
                "description": "专注和认知资源分配能力",
                "weight": 1.0
            },
            "imagination": {
                "name": "想象力",
                "description": "概念构建和虚拟场景能力",
                "weight": 1.0
            }
        }
        
        # 历史数据存储
        self.historical_data = []
        
        # 当前状态
        self.current_status = {}
        
        # 性能基准值
        self.benchmark_scores = {
            "memory": {
                "excellent": 90,  # 优秀
                "good": 75,       # 良好
                "average": 60,    # 一般
                "poor": 40        # 较差
            },
            "thinking": {
                "excellent": 95,
                "good": 80,
                "average": 65,
                "poor": 45
            },
            "creativity": {
                "excellent": 88,
                "good": 72,
                "average": 55,
                "poor": 35
            },
            "observation": {
                "excellent": 92,
                "good": 78,
                "average": 62,
                "poor": 42
            },
            "attention": {
                "excellent": 85,
                "good": 70,
                "average": 55,
                "poor": 38
            },
            "imagination": {
                "excellent": 90,
                "good": 75,
                "average": 58,
                "poor": 40
            }
        }
        
        # 初始化当前状态
        self._initialize_current_status()
        
        # 加载历史数据
        self._load_historical_data()
    
    def _initialize_current_status(self):
        """初始化当前状态"""
        current_time = time.time()
        
        for dimension_key in self.dimensions.keys():
            self.current_status[dimension_key] = {
                "timestamp": current_time,
                "overall_score": self._generate_realistic_score(dimension_key),
                "response_time": self._generate_response_time(dimension_key),
                "stability": self._generate_stability(dimension_key),
                "efficiency": self._generate_efficiency(dimension_key),
                "trend": "stable",  # stable, rising, declining
                "confidence": random.uniform(0.7, 0.95)
            }
    
    def _generate_realistic_score(self, dimension_key: str) -> float:
        """
        生成符合认知能力特点的 réaliste 分数
        
        Args:
            dimension_key: 维度键名
            
        Returns:
            0-100范围内的能力分数
        """
        # 基于基准值生成分数，加入随机波动
        benchmark = self.benchmark_scores[dimension_key]
        
        # 使用正态分布生成分数，均值在良好和优秀之间
        mean_score = (benchmark["good"] + benchmark["excellent"]) / 2
        std_dev = 8.0  # 标准差
        
        score = np.random.normal(mean_score, std_dev)
        
        # 添加时间波动（模拟真实认知能力变化）
        time_factor = np.sin(time.time() / 300) * 3  # 5分钟周期的正弦波
        score += time_factor
        
        # 确保分数在合理范围内
        score = np.clip(score, 0, 100)
        
        return round(score, 1)
    
    def _generate_response_time(self, dimension_key: str) -> float:
        """
        生成响应时间（秒）
        
        Args:
            dimension_key: 维度键名
            
        Returns:
            响应时间（秒）
        """
        # 不同维度有不同的典型响应时间范围
        response_ranges = {
            "memory": (0.1, 0.8),       # 记忆力响应较快
            "thinking": (0.5, 2.5),     # 思维力需要更多思考时间
            "creativity": (1.0, 3.0),   # 创造力响应较慢
            "observation": (0.05, 0.3), # 观察力响应最快
            "attention": (0.1, 0.6),    # 注意力响应中等
            "imagination": (0.8, 2.8)   # 想象力需要构建时间
        }
        
        min_time, max_time = response_ranges[dimension_key]
        response_time = random.uniform(min_time, max_time)
        
        # 添加一些噪声
        response_time += np.random.normal(0, 0.1)
        
        return max(0.01, round(response_time, 3))
    
    def _generate_stability(self, dimension_key: str) -> float:
        """
        生成稳定性指标
        
        Args:
            dimension_key: 维度键名
            
        Returns:
            稳定性指标 (0-1)
        """
        # 稳定性通常在0.7-0.95之间
        base_stability = random.uniform(0.7, 0.95)
        
        # 不同维度有不同的稳定性特征
        stability_factors = {
            "memory": 0.9,      # 记忆力相对稳定
            "thinking": 0.8,    # 思维力有一定波动
            "creativity": 0.7,  # 创造力波动较大
            "observation": 0.95, # 观察力最稳定
            "attention": 0.75,  # 注意力容易受干扰
            "imagination": 0.8   # 想象力中等稳定
        }
        
        stability = base_stability * stability_factors[dimension_key]
        stability += np.random.normal(0, 0.05)  # 添加小量噪声
        
        return round(np.clip(stability, 0, 1), 3)
    
    def _generate_efficiency(self, dimension_key: str) -> float:
        """
        生成效率指标
        
        Args:
            dimension_key: 维度键名
            
        Returns:
            效率指标 (0-100%)
        """
        # 基于当前得分生成效率，效率通常与得分正相关
        current_score = self.current_status.get(dimension_key, {}).get("overall_score", 70)
        
        # 效率 = 基础效率 + 得分影响 + 随机波动
        base_efficiency = 60
        score_factor = (current_score - 50) * 0.3  # 得分对效率的影响
        noise = np.random.normal(0, 5)
        
        efficiency = base_efficiency + score_factor + noise
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def update_dimension_metrics(self, dimension_key: str, **kwargs):
        """
        更新指定维度的指标
        
        Args:
            dimension_key: 维度键名
            **kwargs: 更新的指标值
        """
        if dimension_key not in self.dimensions:
            raise ValueError(f"未知的维度: {dimension_key}")
        
        current_time = time.time()
        
        # 更新指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_status[dimension_key]:
                self.current_status[dimension_key][metric_name] = value
        
        # 更新时间戳
        self.current_status[dimension_key]["timestamp"] = current_time
        
        # 重新计算效率（如果得分有变化）
        if "overall_score" in kwargs:
            self.current_status[dimension_key]["efficiency"] = self._generate_efficiency(dimension_key)
        
        # 分析趋势
        self._analyze_trend(dimension_key)
    
    def _analyze_trend(self, dimension_key: str):
        """
        分析能力趋势
        
        Args:
            dimension_key: 维度键名
        """
        if len(self.historical_data) < 2:
            self.current_status[dimension_key]["trend"] = "stable"
            return
        
        # 获取最近两次得分
        recent_scores = []
        for data_point in reversed(self.historical_data[-10:]):  # 检查最近10个数据点
            if dimension_key in data_point and "overall_score" in data_point[dimension_key]:
                recent_scores.append(data_point[dimension_key]["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 2:
                self.current_status[dimension_key]["trend"] = "rising"
            elif change < -2:
                self.current_status[dimension_key]["trend"] = "declining"
            else:
                self.current_status[dimension_key]["trend"] = "stable"
    
    def get_current_metrics(self, dimension_key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取当前指标
        
        Args:
            dimension_key: 特定维度键名，如果为None则返回所有维度
            
        Returns:
            当前指标字典
        """
        if dimension_key:
            if dimension_key not in self.dimensions:
                raise ValueError(f"未知的维度: {dimension_key}")
            return self.current_status.get(dimension_key, {})
        else:
            return self.current_status.copy()
    
    def get_overall_score(self) -> float:
        """
        计算整体认知能力得分
        
        Returns:
            整体得分 (0-100)
        """
        if not self.current_status:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dim_key, dim_info in self.dimensions.items():
            if dim_key in self.current_status:
                score = self.current_status[dim_key]["overall_score"]
                weight = dim_info["weight"]
                
                total_weighted_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return round(total_weighted_score / total_weight, 1)
    
    def get_performance_level(self, dimension_key: str) -> str:
        """
        获取性能等级
        
        Args:
            dimension_key: 维度键名
            
        Returns:
            性能等级：excellent, good, average, poor
        """
        if dimension_key not in self.current_status:
            return "unknown"
        
        current_score = self.current_status[dimension_key]["overall_score"]
        benchmark = self.benchmark_scores[dimension_key]
        
        if current_score >= benchmark["excellent"]:
            return "excellent"
        elif current_score >= benchmark["good"]:
            return "good"
        elif current_score >= benchmark["average"]:
            return "average"
        else:
            return "poor"
    
    def record_data_point(self):
        """记录当前数据点"""
        data_point = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "dimensions": {}
        }
        
        # 记录各维度数据
        for dim_key in self.dimensions.keys():
            if dim_key in self.current_status:
                data_point["dimensions"][dim_key] = self.current_status[dim_key].copy()
        
        # 记录整体得分
        data_point["overall_score"] = self.get_overall_score()
        
        # 添加到历史数据
        self.historical_data.append(data_point)
        
        # 限制历史数据长度
        max_history = 1000
        if len(self.historical_data) > max_history:
            self.historical_data = self.historical_data[-max_history:]
        
        # 自动保存
        self._auto_save()
    
    def get_historical_data(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        获取指定时间范围内的历史数据
        
        Args:
            hours: 小时数
            
        Returns:
            历史数据列表
        """
        if not self.historical_data:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_data = []
        for data_point in self.historical_data:
            if data_point["timestamp"] >= cutoff_time:
                filtered_data.append(data_point)
        
        return filtered_data
    
    def get_dimension_statistics(self, dimension_key: str, hours: int = 1) -> Dict[str, Any]:
        """
        获取指定维度的统计信息
        
        Args:
            dimension_key: 维度键名
            hours: 小时数
            
        Returns:
            统计信息字典
        """
        historical_data = self.get_historical_data(hours)
        
        if not historical_data:
            return {}
        
        scores = []
        response_times = []
        stabilities = []
        efficiencies = []
        
        for data_point in historical_data:
            if dimension_key in data_point["dimensions"]:
                dim_data = data_point["dimensions"][dimension_key]
                scores.append(dim_data["overall_score"])
                response_times.append(dim_data["response_time"])
                stabilities.append(dim_data["stability"])
                efficiencies.append(dim_data["efficiency"])
        
        if not scores:
            return {}
        
        return {
            "mean_score": round(np.mean(scores), 1),
            "std_score": round(np.std(scores), 1),
            "min_score": round(np.min(scores), 1),
            "max_score": round(np.max(scores), 1),
            "mean_response_time": round(np.mean(response_times), 3),
            "mean_stability": round(np.mean(stabilities), 3),
            "mean_efficiency": round(np.mean(efficiencies), 1),
            "data_points": len(scores)
        }
    
    def _auto_save(self):
        """自动保存数据"""
        if len(self.historical_data) % 50 == 0:  # 每50个数据点保存一次
            self.save_data()
    
    def save_data(self, filename: Optional[str] = None):
        """
        保存数据到文件
        
        Args:
            filename: 文件名，如果为None则使用时间戳命名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"six_dimension_data_{timestamp}.json"
        
        filepath = os.path.join(self.data_path, filename)
        
        save_data = {
            "dimensions": self.dimensions,
            "current_status": self.current_status,
            "historical_data": self.historical_data,
            "benchmark_scores": self.benchmark_scores,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 查找最新的数据文件
            files = [f for f in os.listdir(self.data_path) if f.startswith("six_dimension_data_")]
            if not files:
                return
            
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.data_path, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # 恢复数据
            self.dimensions = save_data.get("dimensions", self.dimensions)
            self.current_status = save_data.get("current_status", self.current_status)
            self.historical_data = save_data.get("historical_data", [])
            self.benchmark_scores = save_data.get("benchmark_scores", self.benchmark_scores)
            
        except Exception as e:
            print(f"加载历史数据失败: {e}")
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择一个维度进行更新
        dimension_keys = list(self.dimensions.keys())
        dimension_to_update = random.choice(dimension_keys)
        
        # 生成新的指标值
        new_score = self._generate_realistic_score(dimension_to_update)
        new_response_time = self._generate_response_time(dimension_to_update)
        new_stability = self._generate_stability(dimension_to_update)
        
        # 更新指标
        self.update_dimension_metrics(
            dimension_to_update,
            overall_score=new_score,
            response_time=new_response_time,
            stability=new_stability
        )
        
        # 记录数据点
        self.record_data_point()
        
        return dimension_to_update


# 测试函数
def test_six_dimension_monitor():
    """测试六维能力监控器"""
    print("开始测试六维能力监控器...")
    
    # 创建监控器实例
    monitor = SixDimensionMonitor()
    
    # 测试获取当前指标
    print("\n1. 当前指标:")
    for dim_key, dim_info in monitor.dimensions.items():
        current_metrics = monitor.get_current_metrics(dim_key)
        print(f"  {dim_info['name']}: {current_metrics.get('overall_score', 0)}% "
              f"(响应时间: {current_metrics.get('response_time', 0):.3f}s)")
    
    # 测试整体得分
    print(f"\n2. 整体认知能力得分: {monitor.get_overall_score():.1f}%")
    
    # 测试性能等级
    print("\n3. 性能等级:")
    for dim_key, dim_info in monitor.dimensions.items():
        level = monitor.get_performance_level(dim_key)
        print(f"  {dim_info['name']}: {level}")
    
    # 模拟多次更新
    print("\n4. 模拟实时更新 (5次):")
    for i in range(5):
        updated_dim = monitor.simulate_real_time_update()
        dim_name = monitor.dimensions[updated_dim]['name']
        print(f"  第{i+1}次更新: {dim_name}")
    
    # 测试统计信息
    print("\n5. 最近1小时统计信息:")
    for dim_key, dim_info in monitor.dimensions.items():
        stats = monitor.get_dimension_statistics(dim_key, hours=1)
        if stats:
            print(f"  {dim_info['name']}: 平均得分 {stats['mean_score']:.1f}%, "
                  f"标准差 {stats['std_score']:.1f}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_six_dimension_monitor()