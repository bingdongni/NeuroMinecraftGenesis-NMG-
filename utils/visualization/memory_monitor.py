"""
记忆力监控模块
=============

专门负责记忆力能力的监控、评估和分析。

记忆力定义：
- 工作记忆：短期信息保持和处理能力
- 长期记忆：信息长期存储和检索能力
- 情景记忆：个人经历和事件的记忆能力
- 程序记忆：技能和习惯的记忆能力
- 语义记忆：概念和知识的记忆能力

主要功能：
- 实时记忆力指标监控
- 记忆性能评估
- 记忆模式分析
- 记忆衰退趋势检测
- 记忆增强建议

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math


class MemoryMonitor:
    """
    记忆力监控器
    
    监控和评估记忆力相关的各种认知能力：
    1. 工作记忆容量和持续时间
    2. 长期记忆检索准确率
    3. 记忆巩固效率
    4. 遗忘曲线特征
    5. 记忆干扰抗性
    6. 记忆编码速度
    """
    
    def __init__(self):
        """初始化记忆力监控器"""
        self.component_name = "记忆力监控器"
        self.last_update_time = time.time()
        
        # 记忆力的子组件
        self.memory_components = {
            "working_memory": {
                "name": "工作记忆",
                "description": "短期信息保持和处理",
                "weight": 0.25
            },
            "long_term_memory": {
                "name": "长期记忆", 
                "description": "长期信息存储和检索",
                "weight": 0.25
            },
            "episodic_memory": {
                "name": "情景记忆",
                "description": "个人经历和事件记忆",
                "weight": 0.20
            },
            "procedural_memory": {
                "name": "程序记忆",
                "description": "技能和习惯记忆", 
                "weight": 0.15
            },
            "semantic_memory": {
                "name": "语义记忆",
                "description": "概念和知识记忆",
                "weight": 0.15
            }
        }
        
        # 记忆性能基准
        self.memory_benchmarks = {
            "working_memory_capacity": {
                "excellent": 8.5,   # 记忆广度评分
                "good": 7.0,
                "average": 5.5,
                "poor": 4.0
            },
            "retrieval_accuracy": {
                "excellent": 0.95,  # 检索准确率
                "good": 0.85,
                "average": 0.75,
                "poor": 0.60
            },
            "encoding_speed": {
                "excellent": 2.5,   # 编码速度（项/秒）
                "good": 2.0,
                "average": 1.5,
                "poor": 1.0
            },
            "memory_consolidation": {
                "excellent": 0.90,  # 记忆巩固率
                "good": 0.80,
                "average": 0.70,
                "poor": 0.55
            },
            "interference_resistance": {
                "excellent": 0.85,  # 抗干扰能力
                "good": 0.75,
                "average": 0.65,
                "poor": 0.50
            }
        }
        
        # 当前记忆状态
        self.current_memory_state = self._initialize_memory_state()
        
        # 记忆历史数据
        self.memory_history = []
        
        # 记忆测试任务
        self.memory_tasks = [
            "数字序列记忆",
            "词汇记忆测试", 
            "空间位置记忆",
            "面孔记忆任务",
            "故事情节记忆",
            "图形记忆测试",
            "配对联想记忆",
            "顺序执行记忆"
        ]
    
    def _initialize_memory_state(self) -> Dict[str, Any]:
        """初始化记忆状态"""
        current_time = time.time()
        
        # 生成初始记忆指标
        working_memory_score = self._generate_working_memory_score()
        long_term_score = self._generate_long_term_memory_score()
        episodic_score = self._generate_episodic_memory_score()
        procedural_score = self._generate_procedural_memory_score()
        semantic_score = self._generate_semantic_memory_score()
        
        # 计算加权综合得分
        overall_score = (
            working_memory_score * 0.25 +
            long_term_score * 0.25 +
            episodic_score * 0.20 +
            procedural_score * 0.15 +
            semantic_score * 0.15
        )
        
        return {
            "timestamp": current_time,
            "overall_score": round(overall_score, 1),
            
            # 工作记忆指标
            "working_memory_capacity": working_memory_score,
            "working_memory_duration": self._generate_working_memory_duration(),
            "working_memory_accuracy": self._generate_working_memory_accuracy(),
            
            # 长期记忆指标
            "long_term_retrieval_accuracy": self._generate_retrieval_accuracy(),
            "long_term_storage_capacity": self._generate_storage_capacity(),
            "long_term_consolidation_rate": self._generate_consolidation_rate(),
            
            # 情景记忆指标
            "episodic_detail_richness": episodic_score,
            "episodic_temporal_sequencing": self._generate_temporal_sequencing(),
            "episodic_context_binding": self._generate_context_binding(),
            
            # 程序记忆指标
            "procedural_automaticity": procedural_score,
            "procedural_skill_retention": self._generate_skill_retention(),
            "procedural_adaptation_speed": self._generate_adaptation_speed(),
            
            # 语义记忆指标
            "semantic_network_connectivity": semantic_score,
            "semantic_concept_retrieval": self._generate_concept_retrieval(),
            "semantic_inference_ability": self._generate_inference_ability(),
            
            # 综合指标
            "memory_interference_resistance": self._generate_interference_resistance(),
            "memory_encoding_speed": self._generate_encoding_speed(),
            "memory_consolidation_rate": self._generate_consolidation_rate(),
            "memory_decay_rate": self._generate_decay_rate(),
            
            # 状态指标
            "memory_fatigue": self._generate_memory_fatigue(),
            "memory_attention_level": self._generate_attention_level(),
            "memory_motivation": self._generate_motivation_level(),
            
            # 响应时间
            "response_time": self._generate_memory_response_time(),
            
            # 稳定性
            "stability": self._generate_memory_stability(),
            
            # 效率
            "efficiency": self._calculate_memory_efficiency(overall_score),
            
            # 趋势
            "trend": "stable"
        }
    
    def _generate_working_memory_score(self) -> float:
        """生成工作记忆得分"""
        # 工作记忆通常在6-9之间（记忆广度）
        base_score = random.uniform(6.5, 8.5)
        
        # 添加时间波动（注意力波动影响）
        time_factor = np.sin(time.time() / 180) * 0.3
        score = base_score + time_factor
        
        # 转换为0-100量表
        normalized_score = (score - 4) * (100 - 0) / (9 - 4) + 0
        
        return round(np.clip(normalized_score, 0, 100), 1)
    
    def _generate_long_term_memory_score(self) -> float:
        """生成长期记忆得分"""
        # 长期记忆受睡眠、年龄、训练等因素影响
        base_score = random.uniform(65, 90)
        
        # 模拟生物钟影响（早晨记忆较好）
        hour = datetime.now().hour
        if 6 <= hour <= 10:
            circadian_boost = 5
        elif 14 <= hour <= 16:
            circadian_boost = 0
        else:
            circadian_boost = -3
        
        score = base_score + circadian_boost
        score += np.random.normal(0, 3)  # 添加随机噪声
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_episodic_memory_score(self) -> float:
        """生成情景记忆得分"""
        # 情景记忆依赖于注意力和情绪状态
        base_score = random.uniform(60, 85)
        
        # 情绪影响（积极情绪有助于情景记忆）
        mood_factor = np.random.choice([-2, 0, 2, 5], p=[0.1, 0.4, 0.3, 0.2])
        score = base_score + mood_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_procedural_memory_score(self) -> float:
        """生成程序记忆得分"""
        # 程序记忆相对稳定，不易受短期波动影响
        base_score = random.uniform(70, 88)
        
        # 程序记忆较为稳定，波动较小
        noise = np.random.normal(0, 2)
        score = base_score + noise
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_semantic_memory_score(self) -> float:
        """生成语义记忆得分"""
        # 语义记忆基于知识和经验积累
        base_score = random.uniform(65, 92)
        
        # 受教育和学习经历影响
        education_factor = np.random.uniform(-3, 8)
        score = base_score + education_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_working_memory_duration(self) -> float:
        """生成工作记忆持续时间（秒）"""
        return round(random.uniform(8, 25), 1)
    
    def _generate_working_memory_accuracy(self) -> float:
        """生成工作记忆准确率"""
        return round(random.uniform(0.75, 0.95), 3)
    
    def _generate_retrieval_accuracy(self) -> float:
        """生成检索准确率"""
        return round(random.uniform(0.80, 0.96), 3)
    
    def _generate_storage_capacity(self) -> int:
        """生成存储容量（项）"""
        return random.randint(8000, 50000)
    
    def _generate_consolidation_rate(self) -> float:
        """生成巩固率"""
        return round(random.uniform(0.70, 0.92), 3)
    
    def _generate_temporal_sequencing(self) -> float:
        """生成时间序列能力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_context_binding(self) -> float:
        """生成情境绑定能力"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_skill_retention(self) -> float:
        """生成技能保持能力"""
        return round(random.uniform(0.80, 0.95), 3)
    
    def _generate_adaptation_speed(self) -> float:
        """生成适应速度"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_concept_retrieval(self) -> float:
        """生成概念检索能力"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_inference_ability(self) -> float:
        """生成推理能力"""
        return round(random.uniform(0.65, 0.88), 3)
    
    def _generate_interference_resistance(self) -> float:
        """生成抗干扰能力"""
        return round(random.uniform(0.60, 0.85), 3)
    
    def _generate_encoding_speed(self) -> float:
        """生成编码速度（项/秒）"""
        return round(random.uniform(1.2, 2.8), 2)
    
    def _generate_decay_rate(self) -> float:
        """生成记忆衰退率"""
        return round(random.uniform(0.05, 0.20), 3)
    
    def _generate_memory_fatigue(self) -> float:
        """生成记忆疲劳度"""
        return round(random.uniform(0.10, 0.60), 3)
    
    def _generate_attention_level(self) -> float:
        """生成注意力水平"""
        return round(random.uniform(0.60, 0.90), 3)
    
    def _generate_motivation_level(self) -> float:
        """生成动机水平"""
        return round(random.uniform(0.50, 0.85), 3)
    
    def _generate_memory_response_time(self) -> float:
        """生成记忆响应时间"""
        return round(random.uniform(0.15, 0.75), 3)
    
    def _generate_memory_stability(self) -> float:
        """生成记忆稳定性"""
        return round(random.uniform(0.80, 0.95), 3)
    
    def _calculate_memory_efficiency(self, overall_score: float) -> float:
        """计算记忆效率"""
        # 效率 = 准确率 * 速度 * 100%
        # 使用生成的值而不是获取当前状态
        base_accuracy = 0.8  # 默认值
        encoding_speed = 2.0  # 默认值
        
        # 标准化速度（假设2.0为100%效率基准）
        speed_factor = min(encoding_speed / 2.0, 1.2)
        
        efficiency = (base_accuracy * speed_factor * overall_score / 100) * 100
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前记忆力指标"""
        return self.current_memory_state.copy()
    
    def update_memory_metrics(self, **kwargs):
        """更新记忆指标"""
        current_time = time.time()
        
        # 更新指定指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_memory_state:
                self.current_memory_state[metric_name] = value
        
        # 更新时间戳
        self.current_memory_state["timestamp"] = current_time
        
        # 重新计算综合得分和效率
        self._recalculate_composite_metrics()
        
        # 分析趋势
        self._analyze_memory_trend()
        
        # 添加到历史记录
        self._add_to_history()
    
    def _recalculate_composite_metrics(self):
        """重新计算综合指标"""
        # 重新计算整体得分
        working_score = self.current_memory_state["working_memory_capacity"]
        long_term_score = self.current_memory_state["long_term_retrieval_accuracy"] * 100
        episodic_score = self.current_memory_state["episodic_detail_richness"]
        procedural_score = self.current_memory_state["procedural_automaticity"]
        semantic_score = self.current_memory_state["semantic_network_connectivity"]
        
        overall_score = (
            working_score * 0.25 +
            long_term_score * 0.25 +
            episodic_score * 0.20 +
            procedural_score * 0.15 +
            semantic_score * 0.15
        )
        
        self.current_memory_state["overall_score"] = round(overall_score, 1)
        
        # 重新计算效率
        self.current_memory_state["efficiency"] = self._calculate_memory_efficiency(overall_score)
    
    def _analyze_memory_trend(self):
        """分析记忆趋势"""
        if len(self.memory_history) < 2:
            self.current_memory_state["trend"] = "stable"
            return
        
        # 获取最近两次整体得分
        recent_scores = []
        for data_point in reversed(self.memory_history[-10:]):
            if "overall_score" in data_point:
                recent_scores.append(data_point["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 3:
                self.current_memory_state["trend"] = "rising"
            elif change < -3:
                self.current_memory_state["trend"] = "declining"
            else:
                self.current_memory_state["trend"] = "stable"
    
    def _add_to_history(self):
        """添加当前状态到历史记录"""
        history_point = self.current_memory_state.copy()
        self.memory_history.append(history_point)
        
        # 限制历史记录长度
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
    
    def simulate_memory_task(self, task_type: str = "random") -> Dict[str, Any]:
        """
        模拟记忆任务执行
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        if task_type == "random":
            task_type = random.choice(self.memory_tasks)
        
        # 根据任务类型调整性能
        task_performance_factors = {
            "数字序列记忆": 0.9,
            "词汇记忆测试": 1.0,
            "空间位置记忆": 0.85,
            "面孔记忆任务": 0.80,
            "故事情节记忆": 0.95,
            "图形记忆测试": 0.88,
            "配对联想记忆": 0.92,
            "顺序执行记忆": 0.87
        }
        
        base_performance = task_performance_factors.get(task_type, 1.0)
        
        # 生成任务结果
        accuracy = base_performance * random.uniform(0.75, 0.98)
        response_time = self.current_memory_state["response_time"] * random.uniform(0.8, 1.3)
        
        task_result = {
            "task_type": task_type,
            "timestamp": time.time(),
            "accuracy": round(accuracy, 3),
            "response_time": round(response_time, 3),
            "difficulty_level": random.choice(["简单", "中等", "困难"]),
            "attention_required": random.uniform(0.6, 0.9),
            "memory_load": random.uniform(0.4, 0.8)
        }
        
        # 根据任务表现调整内存状态
        performance_impact = (accuracy - 0.85) * 10  # 性能影响因子
        self.update_memory_metrics(
            overall_score=self.current_memory_state["overall_score"] + performance_impact,
            memory_fatigue=self.current_memory_state["memory_fatigue"] + 0.05
        )
        
        return task_result
    
    def get_memory_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取记忆能力分析报告
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            记忆分析报告
        """
        if not self.memory_history:
            return {"error": "暂无历史数据"}
        
        # 过滤时间范围内的数据
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [d for d in self.memory_history if d["timestamp"] >= cutoff_time]
        
        if not recent_data:
            return {"error": "指定时间范围内无数据"}
        
        # 计算统计数据
        scores = [d["overall_score"] for d in recent_data]
        accuracies = [d.get("working_memory_accuracy", 0.8) for d in recent_data]
        response_times = [d.get("response_time", 0.5) for d in recent_data]
        
        analysis = {
            "time_range": f"最近{hours}小时",
            "data_points": len(recent_data),
            
            "overall_performance": {
                "mean_score": round(np.mean(scores), 1),
                "std_score": round(np.std(scores), 1),
                "min_score": round(np.min(scores), 1),
                "max_score": round(np.max(scores), 1),
                "trend": self._calculate_trend(scores)
            },
            
            "component_performance": {
                "working_memory": {
                    "mean_accuracy": round(np.mean(accuracies), 3),
                    "stability": round(1 - np.std(accuracies)/np.mean(accuracies), 3)
                },
                "response_speed": {
                    "mean_time": round(np.mean(response_times), 3),
                    "consistency": round(1 - np.std(response_times)/np.mean(response_times), 3)
                }
            },
            
            "memory_characteristics": {
                "dominant_memory_type": self._identify_dominant_memory_type(),
                "fatigue_level": round(np.mean([d.get("memory_fatigue", 0.3) for d in recent_data]), 3),
                "attention_stability": round(np.mean([d.get("memory_attention_level", 0.75) for d in recent_data]), 3)
            },
            
            "recommendations": self._generate_memory_recommendations(recent_data)
        }
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """计算得分趋势"""
        if len(scores) < 2:
            return "stable"
        
        # 线性回归计算趋势
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "declining"
        else:
            return "stable"
    
    def _identify_dominant_memory_type(self) -> str:
        """识别主导记忆类型"""
        current_state = self.current_memory_state
        
        memory_scores = {
            "工作记忆": current_state["working_memory_capacity"],
            "长期记忆": current_state["long_term_retrieval_accuracy"] * 100,
            "情景记忆": current_state["episodic_detail_richness"],
            "程序记忆": current_state["procedural_automaticity"],
            "语义记忆": current_state["semantic_network_connectivity"]
        }
        
        return max(memory_scores, key=memory_scores.get)
    
    def _generate_memory_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """生成记忆改善建议"""
        recommendations = []
        
        # 分析当前状态
        current_state = self.current_memory_state
        avg_fatigue = np.mean([d.get("memory_fatigue", 0.3) for d in recent_data])
        avg_attention = np.mean([d.get("memory_attention_level", 0.75) for d in recent_data])
        
        if avg_fatigue > 0.5:
            recommendations.append("记忆疲劳较高，建议适当休息")
        
        if avg_attention < 0.7:
            recommendations.append("注意力水平偏低，建议进行专注力训练")
        
        if current_state["memory_interference_resistance"] < 0.7:
            recommendations.append("抗干扰能力有待提升，建议在安静环境中进行记忆训练")
        
        if current_state["working_memory_capacity"] < 70:
            recommendations.append("工作记忆容量可加强，建议练习数字序列记忆")
        
        # 根据时间提供建议
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:
            recommendations.append("当前为记忆黄金时间，适合进行重要学习任务")
        elif 22 <= current_hour or current_hour <= 6:
            recommendations.append("当前非最佳记忆时间，建议进行轻松的记忆练习")
        
        return recommendations if recommendations else ["当前记忆状态良好，继续保持"]
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择要更新的指标
        updateable_metrics = [
            "working_memory_capacity",
            "long_term_retrieval_accuracy", 
            "episodic_detail_richness",
            "procedural_automaticity",
            "semantic_network_connectivity"
        ]
        
        metric_to_update = random.choice(updateable_metrics)
        
        # 生成新的值
        if "capacity" in metric_to_update:
            new_value = self._generate_working_memory_score()
        elif "accuracy" in metric_to_update:
            new_value = self._generate_retrieval_accuracy() * 100
        elif "richness" in metric_to_update:
            new_value = self._generate_episodic_memory_score()
        elif "automaticity" in metric_to_update:
            new_value = self._generate_procedural_memory_score()
        elif "connectivity" in metric_to_update:
            new_value = self._generate_semantic_memory_score()
        else:
            new_value = self.current_memory_state.get(metric_to_update, 70)
        
        # 更新指标
        self.update_memory_metrics(**{metric_to_update: new_value})
        
        return metric_to_update


# 测试函数
def test_memory_monitor():
    """测试记忆监控器"""
    print("开始测试记忆监控器...")
    
    # 创建记忆监控器
    memory_monitor = MemoryMonitor()
    
    # 测试当前指标
    print("\n1. 当前记忆指标:")
    current_metrics = memory_monitor.get_current_metrics()
    print(f"  整体得分: {current_metrics['overall_score']:.1f}%")
    print(f"  工作记忆: {current_metrics['working_memory_capacity']:.1f}")
    print(f"  检索准确率: {current_metrics['long_term_retrieval_accuracy']:.1%}")
    print(f"  响应时间: {current_metrics['response_time']:.3f}s")
    
    # 模拟记忆任务
    print("\n2. 模拟记忆任务:")
    for i in range(3):
        result = memory_monitor.simulate_memory_task()
        print(f"  任务{i+1}: {result['task_type']} - 准确率: {result['accuracy']:.1%}")
    
    # 模拟实时更新
    print("\n3. 模拟实时更新 (5次):")
    for i in range(5):
        metric = memory_monitor.simulate_real_time_update()
        print(f"  更新{i+1}: {metric}")
    
    # 生成分析报告
    print("\n4. 记忆能力分析:")
    analysis = memory_monitor.get_memory_analysis(hours=1)
    if "error" not in analysis:
        print(f"  数据点数量: {analysis['data_points']}")
        print(f"  平均得分: {analysis['overall_performance']['mean_score']:.1f}")
        print(f"  趋势: {analysis['overall_performance']['trend']}")
        print(f"  主要建议: {analysis['recommendations'][0]}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_memory_monitor()