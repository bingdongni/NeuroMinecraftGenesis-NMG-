"""
注意力监控模块
=============

专门负责注意力能力的监控、评估和分析。

注意力定义：
- 持续性注意力：长时间保持专注的能力
- 选择性注意力：从多个刺激中选择目标的能力
- 分散性注意力：同时处理多个任务的能力
- 转换性注意力：快速切换注意焦点的能力
- 监控性注意力：维持认知控制和自我调节
- 警觉性：保持清醒和反应灵敏的状态

主要功能：
- 实时注意力指标监控
- 专注能力评估
- 注意缺陷分析
- 注意力训练建议
- 注意调节策略

Author: Claude Code Agent
Date: 2025-11-13
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math


class AttentionMonitor:
    """
    注意力监控器
    
    监控和评估注意力相关的各种认知能力：
    1. 持续性注意力维持能力
    2. 选择性注意力过滤能力
    3. 分散性注意力分配能力
    4. 转换性注意力灵活性
    5. 监控性注意力控制
    6. 警觉性和唤醒水平
    """
    
    def __init__(self):
        """初始化注意力监控器"""
        self.component_name = "注意力监控器"
        self.last_update_time = time.time()
        
        # 注意力的子组件
        self.attention_components = {
            "sustained_attention": {
                "name": "持续性注意力",
                "description": "长时间保持专注的能力",
                "weight": 0.25
            },
            "selective_attention": {
                "name": "选择性注意力",
                "description": "从多个刺激中选择目标",
                "weight": 0.25
            },
            "divided_attention": {
                "name": "分散性注意力",
                "description": "同时处理多个任务的能力",
                "weight": 0.20
            },
            "switching_attention": {
                "name": "转换性注意力",
                "description": "快速切换注意焦点",
                "weight": 0.15
            },
            "monitoring_attention": {
                "name": "监控性注意力",
                "description": "维持认知控制和自我调节",
                "weight": 0.15
            }
        }
        
        # 注意力性能基准
        self.attention_benchmarks = {
            "attention_span": {
                "excellent": 45.0,  # 持续注意力时长（分钟）
                "good": 35.0,
                "average": 25.0,
                "poor": 15.0
            },
            "selective_accuracy": {
                "excellent": 0.92,  # 选择性注意力准确率
                "good": 0.85,
                "average": 0.78,
                "poor": 0.65
            },
            "multitask_efficiency": {
                "excellent": 0.88,  # 多任务效率
                "good": 0.78,
                "average": 0.68,
                "poor": 0.55
            },
            "attention_flexibility": {
                "excellent": 0.90,  # 注意力灵活性
                "good": 0.82,
                "average": 0.74,
                "poor": 0.60
            },
            "vigilance_level": {
                "excellent": 0.90,  # 警觉性水平
                "good": 0.80,
                "average": 0.70,
                "poor": 0.55
            }
        }
        
        # 当前注意力状态
        self.current_attention_state = self._initialize_attention_state()
        
        # 注意力历史数据
        self.attention_history = []
        
        # 注意力测试任务
        self.attention_tasks = [
            "持续注意力测试",
            "选择性注意任务",
            "分散注意力挑战",
            "注意力转换练习",
            "警觉性评估",
            "注意力维持任务",
            "干扰抵抗测试",
            "多任务处理挑战",
            "注意力分配训练",
            "认知控制测试"
        ]
    
    def _initialize_attention_state(self) -> Dict[str, Any]:
        """初始化注意力状态"""
        current_time = time.time()
        
        # 生成初始注意力指标
        sustained_score = self._generate_sustained_attention_score()
        selective_score = self._generate_selective_attention_score()
        divided_score = self._generate_divided_attention_score()
        switching_score = self._generate_switching_attention_score()
        monitoring_score = self._generate_monitoring_attention_score()
        
        # 计算加权综合得分
        overall_score = (
            sustained_score * 0.25 +
            selective_score * 0.25 +
            divided_score * 0.20 +
            switching_score * 0.15 +
            monitoring_score * 0.15
        )
        
        return {
            "timestamp": current_time,
            "overall_score": round(overall_score, 1),
            
            # 持续性注意力指标
            "attention_span": sustained_score,
            "attention_persistence": self._generate_attention_persistence(),
            "focus_stability": self._generate_focus_stability(),
            "distraction_resistance": self._generate_distraction_resistance(),
            "cognitive_endurance": self._generate_cognitive_endurance(),
            
            # 选择性注意力指标
            "selective_accuracy": selective_score,
            "target_detection": self._generate_target_detection(),
            "irrelevant_filtering": self._generate_irrelevant_filtering(),
            "signal_discrimination": self._generate_signal_discrimination(),
            "interference_control": self._generate_interference_control(),
            
            # 分散性注意力指标
            "multitask_capacity": divided_score,
            "task_switching_cost": self._generate_task_switching_cost(),
            "resource_allocation": self._generate_resource_allocation(),
            "parallel_processing": self._generate_parallel_processing(),
            "attention_division": self._generate_attention_division(),
            
            # 转换性注意力指标
            "attention_flexibility": switching_score,
            "switching_speed": self._generate_switching_speed(),
            "cognitive_inhibition": self._generate_cognitive_inhibition(),
            "mental_flexibility": self._generate_mental_flexibility_attention(),
            "attentional_shift": self._generate_attentional_shift(),
            
            # 监控性注意力指标
            "monitoring_attention": monitoring_score,
            "metacognitive_awareness": self._generate_metacognitive_awareness(),
            "error_monitoring": self._generate_error_monitoring(),
            "performance_control": self._generate_performance_control(),
            "self_regulation": self._generate_self_regulation(),
            
            # 警觉性指标
            "vigilance_level": self._generate_vigilance_level(),
            "alertness": self._generate_alertness(),
            "reactivity": self._generate_reactivity(),
            "wakefulness": self._generate_wakefulness(),
            
            # 注意力状态指标
            "attention_fatigue": self._generate_attention_fatigue(),
            "cognitive_load": self._generate_cognitive_load_attention(),
            "mental_energy_attention": self._generate_mental_energy_attention(),
            "stress_impact": self._generate_stress_impact(),
            
            # 注意力性能指标
            "attention_efficiency": self._generate_attention_efficiency(),
            "processing_speed_attention": self._generate_processing_speed_attention(),
            "accuracy_maintenance": self._generate_accuracy_maintenance(),
            "performance_consistency": self._generate_performance_consistency(),
            
            # 响应时间
            "response_time": self._generate_attention_response_time(),
            
            # 稳定性
            "stability": self._generate_attention_stability(),
            
            # 效率
            "efficiency": self._calculate_attention_efficiency(overall_score),
            
            # 趋势
            "trend": "stable"
        }
    
    def _generate_sustained_attention_score(self) -> float:
        """生成持续性注意力得分"""
        # 持续注意力受疲劳和动机影响
        base_score = random.uniform(65, 88)
        
        # 时间因素（注意力在早晨和下午较好）
        hour = datetime.now().hour
        if 8 <= hour <= 11:
            attention_boost = 6  # 早晨注意力佳
        elif 14 <= hour <= 16:
            attention_boost = 3  # 下午一般
        elif 20 <= hour <= 23:
            attention_boost = -5  # 晚间下降
        else:
            attention_boost = -8  # 夜晚较差
        
        # 疲劳因素
        fatigue_factor = np.random.uniform(-8, 2)
        score = base_score + attention_boost + fatigue_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_selective_attention_score(self) -> float:
        """生成选择性注意力得分"""
        # 选择性注意力相对稳定
        base_score = random.uniform(70, 90)
        
        # 干扰因素
        interference_factor = np.random.uniform(-5, 4)
        score = base_score + interference_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_divided_attention_score(self) -> float:
        """生成分散性注意力得分"""
        # 分散注意力比较困难，波动较大
        base_score = random.uniform(60, 82)
        
        # 认知负荷影响
        load_factor = np.random.uniform(-6, 3)
        score = base_score + load_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_switching_attention_score(self) -> float:
        """生成转换性注意力得分"""
        # 转换性注意力受灵活性影响
        base_score = random.uniform(68, 86)
        
        # 灵活性因素
        flexibility_factor = np.random.uniform(-4, 6)
        score = base_score + flexibility_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_monitoring_attention_score(self) -> float:
        """生成监控性注意力得分"""
        # 监控性注意力需要自我意识
        base_score = random.uniform(65, 85)
        
        # 自我调节因素
        self_regulation_factor = np.random.uniform(-3, 5)
        score = base_score + self_regulation_factor
        
        return round(np.clip(score, 0, 100), 1)
    
    def _generate_attention_persistence(self) -> float:
        """生成注意力持续性"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_focus_stability(self) -> float:
        """生成焦点稳定性"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_distraction_resistance(self) -> float:
        """生成抗干扰能力"""
        return round(random.uniform(0.68, 0.88), 3)
    
    def _generate_cognitive_endurance(self) -> float:
        """生成认知耐力"""
        return round(random.uniform(0.65, 0.85), 3)
    
    def _generate_target_detection(self) -> float:
        """生成目标检测能力"""
        return round(random.uniform(0.78, 0.93), 3)
    
    def _generate_irrelevant_filtering(self) -> float:
        """生成无关信息过滤"""
        return round(random.uniform(0.72, 0.89), 3)
    
    def _generate_signal_discrimination(self) -> float:
        """生成信号辨别能力"""
        return round(random.uniform(0.75, 0.91), 3)
    
    def _generate_interference_control(self) -> float:
        """生成干扰控制能力"""
        return round(random.uniform(0.70, 0.87), 3)
    
    def _generate_task_switching_cost(self) -> float:
        """生成任务切换成本"""
        return round(random.uniform(0.15, 0.35), 3)  # 较低成本更好
    
    def _generate_resource_allocation(self) -> float:
        """生成资源分配能力"""
        return round(random.uniform(0.68, 0.88), 3)
    
    def _generate_parallel_processing(self) -> float:
        """生成并行处理能力"""
        return round(random.uniform(0.65, 0.85), 3)
    
    def _generate_attention_division(self) -> float:
        """生成分配注意力能力"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_switching_speed(self) -> float:
        """生成切换速度"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_cognitive_inhibition(self) -> float:
        """生成认知抑制能力"""
        return round(random.uniform(0.72, 0.89), 3)
    
    def _generate_mental_flexibility_attention(self) -> float:
        """生成思维灵活性"""
        return round(random.uniform(0.78, 0.93), 3)
    
    def _generate_attentional_shift(self) -> float:
        """生成注意力转移"""
        return round(random.uniform(0.73, 0.90), 3)
    
    def _generate_metacognitive_awareness(self) -> float:
        """生成元认知意识"""
        return round(random.uniform(0.70, 0.88), 3)
    
    def _generate_error_monitoring(self) -> float:
        """生成错误监控"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_performance_control(self) -> float:
        """生成性能控制"""
        return round(random.uniform(0.72, 0.89), 3)
    
    def _generate_self_regulation(self) -> float:
        """生成自我调节"""
        return round(random.uniform(0.68, 0.87), 3)
    
    def _generate_vigilance_level(self) -> float:
        """生成警觉性水平"""
        hour = datetime.now().hour
        if 8 <= hour <= 11 or 14 <= hour <= 16:
            return round(random.uniform(0.75, 0.92), 3)
        elif 20 <= hour <= 23:
            return round(random.uniform(0.60, 0.80), 3)
        else:
            return round(random.uniform(0.50, 0.70), 3)
    
    def _generate_alertness(self) -> float:
        """生成警觉性"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_reactivity(self) -> float:
        """生成反应性"""
        return round(random.uniform(0.75, 0.93), 3)
    
    def _generate_wakefulness(self) -> float:
        """生成清醒度"""
        hour = datetime.now().hour
        if 8 <= hour <= 22:
            return round(random.uniform(0.80, 0.95), 3)
        else:
            return round(random.uniform(0.60, 0.80), 3)
    
    def _generate_attention_fatigue(self) -> float:
        """生成注意力疲劳"""
        return round(random.uniform(0.20, 0.65), 3)
    
    def _generate_cognitive_load_attention(self) -> float:
        """生成认知负荷"""
        return round(random.uniform(0.30, 0.75), 3)
    
    def _generate_mental_energy_attention(self) -> float:
        """生成心理能量"""
        return round(random.uniform(0.60, 0.88), 3)
    
    def _generate_stress_impact(self) -> float:
        """生成压力影响"""
        return round(random.uniform(0.25, 0.60), 3)
    
    def _generate_attention_efficiency(self) -> float:
        """生成注意力效率"""
        return round(random.uniform(0.70, 0.90), 3)
    
    def _generate_processing_speed_attention(self) -> float:
        """生成处理速度"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _generate_accuracy_maintenance(self) -> float:
        """生成准确性维持"""
        return round(random.uniform(0.78, 0.94), 3)
    
    def _generate_performance_consistency(self) -> float:
        """生成性能一致性"""
        return round(random.uniform(0.72, 0.89), 3)
    
    def _generate_attention_response_time(self) -> float:
        """生成注意力响应时间"""
        return round(random.uniform(0.4, 1.5), 2)
    
    def _generate_attention_stability(self) -> float:
        """生成注意力稳定性"""
        return round(random.uniform(0.75, 0.92), 3)
    
    def _calculate_attention_efficiency(self, overall_score: float) -> float:
        """计算注意力效率"""
        # 效率 = 准确性 * 持续性 * 速度 / 100
        accuracy = self.current_attention_state.get("selective_accuracy", 0.85)
        persistence = (self.current_attention_state.get("attention_span", 30) / 45.0)
        speed = self.current_attention_state.get("processing_speed_attention", 0.85)
        
        efficiency = (accuracy * min(persistence, 1.0) * speed * overall_score / 100) * 100
        
        return round(np.clip(efficiency, 0, 100), 1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前注意力指标"""
        return self.current_attention_state.copy()
    
    def update_attention_metrics(self, **kwargs):
        """更新注意力指标"""
        current_time = time.time()
        
        # 更新指定指标
        for metric_name, value in kwargs.items():
            if metric_name in self.current_attention_state:
                self.current_attention_state[metric_name] = value
        
        # 更新时间戳
        self.current_attention_state["timestamp"] = current_time
        
        # 重新计算综合指标
        self._recalculate_composite_metrics()
        
        # 分析趋势
        self._analyze_attention_trend()
        
        # 添加到历史记录
        self._add_to_history()
    
    def _recalculate_composite_metrics(self):
        """重新计算综合指标"""
        # 重新计算整体得分
        sustained_score = self.current_attention_state["attention_span"]
        selective_score = self.current_attention_state["selective_accuracy"]
        divided_score = self.current_attention_state["multitask_capacity"]
        switching_score = self.current_attention_state["attention_flexibility"]
        monitoring_score = self.current_attention_state["monitoring_attention"]
        
        overall_score = (
            sustained_score * 0.25 +
            selective_score * 0.25 +
            divided_score * 0.20 +
            switching_score * 0.15 +
            monitoring_score * 0.15
        )
        
        self.current_attention_state["overall_score"] = round(overall_score, 1)
        
        # 重新计算效率
        self.current_attention_state["efficiency"] = self._calculate_attention_efficiency(overall_score)
    
    def _analyze_attention_trend(self):
        """分析注意力趋势"""
        if len(self.attention_history) < 2:
            self.current_attention_state["trend"] = "stable"
            return
        
        # 获取最近两次整体得分
        recent_scores = []
        for data_point in reversed(self.attention_history[-10:]):
            if "overall_score" in data_point:
                recent_scores.append(data_point["overall_score"])
                if len(recent_scores) >= 2:
                    break
        
        if len(recent_scores) >= 2:
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2]
            
            change = current_score - previous_score
            
            if change > 2.8:
                self.current_attention_state["trend"] = "rising"
            elif change < -2.8:
                self.current_attention_state["trend"] = "declining"
            else:
                self.current_attention_state["trend"] = "stable"
    
    def _add_to_history(self):
        """添加当前状态到历史记录"""
        history_point = self.current_attention_state.copy()
        self.attention_history.append(history_point)
        
        # 限制历史记录长度
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
    
    def simulate_attention_task(self, task_type: str = "random") -> Dict[str, Any]:
        """
        模拟注意力任务执行
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务执行结果
        """
        if task_type == "random":
            task_type = random.choice(self.attention_tasks)
        
        # 根据任务类型调整性能
        task_performance_factors = {
            "持续注意力测试": 0.92,
            "选择性注意任务": 0.95,
            "分散注意力挑战": 0.85,
            "注意力转换练习": 0.88,
            "警觉性评估": 0.90,
            "注意力维持任务": 0.93,
            "干扰抵抗测试": 0.87,
            "多任务处理挑战": 0.82,
            "注意力分配训练": 0.86,
            "认知控制测试": 0.89
        }
        
        base_performance = task_performance_factors.get(task_type, 1.0)
        
        # 生成任务结果
        accuracy = base_performance * random.uniform(0.72, 0.98)
        persistence = base_performance * random.uniform(0.70, 0.95)
        response_time = self.current_attention_state["response_time"] * random.uniform(0.8, 1.3)
        
        task_result = {
            "task_type": task_type,
            "timestamp": time.time(),
            "accuracy": round(accuracy, 3),
            "persistence": round(persistence, 3),
            "response_time": round(response_time, 2),
            "attention_allocation": random.uniform(0.75, 0.95),
            "focus_quality": random.uniform(0.70, 0.92)
        }
        
        # 根据任务表现调整注意力状态
        performance_impact = (accuracy - 0.80) * 8
        fatigue_increase = random.uniform(0.03, 0.07)
        
        self.update_attention_metrics(
            overall_score=self.current_attention_state["overall_score"] + performance_impact,
            attention_fatigue=self.current_attention_state["attention_fatigue"] + fatigue_increase
        )
        
        return task_result
    
    def get_attention_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取注意力分析报告
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            注意力分析报告
        """
        if not self.attention_history:
            return {"error": "暂无历史数据"}
        
        # 过滤时间范围内的数据
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [d for d in self.attention_history if d["timestamp"] >= cutoff_time]
        
        if not recent_data:
            return {"error": "指定时间范围内无数据"}
        
        # 计算统计数据
        scores = [d["overall_score"] for d in recent_data]
        accuracies = [d.get("selective_accuracy", 0.85) for d in recent_data]
        spans = [d.get("attention_span", 30) for d in recent_data]
        
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
            
            "attention_characteristics": {
                "mean_accuracy": round(np.mean(accuracies), 3),
                "mean_span": round(np.mean(spans), 1),
                "accuracy_consistency": round(1 - np.std(accuracies)/np.mean(accuracies), 3),
                "span_stability": round(1 - np.std(spans)/np.mean(spans), 3)
            },
            
            "attention_state": {
                "average_fatigue": round(np.mean([d.get("attention_fatigue", 0.4) for d in recent_data]), 3),
                "average_energy": round(np.mean([d.get("mental_energy_attention", 0.75) for d in recent_data]), 3),
                "average_vigilance": round(np.mean([d.get("vigilance_level", 0.80) for d in recent_data]), 3)
            },
            
            "component_analysis": self._analyze_attention_components(recent_data),
            
            "attention_insights": self._generate_attention_insights(recent_data),
            
            "recommendations": self._generate_attention_recommendations(recent_data)
        }
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """计算得分趋势"""
        if len(scores) < 2:
            return "stable"
        
        # 线性回归计算趋势
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.35:
            return "improving"
        elif slope < -0.35:
            return "declining"
        else:
            return "stable"
    
    def _analyze_attention_components(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """分析注意力组件性能"""
        component_scores = {
            "sustained_attention": [],
            "selective_attention": [],
            "divided_attention": [],
            "switching_attention": [],
            "monitoring_attention": []
        }
        
        for data_point in recent_data:
            component_scores["sustained_attention"].append(data_point.get("attention_span", 75))
            component_scores["selective_attention"].append(data_point.get("selective_accuracy", 75))
            component_scores["divided_attention"].append(data_point.get("multitask_capacity", 75))
            component_scores["switching_attention"].append(data_point.get("attention_flexibility", 75))
            component_scores["monitoring_attention"].append(data_point.get("monitoring_attention", 75))
        
        component_analysis = {}
        for component, scores in component_scores.items():
            if scores:
                component_analysis[component] = {
                    "mean_score": round(np.mean(scores), 1),
                    "stability": round(1 - np.std(scores)/np.mean(scores), 3),
                    "trend": self._calculate_trend(scores)
                }
        
        return component_analysis
    
    def _generate_attention_insights(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """生成注意力洞察"""
        current_state = self.current_attention_state
        
        # 分析注意力模式
        attention_patterns = []
        
        if current_state["attention_span"] > 35:
            attention_patterns.append("优秀持续注意力 - 能长时间保持专注")
        
        if current_state["selective_accuracy"] > 0.88:
            attention_patterns.append("强选择性注意力 - 善于过滤无关信息")
        
        if current_state["multitask_capacity"] > 75:
            attention_patterns.append("良好分散注意力 - 能同时处理多任务")
        
        if current_state["attention_flexibility"] > 85:
            attention_patterns.append("高注意力灵活性 - 能快速转换焦点")
        
        return {
            "dominant_patterns": attention_patterns,
            "attention_strength": round((current_state["attention_span"] + current_state["selective_accuracy"]) / 2, 1),
            "focus_endurance": current_state["attention_persistence"],
            "attention_control": round((current_state["vigilance_level"] + current_state["self_regulation"]) / 2, 3)
        }
    
    def _generate_attention_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """生成注意力改善建议"""
        recommendations = []
        
        # 分析当前状态
        avg_fatigue = np.mean([d.get("attention_fatigue", 0.4) for d in recent_data])
        avg_energy = np.mean([d.get("mental_energy_attention", 0.75) for d in recent_data])
        avg_vigilance = np.mean([d.get("vigilance_level", 0.80) for d in recent_data])
        
        if avg_fatigue > 0.5:
            recommendations.append("注意力疲劳较高，建议适当休息或进行放松活动")
        
        if avg_energy < 0.65:
            recommendations.append("心理能量偏低，建议补充营养或进行轻度运动")
        
        if avg_vigilance < 0.75:
            recommendations.append("警觉性不足，建议保持环境明亮和通风")
        
        # 基于组件性能提供建议
        current_state = self.current_attention_state
        
        if current_state["attention_span"] < 25:
            recommendations.append("持续注意力有待提升，建议进行专注力训练")
        
        if current_state["selective_accuracy"] < 0.80:
            recommendations.append("选择性注意力需要加强，建议练习信息过滤技巧")
        
        if current_state["multitask_capacity"] < 70:
            recommendations.append("分散注意力可以训练，建议练习简单的多任务处理")
        
        if current_state["attention_flexibility"] < 78:
            recommendations.append("注意力灵活性需要提升，建议进行注意力转换练习")
        
        # 基于时间提供建议
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 11:
            recommendations.append("上午注意力高峰期，适合进行需要高度专注的任务")
        elif 14 <= current_hour <= 16:
            recommendations.append("下午注意力稳定期，适合进行常规专注工作")
        elif 19 <= current_hour <= 21:
            recommendations.append("傍晚注意力下降，建议进行轻松的专注练习")
        
        # 环境建议
        if current_state["distraction_resistance"] < 0.75:
            recommendations.append("抗干扰能力需要改善，建议在安静环境中工作")
        
        if current_state["stress_impact"] > 0.5:
            recommendations.append("压力对注意力影响较大，建议学习压力管理技巧")
        
        return recommendations if recommendations else ["当前注意力状态良好，继续保持专注练习"]
    
    def simulate_real_time_update(self):
        """模拟实时数据更新"""
        # 随机选择要更新的指标
        updateable_metrics = [
            "attention_span",
            "selective_accuracy", 
            "multitask_capacity",
            "attention_flexibility",
            "monitoring_attention"
        ]
        
        metric_to_update = random.choice(updateable_metrics)
        
        # 生成新的值
        if metric_to_update == "attention_span":
            new_value = self._generate_sustained_attention_score()
        elif metric_to_update == "selective_accuracy":
            new_value = self._generate_selective_attention_score()
        elif metric_to_update == "multitask_capacity":
            new_value = self._generate_divided_attention_score()
        elif metric_to_update == "attention_flexibility":
            new_value = self._generate_switching_attention_score()
        elif metric_to_update == "monitoring_attention":
            new_value = self._generate_monitoring_attention_score()
        else:
            new_value = self.current_attention_state.get(metric_to_update, 75)
        
        # 更新指标
        self.update_attention_metrics(**{metric_to_update: new_value})
        
        return metric_to_update


# 测试函数
def test_attention_monitor():
    """测试注意力监控器"""
    print("开始测试注意力监控器...")
    
    # 创建注意力监控器
    attention_monitor = AttentionMonitor()
    
    # 测试当前指标
    print("\n1. 当前注意力指标:")
    current_metrics = attention_monitor.get_current_metrics()
    print(f"  整体得分: {current_metrics['overall_score']:.1f}%")
    print(f"  注意力跨度: {current_metrics['attention_span']:.1f}分钟")
    print(f"  选择性准确率: {current_metrics['selective_accuracy']:.1%}")
    print(f"  警觉性水平: {current_metrics['vigilance_level']:.1%}")
    
    # 模拟注意力任务
    print("\n2. 模拟注意力任务:")
    for i in range(3):
        result = attention_monitor.simulate_attention_task()
        print(f"  任务{i+1}: {result['task_type']} - 准确率: {result['accuracy']:.1%}")
    
    # 模拟实时更新
    print("\n3. 模拟实时更新 (5次):")
    for i in range(5):
        metric = attention_monitor.simulate_real_time_update()
        print(f"  更新{i+1}: {metric}")
    
    # 生成分析报告
    print("\n4. 注意力分析:")
    analysis = attention_monitor.get_attention_analysis(hours=1)
    if "error" not in analysis:
        print(f"  数据点数量: {analysis['data_points']}")
        print(f"  平均得分: {analysis['overall_performance']['mean_score']:.1f}")
        print(f"  注意力洞察: {len(analysis['attention_insights']['dominant_patterns'])}个")
        print(f"  主要建议: {analysis['recommendations'][0]}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_attention_monitor()