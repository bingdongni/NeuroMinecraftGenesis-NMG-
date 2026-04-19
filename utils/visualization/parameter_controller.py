"""
智能体参数控制器
实现智能体参数的实时调节、持久化和应用
"""

import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, asdict
from .parameter_preset import ParameterPresetManager
from .live_feedback import LiveFeedbackSystem
from .parameter_logger import ParameterLogger


@dataclass
class ParameterRange:
    """参数范围定义"""
    min_value: float
    max_value: float
    step: float
    default_value: float
    unit: str = ""
    description: str = ""


class ParameterController:
    """参数控制器主类
    
    负责管理智能体的所有参数，提供实时调节功能，包括：
    - 参数值的更新和验证
    - 预设参数的保存和加载
    - 实时反馈系统集成
    - 参数变更的日志记录
    - 智能体行为的实时影响分析
    """
    
    def __init__(self):
        """初始化参数控制器"""
        self._parameters: Dict[str, Dict[str, Any]] = {}
        self._listeners: List[Callable] = []
        self._current_values: Dict[str, float] = {}
        self._is_active = False
        self._update_lock = threading.Lock()
        
        # 初始化子系统
        self.preset_manager = ParameterPresetManager()
        self.live_feedback = LiveFeedbackSystem()
        self.parameter_logger = ParameterLogger()
        
        # 初始化默认参数
        self._initialize_default_parameters()
        
        print("参数控制器初始化完成")
    
    def _initialize_default_parameters(self):
        """初始化默认参数配置"""
        default_params = {
            # 好奇心相关参数
            "curiosity_weight": ParameterRange(
                min_value=0.0, max_value=2.0, step=0.1, 
                default_value=1.0, unit="权重", 
                description="好奇心权重：控制智能体对新环境的探索倾向"
            ),
            "exploration_rate": ParameterRange(
                min_value=0.0, max_value=1.0, step=0.05,
                default_value=0.1, unit="率", 
                description="探索率：智能体进行随机探索的概率"
            ),
            "novelty_threshold": ParameterRange(
                min_value=0.0, max_value=10.0, step=0.1,
                default_value=2.5, unit="阈值",
                description="新颖性阈值：触发探索行为的新颖度要求"
            ),
            
            # 学习相关参数
            "learning_rate": ParameterRange(
                min_value=0.0001, max_value=0.1, step=0.0001,
                default_value=0.001, unit="率",
                description="学习速率：神经网络权重的更新速度"
            ),
            "memory_capacity": ParameterRange(
                min_value=100, max_value=10000, step=100,
                default_value=1000, unit="条目",
                description="记忆容量：智能体可存储的记忆条目数量"
            ),
            "forgetting_rate": ParameterRange(
                min_value=0.0, max_value=0.1, step=0.001,
                default_value=0.01, unit="率",
                description="遗忘率：记忆衰减的速度"
            ),
            
            # 注意力相关参数
            "attention_span": ParameterRange(
                min_value=0.1, max_value=10.0, step=0.1,
                default_value=1.0, unit="秒",
                description="注意力持续时间：智能体能保持专注的时间"
            ),
            "focus_intensity": ParameterRange(
                min_value=0.0, max_value=1.0, step=0.05,
                default_value=0.8, unit="强度",
                description="注意力强度：智能体专注程度"
            ),
            "distraction_filter": ParameterRange(
                min_value=0.0, max_value=1.0, step=0.05,
                default_value=0.3, unit="过滤器",
                description="干扰过滤器：过滤无关信息的能力"
            ),
            
            # 决策相关参数
            "decision_threshold": ParameterRange(
                min_value=0.0, max_value=1.0, step=0.05,
                default_value=0.7, unit="阈值",
                description="决策阈值：做出行动决策的置信度要求"
            ),
            "risk_tolerance": ParameterRange(
                min_value=0.0, max_value=1.0, step=0.05,
                default_value=0.5, unit="容忍度",
                description="风险容忍度：对不确定结果的接受程度"
            ),
            "patience_level": ParameterRange(
                min_value=0.1, max_value=10.0, step=0.1,
                default_value=2.0, unit="级别",
                description="耐心水平：等待环境变化的等待时间"
            )
        }
        
        for param_name, param_range in default_params.items():
            self._parameters[param_name] = {
                'range': param_range,
                'current_value': param_range.default_value,
                'history': [],
                'last_updated': datetime.now()
            }
            self._current_values[param_name] = param_range.default_value
    
    def create_slider_interface(self) -> Dict[str, Any]:
        """创建滑块界面配置
        
        返回:
            包含所有参数滑块配置的字典
        """
        slider_config = {}
        
        for param_name, param_data in self._parameters.items():
            param_range = param_data['range']
            slider_config[param_name] = {
                'min': param_range.min_value,
                'max': param_range.max_value,
                'step': param_range.step,
                'value': param_data['current_value'],
                'unit': param_range.unit,
                'description': param_range.description,
                'label': self._get_parameter_label(param_name)
            }
        
        return slider_config
    
    def _get_parameter_label(self, param_name: str) -> str:
        """获取参数的中文标签"""
        labels = {
            "curiosity_weight": "好奇心权重",
            "exploration_rate": "探索率",
            "novelty_threshold": "新颖性阈值",
            "learning_rate": "学习速率",
            "memory_capacity": "记忆容量",
            "forgetting_rate": "遗忘率",
            "attention_span": "注意力持续时间",
            "focus_intensity": "注意力强度",
            "distraction_filter": "干扰过滤器",
            "decision_threshold": "决策阈值",
            "risk_tolerance": "风险容忍度",
            "patience_level": "耐心水平"
        }
        return labels.get(param_name, param_name)
    
    def update_parameter(self, parameter_name: str, new_value: float) -> bool:
        """更新参数值
        
        参数:
            parameter_name: 参数名称
            new_value: 新参数值
            
        返回:
            bool: 更新是否成功
        """
        with self._update_lock:
            if parameter_name not in self._parameters:
                print(f"错误：未找到参数 '{parameter_name}'")
                return False
            
            param_data = self._parameters[parameter_name]
            param_range = param_data['range']
            
            # 验证参数值是否在有效范围内
            if not (param_range.min_value <= new_value <= param_range.max_value):
                print(f"错误：参数 '{parameter_name}' 的值 {new_value} 不在有效范围 "
                      f"[{param_range.min_value}, {param_range.max_value}] 内")
                return False
            
            # 保存历史值
            old_value = param_data['current_value']
            param_data['history'].append({
                'value': old_value,
                'timestamp': param_data['last_updated']
            })
            
            # 限制历史记录长度
            if len(param_data['history']) > 100:
                param_data['history'] = param_data['history'][-50:]
            
            # 更新当前值
            param_data['current_value'] = new_value
            param_data['last_updated'] = datetime.now()
            self._current_values[parameter_name] = new_value
            
            # 记录日志
            self.parameter_logger.log_parameter_change(
                parameter_name, old_value, new_value
            )
            
            # 触发实时反馈
            self.live_feedback.notify_parameter_change(
                parameter_name, old_value, new_value
            )
            
            # 通知所有监听器
            self._notify_listeners(parameter_name, new_value)
            
            print(f"参数 '{self._get_parameter_label(parameter_name)}' 已更新: "
                  f"{old_value} -> {new_value}")
            
            return True
    
    def _notify_listeners(self, parameter_name: str, value: float):
        """通知所有参数变更监听器"""
        for listener in self._listeners:
            try:
                listener(parameter_name, value)
            except Exception as e:
                print(f"参数变更监听器执行错误: {e}")
    
    def apply_parameter_change(self, parameter_name: str) -> Dict[str, Any]:
        """应用参数改变到智能体
        
        参数:
            parameter_name: 参数名称
            
        返回:
            包含应用结果的字典
        """
        if parameter_name not in self._parameters:
            return {'success': False, 'error': f'参数 {parameter_name} 不存在'}
        
        current_value = self._parameters[parameter_name]['current_value']
        
        # 模拟参数应用到智能体（实际实现中需要与智能体系统集成）
        try:
            # 这里应该调用实际的智能体参数更新接口
            agent_response = self._simulate_agent_parameter_application(
                parameter_name, current_value
            )
            
            # 记录应用结果
            application_result = {
                'success': True,
                'parameter': parameter_name,
                'value': current_value,
                'timestamp': datetime.now().isoformat(),
                'agent_response': agent_response
            }
            
            self.parameter_logger.log_parameter_application(application_result)
            
            return application_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'parameter': parameter_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.parameter_logger.log_parameter_application(error_result)
            return error_result
    
    def _simulate_agent_parameter_application(self, param_name: str, value: float) -> Dict[str, Any]:
        """模拟参数应用到智能体的过程（开发测试用）"""
        # 模拟不同的响应延迟
        import random
        time.sleep(random.uniform(0.1, 0.5))
        
        # 根据参数类型返回不同的响应
        responses = {
            "curiosity_weight": {
                "exploration_increase": value * 0.1,
                "novelty_detection_sensitivity": value * 0.05
            },
            "learning_rate": {
                "training_speed": value * 1000,
                "convergence_rate": value * 10
            },
            "attention_span": {
                "focus_duration": value,
                "distraction_resistance": 1.0 / value if value > 0 else float('inf')
            }
        }
        
        return responses.get(param_name, {"applied": True})
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前所有参数值
        
        返回:
            当前参数值的字典
        """
        return self._current_values.copy()
    
    def get_parameter_history(self, parameter_name: str) -> List[Dict[str, Any]]:
        """获取参数历史记录
        
        参数:
            parameter_name: 参数名称
            
        返回:
            历史记录列表
        """
        if parameter_name not in self._parameters:
            return []
        return self._parameters[parameter_name]['history']
    
    def add_parameter_change_listener(self, listener: Callable[[str, float], None]):
        """添加参数变更监听器
        
        参数:
            listener: 监听器函数，接受参数名和新值
        """
        self._listeners.append(listener)
    
    def remove_parameter_change_listener(self, listener: Callable[[str, float], None]):
        """移除参数变更监听器
        
        参数:
            listener: 要移除的监听器函数
        """
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def get_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """获取所有参数的取值范围
        
        返回:
            参数范围字典
        """
        return {name: data['range'] for name, data in self._parameters.items()}
    
    def validate_parameters(self) -> Dict[str, Any]:
        """验证当前参数配置的合理性
        
        返回:
            验证结果字典
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'suggestions': []
        }
        
        current_values = self.get_current_parameters()
        
        # 检查参数一致性
        if current_values['exploration_rate'] > 0.8 and current_values['risk_tolerance'] < 0.2:
            validation_result['warnings'].append(
                "高探索率与低风险容忍度可能导致决策困难"
            )
        
        # 检查学习相关参数
        if current_values['learning_rate'] > 0.01 and current_values['memory_capacity'] < 500:
            validation_result['suggestions'].append(
                "高学习速率建议配合更大的记忆容量使用"
            )
        
        # 检查注意力参数
        if current_values['attention_span'] < 0.5 and current_values['focus_intensity'] < 0.5:
            validation_result['warnings'].append(
                "短注意力持续时间和低专注度可能影响学习效果"
            )
        
        if validation_result['warnings'] or validation_result['suggestions']:
            validation_result['valid'] = len(validation_result['warnings']) == 0
        
        return validation_result
    
    def start_monitoring(self):
        """启动参数监控"""
        self._is_active = True
        print("参数监控已启动")
    
    def stop_monitoring(self):
        """停止参数监控"""
        self._is_active = False
        print("参数监控已停止")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态
        
        返回:
            监控状态字典
        """
        return {
            'is_active': self._is_active,
            'monitored_parameters': list(self._parameters.keys()),
            'listener_count': len(self._listeners),
            'total_changes': len(self.parameter_logger.change_log)
        }
    
    def export_parameters(self, file_path: str) -> bool:
        """导出参数配置到文件
        
        参数:
            file_path: 导出文件路径
            
        返回:
            导出是否成功
        """
        try:
            export_data = {
                'parameters': {},
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            for param_name, param_data in self._parameters.items():
                export_data['parameters'][param_name] = {
                    'current_value': param_data['current_value'],
                    'history': param_data['history'][-10:],  # 只保存最近10条记录
                    'range': asdict(param_data['range'])
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"参数配置已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"导出参数配置失败: {e}")
            return False
    
    def import_parameters(self, file_path: str) -> bool:
        """从文件导入参数配置
        
        参数:
            file_path: 导入文件路径
            
        返回:
            导入是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if 'parameters' not in import_data:
                print("错误：文件格式不正确")
                return False
            
            imported_count = 0
            for param_name, param_data in import_data['parameters'].items():
                if param_name in self._parameters:
                    new_value = param_data['current_value']
                    if self.update_parameter(param_name, new_value):
                        imported_count += 1
            
            print(f"成功导入 {imported_count} 个参数配置")
            return imported_count > 0
            
        except Exception as e:
            print(f"导入参数配置失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 创建参数控制器
    controller = ParameterController()
    
    # 创建滑块界面配置
    slider_config = controller.create_slider_interface()
    print("滑块界面配置已创建")
    
    # 启动监控
    controller.start_monitoring()
    
    # 测试参数更新
    controller.update_parameter("curiosity_weight", 1.5)
    controller.update_parameter("learning_rate", 0.002)
    
    # 应用参数改变
    result = controller.apply_parameter_change("curiosity_weight")
    print(f"参数应用结果: {result}")
    
    # 验证参数
    validation = controller.validate_parameters()
    print(f"参数验证结果: {validation}")
    
    # 导出参数
    controller.export_parameters("current_parameters.json")