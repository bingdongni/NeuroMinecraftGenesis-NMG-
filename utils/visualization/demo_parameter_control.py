"""
实时参数调节系统演示
测试和演示整个参数调节界面的功能
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 导入我们创建的组件
from .parameter_controller import ParameterController
from .slider_interface import SliderInterface
from .parameter_preset import ParameterPresetManager
from .live_feedback import LiveFeedbackSystem
from .parameter_logger import ParameterLogger


class ParameterControlDemo:
    """参数调节系统演示类
    
    演示完整的实时参数调节系统，包括：
    - 初始化各个组件
    - 创建用户交互界面
    - 模拟参数变更过程
    - 展示反馈和日志功能
    - 演示预设管理
    """
    
    def __init__(self):
        """初始化演示系统"""
        print("🚀 初始化实时参数调节系统演示...")
        
        # 创建各个组件
        self.parameter_controller = ParameterController()
        self.slider_interface = SliderInterface()
        self.preset_manager = ParameterPresetManager()
        self.feedback_system = LiveFeedbackSystem()
        self.logger = ParameterLogger()
        
        # 启动反馈系统监控
        self.feedback_system.start_monitoring()
        
        # 启动日志会话
        self.logger.start_session("demo_user", ["demo", "演示"])
        
        # 绑定组件间的事件监听
        self._bind_component_events()
        
        print("✅ 实时参数调节系统演示初始化完成")
    
    def _bind_component_events(self):
        """绑定组件间的事件监听"""
        # 监听参数变更
        self.parameter_controller.add_parameter_change_listener(
            self._on_parameter_change
        )
        
        # 监听反馈消息
        self.feedback_system.add_feedback_listener(
            self._on_feedback_message
        )
        
        # 监听日志记录
        self.logger.add_log_listener(
            self._on_log_entry
        )
        
        # 监听预设操作
        self.preset_manager.add_preset_listener(
            self._on_preset_event
        )
    
    def _on_parameter_change(self, parameter_name: str, value: float):
        """处理参数变更事件"""
        print(f"📊 参数变更: {parameter_name} = {value}")
        
        # 记录日志
        self.logger.log_parameter_change(parameter_name, value - 0.1, value)
        
        # 模拟参数应用到智能体
        time.sleep(0.1)  # 模拟处理时间
        application_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'agent_response': {'applied': True}
        }
        
        self.logger.log_parameter_application(parameter_name, value, application_result)
    
    def _on_feedback_message(self, feedback_message):
        """处理反馈消息事件"""
        print(f"💬 反馈: {feedback_message.title} - {feedback_message.message}")
    
    def _on_log_entry(self, log_entry):
        """处理日志条目事件"""
        print(f"📝 日志: {log_entry.parameter_name} 变更记录已保存")
    
    def _on_preset_event(self, event_type: str, preset):
        """处理预设事件"""
        print(f"🎯 预设事件: {event_type} - {preset.name}")
    
    def create_interface_html(self) -> str:
        """创建完整的HTML界面"""
        print("🎨 生成滑块界面HTML...")
        
        # 获取参数配置
        parameter_config = self.parameter_controller.create_slider_interface()
        
        # 创建HTML界面
        html_content = self.slider_interface.create_slider_interface(parameter_config)
        
        return html_content
    
    def demonstrate_parameter_updates(self):
        """演示参数更新过程"""
        print("\n🔄 演示参数更新过程...")
        
        # 获取要演示的参数
        demo_parameters = [
            "curiosity_weight", "learning_rate", "attention_span", 
            "decision_threshold", "risk_tolerance"
        ]
        
        for param_name in demo_parameters:
            # 生成新的参数值
            current_value = self.parameter_controller.get_current_parameters()[param_name]
            
            # 添加一些随机变化
            import random
            change = random.uniform(-0.5, 0.5)
            new_value = max(
                0.1, 
                min(
                    self.parameter_controller.get_parameter_ranges()[param_name].max_value,
                    current_value + change
                )
            )
            
            print(f"  更新参数: {param_name} ({current_value:.3f} → {new_value:.3f})")
            
            # 执行参数更新
            success = self.parameter_controller.update_parameter(param_name, new_value)
            if success:
                # 应用参数改变
                result = self.parameter_controller.apply_parameter_change(param_name)
                print(f"  应用结果: {result['success']}")
                
                # 等待一下再进行下一个
                time.sleep(0.5)
            
            print("  ✅ 参数更新完成\n")
    
    def demonstrate_preset_management(self):
        """演示预设管理功能"""
        print("🎯 演示预设管理功能...")
        
        # 1. 列出所有预设
        print("\n1. 列出所有预设:")
        all_presets = self.preset_manager.list_presets()
        for preset in all_presets:
            print(f"  - {preset.name} ({preset.category}): {preset.description}")
        
        # 2. 加载预设
        print("\n2. 加载平衡型预设:")
        balanced_preset = self.preset_manager.load_preset("平衡型")
        if balanced_preset:
            print(f"  预设参数数量: {len(balanced_preset.parameters)}")
            for param_name, value in list(balanced_preset.parameters.items())[:3]:
                print(f"    {param_name}: {value}")
            print("  ✅ 平衡型预设加载完成")
        
        # 3. 创建自定义预设
        print("\n3. 创建自定义预设:")
        custom_parameters = {
            "curiosity_weight": 1.5,
            "exploration_rate": 0.2,
            "learning_rate": 0.002,
            "memory_capacity": 1500,
            "attention_span": 1.2,
            "decision_threshold": 0.8,
            "risk_tolerance": 0.6,
            "patience_level": 2.5
        }
        
        success = self.preset_manager.save_preset(
            "我的自定义预设",
            custom_parameters,
            "演示用的自定义参数配置",
            ["自定义", "演示", "测试"],
            "custom"
        )
        
        if success:
            print("  ✅ 自定义预设创建成功")
        
        # 4. 搜索预设
        print("\n4. 搜索探索相关预设:")
        search_results = self.preset_manager.search_presets("探索")
        print(f"  找到 {len(search_results)} 个相关预设:")
        for preset in search_results:
            print(f"    - {preset.name}: {preset.description}")
        
        # 5. 获取统计信息
        print("\n5. 预设统计信息:")
        stats = self.preset_manager.get_statistics()
        print(f"  总预设数: {stats['total_presets']}")
        print(f"  总使用次数: {stats['total_usage']}")
        print(f"  平均使用次数: {stats['average_usage']:.2f}")
        
        # 6. 导出预设
        print("\n6. 导出预设:")
        success = self.preset_manager.export_presets("demo_presets.json")
        if success:
            print("  ✅ 预设导出成功")
    
    def demonstrate_feedback_system(self):
        """演示反馈系统"""
        print("\n📊 演示反馈系统...")
        
        # 1. 模拟参数变更并观察反馈
        print("\n1. 模拟大幅参数变更:")
        self.feedback_system.notify_parameter_change(
            "curiosity_weight", 1.0, 2.0  # 大幅增加
        )
        
        # 2. 模拟频繁变更
        print("\n2. 模拟频繁参数变更:")
        for i in range(3):
            self.feedback_system.notify_parameter_change(
                "learning_rate", 0.001, 0.001 + i * 0.0001
            )
            time.sleep(0.1)
        
        # 3. 模拟行为变化
        print("\n3. 模拟行为变化:")
        before_state = {"exploration_rate": 0.1, "focus_level": 0.8}
        after_state = {"exploration_rate": 0.3, "focus_level": 0.6}
        
        self.feedback_system.notify_behavior_change(
            "exploration_behavior",
            before_state,
            after_state,
            ["curiosity_weight", "exploration_rate"]
        )
        
        # 4. 获取最近变化
        print("\n4. 获取最近变化:")
        recent_changes = self.feedback_system.get_recent_changes()
        print(f"  最近参数变更: {len(recent_changes['parameter_changes'])}")
        print(f"  最近行为变化: {len(recent_changes['behavior_changes'])}")
        print(f"  最近反馈消息: {len(recent_changes['feedback_messages'])}")
        
        # 5. 生成性能报告
        print("\n5. 生成性能报告:")
        performance_report = self.feedback_system.generate_performance_report()
        print(f"  稳定性得分: {performance_report['stability']:.3f}")
        print(f"  一致性得分: {performance_report['consistency']:.3f}")
        print(f"  参数变更频率: {performance_report['parameter_change_frequency']}")
        
        # 6. 导出反馈数据
        print("\n6. 导出反馈数据:")
        success = self.feedback_system.export_feedback_data("demo_feedback.json")
        if success:
            print("  ✅ 反馈数据导出成功")
    
    def demonstrate_logging_system(self):
        """演示日志系统"""
        print("\n📋 演示日志系统...")
        
        # 1. 获取参数历史
        print("\n1. 获取参数历史:")
        curiosity_history = self.logger.get_parameter_history("curiosity_weight")
        print(f"  好奇心权重历史记录: {len(curiosity_history)} 条")
        if curiosity_history:
            latest = curiosity_history[0]
            print(f"    最新记录: {latest.timestamp} - {latest.old_value} → {latest.new_value}")
        
        # 2. 获取会话历史
        print("\n2. 获取会话历史:")
        sessions = self.logger.get_session_history()
        print(f"  历史会话: {len(sessions)} 个")
        if sessions:
            latest_session = sessions[0]
            print(f"    最新会话: {latest_session.session_id}")
            print(f"      开始时间: {latest_session.start_time}")
            print(f"      总变更: {latest_session.total_changes}")
        
        # 3. 生成统计报告
        print("\n3. 生成统计报告:")
        stats = self.logger.generate_statistics_report()
        print(f"  总记录数: {stats.total_entries}")
        print(f"  日期范围: {stats.date_range[0]} - {stats.date_range[1]}")
        print(f"  最多变更参数: {stats.most_changed_parameter}")
        print(f"  会话数量: {stats.session_count}")
        print(f"  平均会话时长: {stats.average_session_duration:.2f} 分钟")
        
        # 4. 分析参数趋势
        print("\n4. 分析参数趋势:")
        trend_analysis = self.logger.analyze_parameter_trends("curiosity_weight")
        print(f"  数据点数: {trend_analysis['data_points']}")
        print(f"  趋势方向: {trend_analysis['trend_direction']}")
        print(f"  稳定性得分: {trend_analysis['stability_score']:.3f}")
        print(f"  当前值: {trend_analysis['current_value']}")
        
        # 5. 导出日志数据
        print("\n5. 导出日志数据:")
        success = self.logger.export_logs("demo_logs.json", "json")
        if success:
            print("  ✅ 日志数据导出成功")
        
        # 6. 清理旧日志
        print("\n6. 清理旧日志:")
        cleaned_count = self.logger.cleanup_old_logs(1)  # 保留1天
        print(f"  清理文件数量: {cleaned_count}")
    
    def run_full_demo(self):
        """运行完整演示"""
        print("🎪 开始完整演示...")
        
        try:
            # 1. 创建界面HTML
            html_content = self.create_interface_html()
            with open("parameter_control_demo.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("✅ 界面HTML已生成: parameter_control_demo.html")
            
            # 2. 演示参数更新
            self.demonstrate_parameter_updates()
            
            # 3. 演示预设管理
            self.demonstrate_preset_management()
            
            # 4. 演示反馈系统
            self.demonstrate_feedback_system()
            
            # 5. 演示日志系统
            self.demonstrate_logging_system()
            
            # 6. 展示组件间协作
            print("\n🔗 展示组件间协作:")
            self._demonstrate_component_integration()
            
            print("\n🎉 演示完成!")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
        
        finally:
            # 清理资源
            self.cleanup()
    
    def _demonstrate_component_integration(self):
        """演示组件间集成"""
        print("\n  1. 参数变更触发完整流程:")
        
        # 模拟参数变更，触发完整的处理流程
        self.parameter_controller.update_parameter("learning_rate", 0.003)
        result = self.parameter_controller.apply_parameter_change("learning_rate")
        
        print(f"    参数更新结果: {result}")
        
        # 展示参数验证
        print("\n  2. 参数配置验证:")
        validation = self.parameter_controller.validate_parameters()
        print(f"    配置有效性: {validation['valid']}")
        if validation['warnings']:
            print(f"    警告: {validation['warnings']}")
        if validation['suggestions']:
            print(f"    建议: {validation['suggestions']}")
        
        # 展示监控状态
        print("\n  3. 系统监控状态:")
        monitoring_status = self.parameter_controller.get_monitoring_status()
        print(f"    监控状态: {monitoring_status['is_active']}")
        print(f"    监控参数数: {len(monitoring_status['monitored_parameters'])}")
        print(f"    监听器数: {monitoring_status['listener_count']}")
        print(f"    总变更数: {monitoring_status['total_changes']}")
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理演示资源...")
        
        # 停止反馈监控
        self.feedback_system.stop_monitoring()
        
        # 结束日志会话
        self.logger.end_session()
        
        print("✅ 资源清理完成")
    
    def generate_report(self) -> Dict[str, Any]:
        """生成演示报告"""
        print("\n📄 生成演示报告...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "demo_components": {
                "parameter_controller": "✅",
                "slider_interface": "✅", 
                "preset_manager": "✅",
                "feedback_system": "✅",
                "logger": "✅"
            },
            "features_tested": [
                "参数实时更新",
                "滑块界面生成",
                "预设管理",
                "反馈系统",
                "日志记录",
                "参数验证",
                "性能监控",
                "数据导出"
            ],
            "system_capabilities": {
                "实时参数调节": True,
                "参数持久化": True,
                "预设管理": True,
                "实时反馈": True,
                "行为分析": True,
                "日志记录": True,
                "趋势分析": True,
                "性能监控": True
            },
            "component_statistics": {
                "parameters_managed": len(self.parameter_controller.get_current_parameters()),
                "presets_available": len(self.preset_manager.list_presets()),
                "feedback_messages": len(self.feedback_system.feedback_messages),
                "log_entries": self.logger.log_statistics['total_entries']
            }
        }
        
        return report


def main():
    """主演示函数"""
    print("🧠 智能体参数实时调节系统演示")
    print("=" * 50)
    
    # 创建演示实例
    demo = ParameterControlDemo()
    
    try:
        # 运行完整演示
        demo.run_full_demo()
        
        # 生成演示报告
        report = demo.generate_report()
        print("\n📊 演示报告:")
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 保存报告
        import json
        with open("demo_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print("\n💾 演示报告已保存: demo_report.json")
        
        print("\n🌐 您可以通过浏览器打开 parameter_control_demo.html 查看界面效果")
        print("📁 查看生成的文件:")
        print("  - parameter_control_demo.html (界面)")
        print("  - demo_presets.json (预设数据)")
        print("  - demo_feedback.json (反馈数据)")
        print("  - demo_logs.json (日志数据)")
        print("  - demo_report.json (演示报告)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示出现错误: {e}")
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()