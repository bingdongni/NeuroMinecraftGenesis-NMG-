"""
滑块界面组件
提供参数调节的用户交互界面
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class SliderConfig:
    """滑块配置"""
    name: str
    label: str
    min_value: float
    max_value: float
    step: float
    current_value: float
    unit: str = ""
    description: str = ""
    category: str = "general"
    color: str = "#007bff"


class SliderInterface:
    """滑块界面组件
    
    负责创建和管理参数调节的滑块界面，包括：
    - 滑块的创建和配置
    - 实时值显示
    - 参数分类和布局
    - 预设按钮和快速调节
    - 响应式设计
    """
    
    def __init__(self):
        """初始化滑块界面组件"""
        self.sliders: Dict[str, SliderConfig] = {}
        self.slider_callbacks: Dict[str, Callable] = {}
        self.categories = {
            "curiosity": {
                "name": "好奇心参数",
                "description": "控制智能体探索行为的相关参数",
                "color": "#e74c3c"
            },
            "learning": {
                "name": "学习参数", 
                "description": "影响智能体学习能力的参数",
                "color": "#3498db"
            },
            "attention": {
                "name": "注意力参数",
                "description": "控制智能体专注程度的参数", 
                "color": "#2ecc71"
            },
            "decision": {
                "name": "决策参数",
                "description": "影响智能体决策过程的参数",
                "color": "#f39c12"
            },
            "general": {
                "name": "通用参数",
                "description": "其他参数设置",
                "color": "#9b59b6"
            }
        }
    
    def create_slider_interface(self, parameter_config: Dict[str, Any]) -> str:
        """创建滑块界面的HTML代码
        
        参数:
            parameter_config: 参数配置字典
            
        返回:
            HTML代码字符串
        """
        self._initialize_sliders(parameter_config)
        return self._generate_html_interface()
    
    def _initialize_sliders(self, parameter_config: Dict[str, Any]):
        """根据参数配置初始化滑块"""
        for param_name, config in parameter_config.items():
            slider_config = SliderConfig(
                name=param_name,
                label=config.get('label', param_name),
                min_value=config.get('min', 0.0),
                max_value=config.get('max', 1.0),
                step=config.get('step', 0.1),
                current_value=config.get('value', 0.0),
                unit=config.get('unit', ''),
                description=config.get('description', ''),
                category=self._get_parameter_category(param_name),
                color=self.categories[self._get_parameter_category(param_name)]["color"]
            )
            self.sliders[param_name] = slider_config
    
    def _get_parameter_category(self, param_name: str) -> str:
        """根据参数名称确定参数分类"""
        category_mapping = {
            "curiosity_weight": "curiosity",
            "exploration_rate": "curiosity", 
            "novelty_threshold": "curiosity",
            "learning_rate": "learning",
            "memory_capacity": "learning",
            "forgetting_rate": "learning",
            "attention_span": "attention",
            "focus_intensity": "attention",
            "distraction_filter": "attention",
            "decision_threshold": "decision",
            "risk_tolerance": "decision",
            "patience_level": "decision"
        }
        return category_mapping.get(param_name, "general")
    
    def _generate_html_interface(self) -> str:
        """生成完整的HTML界面"""
        html = self._generate_html_header()
        html += self._generate_control_panel()
        html += self._generate_slider_sections()
        html += self._generate_preset_buttons()
        html += self._generate_status_display()
        html += self._generate_javascript_code()
        html += self._generate_html_footer()
        return html
    
    def _generate_html_header(self) -> str:
        """生成HTML头部"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能体参数实时调节界面</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
    """
    
    def _generate_control_panel(self) -> str:
        """生成控制面板"""
        return """
        .control-panel {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background: #1e7e34;
            transform: translateY(-2px);
        }
        
        .btn-warning {
            background: #ffc107;
            color: #212529;
        }
        
        .btn-warning:hover {
            background: #e0a800;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #545b62;
            transform: translateY(-2px);
        }
    """
    
    def _generate_slider_sections(self) -> str:
        """生成滑块区域"""
        return """
        .slider-sections {
            padding: 30px;
            max-height: 70vh;
            overflow-y: auto;
        }
        
        .category-section {
            margin-bottom: 40px;
        }
        
        .category-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .category-color {
            width: 4px;
            height: 30px;
            border-radius: 2px;
            margin-right: 15px;
        }
        
        .category-info h3 {
            margin: 0;
            font-size: 1.3rem;
        }
        
        .category-info p {
            margin: 5px 0 0 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        .slider-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            border-left: 4px solid var(--slider-color);
        }
        
        .slider-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
        }
        
        .slider-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .slider-label {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
        }
        
        .slider-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--slider-color);
            background: rgba(0, 123, 255, 0.1);
            padding: 5px 15px;
            border-radius: 20px;
            min-width: 80px;
            text-align: center;
        }
        
        .slider-description {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 15px;
            font-style: italic;
        }
        
        .slider-control {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .slider {
            flex: 1;
            height: 8px;
            border-radius: 4px;
            background: #e9ecef;
            outline: none;
            transition: all 0.3s ease;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--slider-color);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        
        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--slider-color);
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }
        
        .slider-value-input {
            width: 80px;
            padding: 8px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            text-align: center;
            font-size: 0.9rem;
            transition: border-color 0.3s ease;
        }
        
        .slider-value-input:focus {
            outline: none;
            border-color: var(--slider-color);
        }
    """
    
    def _generate_preset_buttons(self) -> str:
        """生成预设按钮区域"""
        return """
        .preset-panel {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .preset-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .preset-btn {
            padding: 8px 16px;
            border: 2px solid transparent;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .preset-btn:hover {
            border-color: var(--slider-color);
            background: rgba(0, 123, 255, 0.05);
        }
        
        .preset-btn.active {
            background: var(--slider-color);
            color: white;
        }
    """
    
    def _generate_status_display(self) -> str:
        """生成状态显示区域"""
        return """
        .status-panel {
            background: #e9ecef;
            padding: 15px 20px;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }
        
        .status-indicator.warning {
            background: #ffc107;
        }
        
        .status-indicator.error {
            background: #dc3545;
        }
        
        .status-value {
            font-weight: 600;
            color: #495057;
        }
    """
    
    def _generate_javascript_code(self) -> str:
        """生成JavaScript代码"""
        return """
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .change-animation {
            animation: valueChange 0.5s ease-in-out;
        }
        
        @keyframes valueChange {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .feedback-message {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .feedback-message.show {
            transform: translateX(0);
        }
        
        .feedback-success {
            background: #28a745;
        }
        
        .feedback-warning {
            background: #ffc107;
            color: #212529;
        }
        
        .feedback-error {
            background: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>智能体参数实时调节</h1>
            <p>通过拖动滑块实时调节智能体行为参数</p>
        </div>
    """
    
    def _generate_slider_content(self) -> str:
        """生成滑块内容"""
        content = '<div class="slider-sections">\n'
        
        # 按分类组织滑块
        categorized_sliders = {}
        for param_name, slider in self.sliders.items():
            category = slider.category
            if category not in categorized_sliders:
                categorized_sliders[category] = []
            categorized_sliders[category].append((param_name, slider))
        
        # 生成每个分类的内容
        for category, sliders in categorized_sliders.items():
            category_info = self.categories[category]
            
            content += f"""
        <div class="category-section">
            <div class="category-header">
                <div class="category-color" style="background-color: {category_info['color']};"></div>
                <div class="category-info">
                    <h3>{category_info['name']}</h3>
                    <p>{category_info['description']}</p>
                </div>
            </div>
            <div class="slider-grid">
"""
            
            for param_name, slider in sliders:
                content += self._generate_single_slider(slider)
            
            content += """
            </div>
        </div>
"""
        
        content += '</div>\n'
        return content
    
    def _generate_single_slider(self, slider: SliderConfig) -> str:
        """生成单个滑块的HTML"""
        return f"""
            <div class="slider-container" data-slider="{slider.name}">
                <div class="slider-header">
                    <div class="slider-label">{slider.label}</div>
                    <div class="slider-value" id="{slider.name}-value">
                        {slider.current_value:.2f}{slider.unit}
                    </div>
                </div>
                <div class="slider-description">{slider.description}</div>
                <div class="slider-control">
                    <span class="slider-min">{slider.min_value}</span>
                    <input type="range" 
                           class="slider" 
                           id="{slider.name}" 
                           min="{slider.min_value}" 
                           max="{slider.max_value}" 
                           step="{slider.step}" 
                           value="{slider.current_value}"
                           style="--slider-color: {slider.color}">
                    <span class="slider-max">{slider.max_value}</span>
                    <input type="number" 
                           class="slider-value-input" 
                           id="{slider.name}-input" 
                           min="{slider.min_value}" 
                           max="{slider.max_value}" 
                           step="{slider.step}" 
                           value="{slider.current_value}">
                </div>
            </div>
"""
    
    def _generate_html_footer(self) -> str:
        """生成HTML尾部"""
        return f"""
        <div class="preset-panel">
            <div class="control-group">
                <h4 style="margin: 0; color: #495057;">快速预设：</h4>
                <div class="preset-group">
                    <button class="preset-btn" onclick="applyPreset('conservative')">保守型</button>
                    <button class="preset-btn" onclick="applyPreset('balanced')">平衡型</button>
                    <button class="preset-btn" onclick="applyPreset('aggressive')">激进型</button>
                    <button class="preset-btn" onclick="applyPreset('explorer')">探索型</button>
                </div>
            </div>
            <div class="control-group">
                <button class="btn btn-success" onclick="exportParameters()">导出参数</button>
                <button class="btn btn-warning" onclick="importParameters()">导入参数</button>
                <button class="btn btn-secondary" onclick="resetToDefaults()">重置默认</button>
            </div>
        </div>
        
        <div class="status-panel">
            <div class="status-item">
                <div class="status-indicator" id="connection-status"></div>
                <span>连接状态：</span>
                <span class="status-value" id="connection-text">已连接</span>
            </div>
            <div class="status-item">
                <span>最后更新：</span>
                <span class="status-value" id="last-update">--</span>
            </div>
            <div class="status-item">
                <span>待处理更改：</span>
                <span class="status-value" id="pending-changes">0</span>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let parameterController = null;
        let isConnected = false;
        let pendingChanges = new Set();
        
        // 初始化界面
        function initializeInterface() {{
            // 创建WebSocket连接（如果需要）
            setupWebSocket();
            
            // 绑定滑块事件
            bindSliderEvents();
            
            // 绑定预设事件
            bindPresetEvents();
            
            // 开始状态监控
            startStatusMonitor();
            
            console.log('参数调节界面初始化完成');
        }}
        
        // 设置WebSocket连接
        function setupWebSocket() {{
            try {{
                // 这里应该连接到实际的参数控制器
                // WebSocket连接逻辑
                isConnected = true;
                updateConnectionStatus(true);
            }} catch (error) {{
                console.warn('WebSocket连接失败，使用模拟模式');
                updateConnectionStatus(false);
            }}
        }}
        
        // 绑定滑块事件
        function bindSliderEvents() {{
            // 为所有滑块添加事件监听
            const sliders = document.querySelectorAll('.slider');
            sliders.forEach(slider => {{
                slider.addEventListener('input', (e) => {{
                    handleSliderChange(e.target.id, parseFloat(e.target.value));
                }});
                
                slider.addEventListener('change', (e) => {{
                    commitSliderChange(e.target.id, parseFloat(e.target.value));
                }});
            }});
            
            // 为输入框添加事件监听
            const inputs = document.querySelectorAll('.slider-value-input');
            inputs.forEach(input => {{
                input.addEventListener('change', (e) => {{
                    const value = parseFloat(e.target.value);
                    const sliderId = e.target.id.replace('-input', '');
                    handleSliderChange(sliderId, value);
                    commitSliderChange(sliderId, value);
                }});
            }});
        }}
        
        // 处理滑块变化（实时更新）
        function handleSliderChange(sliderId, value) {{
            // 更新显示值
            updateSliderDisplay(sliderId, value);
            
            // 添加到待处理更改
            pendingChanges.add(sliderId);
            updatePendingChanges();
            
            // 实时反馈
            if (isConnected) {{
                sendParameterUpdate(sliderId, value);
            }}
        }}
        
        // 提交滑块更改
        function commitSliderChange(sliderId, value) {{
            // 从待处理中移除
            pendingChanges.delete(sliderId);
            updatePendingChanges();
            
            // 应用参数更改
            applyParameterChange(sliderId, value);
            
            // 显示成功消息
            showFeedbackMessage('参数已更新', 'success');
        }}
        
        // 更新滑块显示
        function updateSliderDisplay(sliderId, value) {{
            const valueDisplay = document.getElementById(`${{sliderId}}-value`);
            const slider = document.getElementById(sliderId);
            const input = document.getElementById(`${{sliderId}}-input`);
            
            if (valueDisplay && slider && input) {{
                valueDisplay.textContent = value.toFixed(2) + getSliderUnit(sliderId);
                slider.value = value;
                input.value = value;
                valueDisplay.classList.add('change-animation');
                
                setTimeout(() => {{
                    valueDisplay.classList.remove('change-animation');
                }}, 500);
            }}
        }}
        
        // 获取滑块单位
        function getSliderUnit(sliderId) {{
            const units = {{
                'curiosity_weight': '权重',
                'exploration_rate': '率',
                'novelty_threshold': '阈值',
                'learning_rate': '率',
                'memory_capacity': '条目',
                'forgetting_rate': '率',
                'attention_span': '秒',
                'focus_intensity': '强度',
                'distraction_filter': '过滤器',
                'decision_threshold': '阈值',
                'risk_tolerance': '容忍度',
                'patience_level': '级别'
            }};
            return units[sliderId] || '';
        }}
        
        // 发送参数更新
        function sendParameterUpdate(parameterName, value) {{
            if (!isConnected) return;
            
            const message = {{
                type: 'parameter_update',
                parameter: parameterName,
                value: value,
                timestamp: Date.now()
            }};
            
            // WebSocket发送逻辑
            console.log('发送参数更新:', message);
        }}
        
        // 应用参数更改
        function applyParameterChange(parameterName, value) {{
            // 这里调用实际的参数控制器API
            fetch('/api/parameters/${{parameterName}}', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{ value: value }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    console.log('参数应用成功:', parameterName, value);
                }} else {{
                    console.error('参数应用失败:', data.error);
                    showFeedbackMessage('参数应用失败: ' + data.error, 'error');
                }}
            }})
            .catch(error => {{
                console.error('参数应用请求失败:', error);
                showFeedbackMessage('参数应用请求失败', 'error');
            }});
        }}
        
        // 绑定预设事件
        function bindPresetEvents() {{
            // 预设按钮事件已通过onclick属性绑定
        }}
        
        // 应用预设
        function applyPreset(presetName) {{
            const presets = {{
                'conservative': {{
                    'curiosity_weight': 0.3,
                    'exploration_rate': 0.05,
                    'learning_rate': 0.0005,
                    'risk_tolerance': 0.2
                }},
                'balanced': {{
                    'curiosity_weight': 1.0,
                    'exploration_rate': 0.1,
                    'learning_rate': 0.001,
                    'risk_tolerance': 0.5
                }},
                'aggressive': {{
                    'curiosity_weight': 1.8,
                    'exploration_rate': 0.2,
                    'learning_rate': 0.005,
                    'risk_tolerance': 0.8
                }},
                'explorer': {{
                    'curiosity_weight': 2.0,
                    'exploration_rate': 0.3,
                    'novelty_threshold': 1.0,
                    'risk_tolerance': 0.9
                }}
            }};
            
            const preset = presets[presetName];
            if (!preset) return;
            
            // 更新预设按钮状态
            document.querySelectorAll('.preset-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            // 应用预设参数
            for (const [paramName, value] of Object.entries(preset)) {{
                updateSliderDisplay(paramName, value);
                handleSliderChange(paramName, value);
                commitSliderChange(paramName, value);
            }}
            
            showFeedbackMessage(`已应用${{presetName}}预设`, 'success');
        }}
        
        // 导出参数
        function exportParameters() {{
            const parameters = {{}};
            document.querySelectorAll('.slider').forEach(slider => {{
                parameters[slider.id] = parseFloat(slider.value);
            }});
            
            const dataStr = JSON.stringify(parameters, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `parameters_${{new Date().toISOString().split('T')[0]}}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
            showFeedbackMessage('参数导出成功', 'success');
        }}
        
        // 导入参数
        function importParameters() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            
            input.onchange = (e) => {{
                const file = e.target.files[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = (e) => {{
                    try {{
                        const parameters = JSON.parse(e.target.result);
                        
                        Object.entries(parameters).forEach(([paramName, value]) => {{
                            if (document.getElementById(paramName)) {{
                                updateSliderDisplay(paramName, value);
                                handleSliderChange(paramName, value);
                                commitSliderChange(paramName, value);
                            }}
                        }});
                        
                        showFeedbackMessage('参数导入成功', 'success');
                    }} catch (error) {{
                        showFeedbackMessage('参数导入失败：文件格式错误', 'error');
                    }}
                }};
                reader.readAsText(file);
            }};
            
            input.click();
        }}
        
        // 重置为默认值
        function resetToDefaults() {{
            if (confirm('确定要重置所有参数为默认值吗？')) {{
                // 这里应该调用默认参数API
                document.querySelectorAll('.slider').forEach(slider => {{
                    const defaultValue = parseFloat(slider.getAttribute('value'));
                    updateSliderDisplay(slider.id, defaultValue);
                    handleSliderChange(slider.id, defaultValue);
                    commitSliderChange(slider.id, defaultValue);
                }});
                
                showFeedbackMessage('已重置为默认参数', 'success');
            }}
        }}
        
        // 开始状态监控
        function startStatusMonitor() {{
            setInterval(() => {{
                updateTimestamp();
            }}, 1000);
        }}
        
        // 更新时间戳
        function updateTimestamp() {{
            const now = new Date();
            const timeString = now.toLocaleTimeString('zh-CN');
            document.getElementById('last-update').textContent = timeString;
        }}
        
        // 更新连接状态
        function updateConnectionStatus(connected) {{
            const statusIndicator = document.getElementById('connection-status');
            const statusText = document.getElementById('connection-text');
            
            if (connected) {{
                statusIndicator.className = 'status-indicator';
                statusText.textContent = '已连接';
            }} else {{
                statusIndicator.className = 'status-indicator error';
                statusText.textContent = '离线模式';
            }}
        }}
        
        // 更新待处理更改计数
        function updatePendingChanges() {{
            document.getElementById('pending-changes').textContent = pendingChanges.size;
        }}
        
        // 显示反馈消息
        function showFeedbackMessage(message, type = 'info') {{
            const feedback = document.createElement('div');
            feedback.className = `feedback-message feedback-${{type}}`;
            feedback.textContent = message;
            
            document.body.appendChild(feedback);
            
            // 显示动画
            setTimeout(() => {{
                feedback.classList.add('show');
            }}, 100);
            
            // 自动隐藏
            setTimeout(() => {{
                feedback.classList.remove('show');
                setTimeout(() => {{
                    document.body.removeChild(feedback);
                }}, 300);
            }}, 3000);
        }}
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', initializeInterface);
    </script>
</body>
</html>
"""
    
    def get_slider_config_by_category(self, category: str) -> Dict[str, SliderConfig]:
        """获取指定分类的滑块配置
        
        参数:
            category: 参数分类
            
        返回:
            该分类下的滑块配置字典
        """
        result = {}
        for param_name, slider in self.sliders.items():
            if slider.category == category:
                result[param_name] = slider
        return result
    
    def update_slider_callback(self, param_name: str, callback: Callable):
        """更新滑块回调函数
        
        参数:
            param_name: 参数名称
            callback: 回调函数
        """
        self.slider_callbacks[param_name] = callback
    
    def get_categories(self) -> Dict[str, Dict[str, str]]:
        """获取所有参数分类信息
        
        返回:
            分类信息字典
        """
        return self.categories.copy()
    
    def export_interface_config(self, file_path: str) -> bool:
        """导出界面配置到文件
        
        参数:
            file_path: 导出文件路径
            
        返回:
            导出是否成功
        """
        try:
            config_data = {
                'sliders': {},
                'categories': self.categories,
                'export_timestamp': str(datetime.now()),
                'version': '1.0'
            }
            
            for param_name, slider in self.sliders.items():
                config_data['sliders'][param_name] = {
                    'label': slider.label,
                    'min_value': slider.min_value,
                    'max_value': slider.max_value,
                    'step': slider.step,
                    'current_value': slider.current_value,
                    'unit': slider.unit,
                    'description': slider.description,
                    'category': slider.category,
                    'color': slider.color
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"导出界面配置失败: {e}")
            return False
    
    def import_interface_config(self, file_path: str) -> bool:
        """从文件导入界面配置
        
        参数:
            file_path: 导入文件路径
            
        返回:
            导入是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            if 'sliders' not in config_data:
                print("错误：配置文件格式不正确")
                return False
            
            # 清除现有滑块
            self.sliders.clear()
            
            # 导入滑块配置
            for param_name, config in config_data['sliders'].items():
                slider_config = SliderConfig(
                    name=param_name,
                    label=config['label'],
                    min_value=config['min_value'],
                    max_value=config['max_value'],
                    step=config['step'],
                    current_value=config['current_value'],
                    unit=config['unit'],
                    description=config['description'],
                    category=config['category'],
                    color=config['color']
                )
                self.sliders[param_name] = slider_config
            
            print(f"成功导入 {len(self.sliders)} 个滑块配置")
            return True
            
        except Exception as e:
            print(f"导入界面配置失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    from datetime import datetime
    
    # 创建滑块界面
    interface = SliderInterface()
    
    # 示例参数配置
    example_config = {
        'curiosity_weight': {
            'label': '好奇心权重',
            'min': 0.0,
            'max': 2.0,
            'step': 0.1,
            'value': 1.0,
            'unit': '权重',
            'description': '控制智能体对新环境的探索倾向'
        },
        'learning_rate': {
            'label': '学习速率',
            'min': 0.0001,
            'max': 0.1,
            'step': 0.0001,
            'value': 0.001,
            'unit': '率',
            'description': '神经网络权重的更新速度'
        }
    }
    
    # 创建HTML界面
    html_content = interface.create_slider_interface(example_config)
    print("滑块界面HTML已生成")
    
    # 保存到文件
    with open('slider_interface.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("界面已保存到 slider_interface.html")