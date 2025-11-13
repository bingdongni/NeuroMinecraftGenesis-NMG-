"""
参数预设管理系统
实现参数组合的预设保存、加载和应用功能
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import copy


@dataclass
class ParameterPreset:
    """参数预设数据类"""
    name: str
    description: str
    parameters: Dict[str, float]
    tags: List[str]
    created_at: str
    updated_at: str
    usage_count: int = 0
    category: str = "custom"
    color: str = "#6c757d"
    is_default: bool = False


class ParameterPresetManager:
    """参数预设管理类
    
    负责管理智能体参数的各种预设配置，包括：
    - 预设的创建、保存和加载
    - 预设分类和标签管理
    - 预设使用统计和推荐
    - 预设导入导出功能
    - 预设验证和冲突检测
    """
    
    def __init__(self, storage_path: str = "parameter_presets"):
        """初始化参数预设管理器
        
        参数:
            storage_path: 预设文件存储路径
        """
        self.storage_path = storage_path
        self.presets: Dict[str, ParameterPreset] = {}
        self.preset_callbacks: List[Callable] = []
        self.categories = {
            "exploration": {
                "name": "探索型",
                "description": "高好奇心和探索率的配置",
                "color": "#e74c3c",
                "icon": "🔍"
            },
            "learning": {
                "name": "学习型", 
                "description": "高学习速率和记忆容量的配置",
                "color": "#3498db",
                "icon": "📚"
            },
            "attention": {
                "name": "专注型",
                "description": "高专注度和低干扰的配置",
                "color": "#2ecc71",
                "icon": "🎯"
            },
            "decision": {
                "name": "决策型",
                "description": "快速决策和风险承受的配置",
                "color": "#f39c12",
                "icon": "⚡"
            },
            "balanced": {
                "name": "平衡型",
                "description": "各项指标均衡的配置",
                "color": "#9b59b6",
                "icon": "⚖️"
            },
            "conservative": {
                "name": "保守型",
                "description": "低风险和稳定性的配置",
                "color": "#95a5a6",
                "icon": "🛡️"
            },
            "custom": {
                "name": "自定义",
                "description": "用户自定义的配置",
                "color": "#34495e",
                "icon": "🔧"
            }
        }
        
        # 确保存储目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 初始化默认预设
        self._initialize_default_presets()
        
        print("参数预设管理器初始化完成")
    
    def _initialize_default_presets(self):
        """初始化默认预设"""
        default_presets = {
            "保守型": ParameterPreset(
                name="保守型",
                description="低风险偏好，注重稳定性，适用于安全关键场景",
                parameters={
                    "curiosity_weight": 0.3,
                    "exploration_rate": 0.05,
                    "novelty_threshold": 4.0,
                    "learning_rate": 0.0005,
                    "memory_capacity": 2000,
                    "forgetting_rate": 0.005,
                    "attention_span": 2.0,
                    "focus_intensity": 0.9,
                    "distraction_filter": 0.8,
                    "decision_threshold": 0.9,
                    "risk_tolerance": 0.2,
                    "patience_level": 5.0
                },
                tags=["稳定", "低风险", "安全"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="conservative",
                color=self.categories["conservative"]["color"],
                is_default=True,
                usage_count=0
            ),
            
            "平衡型": ParameterPreset(
                name="平衡型",
                description="各项参数均衡，适用于大多数场景的通用配置",
                parameters={
                    "curiosity_weight": 1.0,
                    "exploration_rate": 0.1,
                    "novelty_threshold": 2.5,
                    "learning_rate": 0.001,
                    "memory_capacity": 1000,
                    "forgetting_rate": 0.01,
                    "attention_span": 1.0,
                    "focus_intensity": 0.8,
                    "distraction_filter": 0.3,
                    "decision_threshold": 0.7,
                    "risk_tolerance": 0.5,
                    "patience_level": 2.0
                },
                tags=["通用", "均衡", "标准"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="balanced",
                color=self.categories["balanced"]["color"],
                is_default=True,
                usage_count=0
            ),
            
            "探索型": ParameterPreset(
                name="探索型",
                description="高好奇心和探索率，适用于需要广泛探索的场景",
                parameters={
                    "curiosity_weight": 2.0,
                    "exploration_rate": 0.3,
                    "novelty_threshold": 1.0,
                    "learning_rate": 0.002,
                    "memory_capacity": 500,
                    "forgetting_rate": 0.02,
                    "attention_span": 0.5,
                    "focus_intensity": 0.6,
                    "distraction_filter": 0.1,
                    "decision_threshold": 0.5,
                    "risk_tolerance": 0.8,
                    "patience_level": 1.0
                },
                tags=["探索", "创新", "发现"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="exploration",
                color=self.categories["exploration"]["color"],
                is_default=True,
                usage_count=0
            ),
            
            "学习型": ParameterPreset(
                name="学习型",
                description="高学习速率和记忆容量，适用于学习和适应场景",
                parameters={
                    "curiosity_weight": 1.2,
                    "exploration_rate": 0.15,
                    "novelty_threshold": 2.0,
                    "learning_rate": 0.005,
                    "memory_capacity": 5000,
                    "forgetting_rate": 0.003,
                    "attention_span": 1.5,
                    "focus_intensity": 0.95,
                    "distraction_filter": 0.6,
                    "decision_threshold": 0.8,
                    "risk_tolerance": 0.4,
                    "patience_level": 3.0
                },
                tags=["学习", "适应", "记忆"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="learning",
                color=self.categories["learning"]["color"],
                is_default=True,
                usage_count=0
            ),
            
            "专注型": ParameterPreset(
                name="专注型",
                description="高专注度和强干扰过滤，适用于需要深度思考的场景",
                parameters={
                    "curiosity_weight": 0.8,
                    "exploration_rate": 0.08,
                    "novelty_threshold": 3.0,
                    "learning_rate": 0.0008,
                    "memory_capacity": 1500,
                    "forgetting_rate": 0.008,
                    "attention_span": 3.0,
                    "focus_intensity": 0.95,
                    "distraction_filter": 0.9,
                    "decision_threshold": 0.85,
                    "risk_tolerance": 0.3,
                    "patience_level": 4.0
                },
                tags=["专注", "深度", "思考"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="attention",
                color=self.categories["attention"]["color"],
                is_default=True,
                usage_count=0
            ),
            
            "决策型": ParameterPreset(
                name="决策型",
                description="快速决策和风险承受，适用于需要快速响应的场景",
                parameters={
                    "curiosity_weight": 1.5,
                    "exploration_rate": 0.25,
                    "novelty_threshold": 1.5,
                    "learning_rate": 0.003,
                    "memory_capacity": 800,
                    "forgetting_rate": 0.015,
                    "attention_span": 0.8,
                    "focus_intensity": 0.7,
                    "distraction_filter": 0.2,
                    "decision_threshold": 0.6,
                    "risk_tolerance": 0.9,
                    "patience_level": 0.5
                },
                tags=["决策", "快速", "响应"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category="decision",
                color=self.categories["decision"]["color"],
                is_default=True,
                usage_count=0
            )
        }
        
        # 添加默认预设到管理器
        for preset_name, preset_data in default_presets.items():
            self.presets[preset_name] = preset_data
        
        print(f"已初始化 {len(default_presets)} 个默认预设")
    
    def save_preset(self, name: str, parameters: Dict[str, float], 
                   description: str = "", tags: List[str] = None,
                   category: str = "custom") -> bool:
        """保存参数预设
        
        参数:
            name: 预设名称
            parameters: 参数字典
            description: 预设描述
            tags: 标签列表
            category: 预设分类
            
        返回:
            保存是否成功
        """
        try:
            # 验证预设名称
            if not name or not name.strip():
                print("错误：预设名称不能为空")
                return False
            
            # 检查名称是否已存在
            if name in self.presets:
                # 如果是覆盖操作，需要确认
                existing_preset = self.presets[name]
                if not existing_preset.is_default:
                    print(f"警告：预设 '{name}' 已存在，将被覆盖")
                else:
                    print("错误：无法覆盖默认预设")
                    return False
            
            # 确定分类颜色
            preset_color = self.categories.get(category, {}).get("color", "#6c757d")
            
            # 创建或更新预设
            if name in self.presets:
                preset = self.presets[name]
                preset.parameters = copy.deepcopy(parameters)
                preset.description = description
                preset.tags = tags or []
                preset.category = category
                preset.color = preset_color
                preset.updated_at = datetime.now().isoformat()
                if not preset.is_default:
                    preset.usage_count = 0  # 重置使用次数
            else:
                preset = ParameterPreset(
                    name=name,
                    description=description,
                    parameters=copy.deepcopy(parameters),
                    tags=tags or [],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    category=category,
                    color=preset_color,
                    is_default=False,
                    usage_count=0
                )
            
            self.presets[name] = preset
            
            # 通知监听器
            self._notify_preset_listeners('save', preset)
            
            # 保存到文件
            self._save_preset_to_file(name, preset)
            
            print(f"参数预设 '{name}' 保存成功")
            return True
            
        except Exception as e:
            print(f"保存参数预设失败: {e}")
            return False
    
    def load_preset(self, name: str) -> Optional[ParameterPreset]:
        """加载参数预设
        
        参数:
            name: 预设名称
            
        返回:
            加载的预设对象，如果不存在则返回None
        """
        try:
            if name not in self.presets:
                print(f"错误：未找到预设 '{name}'")
                return None
            
            preset = self.presets[name]
            
            # 更新使用统计
            preset.usage_count += 1
            preset.updated_at = datetime.now().isoformat()
            
            # 通知监听器
            self._notify_preset_listeners('load', preset)
            
            print(f"参数预设 '{name}' 加载成功，使用次数: {preset.usage_count}")
            return copy.deepcopy(preset)
            
        except Exception as e:
            print(f"加载参数预设失败: {e}")
            return None
    
    def delete_preset(self, name: str) -> bool:
        """删除参数预设
        
        参数:
            name: 预设名称
            
        返回:
            删除是否成功
        """
        try:
            if name not in self.presets:
                print(f"错误：未找到预设 '{name}'")
                return False
            
            preset = self.presets[name]
            
            # 检查是否是默认预设
            if preset.is_default:
                print("错误：无法删除默认预设")
                return False
            
            # 从管理器中删除
            del self.presets[name]
            
            # 删除文件
            self._delete_preset_file(name)
            
            # 通知监听器
            self._notify_preset_listeners('delete', preset)
            
            print(f"参数预设 '{name}' 删除成功")
            return True
            
        except Exception as e:
            print(f"删除参数预设失败: {e}")
            return False
    
    def list_presets(self, category: str = None, tags: List[str] = None) -> List[ParameterPreset]:
        """列出参数预设
        
        参数:
            category: 按分类过滤
            tags: 按标签过滤
            
        返回:
            符合条件的预设列表
        """
        result = list(self.presets.values())
        
        # 按分类过滤
        if category:
            result = [p for p in result if p.category == category]
        
        # 按标签过滤
        if tags:
            result = [p for p in result if any(tag in p.tags for tag in tags)]
        
        # 按使用次数排序
        result.sort(key=lambda p: p.usage_count, reverse=True)
        
        return result
    
    def search_presets(self, query: str) -> List[ParameterPreset]:
        """搜索参数预设
        
        参数:
            query: 搜索关键词
            
        返回:
            匹配的预设列表
        """
        query = query.lower()
        result = []
        
        for preset in self.presets.values():
            # 搜索名称
            if query in preset.name.lower():
                result.append(preset)
                continue
            
            # 搜索描述
            if query in preset.description.lower():
                result.append(preset)
                continue
            
            # 搜索标签
            if any(query in tag.lower() for tag in preset.tags):
                result.append(preset)
                continue
        
        return result
    
    def get_preset_by_category(self, category: str) -> Dict[str, ParameterPreset]:
        """获取指定分类的所有预设
        
        参数:
            category: 分类名称
            
        返回:
            该分类下的预设字典
        """
        result = {}
        for name, preset in self.presets.items():
            if preset.category == category:
                result[name] = preset
        return result
    
    def get_most_used_presets(self, limit: int = 5) -> List[ParameterPreset]:
        """获取最常用的预设
        
        参数:
            limit: 返回数量限制
            
        返回:
            使用次数最多的预设列表
        """
        sorted_presets = sorted(
            self.presets.values(), 
            key=lambda p: p.usage_count, 
            reverse=True
        )
        return sorted_presets[:limit]
    
    def get_recent_presets(self, limit: int = 5) -> List[ParameterPreset]:
        """获取最近的预设
        
        参数:
            limit: 返回数量限制
            
        返回:
            最近更新的预设列表
        """
        sorted_presets = sorted(
            self.presets.values(), 
            key=lambda p: p.updated_at, 
            reverse=True
        )
        return sorted_presets[:limit]
    
    def duplicate_preset(self, source_name: str, target_name: str) -> bool:
        """复制预设
        
        参数:
            source_name: 源预设名称
            target_name: 目标预设名称
            
        返回:
            复制是否成功
        """
        try:
            if source_name not in self.presets:
                print(f"错误：源预设 '{source_name}' 不存在")
                return False
            
            if target_name in self.presets:
                print(f"错误：目标预设 '{target_name}' 已存在")
                return False
            
            source_preset = self.presets[source_name]
            
            # 创建复制的新预设
            new_preset = ParameterPreset(
                name=target_name,
                description=f"复制自: {source_preset.description}",
                parameters=copy.deepcopy(source_preset.parameters),
                tags=source_preset.tags + ["复制"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                category=source_preset.category,
                color=source_preset.color,
                is_default=False,
                usage_count=0
            )
            
            self.presets[target_name] = new_preset
            
            # 通知监听器
            self._notify_preset_listeners('duplicate', new_preset)
            
            print(f"预设 '{source_name}' 已复制为 '{target_name}'")
            return True
            
        except Exception as e:
            print(f"复制预设失败: {e}")
            return False
    
    def export_presets(self, file_path: str, preset_names: List[str] = None) -> bool:
        """导出预设到文件
        
        参数:
            file_path: 导出文件路径
            preset_names: 要导出的预设名称列表，None表示导出全部
            
        返回:
            导出是否成功
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'presets': {}
            }
            
            # 确定要导出的预设
            presets_to_export = self.presets
            if preset_names:
                presets_to_export = {name: self.presets[name] for name in preset_names if name in self.presets}
            
            # 导出预设数据
            for name, preset in presets_to_export.items():
                export_data['presets'][name] = asdict(preset)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功导出 {len(presets_to_export)} 个预设到: {file_path}")
            return True
            
        except Exception as e:
            print(f"导出预设失败: {e}")
            return False
    
    def import_presets(self, file_path: str, overwrite: bool = False) -> int:
        """从文件导入预设
        
        参数:
            file_path: 导入文件路径
            overwrite: 是否覆盖已存在的预设
            
        返回:
            成功导入的预设数量
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if 'presets' not in import_data:
                print("错误：文件格式不正确")
                return 0
            
            imported_count = 0
            
            for name, preset_data in import_data['presets'].items():
                # 检查是否已存在
                if name in self.presets and not overwrite:
                    print(f"跳过已存在的预设: {name}")
                    continue
                
                # 重建预设对象
                try:
                    preset = ParameterPreset(**preset_data)
                    
                    # 如果是覆盖，保留使用次数
                    if name in self.presets and overwrite:
                        preset.usage_count = self.presets[name].usage_count
                    
                    self.presets[name] = preset
                    imported_count += 1
                    
                except Exception as e:
                    print(f"导入预设 '{name}' 失败: {e}")
                    continue
            
            print(f"成功导入 {imported_count} 个预设")
            return imported_count
            
        except Exception as e:
            print(f"导入预设失败: {e}")
            return 0
    
    def get_categories(self) -> Dict[str, Dict[str, str]]:
        """获取所有预设分类信息
        
        返回:
            分类信息字典
        """
        return copy.deepcopy(self.categories)
    
    def validate_preset(self, preset: ParameterPreset) -> Dict[str, Any]:
        """验证预设参数的有效性
        
        参数:
            preset: 要验证的预设
            
        返回:
            验证结果字典
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # 检查必要参数
        required_params = [
            'curiosity_weight', 'exploration_rate', 'learning_rate',
            'memory_capacity', 'attention_span', 'decision_threshold'
        ]
        
        missing_params = [param for param in required_params if param not in preset.parameters]
        if missing_params:
            validation_result['errors'].append(f"缺少必要参数: {', '.join(missing_params)}")
            validation_result['valid'] = False
        
        # 检查参数范围
        parameter_ranges = {
            'curiosity_weight': (0.0, 2.0),
            'exploration_rate': (0.0, 1.0),
            'learning_rate': (0.0001, 0.1),
            'memory_capacity': (100, 10000),
            'attention_span': (0.1, 10.0),
            'decision_threshold': (0.0, 1.0)
        }
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            if param_name in preset.parameters:
                value = preset.parameters[param_name]
                if not (min_val <= value <= max_val):
                    validation_result['warnings'].append(
                        f"参数 '{param_name}' 的值 {value} 超出建议范围 [{min_val}, {max_val}]"
                    )
        
        # 检查参数一致性
        params = preset.parameters
        
        # 好奇心和探索率的一致性
        if params.get('curiosity_weight', 0) > 1.5 and params.get('exploration_rate', 0) > 0.2:
            validation_result['suggestions'].append(
                "高好奇心与高探索率可能导致过度探索，建议调整平衡"
            )
        
        # 学习速率和记忆容量的关系
        if params.get('learning_rate', 0) > 0.01 and params.get('memory_capacity', 0) < 500:
            validation_result['suggestions'].append(
                "高学习速率建议配合更大的记忆容量使用"
            )
        
        # 决策阈值和风险容忍度
        if params.get('decision_threshold', 0) > 0.8 and params.get('risk_tolerance', 0) < 0.3:
            validation_result['warnings'].append(
                "高决策阈值与低风险容忍度可能导致决策延迟"
            )
        
        return validation_result
    
    def add_preset_listener(self, listener: Callable[[str, ParameterPreset], None]):
        """添加预设事件监听器
        
        参数:
            listener: 监听器函数，接受事件类型和预设对象
        """
        self.preset_callbacks.append(listener)
    
    def remove_preset_listener(self, listener: Callable[[str, ParameterPreset], None]):
        """移除预设事件监听器
        
        参数:
            listener: 要移除的监听器函数
        """
        if listener in self.preset_callbacks:
            self.preset_callbacks.remove(listener)
    
    def _notify_preset_listeners(self, event_type: str, preset: ParameterPreset):
        """通知所有预设事件监听器"""
        for listener in self.preset_callbacks:
            try:
                listener(event_type, preset)
            except Exception as e:
                print(f"预设事件监听器执行错误: {e}")
    
    def _save_preset_to_file(self, name: str, preset: ParameterPreset):
        """保存预设到文件"""
        try:
            file_path = os.path.join(self.storage_path, f"{name}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(preset), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存预设文件失败: {e}")
    
    def _delete_preset_file(self, name: str):
        """删除预设文件"""
        try:
            file_path = os.path.join(self.storage_path, f"{name}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"删除预设文件失败: {e}")
    
    def load_all_presets_from_files(self):
        """从文件加载所有预设"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            preset_data = json.load(f)
                        
                        preset = ParameterPreset(**preset_data)
                        self.presets[preset.name] = preset
                        
                    except Exception as e:
                        print(f"加载预设文件 {filename} 失败: {e}")
            
            print(f"从文件加载了 {len(self.presets)} 个预设")
            
        except Exception as e:
            print(f"加载预设文件失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取预设统计信息
        
        返回:
            统计信息字典
        """
        total_presets = len(self.presets)
        total_usage = sum(preset.usage_count for preset in self.presets.values())
        
        category_stats = {}
        for category in self.categories.keys():
            category_stats[category] = len(self.get_preset_by_category(category))
        
        most_used = self.get_most_used_presets(1)
        most_used_name = most_used[0].name if most_used else None
        
        return {
            'total_presets': total_presets,
            'total_usage': total_usage,
            'average_usage': total_usage / max(total_presets, 1),
            'category_distribution': category_stats,
            'most_used_preset': most_used_name,
            'default_presets': len([p for p in self.presets.values() if p.is_default]),
            'custom_presets': len([p for p in self.presets.values() if not p.is_default])
        }


# 使用示例
if __name__ == "__main__":
    # 创建预设管理器
    manager = ParameterPresetManager()
    
    # 获取所有分类
    categories = manager.get_categories()
    print("预设分类:", categories.keys())
    
    # 列出所有预设
    all_presets = manager.list_presets()
    print(f"共有 {len(all_presets)} 个预设")
    
    # 加载平衡型预设
    balanced = manager.load_preset("平衡型")
    if balanced:
        print("平衡型预设参数:", balanced.parameters)
    
    # 创建自定义预设
    custom_params = {
        "curiosity_weight": 1.3,
        "exploration_rate": 0.15,
        "learning_rate": 0.002,
        "memory_capacity": 1200,
        "attention_span": 1.2,
        "decision_threshold": 0.75
    }
    
    manager.save_preset("我的测试预设", custom_params, "测试用自定义预设", ["测试", "自定义"])
    
    # 搜索预设
    search_results = manager.search_presets("探索")
    print(f"搜索'探索'的结果: {[p.name for p in search_results]}")
    
    # 获取统计信息
    stats = manager.get_statistics()
    print("预设统计:", stats)
    
    # 导出预设
    manager.export_presets("my_presets.json", ["平衡型", "探索型"])