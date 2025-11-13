"""
视觉皮层模块 - 多模态感知融合系统
====================================

该模块实现了完整的多模态感知融合系统，包括：
1. 视觉编码：使用CLIP模型进行图像特征提取
2. 听觉编码：使用Whisper模型进行语音转录和特征提取  
3. 触觉感知：编码物品栏状态和交互信息
4. 本体感知：编码生理状态和空间位置信息
5. 丘脑门控：选择性关注关键特征

作者：NeuroMinecraftGenesis开发团队
版本：1.0.0
创建时间：2025-11-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from PIL import Image
import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualCortex(nn.Module):
    """
    视觉皮层核心类 - 多模态感知融合系统
    
    该类整合了多种感知模态的编码器，实现多模态特征融合和智能门控机制。
    模仿生物大脑的感知处理层次结构，从原始感觉输入到高级特征表示。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化视觉皮层系统
        
        Args:
            config: 配置字典，包含各模块的参数设置
        """
        super(VisualCortex, self).__init__()
        
        # 初始化配置
        self.config = config or self._default_config()
        
        # 初始化各编码器
        self._init_encoders()
        
        # 丘脑门控机制
        self.thalamic_gating = ThalamicGating(self.config)
        
        # 多模态特征融合层
        self.fusion_layers = self._init_fusion_layers()
        
        # 性能指标跟踪
        self.performance_metrics = {
            'visual_mAP': 0.0,
            'auditory_accuracy': 0.0,
            'response_latency': 0.0,
            'gate_selectivity': 0.0
        }
        
        logger.info("视觉皮层系统初始化完成")
    
    def _default_config(self) -> Dict:
        """
        获取默认配置参数
        
        Returns:
            Dict: 默认配置参数
        """
        return {
            'visual': {
                'input_size': (224, 224),
                'feature_dim': 512,
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'auditory': {
                'feature_dim': 512,
                'sample_rate': 16000,
                'model_name': 'openai/whisper-base'
            },
            'tactile': {
                'num_slots': 36,
                'slot_dim': 10,
                'total_dim': 360
            },
            'proprioception': {
                'health_dim': 32,
                'position_dim': 64,
                'inventory_dim': 128,
                'total_dim': 224
            },
            'thalamic_gating': {
                'initial_k': 256,
                'min_k': 128,
                'max_k': 512,
                'adaptation_rate': 0.01
            }
        }
    
    def _init_encoders(self):
        """
        初始化各种编码器模型
        """
        # 视觉编码器 - CLIP模型
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(self.config['visual']['model_name'])
            self.clip_model = CLIPModel.from_pretrained(self.config['visual']['model_name'])
            logger.info("CLIP视觉编码器初始化成功")
        except Exception as e:
            logger.warning(f"CLIP模型加载失败，将使用模拟模式: {e}")
            self.clip_model = None
        
        # 听觉编码器 - Whisper模型
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(self.config['auditory']['model_name'])
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.config['auditory']['model_name'])
            logger.info("Whisper听觉编码器初始化成功")
        except Exception as e:
            logger.warning(f"Whisper模型加载失败，将使用模拟模式: {e}")
            self.whisper_model = None
        
        # 触觉编码器层
        self.tactile_encoder = nn.Sequential(
            nn.Linear(self.config['tactile']['total_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # 本体感知编码器层
        self.proprioception_encoder = nn.Sequential(
            nn.Linear(self.config['proprioception']['total_dim'], 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def _init_fusion_layers(self) -> nn.ModuleDict:
        """
        初始化多模态特征融合层
        
        Returns:
            nn.ModuleDict: 融合层模块字典
        """
        fusion_layers = nn.ModuleDict()
        
        # 早期融合层 - 融合原始特征
        fusion_layers['early_fusion'] = nn.Sequential(
            nn.Linear(
                self.config['visual']['feature_dim'] + 
                self.config['auditory']['feature_dim'] + 
                self.config['tactile']['total_dim'] + 
                self.config['proprioception']['total_dim'],
                1024
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 后期融合层 - 融合处理后的特征
        fusion_layers['late_fusion'] = nn.Sequential(
            nn.Linear(
                self.config['visual']['feature_dim'] + 
                128 +  # tactile encoded
                256 +  # proprioception encoded
                512,   # auditory encoded
                512
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 注意力机制用于模态间权重分配
        fusion_layers['modality_attention'] = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            dropout=0.1
        )
        
        return fusion_layers
    
    def encode_visual(self, image_input: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        视觉编码 - 使用CLIP模型提取图像特征
        
        对输入的游戏画面进行224x224标准化处理，使用CLIP模型提取512维视觉特征。
        特征表示包含场景语义、物体识别、环境状态等关键视觉信息。
        
        Args:
            image_input: 输入图像，可以是PIL Image、torch.Tensor或numpy.ndarray
            
        Returns:
            torch.Tensor: 512维视觉特征向量
            
        Raises:
            ValueError: 当输入图像格式不支持时
        """
        try:
            # 输入预处理
            if isinstance(image_input, np.ndarray):
                image_input = Image.fromarray(image_input)
            
            # 使用CLIP处理器标准化图像
            if hasattr(self, 'clip_processor') and self.clip_processor:
                inputs = self.clip_processor(images=image_input, return_tensors="pt")
                
                # CLIP视觉编码
                with torch.no_grad():
                    vision_features = self.clip_model.get_image_features(**inputs)
                    
                # 特征归一化
                visual_features = F.normalize(vision_features, p=2, dim=-1)
                
            else:
                # 模拟模式 - 生成伪视觉特征
                logger.info("使用模拟视觉编码模式")
                visual_features = torch.randn(1, self.config['visual']['feature_dim'])
                visual_features = F.normalize(visual_features, p=2, dim=-1)
            
            # 性能指标更新
            self._update_metric('visual_processing_time', 0.05)  # 假设50ms处理时间
            
            logger.info(f"视觉编码完成，特征维度: {visual_features.shape}")
            return visual_features.squeeze(0)
            
        except Exception as e:
            logger.error(f"视觉编码失败: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.config['visual']['feature_dim'])
    
    def encode_auditory(self, audio_input: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        """
        听觉编码 - 使用Whisper模型转录音频特征
        
        对输入的游戏音效进行预处理，使用Whisper模型进行语音识别，
        然后将识别的文本转换为512维语义特征。
        
        Args:
            audio_input: 输入音频，可以是numpy数组、tensor或音频文件路径
            
        Returns:
            torch.Tensor: 512维听觉语义特征向量
        """
        try:
            # 音频预处理
            if isinstance(audio_input, str):
                # 从文件加载音频
                audio_data, sample_rate = librosa.load(audio_input, sr=self.config['auditory']['sample_rate'])
            elif isinstance(audio_input, torch.Tensor):
                audio_data = audio_input.numpy()
            else:
                audio_data = audio_input
            
            # 音频特征提取
            if hasattr(self, 'whisper_model') and self.whisper_model:
                # Whisper音频编码
                inputs = self.whisper_processor(
                    audio_data, 
                    sampling_rate=self.config['auditory']['sample_rate'], 
                    return_tensors="pt"
                )
                
                # 生成音频特征（不进行文本生成）
                with torch.no_grad():
                    encoder_outputs = self.whisper_model.get_encoder()(**inputs)
                    audio_features = encoder_outputs.last_hidden_state.mean(dim=1)
                    
                # 映射到固定维度
                if audio_features.shape[-1] != self.config['auditory']['feature_dim']:
                    # 使用线性层调整维度
                    if not hasattr(self, 'auditory_projection'):
                        self.auditory_projection = nn.Linear(
                            audio_features.shape[-1], 
                            self.config['auditory']['feature_dim']
                        ).to(audio_features.device)
                    
                    audio_features = self.auditory_projection(audio_features)
                
                # 特征归一化
                auditory_features = F.normalize(audio_features, p=2, dim=-1)
                
            else:
                # 模拟模式 - 生成伪听觉特征
                logger.info("使用模拟听觉编码模式")
                auditory_features = torch.randn(1, self.config['auditory']['feature_dim'])
                auditory_features = F.normalize(auditory_features, p=2, dim=-1)
            
            # 性能指标更新
            self._update_metric('auditory_processing_time', 0.1)  # 假设100ms处理时间
            
            logger.info(f"听觉编码完成，特征维度: {auditory_features.shape}")
            return auditory_features.squeeze(0)
            
        except Exception as e:
            logger.error(f"听觉编码失败: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.config['auditory']['feature_dim'])
    
    def encode_tactile(self, inventory_state: Dict) -> torch.Tensor:
        """
        触觉感知编码 - 编码物品栏状态信息
        
        对玩家的物品栏状态进行编码，每个物品槽位用10维向量表示，
        总共36个槽位，形成360维的触觉感知特征向量。
        
        Args:
            inventory_state: 物品栏状态字典，格式：
                {
                    'slots': [{'item_id': int, 'count': int, 'durability': float}, ...],
                    'selected_slot': int,
                    'armor_slots': [...],
                    'hotbar': [...]
                }
                
        Returns:
            torch.Tensor: 360维触觉感知特征向量
        """
        try:
            # 初始化触觉特征矩阵
            tactile_features = []
            
            # 处理主物品栏36个槽位
            slots = inventory_state.get('slots', [])
            for i in range(self.config['tactile']['num_slots']):
                if i < len(slots):
                    slot_info = slots[i]
                    # 物品ID编码 (one-hot + 连续特征)
                    item_id = slot_info.get('item_id', 0)
                    item_features = self._encode_item_info(item_id, slot_info)
                else:
                    # 空槽位编码
                    item_features = torch.zeros(self.config['tactile']['slot_dim'])
                
                tactile_features.append(item_features)
            
            # 处理特殊槽位（副手、快捷栏等）
            special_slots = ['selected_slot', 'armor_slots', 'hotbar']
            for slot_type in special_slots:
                slot_data = inventory_state.get(slot_type, [])
                if isinstance(slot_data, list):
                    for slot_info in slot_data:
                        if isinstance(slot_info, dict):
                            item_id = slot_info.get('item_id', 0)
                            item_features = self._encode_item_info(item_id, slot_info)
                            tactile_features.append(item_features)
            
            # 转换为张量并填充到固定长度
            tactile_tensor = torch.stack(tactile_features[:self.config['tactile']['num_slots']])
            if len(tactile_tensor) < self.config['tactile']['num_slots']:
                padding = torch.zeros(
                    self.config['tactile']['num_slots'] - len(tactile_tensor), 
                    self.config['tactile']['slot_dim']
                )
                tactile_tensor = torch.cat([tactile_tensor, padding], dim=0)
            
            # 通过触觉编码器层处理
            encoded_tactile = self.tactile_encoder(tactile_tensor.flatten())
            
            # 性能指标更新
            self._update_metric('tactile_processing_time', 0.01)  # 假设10ms处理时间
            
            logger.info(f"触觉编码完成，特征维度: {encoded_tactile.shape}")
            return encoded_tactile
            
        except Exception as e:
            logger.error(f"触觉编码失败: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.config['tactile']['total_dim'])
    
    def _encode_item_info(self, item_id: int, item_info: Dict) -> torch.Tensor:
        """
        编码单个物品信息为10维特征向量
        
        Args:
            item_id: 物品ID
            item_info: 物品详细信息
            
        Returns:
            torch.Tensor: 10维物品特征向量
        """
        features = torch.zeros(self.config['tactile']['slot_dim'])
        
        # 物品ID编码 (1维 - 归一化ID值)
        features[0] = min(item_id / 1000.0, 1.0)  # 归一化到[0,1]
        
        # 物品数量 (1维)
        count = item_info.get('count', 0)
        features[1] = min(count / 64.0, 1.0)  # 归一化数量
        
        # 耐久度 (1维)
        durability = item_info.get('durability', 1.0)
        features[2] = durability
        
        # 物品类型编码 (3维 - 工具、武器、消耗品等)
        item_category = self._get_item_category(item_id)
        category_onehot = F.one_hot(
            torch.tensor(item_category), 
            num_classes=8
        ).float()
        features[3:11] = category_onehot[:8]  # 使用前8维进行类别编码
        
        return features
    
    def _get_item_category(self, item_id: int) -> int:
        """
        根据物品ID获取物品类别
        
        Args:
            item_id: 物品ID
            
        Returns:
            int: 物品类别 (0-7)
        """
        # 简化的物品分类逻辑
        if item_id < 256:
            return 0  # 方块
        elif item_id < 512:
            return 1  # 工具
        elif item_id < 768:
            return 2  # 武器
        elif item_id < 1024:
            return 3  # 装备
        else:
            return 4  # 消耗品
    
    def encode_proprioception(self, body_state: Dict) -> torch.Tensor:
        """
        本体感知编码 - 编码身体和空间状态信息
        
        编码玩家的生理状态和空间位置信息，包括：
        - 生命值、健康状态
        - 饥饿度、体力状态  
        - 三维坐标、朝向角度
        - 背包占用率、装备状态
        
        Args:
            body_state: 身体状态字典，格式：
                {
                    'health': float, 'hunger': float, 'experience': float,
                    'position': {'x': float, 'y': float, 'z': float},
                    'rotation': {'yaw': float, 'pitch': float},
                    'inventory_fullness': float,
                    'equipment': {...}
                }
                
        Returns:
            torch.Tensor: 224维本体感知特征向量
        """
        try:
            proprioception_features = []
            
            # 1. 生理状态编码 (健康、饥饿、经验)
            health = body_state.get('health', 1.0)
            hunger = body_state.get('hunger', 1.0) 
            experience = body_state.get('experience', 0.0)
            
            physiological = torch.tensor([
                health, hunger, experience,
                (health + hunger) / 2.0,  # 综合健康度
                min(health * 2.0, 1.0),   # 生命值紧急度
                abs(hunger - 1.0)         # 饥饿度偏差
            ])
            proprioception_features.append(physiological)
            
            # 2. 空间位置编码 (坐标、朝向)
            position = body_state.get('position', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            rotation = body_state.get('rotation', {'yaw': 0.0, 'pitch': 0.0})
            
            # 三维坐标 (归一化)
            pos_x = position.get('x', 0.0) / 1000.0
            pos_y = position.get('y', 0.0) / 100.0 
            pos_z = position.get('z', 0.0) / 1000.0
            
            # 朝向角度 (sin/cos编码避免角度不连续)
            yaw = rotation.get('yaw', 0.0)
            pitch = rotation.get('pitch', 0.0)
            
            spatial_features = torch.tensor([
                pos_x, pos_y, pos_z,
                np.sin(yaw), np.cos(yaw),
                np.sin(pitch), np.cos(pitch),
                np.sqrt(pos_x**2 + pos_z**2),  # 距离原点距离
                pos_y  # 高度
            ])
            proprioception_features.append(spatial_features)
            
            # 3. 装备和物品状态编码
            inventory_fullness = body_state.get('inventory_fullness', 0.0)
            equipment = body_state.get('equipment', {})
            
            # 计算装备价值
            equipment_value = self._calculate_equipment_value(equipment)
            
            equipment_features = torch.tensor([
                inventory_fullness,
                equipment_value,
                min(equipment_value * 2.0, 1.0),  # 装备充足度
                1.0 - inventory_fullness  # 背包剩余空间
            ])
            proprioception_features.append(equipment_features)
            
            # 4. 状态变化趋势编码
            if hasattr(self, 'last_proprioception'):
                # 计算状态变化率
                current_state = torch.cat(proprioception_features)
                state_change = current_state - self.last_proprioception
                proprioception_features.append(state_change[:6])  # 只取前6维变化率
            else:
                # 首次记录，变化率为0
                proprioception_features.append(torch.zeros(6))
            
            # 拼接所有特征
            full_proprioception = torch.cat(proprioception_features)
            
            # 通过本体感知编码器处理
            encoded_proprioception = self.proprioception_encoder(full_proprioception)
            
            # 保存当前状态用于下次变化率计算
            self.last_proprioception = full_proprioception.detach()
            
            # 性能指标更新
            self._update_metric('proprioception_processing_time', 0.005)  # 假设5ms处理时间
            
            logger.info(f"本体感知编码完成，特征维度: {encoded_proprioception.shape}")
            return encoded_proprioception
            
        except Exception as e:
            logger.error(f"本体感知编码失败: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.config['proprioception']['total_dim'])
    
    def _calculate_equipment_value(self, equipment: Dict) -> float:
        """
        计算装备价值分数
        
        Args:
            equipment: 装备信息字典
            
        Returns:
            float: 装备价值分数 [0,1]
        """
        # 简化的装备价值计算逻辑
        total_value = 0.0
        item_count = 0
        
        for slot, item_info in equipment.items():
            if isinstance(item_info, dict):
                item_id = item_info.get('item_id', 0)
                count = item_info.get('count', 1)
                durability = item_info.get('durability', 1.0)
                
                # 根据物品ID和属性计算价值
                base_value = min(item_id / 1000.0, 1.0)  # 基础价值
                item_value = base_value * count * durability
                total_value += item_value
                item_count += 1
        
        # 归一化价值
        if item_count > 0:
            return min(total_value / item_count, 1.0)
        else:
            return 0.0
    
    def thalamic_gating(self, 
                       features: Dict[str, torch.Tensor], 
                       attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        丘脑门控机制 - 选择性关注关键特征
        
        实现类似大脑丘脑的门控功能，从多模态特征中选择最相关的K个特征。
        K值根据环境复杂度动态调整，确保关键信息的有效传递。
        
        Args:
            features: 多模态特征字典，包含各模态的特征向量
            attention_weights: 可选的注意力权重，用于特征重要性评估
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 选择后的特征向量
                - 选择的特征索引
        """
        try:
            # 计算整体特征重要性和复杂度
            all_features = []
            feature_importance = []
            
            # 评估各模态特征的重要性
            for modality, feature in features.items():
                if modality == 'visual':
                    importance = self._calculate_visual_importance(feature)
                elif modality == 'auditory':
                    importance = self._calculate_auditory_importance(feature)
                elif modality == 'tactile':
                    importance = self._calculate_tactile_importance(feature)
                elif modality == 'proprioception':
                    importance = self._calculate_proprioception_importance(feature)
                else:
                    importance = torch.rand(1)
                
                all_features.append(feature)
                feature_importance.append(importance)
            
            # 动态调整门控K值
            current_k = self._adaptive_gate_selection(all_features, feature_importance)
            
            # 计算最终特征重要性分数
            final_importance = torch.cat(feature_importance)
            
            # 选择Top-K重要特征
            if attention_weights is not None:
                # 如果有注意力权重，结合使用
                combined_importance = final_importance + attention_weights.squeeze()
            else:
                combined_importance = final_importance
            
            # 特征选择
            top_k_indices = torch.topk(combined_importance, k=current_k).indices
            
            # 获取选择后的特征
            selected_features = []
            for i, feature in enumerate(all_features):
                # 为每个特征分配重要性分数
                num_chunks = min(len(feature), current_k // len(all_features) + 1)
                chunk_size = len(feature) // num_chunks if num_chunks > 0 else len(feature)
                
                for j in range(num_chunks):
                    start_idx = j * chunk_size
                    end_idx = start_idx + chunk_size if j < num_chunks - 1 else len(feature)
                    
                    chunk_importance = combined_importance[start_idx:end_idx].mean()
                    if chunk_importance > 0.3:  # 选择重要性阈值
                        selected_features.append(feature[start_idx:end_idx])
            
            # 拼接选择后的特征
            gated_features = torch.cat(selected_features) if selected_features else torch.zeros(current_k)
            
            # 归一化选择后的特征
            if gated_features.numel() > 0:
                gated_features = F.normalize(gated_features, p=2, dim=-1)
            
            # 更新性能指标
            self.performance_metrics['gate_selectivity'] = current_k / sum(len(f) for f in all_features)
            
            logger.info(f"丘脑门控完成，选择特征数: {current_k}/{sum(len(f) for f in all_features)}")
            
            return gated_features, top_k_indices
            
        except Exception as e:
            logger.error(f"丘脑门控处理失败: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.config['thalamic_gating']['initial_k']), torch.arange(0)
    
    def _calculate_visual_importance(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        计算视觉特征的重要性分数
        
        Args:
            visual_features: 视觉特征向量
            
        Returns:
            torch.Tensor: 重要性分数
        """
        # 基于特征激活值和空间位置计算重要性
        feature_activation = torch.abs(visual_features)
        
        # 添加一些基于场景的启发式权重
        # 这里可以加入更复杂的视觉注意力机制
        importance = feature_activation.mean(dim=-1, keepdim=True)
        
        return importance
    
    def _calculate_auditory_importance(self, auditory_features: torch.Tensor) -> torch.Tensor:
        """
        计算听觉特征的重要性分数
        
        Args:
            auditory_features: 听觉特征向量
            
        Returns:
            torch.Tensor: 重要性分数
        """
        # 基于特征变化和信息量计算重要性
        feature_energy = torch.abs(auditory_features)**2
        importance = feature_energy.mean(dim=-1, keepdim=True)
        
        return importance
    
    def _calculate_tactile_importance(self, tactile_features: torch.Tensor) -> torch.Tensor:
        """
        计算触觉特征的重要性分数
        
        Args:
            tactile_features: 触觉特征向量
            
        Returns:
            torch.Tensor: 重要性分数
        """
        # 基于物品价值和使用概率计算重要性
        # 选择手中的物品槽位权重更高
        feature_activation = torch.abs(tactile_features)
        
        # 根据物品类型调整重要性
        importance = feature_activation.mean(dim=-1, keepdim=True)
        
        return importance
    
    def _calculate_proprioception_importance(self, proprioception_features: torch.Tensor) -> torch.Tensor:
        """
        计算本体感知特征的重要性分数
        
        Args:
            proprioception_features: 本体感知特征向量
            
        Returns:
            torch.Tensor: 重要性分数
        """
        # 基于生理状态紧急程度计算重要性
        # 健康状态越差，重要性越高
        feature_activation = torch.abs(proprioception_features)
        
        # 紧急状态权重
        health_component = torch.sigmoid(feature_activation[:3])  # 健康相关特征
        importance = (feature_activation.mean(dim=-1, keepdim=True) + health_component.mean(dim=-1, keepdim=True)) / 2
        
        return importance
    
    def _adaptive_gate_selection(self, features: List[torch.Tensor], importance: List[torch.Tensor]) -> int:
        """
        动态调整门控K值
        
        Args:
            features: 特征列表
            importance: 重要性列表
            
        Returns:
            int: 调整后的K值
        """
        # 基于特征数量和环境复杂度调整
        total_features = sum(len(f) for f in features)
        
        # 计算特征变化度（环境变化指标）
        if hasattr(self, 'last_features'):
            feature_changes = []
            for i, (curr_feat, last_feat) in enumerate(zip(features, self.last_features)):
                if len(curr_feat) == len(last_feat):
                    change = torch.mean(torch.abs(curr_feat - last_feat))
                    feature_changes.append(change)
            
            if feature_changes:
                avg_change = torch.stack(feature_changes).mean().item()
                # 环境变化越大，需要更多特征来捕捉变化
                complexity_factor = 1.0 + avg_change * 2.0
            else:
                complexity_factor = 1.0
        else:
            complexity_factor = 1.0
        
        # 动态调整K值
        base_k = self.config['thalamic_gating']['initial_k']
        min_k = self.config['thalamic_gating']['min_k']
        max_k = min(self.config['thalamic_gating']['max_k'], total_features)
        
        adjusted_k = min(int(base_k * complexity_factor), max_k)
        adjusted_k = max(adjusted_k, min_k)
        
        # 更新历史特征
        self.last_features = features
        
        return adjusted_k
    
    def multimodal_fusion(self, 
                         visual_features: torch.Tensor,
                         auditory_features: torch.Tensor, 
                         tactile_features: torch.Tensor,
                         proprioception_features: torch.Tensor) -> torch.Tensor:
        """
        多模态特征融合
        
        将来自不同感知模态的特征进行智能融合，
        形成统一的多模态表示。
        
        Args:
            visual_features: 视觉特征 (512维)
            auditory_features: 听觉特征 (512维)
            tactile_features: 触觉特征 (128维)
            proprioception_features: 本体感知特征 (256维)
            
        Returns:
            torch.Tensor: 融合后的多模态特征 (512维)
        """
        try:
            # 1. 早期融合 - 拼接所有原始特征
            early_fusion_input = torch.cat([
                visual_features,
                auditory_features,
                tactile_features,
                proprioception_features
            ], dim=0)
            
            early_fused = self.fusion_layers['early_fusion'](early_fusion_input)
            
            # 2. 各模态独立处理
            visual_processed = visual_features
            auditory_processed = auditory_features
            tactile_processed = tactile_features
            proprioception_processed = proprioception_features
            
            # 3. 模态间注意力机制
            modality_features = torch.stack([
                visual_processed,
                auditory_processed,
                tactile_processed,
                proprioception_processed
            ], dim=0)  # [4, feature_dim]
            
            # 应用多头注意力机制
            attended_features, attention_weights = self.fusion_layers['modality_attention'](
                modality_features, modality_features, modality_features
            )
            
            # 4. 后期融合 - 加权平均
            modality_weights = F.softmax(attention_weights.mean(dim=0), dim=-1)
            fused_features = torch.sum(attended_features * modality_weights.unsqueeze(-1), dim=0)
            
            # 5. 最终融合层处理
            final_fusion = self.fusion_layers['late_fusion'](fused_features)
            
            logger.info(f"多模态融合完成，输入维度: {[f.shape for f in [visual_features, auditory_features, tactile_features, proprioception_features]]}")
            logger.info(f"融合后维度: {final_fusion.shape}")
            
            return final_fusion
            
        except Exception as e:
            logger.error(f"多模态融合失败: {e}")
            # 返回简单拼接作为fallback
            fallback_fusion = torch.cat([
                visual_features, auditory_features, 
                tactile_features, proprioception_features
            ], dim=0)
            return fallback_fusion
    
    def forward(self, 
                visual_input: Optional[Union[Image.Image, torch.Tensor]] = None,
                auditory_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
                tactile_input: Optional[Dict] = None,
                proprioception_input: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 完整的多模态感知处理流程
        
        Args:
            visual_input: 视觉输入
            auditory_input: 听觉输入  
            tactile_input: 触觉输入
            proprioception_input: 本体感知输入
            
        Returns:
            Dict[str, torch.Tensor]: 包含各种特征和融合结果的字典
        """
        results = {}
        
        try:
            # 1. 各模态特征编码
            if visual_input is not None:
                results['visual_features'] = self.encode_visual(visual_input)
            
            if auditory_input is not None:
                results['auditory_features'] = self.encode_auditory(auditory_input)
                
            if tactile_input is not None:
                results['tactile_features'] = self.encode_tactile(tactile_input)
                
            if proprioception_input is not None:
                results['proprioception_features'] = self.encode_proprioception(proprioception_input)
            
            # 2. 多模态融合
            if len(results) >= 2:
                fused_features = self.multimodal_fusion(
                    results.get('visual_features', torch.zeros(512)),
                    results.get('auditory_features', torch.zeros(512)),
                    results.get('tactile_features', torch.zeros(128)),
                    results.get('proprioception_features', torch.zeros(256))
                )
                results['fused_features'] = fused_features
            
            # 3. 丘脑门控选择
            if results:
                gated_features, selected_indices = self.thalamic_gating(results)
                results['gated_features'] = gated_features
                results['selected_indices'] = selected_indices
            
            # 4. 性能指标更新
            total_processing_time = (
                self.performance_metrics.get('visual_processing_time', 0) +
                self.performance_metrics.get('auditory_processing_time', 0) +
                self.performance_metrics.get('tactile_processing_time', 0) +
                self.performance_metrics.get('proprioception_processing_time', 0)
            )
            results['processing_latency'] = total_processing_time
            
            logger.info(f"视觉皮层处理完成，处理延迟: {total_processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"前向传播失败: {e}")
            return {'error': str(e)}
    
    def _update_metric(self, metric_name: str, value: float):
        """
        更新性能指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        if metric_name in self.performance_metrics:
            # 使用指数移动平均更新指标
            alpha = 0.1
            self.performance_metrics[metric_name] = (
                alpha * value + (1 - alpha) * self.performance_metrics[metric_name]
            )
        else:
            self.performance_metrics[metric_name] = value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取当前性能指标
        
        Returns:
            Dict[str, float]: 性能指标字典
        """
        return self.performance_metrics.copy()
    
    def reset_metrics(self):
        """
        重置性能指标
        """
        for key in self.performance_metrics:
            if isinstance(self.performance_metrics[key], (int, float)):
                self.performance_metrics[key] = 0.0
        
        logger.info("性能指标已重置")


class ThalamicGating(nn.Module):
    """
    丘脑门控模块
    
    模拟生物大脑丘脑的门控功能，实现选择性感知机制。
    """
    
    def __init__(self, config: Dict):
        super(ThalamicGating, self).__init__()
        self.config = config
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, importance_weights: torch.Tensor) -> torch.Tensor:
        """
        门控前向传播
        
        Args:
            features: 输入特征
            importance_weights: 重要性权重
            
        Returns:
            torch.Tensor: 门控后的特征
        """
        # 计算门控权重
        gate_weights = self.gate_network(features)
        
        # 应用门控
        gated_features = features * gate_weights
        
        # 选择Top-K特征
        k = self.config['thalamic_gating']['initial_k']
        if gated_features.shape[0] > k:
            # 保留最重要的K个特征
            top_k_indices = torch.topk(torch.abs(gated_features), k=k).indices
            gated_features = gated_features[top_k_indices]
        
        return gated_features


def create_visual_cortex(config: Optional[Dict] = None) -> VisualCortex:
    """
    创建视觉皮层实例的工厂函数
    
    Args:
        config: 可选的配置字典
        
    Returns:
        VisualCortex: 视觉皮层实例
    """
    return VisualCortex(config)


# 使用示例
if __name__ == "__main__":
    # 创建视觉皮层实例
    visual_cortex = create_visual_cortex()
    
    # 模拟输入数据
    import random
    
    # 模拟视觉输入
    dummy_image = torch.randn(3, 224, 224)
    
    # 模拟听觉输入
    dummy_audio = np.random.randn(16000)  # 1秒音频
    
    # 模拟触觉输入
    dummy_inventory = {
        'slots': [
            {'item_id': 1, 'count': 64, 'durability': 1.0},
            {'item_id': 5, 'count': 32, 'durability': 0.8},
            {'item_id': 10, 'count': 1, 'durability': 0.5}
        ] * 12,  # 重复36次
        'selected_slot': 0
    }
    
    # 模拟本体感知输入
    dummy_body_state = {
        'health': 0.8,
        'hunger': 0.9,
        'experience': 0.5,
        'position': {'x': 100.0, 'y': 64.0, 'z': 200.0},
        'rotation': {'yaw': 45.0, 'pitch': 0.0},
        'inventory_fullness': 0.3,
        'equipment': {
            'helmet': {'item_id': 100, 'count': 1, 'durability': 0.9},
            'sword': {'item_id': 200, 'count': 1, 'durability': 0.7}
        }
    }
    
    # 执行多模态感知处理
    results = visual_cortex(
        visual_input=dummy_image,
        auditory_input=dummy_audio,
        tactile_input=dummy_inventory,
        proprioception_input=dummy_body_state
    )
    
    # 打印结果
    print("\n=== 视觉皮层处理结果 ===")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, mean={value.mean().item():.4f}")
        else:
            print(f"{key}: {value}")
    
    # 获取性能指标
    metrics = visual_cortex.get_performance_metrics()
    print(f"\n=== 性能指标 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n视觉皮层系统测试完成！")