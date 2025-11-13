"""
创造力记忆系统 - 核心创新算法实现 (升级版)

本模块实现了基于扩散模型和生成对抗网络的先进创意记忆系统，
包含新颖性检测、联想变异、组合创新等核心算法，为AI系统提供创造性思维能力。

升级功能：
1. 扩散模型生成机制 - 用于创意内容的生成和细化
2. 生成对抗网络(GAN) - 用于创意质量评估和优化
3. 创意生成和新颖性评估 - 多层次的新颖性检测
4. 多模态创意融合 - 跨模态的创意组合
5. 创意质量评价和优化 - 基于GAN的自动优化

作者: AI创造力系统
创建时间: 2025-11-13
版本: 2.0 (升级版)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import copy


# ==================== 扩散模型实现 ====================

class DiffusionModel:
    """扩散模型用于创意生成和细化"""
    
    def __init__(self, feature_dim: int = 128, timesteps: int = 1000, device: str = 'cpu'):
        """
        初始化扩散模型
        
        Args:
            feature_dim: 特征向量维度
            timesteps: 扩散时间步数
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.feature_dim = feature_dim
        self.timesteps = timesteps
        self.device = device
        self.model = None
        self.noise_schedule = np.linspace(0.001, 0.02, timesteps)
        self.initialize_model()
    
    def initialize_model(self):
        """初始化扩散模型网络"""
        class DiffusionUNet(nn.Module):
            def __init__(self, feature_dim, timesteps):
                super().__init__()
                self.feature_dim = feature_dim
                self.time_embed = nn.Embedding(timesteps, 64)
                self.net = nn.Sequential(
                    nn.Linear(feature_dim + 64, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, feature_dim)
                )
            
            def forward(self, x, t):
                time_embed = self.time_embed(t)
                x_embed = torch.cat([x, time_embed], dim=-1)
                return self.net(x_embed)
        
        self.model = DiffusionUNet(self.feature_dim, self.timesteps).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.initialized = True
    
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程：添加噪声
        
        Args:
            x0: 原始特征
            t: 时间步
            
        Returns:
            添加噪声的特征和噪声
        """
        noise = torch.randn_like(x0).to(self.device)
        alpha_t = self.noise_schedule[t]
        
        # 添加噪声
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        
        return xt, noise
    
    def denoise_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        去噪步骤
        
        Args:
            xt: 当前特征
            t: 当前时间步
            
        Returns:
            去噪后的特征
        """
        t_tensor = torch.tensor([t] * xt.shape[0]).to(self.device)
        predicted_noise = self.model(xt, t_tensor)
        
        # 计算去噪特征
        alpha_t = self.noise_schedule[t]
        alpha_t_prev = self.noise_schedule[t-1] if t > 0 else 0.001
        
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # DDPM采样
        noise = torch.randn_like(xt).to(self.device) if t > 0 else 0
        xt_prev = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise
        
        return xt_prev
    
    def generate_creative_content(self, batch_size: int = 1) -> np.ndarray:
        """
        生成创意内容
        
        Args:
            batch_size: 生成批次大小
            
        Returns:
            生成的创意特征
        """
        self.model.eval()
        with torch.no_grad():
            # 从纯噪声开始
            x = torch.randn(batch_size, self.feature_dim).to(self.device)
            
            # 逐步去噪
            for t in reversed(range(self.timesteps)):
                if t % 100 == 0:
                    print(f"  扩散模型采样进度: {t}/{self.timesteps}")
                x = self.denoise_step(x, t)
            
            # 归一化
            x = torch.tanh(x)
        
        return x.cpu().numpy()
    
    def train_step(self, real_features: torch.Tensor) -> Dict[str, float]:
        """
        训练扩散模型
        
        Args:
            real_features: 真实特征数据
            
        Returns:
            训练损失信息
        """
        self.model.train()
        
        # 随机时间步
        t = torch.randint(0, self.timesteps, (real_features.shape[0],)).to(self.device)
        
        # 前向扩散
        xt, noise = self.forward_diffusion(real_features, t)
        
        # 预测噪声
        predicted_noise = self.model(xt, t)
        
        # 计算损失
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'diffusion_loss': loss.item()}


# ==================== 生成对抗网络实现 ====================

class CreativeGAN:
    """用于创意质量评估和优化的生成对抗网络"""
    
    def __init__(self, feature_dim: int = 128, device: str = 'cpu'):
        """
        初始化GAN
        
        Args:
            feature_dim: 特征维度
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.device = device
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.training_history = {
            'gen_losses': [],
            'disc_losses': [],
            'quality_scores': []
        }
    
    def _build_generator(self) -> nn.Module:
        """构建生成器网络"""
        class Generator(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, feature_dim),
                    nn.Tanh()
                )
            
            def forward(self, z):
                return self.net(z)
        
        return Generator(self.feature_dim).to(self.device)
    
    def _build_discriminator(self) -> nn.Module:
        """构建判别器网络"""
        class Discriminator(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        return Discriminator(self.feature_dim).to(self.device)
    
    def train_step(self, real_features: torch.Tensor, batch_size: int = 32) -> Dict[str, float]:
        """
        GAN训练步骤
        
        Args:
            real_features: 真实特征数据
            batch_size: 批次大小
            
        Returns:
            训练损失信息
        """
        self.generator.train()
        self.discriminator.train()
        
        # 训练判别器
        self.discriminator.zero_grad()
        
        # 真实数据
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_features[:batch_size])
        real_loss = self.criterion(real_output, real_labels)
        
        # 生成数据
        noise = torch.randn(batch_size, 128).to(self.device)
        fake_features = self.generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_features)
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # 总判别器损失
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # 训练生成器
        self.generator.zero_grad()
        
        # 重新生成数据
        noise = torch.randn(batch_size, 128).to(self.device)
        fake_features = self.generator(noise)
        fake_output = self.discriminator(fake_features)
        
        # 生成器希望判别器认为生成数据是真实的
        gen_loss = self.criterion(fake_output, real_labels)
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # 记录训练历史
        self.training_history['gen_losses'].append(gen_loss.item())
        self.training_history['disc_losses'].append(disc_loss.item())
        
        return {
            'generator_loss': gen_loss.item(),
            'discriminator_loss': disc_loss.item()
        }
    
    def generate_creative_features(self, num_samples: int = 1) -> np.ndarray:
        """
        生成创意特征
        
        Args:
            num_samples: 生成样本数量
            
        Returns:
            生成的创意特征
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, 128).to(self.device)
            generated_features = self.generator(noise)
            # 添加tanh激活确保范围在[-1, 1]
            generated_features = torch.tanh(generated_features)
        
        return generated_features.cpu().numpy()
    
    def evaluate_quality(self, features: np.ndarray) -> float:
        """
        评估创意质量
        
        Args:
            features: 待评估的特征
            
        Returns:
            质量分数 [0, 1]
        """
        self.discriminator.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features).to(self.device)
            quality_score = self.discriminator(features_tensor).mean().item()
        
        return quality_score
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标"""
        if not self.training_history['gen_losses']:
            return {'status': 'no_training_data'}
        
        return {
            'avg_generator_loss': np.mean(self.training_history['gen_losses'][-100:]),
            'avg_discriminator_loss': np.mean(self.training_history['disc_losses'][-100:]),
            'latest_quality_score': self.training_history['quality_scores'][-1] if self.training_history['quality_scores'] else 0.0,
            'training_steps': len(self.training_history['gen_losses'])
        }


# ==================== 多模态创意融合器 ====================

class MultimodalCreativeFusion:
    """多模态创意融合器"""
    
    def __init__(self, modal_dims: Dict[str, int]):
        """
        初始化多模态融合器
        
        Args:
            modal_dims: 各模态的维度字典 {'text': 512, 'image': 1024, 'audio': 256}
        """
        self.modal_dims = modal_dims
        self.fusion_weights = {modal: 1.0 for modal in modal_dims.keys()}
        self.cross_modal_attention = {}
        self._initialize_attention()
    
    def _initialize_attention(self):
        """初始化跨模态注意力机制"""
        for modal1 in self.modal_dims.keys():
            for modal2 in self.modal_dims.keys():
                if modal1 != modal2:
                    key = f"{modal1}_to_{modal2}"
                    self.cross_modal_attention[key] = CrossModalAttention(
                        self.modal_dims[modal1], 
                        self.modal_dims[modal2]
                    )
    
    def fuse_creative_concepts(self, modal_data: Dict[str, np.ndarray], 
                              creative_type: str = 'innovation') -> Dict[str, np.ndarray]:
        """
        融合多模态创意概念
        
        Args:
            modal_data: 各模态的数据 {'text': text_features, 'image': image_features}
            creative_type: 创意类型 ('innovation', 'imagination', 'combination')
            
        Returns:
            融合后的多模态特征
        """
        fused_features = {}
        
        # 1. 模态内融合
        intra_fused = self._intra_modal_fusion(modal_data, creative_type)
        
        # 2. 跨模态注意力融合
        inter_fused = self._inter_modal_attention(intra_fused)
        
        # 3. 创意类型特定融合
        creative_fused = self._creative_type_specific_fusion(inter_fused, creative_type)
        
        return creative_fused
    
    def _intra_modal_fusion(self, modal_data: Dict[str, np.ndarray], 
                           creative_type: str) -> Dict[str, np.ndarray]:
        """模态内融合"""
        fused = {}
        
        for modal, data in modal_data.items():
            if modal in self.modal_dims:
                # 应用模态特定的处理
                if len(data.shape) > 1:
                    # 如果是多实例，取平均
                    fused[modal] = np.mean(data, axis=0)
                else:
                    fused[modal] = data
                
                # 模态特定增强
                fused[modal] = self._enhance_modal_features(fused[modal], modal, creative_type)
        
        return fused
    
    def _inter_modal_attention(self, modal_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """跨模态注意力融合"""
        enhanced_features = {}
        
        for target_modal in modal_features.keys():
            enhanced_features[target_modal] = modal_features[target_modal].copy()
            
            # 聚合来自其他模态的信息
            attention_sum = np.zeros_like(modal_features[target_modal])
            attention_count = 0
            
            for source_modal, source_features in modal_features.items():
                if source_modal != target_modal:
                    attention_key = f"{target_modal}_to_{source_modal}"
                    if attention_key in self.cross_modal_attention:
                        attention_weights = self.cross_modal_attention[attention_key].compute_attention(
                            modal_features[target_modal], source_features
                        )
                        
                        # 应用注意力权重
                        attended_features = source_features * attention_weights
                        attention_sum += attended_features
                        attention_count += 1
            
            # 融合注意力结果
            if attention_count > 0:
                attention_sum /= attention_count
                fusion_ratio = 0.3  # 注意力融合比例
                enhanced_features[target_modal] = (
                    (1 - fusion_ratio) * enhanced_features[target_modal] + 
                    fusion_ratio * attention_sum
                )
        
        return enhanced_features
    
    def _creative_type_specific_fusion(self, modal_features: Dict[str, np.ndarray], 
                                     creative_type: str) -> Dict[str, np.ndarray]:
        """创意类型特定融合"""
        
        if creative_type == 'innovation':
            # 创新型：强调差异性和新颖性
            return self._innovation_fusion(modal_features)
        elif creative_type == 'imagination':
            # 想象型：强调联想和组合
            return self._imagination_fusion(modal_features)
        elif creative_type == 'combination':
            # 组合型：强调协同和整合
            return self._combination_fusion(modal_features)
        else:
            return modal_features
    
    def _innovation_fusion(self, modal_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """创新型融合策略"""
        # 增强特征差异性
        fused = {}
        feature_list = list(modal_features.values())
        
        if len(feature_list) > 1:
            # 计算特征间差异
            for i, (modal, features) in enumerate(modal_features.items()):
                # 与其他模态的差异
                differences = []
                for j, other_features in enumerate(feature_list):
                    if i != j:
                        diff = np.linalg.norm(features - other_features)
                        differences.append(diff)
                
                # 增加新颖性权重
                novelty_weight = np.mean(differences) if differences else 0
                enhanced_features = features * (1 + novelty_weight * 0.1)
                fused[modal] = enhanced_features
        else:
            fused = modal_features
        
        return fused
    
    def _imagination_fusion(self, modal_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """想象型融合策略"""
        # 增强联想性
        fused = {}
        feature_array = np.array(list(modal_features.values()))
        
        # 计算联想向量（特征间的中间值）
        if len(feature_array) > 1:
            associative_vector = np.mean(feature_array, axis=0)
            
            for modal, features in modal_features.items():
                # 增强联想性
                associative_enhancement = np.tanh((associative_vector - features) * 2)
                enhanced_features = features + associative_enhancement * 0.2
                fused[modal] = enhanced_features
        else:
            fused = modal_features
        
        return fused
    
    def _combination_fusion(self, modal_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """组合型融合策略"""
        # 强调协同性
        fused = {}
        feature_list = list(modal_features.values())
        
        if len(feature_list) > 1:
            # 计算协同向量
           协同_vector = np.sum(feature_list, axis=0)
            
            for modal, features in modal_features.items():
                # 协同增强
                synergy_factor = np.dot(features, 协同_vector) / (np.linalg.norm(features) * np.linalg.norm(协同_vector) + 1e-8)
                enhanced_features = features * (1 + synergy_factor * 0.15)
                fused[modal] = enhanced_features
        else:
            fused = modal_features
        
        return fused
    
    def _enhance_modal_features(self, features: np.ndarray, modal: str, creative_type: str) -> np.ndarray:
        """增强模态特定特征"""
        # 模态特定增强逻辑
        if modal == 'text':
            # 文本模态：增强语义丰富度
            feature_variance = np.var(features)
            enhanced = features * (1 + feature_variance * 0.1)
        elif modal == 'image':
            # 图像模态：增强空间多样性
            enhanced = features * (1 + np.random.normal(0, 0.05, len(features)))
        elif modal == 'audio':
            # 音频模态：增强频率特性
            enhanced = np.tanh(features)  # 使用tanh激活增强动态范围
        else:
            enhanced = features
        
        return enhanced


class CrossModalAttention:
    """跨模态注意力机制"""
    
    def __init__(self, query_dim: int, key_dim: int):
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.attention_weights = None
    
    def compute_attention(self, query_features: np.ndarray, key_features: np.ndarray) -> np.ndarray:
        """
        计算注意力权重
        
        Args:
            query_features: 查询特征
            key_features: 键特征
            
        Returns:
            注意力权重
        """
        # 简化的注意力计算
        similarity = np.dot(query_features, key_features) / (
            np.linalg.norm(query_features) * np.linalg.norm(key_features) + 1e-8
        )
        
        # Softmax激活
        attention_weights = np.exp(similarity * 2) / (1 + np.exp(similarity * 2))
        
        return attention_weights


# ==================== 主要的创意记忆系统类 ====================

class CreativeMemory:
    """
    创造力记忆系统核心类 (升级版)
    
    升级功能：
    1. 扩散模型生成机制 - 用于创意内容的生成和细化
    2. 生成对抗网络(GAN) - 用于创意质量评估和优化
    3. 创意生成和新颖性评估 - 多层次的新颖性检测
    4. 多模态创意融合 - 跨模态的创意组合
    5. 创意质量评价和优化 - 基于GAN的自动优化
    
    该系统模拟大脑的创造性思维机制，通过先进的深度学习模型，
    产生具有创新性、多样性和高质量的创意内容。
    """
    
    def __init__(self, memory_capacity: int = 10000, novelty_threshold: float = 0.4, 
                 device: str = 'cpu', modal_dims: Optional[Dict[str, int]] = None):
        """
        初始化创造力记忆系统
        
        Args:
            memory_capacity: 记忆容量上限
            novelty_threshold: 新颖性检测阈值
            device: 计算设备
            modal_dims: 多模态维度配置
        """
        # 基础存储
        self.memories = []  # 存储所有记忆条目
        self.action_library = {}  # 行为动作库
        self.novelty_memory = deque(maxlen=1000)  # 新颖性记忆缓存
        
        # 升级功能组件
        self.diffusion_model = DiffusionModel(feature_dim=128, device=device)
        self.creative_gan = CreativeGAN(feature_dim=128, device=device)
        
        # 多模态配置
        if modal_dims is None:
            modal_dims = {'text': 512, 'image': 1024, 'audio': 256, 'sensor': 128}
        self.modal_dims = modal_dims
        self.multimodal_fusion = MultimodalCreativeFusion(modal_dims)
        
        # 系统参数
        self.memory_capacity = memory_capacity
        self.novelty_threshold = novelty_threshold
        self.mutation_threshold = 0.3  # 联想变异相似度阈值
        self.device = device
        
        # 创新统计
        self.innovation_stats = {
            'total_actions': 0,
            'innovative_actions': 0,
            'novel_behaviors': 0,
            'diffusion_generations': 0,
            'gan_generations': 0,
            'quality_optimizations': 0,
            'multimodal_fusions': 0,
            'start_time': datetime.now()
        }
        
        # 质量评估历史
        self.quality_history = deque(maxlen=500)
        self.creative_quality_scores = []
        
        # 行为频率跟踪
        self.behavior_frequency = defaultdict(int)
        self.hourly_novel_behaviors = deque(maxlen=24)
        
        # 记忆特征向量维度
        self.feature_dim = 128
        
        # 训练计数器
        self.training_step = 0
        
        print(f"🎨 创造力记忆系统升级版初始化完成")
        print(f"   扩散模型: ✅ 已启用")
        print(f"   GAN网络: ✅ 已启用")
        print(f"   多模态融合: ✅ 已启用")
        print(f"   记忆容量: {memory_capacity}")
        print(f"   新颖性阈值: {novelty_threshold}")
        print(f"   计算设备: {device}")
    
    def enhanced_novelty_detection(self, current_perception: np.ndarray, 
                                 modal_type: str = 'sensor') -> Dict[str, Any]:
        """
        增强版新颖性检测算法
        
        使用多层次检测机制：
        1. 基础相似度检测
        2. 扩散模型生成对比
        3. GAN质量评估
        
        Args:
            current_perception: 当前感知特征向量
            modal_type: 模态类型 ('text', 'image', 'audio', 'sensor')
            
        Returns:
            Dict containing enhanced novelty assessment
        """
        # 1. 基础新颖性检测
        basic_novelty = self._basic_novelty_detection(current_perception)
        
        # 2. 扩散模型生成对比
        diffusion_novelty = self._diffusion_based_novelty(current_perception)
        
        # 3. GAN质量评估
        gan_quality = self._gan_based_quality_assessment(current_perception)
        
        # 4. 多模态融合检测
        multimodal_novelty = self._multimodal_novelty_detection(current_perception, modal_type)
        
        # 5. 综合新颖性评分
        novelty_score = self._compute_enhanced_novelty_score(
            basic_novelty, diffusion_novelty, gan_quality, multimodal_novelty
        )
        
        # 6. 多巴胺调制
        dopamine_level = self._compute_enhanced_dopamine_level(novelty_score, gan_quality)
        
        result = {
            'novelty_score': novelty_score,
            'dopamine_level': dopamine_level,
            'is_highly_novel': novelty_score > self.novelty_threshold,
            'quality_score': gan_quality,
            'modal_type': modal_type,
            'component_scores': {
                'basic': basic_novelty,
                'diffusion': diffusion_novelty,
                'gan_quality': gan_quality,
                'multimodal': multimodal_novelty
            }
        }
        
        # 更新新颖性记忆缓存
        self.novelty_memory.append({
            'timestamp': datetime.now(),
            'novelty_score': novelty_score,
            'dopamine_level': dopamine_level,
            'is_highly_novel': result['is_highly_novel'],
            'quality_score': gan_quality
        })
        
        return result
    
    def _basic_novelty_detection(self, current_perception: np.ndarray) -> Dict[str, Any]:
        """基础新颖性检测"""
        if len(self.memories) == 0:
            return {'novelty_score': 1.0, 'max_similarity': 0.0}
        
        similarities = []
        for memory in self.memories:
            similarity = self._cosine_similarity(current_perception, memory['features'])
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        novelty_score = 1.0 - max_similarity
        
        return {'novelty_score': novelty_score, 'max_similarity': max_similarity}
    
    def _diffusion_based_novelty(self, current_perception: np.ndarray) -> Dict[str, Any]:
        """基于扩散模型的新颖性检测"""
        try:
            # 生成扩散模型样本进行对比
            generated_samples = self.diffusion_model.generate_creative_content(batch_size=5)
            
            # 计算与生成样本的差异
            diffusion_distances = []
            for sample in generated_samples:
                distance = np.linalg.norm(current_perception - sample)
                diffusion_distances.append(distance)
            
            # 距离越小，新颖性越高
            avg_distance = np.mean(diffusion_distances)
            max_possible_distance = np.sqrt(2 * len(current_perception))  # 理论最大距离
            
            diffusion_novelty = min(avg_distance / max_possible_distance, 1.0)
            
            return {
                'novelty_score': diffusion_novelty,
                'avg_distance': avg_distance,
                'generated_samples': len(generated_samples)
            }
        except Exception as e:
            print(f"扩散模型新颖性检测失败: {e}")
            return {'novelty_score': 0.5, 'avg_distance': 0.5, 'generated_samples': 0}
    
    def _gan_based_quality_assessment(self, current_perception: np.ndarray) -> float:
        """基于GAN的质量评估"""
        try:
            # 将感知转换为GAN可评估的格式
            if len(current_perception) != 128:
                # 调整维度
                if len(current_perception) < 128:
                    padded = np.pad(current_perception, (0, 128 - len(current_perception)))
                    gan_input = padded
                else:
                    gan_input = current_perception[:128]
            else:
                gan_input = current_perception
            
            # 使用判别器评估质量
            quality_score = self.creative_gan.evaluate_quality(gan_input.reshape(1, -1))
            
            # 更新质量历史
            self.quality_history.append({
                'timestamp': datetime.now(),
                'quality_score': quality_score,
                'perception_norm': np.linalg.norm(current_perception)
            })
            
            return float(quality_score)
        except Exception as e:
            print(f"GAN质量评估失败: {e}")
            return 0.5
    
    def _multimodal_novelty_detection(self, current_perception: np.ndarray, modal_type: str) -> Dict[str, Any]:
        """多模态新颖性检测"""
        try:
            # 创建模态特定表示
            modal_data = {modal_type: current_perception}
            
            # 使用多模态融合器进行新颖性检测
            if len(self.memories) > 0:
                # 从记忆中提取相同模态的数据
                same_modal_memories = [
                    m for m in self.memories 
                    if m.get('modal_type') == modal_type
                ]
                
                if same_modal_memories:
                    memory_features = np.array([m['features'] for m in same_modal_memories[-10:]])
                    modal_data['memory_reference'] = np.mean(memory_features, axis=0)
                    
                    # 计算多模态新颖性
                    reference_features = modal_data['memory_reference']
                    distance = np.linalg.norm(current_perception - reference_features)
                    max_distance = np.sqrt(2 * len(current_perception))
                    multimodal_novelty = min(distance / max_distance, 1.0)
                else:
                    multimodal_novelty = 0.8  # 默认新颖性
            else:
                multimodal_novelty = 1.0  # 完全新颖
            
            return {
                'novelty_score': multimodal_novelty,
                'modal_type': modal_type,
                'memory_references': len([m for m in self.memories if m.get('modal_type') == modal_type])
            }
        except Exception as e:
            print(f"多模态新颖性检测失败: {e}")
            return {'novelty_score': 0.5, 'modal_type': modal_type, 'memory_references': 0}
    
    def _compute_enhanced_novelty_score(self, basic: Dict, diffusion: Dict, 
                                      gan_quality: float, multimodal: Dict) -> float:
        """计算增强版新颖性分数"""
        # 权重配置
        weights = {
            'basic': 0.3,
            'diffusion': 0.3,
            'quality_penalty': 0.2,  # 质量高时新颖性降低
            'multimodal': 0.2
        }
        
        # 基础新颖性
        basic_score = basic['novelty_score']
        
        # 扩散新颖性
        diffusion_score = diffusion['novelty_score']
        
        # 质量惩罚（高质量样本新颖性稍降低）
        quality_score = gan_quality
        quality_penalty = quality_score * 0.1  # 质量越高，新颖性略微降低
        
        # 多模态新颖性
        multimodal_score = multimodal['novelty_score']
        
        # 综合评分
        enhanced_score = (
            weights['basic'] * basic_score +
            weights['diffusion'] * diffusion_score +
            weights['quality_penalty'] * (1 - quality_penalty) +
            weights['multimodal'] * multimodal_score
        )
        
        return min(enhanced_score, 1.0)
    
    def _compute_enhanced_dopamine_level(self, novelty_score: float, quality_score: float) -> float:
        """计算增强版多巴胺水平"""
        # 基础多巴胺（新颖性驱动）
        base_dopamine = novelty_score * 0.8
        
        # 质量奖励（高质量创意额外奖励）
        quality_reward = quality_score * 0.3
        
        # 多巴胺总和
        total_dopamine = base_dopamine + quality_reward
        
        # 限制范围
        return min(total_dopamine, 2.0)
    
    def generate_creative_content_advanced(self, num_samples: int = 1, 
                                         generation_method: str = 'diffusion',
                                         quality_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        高级创意内容生成
        
        Args:
            num_samples: 生成样本数量
            generation_method: 生成方法 ('diffusion', 'gan', 'hybrid')
            quality_threshold: 质量阈值
            
        Returns:
            生成的创意内容列表
        """
        generated_samples = []
        attempts = 0
        max_attempts = num_samples * 5  # 最多尝试次数
        
        while len(generated_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            if generation_method == 'diffusion':
                features = self.diffusion_model.generate_creative_content(1)[0]
                method = 'diffusion'
            elif generation_method == 'gan':
                features = self.creative_gan.generate_creative_features(1)[0]
                method = 'gan'
            else:  # hybrid
                # 混合生成：GAN生成 + 扩散模型细化
                gan_features = self.creative_gan.generate_creative_features(1)[0]
                diffusion_features = self.diffusion_model.generate_creative_content(1)[0]
                features = (gan_features + diffusion_features) / 2
                method = 'hybrid'
            
            # 质量评估
            quality_score = self.creative_gan.evaluate_quality(features.reshape(1, -1))
            
            # 如果质量达到阈值，保留样本
            if quality_score >= quality_threshold:
                sample = {
                    'features': features,
                    'quality_score': quality_score,
                    'generation_method': method,
                    'timestamp': datetime.now(),
                    'novelty_assessment': self._basic_novelty_detection(features)
                }
                generated_samples.append(sample)
                
                # 更新统计
                self.innovation_stats[f'{method}_generations'] += 1
        
        print(f"✅ 成功生成 {len(generated_samples)} 个高质量创意样本")
        print(f"   生成方法: {generation_method}")
        print(f"   尝试次数: {attempts}")
        print(f"   成功率: {len(generated_samples)/attempts:.1%}")
        
        return generated_samples
    
    def advanced_associative_mutation(self, current_features: np.ndarray, 
                                    dopamine_level: float, 
                                    mutation_type: str = 'diffusion_enhanced') -> np.ndarray:
        """
        高级联想变异算法
        
        Args:
            current_features: 当前特征向量
            dopamine_level: 多巴胺水平
            mutation_type: 变异类型 ('basic', 'diffusion_enhanced', 'gan_optimized', 'multimodal')
            
        Returns:
            变异后的新特征向量
        """
        if mutation_type == 'basic':
            # 基础变异
            return self._basic_associative_mutation(current_features, dopamine_level)
        elif mutation_type == 'diffusion_enhanced':
            # 扩散模型增强变异
            return self._diffusion_enhanced_mutation(current_features, dopamine_level)
        elif mutation_type == 'gan_optimized':
            # GAN优化变异
            return self._gan_optimized_mutation(current_features, dopamine_level)
        elif mutation_type == 'multimodal':
            # 多模态变异
            return self._multimodal_mutation(current_features, dopamine_level)
        else:
            # 默认使用扩散增强
            return self._diffusion_enhanced_mutation(current_features, dopamine_level)
    
    def _diffusion_enhanced_mutation(self, current_features: np.ndarray, dopamine_level: float) -> np.ndarray:
        """扩散模型增强变异"""
        try:
            # 生成扩散模型样本
            diffusion_samples = self.diffusion_model.generate_creative_content(batch_size=3)
            
            # 选择最相似的样本作为变异基础
            similarities = []
            for sample in diffusion_samples:
                similarity = self._cosine_similarity(current_features, sample)
                similarities.append(similarity)
            
            # 选择相似度适中的样本（既相似又有差异）
            target_similarity = 0.3 + (dopamine_level * 0.2)  # 多巴胺越高，容忍差异越大
            best_idx = np.argmin([abs(sim - target_similarity) for sim in similarities])
            
            base_sample = diffusion_samples[best_idx]
            
            # 基于多巴胺水平的变异强度
            mutation_strength = min(dopamine_level / 2.0, 1.0)
            
            # 执行变异
            mutated_features = (
                current_features * (1 - mutation_strength * 0.5) + 
                base_sample * (mutation_strength * 0.5)
            )
            
            # 添加噪声
            noise_scale = mutation_strength * 0.1
            noise = np.random.normal(0, noise_scale, len(mutated_features))
            mutated_features += noise
            
            # 归一化
            mutated_features = np.clip(mutated_features, -1, 1)
            
            return mutated_features
            
        except Exception as e:
            print(f"扩散增强变异失败，使用基础变异: {e}")
            return self._basic_associative_mutation(current_features, dopamine_level)
    
    def _gan_optimized_mutation(self, current_features: np.ndarray, dopamine_level: float) -> np.ndarray:
        """GAN优化变异"""
        try:
            # 生成多个GAN样本
            gan_samples = self.creative_gan.generate_creative_features(num_samples=5)
            
            # 选择质量最高且有一定新颖性的样本
            best_sample = None
            best_score = -1
            
            for sample in gan_samples:
                quality = self.creative_gan.evaluate_quality(sample.reshape(1, -1))
                novelty = 1 - self._cosine_similarity(current_features, sample)
                
                # 综合评分：质量 + 新颖性
                composite_score = quality * 0.7 + novelty * 0.3
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_sample = sample
            
            if best_sample is not None:
                # 基于多巴胺水平决定融合程度
                fusion_ratio = min(dopamine_level / 2.0, 0.8)
                
                mutated_features = (
                    current_features * (1 - fusion_ratio) + 
                    best_sample * fusion_ratio
                )
                
                return mutated_features
            else:
                return current_features.copy()
                
        except Exception as e:
            print(f"GAN优化变异失败: {e}")
            return current_features.copy()
    
    def _multimodal_mutation(self, current_features: np.ndarray, dopamine_level: float) -> np.ndarray:
        """多模态变异"""
        try:
            # 模拟多模态输入
            modal_data = {
                'sensor': current_features,
                'text': self.creative_gan.generate_creative_features(1)[0][:self.modal_dims['text']],
                'image': self.creative_gan.generate_creative_features(1)[0][:self.modal_dims['image']]
            }
            
            # 使用多模态融合器进行变异
            fused_features_dict = self.multimodal_fusion.fuse_creative_concepts(
                modal_data, 
                creative_type='imagination'
            )
            
            # 提取融合后的传感器模态特征
            if 'sensor' in fused_features_dict:
                sensor_fused = fused_features_dict['sensor']
                
                # 基于多巴胺水平调整变异强度
                mutation_strength = min(dopamine_level / 2.0, 1.0)
                
                mutated_features = (
                    current_features * (1 - mutation_strength * 0.3) + 
                    sensor_fused * (mutation_strength * 0.7)
                )
                
                return mutated_features
            else:
                return current_features.copy()
                
        except Exception as e:
            print(f"多模态变异失败: {e}")
            return current_features.copy()
    
    def _basic_associative_mutation(self, current_features: np.ndarray, dopamine_level: float) -> np.ndarray:
        """基础联想变异（保留原有实现）"""
        if len(self.memories) < 2:
            return current_features.copy()
        
        # 筛选远距离记忆
        distant_memories = []
        for memory in self.memories:
            similarity = self._cosine_similarity(current_features, memory['features'])
            if similarity < self.mutation_threshold:
                distant_memories.append((memory, similarity))
        
        if not distant_memories:
            return current_features.copy()
        
        # 根据多巴胺水平决定变异强度
        mutation_intensity = min(dopamine_level / 2.0, 1.0)
        num_selections = min(int(1 + mutation_intensity * 2), len(distant_memories))
        selected_memories = random.sample(distant_memories, min(num_selections, len(distant_memories)))
        
        # 执行变异
        mutated_features = current_features.copy()
        
        for memory, similarity in selected_memories:
            fusion_weight = (1.0 - similarity) * mutation_intensity * 0.3
            memory_features = memory['features']
            mutated_features = mutated_features * (1 - fusion_weight) + memory_features * fusion_weight
            
            noise_scale = fusion_weight * 0.1
            noise = np.random.normal(0, noise_scale, len(mutated_features))
            mutated_features += noise
        
        mutated_features = np.clip(mutated_features, -1, 1)
        return mutated_features
    
    def train_creative_models(self, training_data: np.ndarray, epochs: int = 10) -> Dict[str, Any]:
        """
        训练创意生成模型
        
        Args:
            training_data: 训练数据
            epochs: 训练轮数
            
        Returns:
            训练结果摘要
        """
        print(f"🚀 开始训练创意生成模型...")
        print(f"   训练数据大小: {training_data.shape}")
        print(f"   训练轮数: {epochs}")
        
        training_results = {
            'diffusion_losses': [],
            'gan_losses': [],
            'quality_improvements': [],
            'training_time': []
        }
        
        # 转换数据格式
        if isinstance(training_data, np.ndarray):
            training_tensor = torch.tensor(training_data, dtype=torch.float32).to(self.device)
        else:
            training_tensor = training_data.to(self.device)
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # 训练扩散模型
            if len(training_tensor) > 0:
                # 随机采样批次
                batch_size = min(32, len(training_tensor))
                indices = torch.randperm(len(training_tensor))[:batch_size]
                batch_data = training_tensor[indices]
                
                diffusion_loss_dict = self.diffusion_model.train_step(batch_data)
                training_results['diffusion_losses'].append(diffusion_loss_dict['diffusion_loss'])
            
            # 训练GAN
            if len(training_tensor) > 0:
                gan_loss_dict = self.creative_gan.train_step(training_tensor[:batch_size])
                training_results['gan_losses'].append(gan_loss_dict)
            
            # 评估质量改进
            if len(training_tensor) > 0:
                sample_quality = []
                for i in range(min(10, len(training_tensor))):
                    quality = self.creative_gan.evaluate_quality(
                        training_tensor[i].cpu().numpy().reshape(1, -1)
                    )
                    sample_quality.append(quality)
                
                avg_quality = np.mean(sample_quality)
                training_results['quality_improvements'].append(avg_quality)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            training_results['training_time'].append(epoch_time)
            
            self.training_step += 1
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch+1}/{epochs} 完成")
                print(f"   扩散模型损失: {training_results['diffusion_losses'][-1]:.4f}")
                print(f"   GAN生成器损失: {training_results['gan_losses'][-1]['generator_loss']:.4f}")
                print(f"   GAN判别器损失: {training_results['gan_losses'][-1]['discriminator_loss']:.4f}")
                print(f"   平均质量分数: {training_results['quality_improvements'][-1]:.3f}")
        
        print("✅ 创意模型训练完成！")
        
        return {
            'total_epochs': epochs,
            'final_diffusion_loss': np.mean(training_results['diffusion_losses'][-5:]),
            'final_gan_generator_loss': np.mean([l['generator_loss'] for l in training_results['gan_losses'][-5:]]),
            'final_quality_score': np.mean(training_results['quality_improvements'][-5:]),
            'training_results': training_results
        }
    
    def create_multimodal_creative_concept(self, concept_description: str, 
                                         modal_inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        创建多模态创意概念
        
        Args:
            concept_description: 概念描述
            modal_inputs: 各模态输入数据
            
        Returns:
            多模态创意概念
        """
        print(f"🎨 创建多模态创意概念: {concept_description}")
        
        # 1. 多模态数据预处理
        processed_modal_data = {}
        for modal, data in modal_inputs.items():
            if modal in self.modal_dims:
                # 调整维度
                if len(data) != self.modal_dims[modal]:
                    if len(data) < self.modal_dims[modal]:
                        padded = np.pad(data, (0, self.modal_dims[modal] - len(data)))
                        processed_modal_data[modal] = padded
                    else:
                        processed_modal_data[modal] = data[:self.modal_dims[modal]]
                else:
                    processed_modal_data[modal] = data
        
        # 2. 多模态融合
        fused_concept = self.multimodal_fusion.fuse_creative_concepts(
            processed_modal_data, 
            creative_type='innovation'
        )
        
        # 3. 生成创意内容
        creative_samples = self.generate_creative_content_advanced(
            num_samples=3, 
            generation_method='hybrid',
            quality_threshold=0.6
        )
        
        # 4. 质量评估
        concept_quality_scores = []
        for modal, features in fused_concept.items():
            quality = self.creative_gan.evaluate_quality(features.reshape(1, -1))
            concept_quality_scores.append(quality)
        
        overall_quality = np.mean(concept_quality_scores)
        
        # 5. 创建完整概念
        creative_concept = {
            'description': concept_description,
            'modal_features': fused_concept,
            'generated_samples': creative_samples,
            'quality_score': overall_quality,
            'modal_contributions': {
                modal: self._compute_modal_contribution(features, concept_description)
                for modal, features in fused_concept.items()
            },
            'creation_timestamp': datetime.now(),
            'innovation_potential': self._assess_innovation_potential(fused_concept, concept_description)
        }
        
        # 更新统计
        self.innovation_stats['multimodal_fusions'] += 1
        
        print(f"✅ 多模态创意概念创建完成")
        print(f"   参与模态: {list(fused_concept.keys())}")
        print(f"   综合质量分数: {overall_quality:.3f}")
        print(f"   创新潜力: {creative_concept['innovation_potential']:.3f}")
        
        return creative_concept
    
    def _compute_modal_contribution(self, features: np.ndarray, concept_description: str) -> Dict[str, Any]:
        """计算模态贡献度"""
        # 特征统计
        feature_variance = np.var(features)
        feature_entropy = self._compute_feature_entropy(features)
        feature_magnitude = np.linalg.norm(features)
        
        # 模态特异性分析
        uniqueness_score = min(feature_variance * 2, 1.0)
        diversity_score = min(feature_entropy / 10.0, 1.0)  # 假设最大熵为10
        
        # 综合贡献度
        contribution_score = (uniqueness_score * 0.4 + diversity_score * 0.6) * \
                           (1 + feature_magnitude / 10.0)  # 幅度因子
        
        return {
            'uniqueness': uniqueness_score,
            'diversity': diversity_score,
            'magnitude': feature_magnitude,
            'contribution_score': min(contribution_score, 1.0),
            'feature_stats': {
                'variance': feature_variance,
                'entropy': feature_entropy,
                'mean': np.mean(features),
                'std': np.std(features)
            }
        }
    
    def _compute_feature_entropy(self, features: np.ndarray) -> float:
        """计算特征熵"""
        # 简化的熵计算
        normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
        
        # 分桶计算
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(normalized_features, bins=bins)
        
        # 计算熵
        hist = hist + 1e-8  # 避免log(0)
        probabilities = hist / np.sum(hist)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def _assess_innovation_potential(self, fused_features: Dict[str, np.ndarray], 
                                   concept_description: str) -> float:
        """评估创新潜力"""
        # 基于特征多样性和新颖性的创新潜力评估
        feature_vectors = list(fused_features.values())
        
        if len(feature_vectors) < 2:
            return 0.7  # 单模态默认潜力
        
        # 计算特征间距离
        distances = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                distance = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
                distances.append(distance)
        
        # 创新潜力 = 平均距离 + 特征熵
        avg_distance = np.mean(distances) if distances else 0.5
        
        # 组合多样性
        entropy_scores = []
        for features in feature_vectors:
            entropy = self._compute_feature_entropy(features)
            entropy_scores.append(entropy)
        
        avg_entropy = np.mean(entropy_scores)
        
        # 综合创新潜力
        innovation_potential = (avg_distance * 0.6 + (avg_entropy / 10.0) * 0.4)
        
        return min(innovation_potential, 1.0)
    
    def optimize_creative_quality(self, target_features: np.ndarray, 
                                optimization_steps: int = 50) -> Dict[str, Any]:
        """
        基于GAN的创意质量优化
        
        Args:
            target_features: 目标特征
            optimization_steps: 优化步数
            
        Returns:
            优化结果
        """
        print(f"🔧 开始创意质量优化...")
        print(f"   目标特征维度: {len(target_features)}")
        print(f"   优化步数: {optimization_steps}")
        
        # 初始化优化变量
        current_features = target_features.copy()
        initial_quality = self.creative_gan.evaluate_quality(current_features.reshape(1, -1))
        
        quality_history = [initial_quality]
        feature_history = [current_features.copy()]
        
        for step in range(optimization_steps):
            # 质量梯度估计
            quality_score = self.creative_gan.evaluate_quality(current_features.reshape(1, -1))
            
            # 生成多个候选样本
            candidates = self.creative_gan.generate_creative_features(5)
            candidate_qualities = [self.creative_gan.evaluate_quality(c.reshape(1, -1)) for c in candidates]
            
            # 选择质量最高的候选样本
            best_candidate_idx = np.argmax(candidate_qualities)
            best_candidate = candidates[best_candidate_idx]
            best_quality = candidate_qualities[best_candidate_idx]
            
            # 如果候选质量更好，则更新
            if best_quality > quality_score:
                # 插值更新
                alpha = 0.3  # 更新强度
                current_features = (1 - alpha) * current_features + alpha * best_candidate
                quality_history.append(best_quality)
                feature_history.append(current_features.copy())
                
                if step % 10 == 0:
                    print(f"   优化步数 {step}: 质量 {best_quality:.3f} (提升 {best_quality - quality_score:.3f})")
            else:
                quality_history.append(quality_score)
                feature_history.append(current_features.copy())
            
            # 添加轻微噪声探索
            noise_scale = 0.01 * (1 - step / optimization_steps)  # 递减噪声
            noise = np.random.normal(0, noise_scale, len(current_features))
            current_features += noise
            current_features = np.clip(current_features, -1, 1)
        
        final_quality = self.creative_gan.evaluate_quality(current_features.reshape(1, -1))
        
        optimization_result = {
            'optimized_features': current_features,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'quality_improvement': final_quality - initial_quality,
            'optimization_steps': optimization_steps,
            'quality_history': quality_history,
            'convergence_info': {
                'max_quality_reached': max(quality_history),
                'improvement_rate': (final_quality - initial_quality) / optimization_steps,
                'final_improvement': final_quality - initial_quality
            }
        }
        
        # 更新统计
        self.innovation_stats['quality_optimizations'] += 1
        self.creative_quality_scores.append({
            'timestamp': datetime.now(),
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'improvement': final_quality - initial_quality
        })
        
        print(f"✅ 质量优化完成")
        print(f"   初始质量: {initial_quality:.3f}")
        print(f"   最终质量: {final_quality:.3f}")
        print(f"   质量提升: {final_quality - initial_quality:.3f}")
        print(f"   优化成功率: {(len([q for q in quality_history[1:] if q > quality_history[0]]) / (optimization_steps)):.1%}")
        
        return optimization_result
    
    # 保留原有的其他方法...
    
    def combine_innovations(self, primary_action: str, secondary_action: str) -> Dict[str, Any]:
        """
        组合创新算法 - 远距离记忆动作组合 (保留原有实现，增强版)
        
        将远距离记忆的动作部分进行创新性组合，生成新的复合行为。
        """
        # 获取动作库中的动作信息
        primary_info = self.action_library.get(primary_action, {})
        secondary_info = self.action_library.get(secondary_action, {})
        
        if not primary_info or not secondary_info:
            return {
                'combined_action': f"{primary_action}_{secondary_action}",
                'innovation_type': 'simple_combination',
                'feasibility_score': 0.5,
                'description': f'简单组合 {primary_action} 和 {secondary_action}',
                'steps': [primary_action, secondary_action]
            }
        
        # 分析动作特征
        primary_features = primary_info.get('features', np.zeros(self.feature_dim))
        secondary_features = secondary_info.get('features', np.zeros(self.feature_dim))
        
        # 使用GAN评估组合质量
        combined_features = (primary_features + secondary_features) / 2
        combined_quality = self.creative_gan.evaluate_quality(combined_features.reshape(1, -1))
        
        # 计算动作间的协同性
        synergy_score = self._calculate_action_synergy(primary_features, secondary_features)
        
        # 生成创新组合
        innovation_type = self._determine_innovation_type(primary_info, secondary_info, synergy_score)
        
        combined_action = f"{primary_action}_innovated_{secondary_action}"
        description = self._generate_combination_description(primary_action, secondary_action, innovation_type)
        
        # 生成执行步骤
        steps = self._create_combination_steps(primary_action, secondary_action, innovation_type)
        
        return {
            'combined_action': combined_action,
            'innovation_type': innovation_type,
            'synergy_score': synergy_score,
            'description': description,
            'steps': steps,
            'primary_action': primary_action,
            'secondary_action': secondary_action,
            'gan_quality_score': combined_quality,
            'enhanced_innovation': True
        }
    
    def get_enhanced_innovation_metrics(self) -> Dict[str, Any]:
        """
        获取增强版创新系统量化指标
        
        Returns:
            包含所有创新指标的字典
        """
        current_time = datetime.now()
        elapsed_hours = (current_time - self.innovation_stats['start_time']).total_seconds() / 3600
        
        # 基础指标
        basic_metrics = self.get_innovation_metrics()
        
        # 扩散模型指标
        diffusion_stats = {
            'total_diffusion_generations': self.innovation_stats['diffusion_generations'],
            'diffusion_efficiency': self.innovation_stats['diffusion_generations'] / max(elapsed_hours, 1)
        }
        
        # GAN指标
        gan_metrics = self.creative_gan.get_training_metrics()
        gan_stats = {
            'total_gan_generations': self.innovation_stats['gan_generations'],
            'gan_training_steps': gan_metrics.get('training_steps', 0),
            'latest_gan_quality': gan_metrics.get('latest_quality_score', 0.0)
        }
        
        # 质量优化指标
        quality_optimization_stats = {
            'total_optimizations': self.innovation_stats['quality_optimizations'],
            'optimization_rate': self.innovation_stats['quality_optimizations'] / max(elapsed_hours, 1),
            'avg_quality_improvement': np.mean([q['improvement'] for q in self.creative_quality_scores[-10:]]) if self.creative_quality_scores else 0.0
        }
        
        # 多模态融合指标
        multimodal_stats = {
            'total_multimodal_fusions': self.innovation_stats['multimodal_fusions'],
            'multimodal_fusion_rate': self.innovation_stats['multimodal_fusions'] / max(elapsed_hours, 1),
            'active_modalities': len(self.modal_dims)
        }
        
        # 整体质量评估
        recent_quality_scores = [q['final_quality'] for q in self.creative_quality_scores[-20:]] if self.creative_quality_scores else [0.5]
        avg_quality_score = np.mean(recent_quality_scores)
        
        # 升级版目标评估
        enhanced_targets = {
            'innovation_ratio_target': basic_metrics['innovative_action_ratio'] > 0.30,
            'frequency_target': basic_metrics['novel_behavior_frequency_per_hour'] > 10,
            'quality_target': avg_quality_score > 0.7,
            'multimodal_target': self.innovation_stats['multimodal_fusions'] > 5,
            'diffusion_target': self.innovation_stats['diffusion_generations'] > 20
        }
        
        # 创新活跃度评分
        innovation_activity_score = (
            basic_metrics['innovative_action_ratio'] * 0.3 +
            min(avg_quality_score, 1.0) * 0.3 +
            min(self.innovation_stats['diffusion_generations'] / 50.0, 1.0) * 0.2 +
            min(self.innovation_stats['multimodal_fusions'] / 10.0, 1.0) * 0.2
        )
        
        return {
            **basic_metrics,
            'enhanced_metrics': {
                'diffusion_stats': diffusion_stats,
                'gan_stats': gan_stats,
                'quality_optimization_stats': quality_optimization_stats,
                'multimodal_stats': multimodal_stats,
                'avg_quality_score': avg_quality_score,
                'innovation_activity_score': innovation_activity_score,
                'enhanced_targets_met': enhanced_targets
            },
            'training_progress': {
                'training_step': self.training_step,
                'diffusion_trained': self.training_step > 0,
                'gan_trained': self.creative_gan.get_training_metrics().get('training_steps', 0) > 0
            },
            'system_capabilities': {
                'diffusion_generation': True,
                'gan_quality_assessment': True,
                'multimodal_fusion': True,
                'quality_optimization': True,
                'advanced_novelty_detection': True
            }
        }
    
    def export_enhanced_innovation_report(self) -> str:
        """
        导出增强版创新系统详细报告
        
        Returns:
            格式化的创新报告文本
        """
        metrics = self.get_enhanced_innovation_metrics()
        summary = self.get_memory_summary()
        
        # 获取各种统计数据
        recent_quality_items = list(self.quality_history)[-10:] if self.quality_history else []
        quality_scores = [item['quality_score'] for item in recent_quality_items]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        enhanced_targets = metrics['enhanced_metrics']['enhanced_targets_met']
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                创造力记忆系统升级版创新报告                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ 📊 系统概览
║   ├─ 总记忆数: {summary['total_memories']:,}
║   ├─ 动作库规模: {summary['action_library_size']:,}
║   ├─ 记忆利用率: {summary['innovation_metrics']['memory_utilization']:.1%}
║   ├─ 系统运行时间: {summary['innovation_metrics']['system_uptime_hours']:.1f}小时
║   ├─ 训练步数: {metrics['training_progress']['training_step']:,}
║   └─ 活跃模态数: {metrics['enhanced_metrics']['multimodal_stats']['active_modalities']}

║ 🎨 基础创新性能
║   ├─ 总动作数: {metrics['total_actions']:,}
║   ├─ 创新动作数: {metrics['innovative_actions']:,}
║   ├─ 创新性动作占比: {metrics['innovative_action_ratio']:.1%}
║   ├─ 新颖行为总数: {metrics['total_novel_behaviors']:,}
║   ├─ 新颖行为频率(每小时): {metrics['novel_behavior_frequency_per_hour']:.1f}
║   └─ 最近一小时新颖行为: {metrics['recent_novel_behaviors']}

║ 🤖 扩散模型性能
║   ├─ 扩散生成总数: {metrics['enhanced_metrics']['diffusion_stats']['total_diffusion_generations']:,}
║   ├─ 扩散效率(个/小时): {metrics['enhanced_metrics']['diffusion_stats']['diffusion_efficiency']:.1f}
║   └─ 训练状态: {'✅ 已训练' if metrics['training_progress']['diffusion_trained'] else '❌ 未训练'}

║ 🎭 GAN网络性能  
║   ├─ GAN生成总数: {metrics['enhanced_metrics']['gan_stats']['total_gan_generations']:,}
║   ├─ GAN训练步数: {metrics['enhanced_metrics']['gan_stats']['gan_training_steps']:,}
║   ├─ 质量评估能力: {'✅ 已启用' if metrics['enhanced_metrics']['gan_stats']['latest_gan_quality'] > 0 else '❌ 未就绪'}
║   └─ 最新质量分数: {metrics['enhanced_metrics']['gan_stats']['latest_gan_quality']:.3f}

║ 🔄 质量优化性能
║   ├─ 总优化次数: {metrics['enhanced_metrics']['quality_optimization_stats']['total_optimizations']:,}
║   ├─ 优化效率(次/小时): {metrics['enhanced_metrics']['quality_optimization_stats']['optimization_rate']:.1f}
║   ├─ 平均质量提升: {metrics['enhanced_metrics']['quality_optimization_stats']['avg_quality_improvement']:.3f}
║   └─ 当前平均质量: {metrics['enhanced_metrics']['avg_quality_score']:.3f}

║ 🌈 多模态融合性能
║   ├─ 融合操作总数: {metrics['enhanced_metrics']['multimodal_stats']['total_multimodal_fusions']:,}
║   ├─ 融合效率(次/小时): {metrics['enhanced_metrics']['multimodal_stats']['multimodal_fusion_rate']:.1f}
║   └─ 支持模态: {', '.join(self.modal_dims.keys())}

║ 🎯 目标达成情况 (升级版)
║   ├─ 创新性动作占比目标(>30%): {'✅ 已达成' if enhanced_targets['innovation_ratio_target'] else '❌ 未达成'}
║   ├─ 新颖行为频率目标(>10次/小时): {'✅ 已达成' if enhanced_targets['frequency_target'] else '❌ 未达成'}
║   ├─ 创意质量目标(>0.7): {'✅ 已达成' if enhanced_targets['quality_target'] else '❌ 未达成'}
║   ├─ 多模态融合目标(>5次): {'✅ 已达成' if enhanced_targets['multimodal_target'] else '❌ 未达成'}
║   └─ 扩散生成目标(>20次): {'✅ 已达成' if enhanced_targets['diffusion_target'] else '❌ 未达成'}

║ 📈 创新活跃度评估
║   └─ 综合活跃度分数: {metrics['enhanced_metrics']['innovation_activity_score']:.3f} 
║     (创新占比30% + 质量30% + 扩散生成20% + 多模态融合20%)

║ 🛠️ 系统能力状态
║   ├─ 扩散模型生成: {'✅' if metrics['system_capabilities']['diffusion_generation'] else '❌'}
║   ├─ GAN质量评估: {'✅' if metrics['system_capabilities']['gan_quality_assessment'] else '❌'}
║   ├─ 多模态融合: {'✅' if metrics['system_capabilities']['multimodal_fusion'] else '❌'}
║   ├─ 质量自动优化: {'✅' if metrics['system_capabilities']['quality_optimization'] else '❌'}
║   └─ 高级新颖性检测: {'✅' if metrics['system_capabilities']['advanced_novelty_detection'] else '❌'}
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        return report.strip()
    
    # 保留原有的辅助方法
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_action_synergy(self, features1, features2) -> float:
        """计算动作间的协同性"""
        if features1 is None or features2 is None:
            return 0.5
        
        if hasattr(features1, '__len__') and hasattr(features2, '__len__'):
            if len(features1) == 0 or len(features2) == 0:
                return 0.5
            return self._cosine_similarity(features1, features2)
        
        try:
            overlap = len(set(features1) & set(features2))
            total_unique = len(set(features1) | set(features2))
            
            if total_unique == 0:
                return 0.0
            
            return overlap / total_unique
        except:
            return 0.5
    
    def _determine_innovation_type(self, action1: Dict, action2: Dict, synergy: float) -> str:
        """确定创新类型"""
        if synergy > 0.7:
            return 'high_synergy_combination'
        elif synergy > 0.4:
            return 'moderate_combination'
        else:
            return 'novel_combination'
    
    def _generate_combination_description(self, action1: str, action2: str, innovation_type: str) -> str:
        """生成组合描述"""
        descriptions = {
            'high_synergy_combination': f'高度协同的{action1}和{action2}组合',
            'moderate_combination': f'适度融合的{action1}和{action2}组合',
            'novel_combination': f'创新性的{action1}和{action2}组合'
        }
        return descriptions.get(innovation_type, f'{action1}与{action2}的组合')
    
    def _create_combination_steps(self, action1: str, action2: str, innovation_type: str) -> List[str]:
        """创建组合步骤"""
        if innovation_type == 'high_synergy_combination':
            return [f'同时执行{action1}', f'无缝过渡到{action2}', f'完成复合动作']
        else:
            return [f'执行{action1}', f'基于结果调整', f'执行{action2}', f'评估最终效果']
    
    def get_innovation_metrics(self) -> Dict[str, Any]:
        """获取基础创新系统量化指标 (保留原方法)"""
        current_time = datetime.now()
        elapsed_hours = (current_time - self.innovation_stats['start_time']).total_seconds() / 3600
        
        innovative_ratio = (self.innovation_stats['innovative_actions'] / 
                          max(self.innovation_stats['total_actions'], 1))
        
        novel_behavior_frequency = (self.innovation_stats['novel_behaviors'] / 
                                  max(elapsed_hours, 1))
        
        recent_novel_behaviors = sum(1 for item in self.novelty_memory 
                                   if (current_time - item['timestamp']).total_seconds() < 3600)
        
        return {
            'innovative_action_ratio': innovative_ratio,
            'novel_behavior_frequency_per_hour': novel_behavior_frequency,
            'recent_novel_behaviors': recent_novel_behaviors,
            'total_actions': self.innovation_stats['total_actions'],
            'innovative_actions': self.innovation_stats['innovative_actions'],
            'total_novel_behaviors': self.innovation_stats['novel_behaviors'],
            'memory_utilization': len(self.memories) / self.memory_capacity,
            'action_library_size': len(self.action_library),
            'system_uptime_hours': elapsed_hours,
            'targets_met': {
                'innovation_ratio_target': innovative_ratio > 0.30,
                'frequency_target': novel_behavior_frequency > 10
            }
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        return {
            'total_memories': len(self.memories),
            'action_library_size': len(self.action_library),
            'innovation_metrics': self.get_innovation_metrics(),
            'recent_novelty': list(self.novelty_memory)[-5:] if self.novelty_memory else []
        }
    
    def store_memory(self, features: np.ndarray, metadata: Dict[str, Any]):
        """存储新记忆"""
        memory = {
            'features': features,
            'timestamp': datetime.now(),
            'metadata': metadata,
            'modal_type': metadata.get('modal_type', 'sensor')
        }
        
        self.memories.append(memory)
        
        # 记忆容量管理
        if len(self.memories) > self.memory_capacity:
            self.memories.pop(0)


# ==================== 使用示例和演示 ====================

def demonstrate_enhanced_creative_system():
    """演示增强版创意记忆系统的完整功能"""
    
    print("=" * 80)
    print("🎨 创造力记忆系统升级版完整演示")
    print("=" * 80)
    
    # 1. 初始化增强版系统
    print("\n🚀 初始化增强版创造力记忆系统...")
    
    # 配置多模态维度
    modal_dims = {
        'text': 256,
        'image': 512, 
        'audio': 128,
        'sensor': 128
    }
    
    creative_system = CreativeMemory(
        memory_capacity=5000, 
        novelty_threshold=0.4,
        device='cpu',
        modal_dims=modal_dims
    )
    
    # 2. 生成训练数据并训练模型
    print("\n📚 生成训练数据并训练创意模型...")
    training_data = np.random.randn(100, 128)  # 100个128维特征样本
    training_result = creative_system.train_creative_models(training_data, epochs=10)
    
    print(f"   训练完成：扩散损失 {training_result['final_diffusion_loss']:.4f}")
    print(f"   GAN质量：{training_result['final_quality_score']:.3f}")
    
    # 3. 演示增强新颖性检测
    print("\n🔍 增强版新颖性检测演示...")
    
    sample_perception = np.random.randn(128)
    enhanced_novelty = creative_system.enhanced_novelty_detection(
        sample_perception, 
        modal_type='sensor'
    )
    
    print(f"   新颖性分数: {enhanced_novelty['novelty_score']:.3f}")
    print(f"   多巴胺水平: {enhanced_novelty['dopamine_level']:.3f}")
    print(f"   质量分数: {enhanced_novelty['quality_score']:.3f}")
    print(f"   高度新颖: {enhanced_novelty['is_highly_novel']}")
    
    # 4. 演示高级创意生成
    print("\n🎭 高级创意内容生成演示...")
    
    # 扩散模型生成
    diffusion_samples = creative_system.generate_creative_content_advanced(
        num_samples=3,
        generation_method='diffusion'
    )
    print(f"   扩散模型生成: {len(diffusion_samples)} 个样本")
    
    # GAN生成
    gan_samples = creative_system.generate_creative_content_advanced(
        num_samples=3,
        generation_method='gan'
    )
    print(f"   GAN生成: {len(gan_samples)} 个样本")
    
    # 混合生成
    hybrid_samples = creative_system.generate_creative_content_advanced(
        num_samples=3,
        generation_method='hybrid'
    )
    print(f"   混合生成: {len(hybrid_samples)} 个样本")
    
    # 5. 演示高级联想变异
    print("\n🧬 高级联想变异演示...")
    
    mutation_types = ['diffusion_enhanced', 'gan_optimized', 'multimodal']
    for mut_type in mutation_types:
        mutated = creative_system.advanced_associative_mutation(
            sample_perception, 
            enhanced_novelty['dopamine_level'],
            mutation_type=mut_type
        )
        similarity = creative_system._cosine_similarity(sample_perception, mutated)
        print(f"   {mut_type}: 相似度 {similarity:.3f}")
    
    # 6. 演示多模态创意概念创建
    print("\n🌈 多模态创意概念创建演示...")
    
    # 模拟多模态输入
    modal_inputs = {
        'text': np.random.randn(modal_dims['text']),
        'image': np.random.randn(modal_dims['image']),
        'audio': np.random.randn(modal_dims['audio']),
        'sensor': sample_perception
    }
    
    creative_concept = creative_system.create_multimodal_creative_concept(
        concept_description="智能环境感知与响应系统",
        modal_inputs=modal_inputs
    )
    
    print(f"   创意概念: {creative_concept['description']}")
    print(f"   综合质量: {creative_concept['quality_score']:.3f}")
    print(f"   创新潜力: {creative_concept['innovation_potential']:.3f}")
    print(f"   参与模态: {list(creative_concept['modal_features'].keys())}")
    
    # 7. 演示质量优化
    print("\n🔧 创意质量优化演示...")
    
    optimization_result = creative_system.optimize_creative_quality(
        target_features=sample_perception,
        optimization_steps=20
    )
    
    print(f"   初始质量: {optimization_result['initial_quality']:.3f}")
    print(f"   最终质量: {optimization_result['final_quality']:.3f}")
    print(f"   质量提升: {optimization_result['quality_improvement']:.3f}")
    
    # 8. 更新动作库
    print("\n📚 更新动作库...")
    
    basic_actions = [
        ('智能建造', {'features': np.random.randn(128), 'success_rate': 0.8}),
        ('自适应学习', {'features': np.random.randn(128), 'success_rate': 0.7}),
        ('多模态感知', {'features': np.random.randn(128), 'success_rate': 0.9}),
        ('创意生成', {'features': np.random.randn(128), 'success_rate': 0.6}),
        ('质量优化', {'features': np.random.randn(128), 'success_rate': 0.5})
    ]
    
    for action_name, action_data in basic_actions:
        creative_system.action_library[action_name] = {
            'created_time': datetime.now(),
            'features': action_data['features'],
            'success_rate': action_data['success_rate'],
            'usage_count': 0,
            'innovation_score': random.uniform(0.3, 0.7),
            'feasibility_score': random.uniform(0.5, 0.9)
        }
    
    # 9. 演示组合创新（升级版）
    print("\n🔗 升级版组合创新演示...")
    
    combination = creative_system.combine_innovations('智能建造', '创意生成')
    print(f"   组合动作: {combination['combined_action']}")
    print(f"   创新类型: {combination['innovation_type']}")
    print(f"   GAN质量分数: {combination['gan_quality_score']:.3f}")
    print(f"   描述: {combination['description']}")
    
    # 10. 获取增强版创新指标
    print("\n📊 增强版创新系统指标...")
    
    enhanced_metrics = creative_system.get_enhanced_innovation_metrics()
    
    print(f"   基础创新:")
    print(f"     创新性动作占比: {enhanced_metrics['innovative_action_ratio']:.1%}")
    print(f"     新颖行为频率: {enhanced_metrics['novel_behavior_frequency_per_hour']:.1f}/小时")
    
    print(f"   扩散模型:")
    print(f"     生成总数: {enhanced_metrics['enhanced_metrics']['diffusion_stats']['total_diffusion_generations']}")
    print(f"     效率: {enhanced_metrics['enhanced_metrics']['diffusion_stats']['diffusion_efficiency']:.1f}/小时")
    
    print(f"   GAN网络:")
    print(f"     生成总数: {enhanced_metrics['enhanced_metrics']['gan_stats']['total_gan_generations']}")
    print(f"     训练步数: {enhanced_metrics['enhanced_metrics']['gan_stats']['gan_training_steps']}")
    
    print(f"   质量优化:")
    print(f"     优化次数: {enhanced_metrics['enhanced_metrics']['quality_optimization_stats']['total_optimizations']}")
    print(f"     平均提升: {enhanced_metrics['enhanced_metrics']['quality_optimization_stats']['avg_quality_improvement']:.3f}")
    
    print(f"   多模态融合:")
    print(f"     融合次数: {enhanced_metrics['enhanced_metrics']['multimodal_stats']['total_multimodal_fusions']}")
    print(f"     支持模态: {', '.join(enhanced_metrics['enhanced_metrics']['multimodal_stats']['active_modalities'])}")
    
    # 11. 目标达成情况
    print("\n🎯 升级版目标达成情况...")
    
    targets = enhanced_metrics['enhanced_metrics']['enhanced_targets_met']
    print(f"   创新性动作占比目标(>30%): {'✅' if targets['innovation_ratio_target'] else '❌'}")
    print(f"   新颖行为频率目标(>10次/小时): {'✅' if targets['frequency_target'] else '❌'}")
    print(f"   创意质量目标(>0.7): {'✅' if targets['quality_target'] else '❌'}")
    print(f"   多模态融合目标(>5次): {'✅' if targets['multimodal_target'] else '❌'}")
    print(f"   扩散生成目标(>20次): {'✅' if targets['diffusion_target'] else '❌'}")
    
    # 12. 生成详细报告
    print("\n📋 生成详细创新报告...")
    detailed_report = creative_system.export_enhanced_innovation_report()
    print(detailed_report)
    
    print("\n🎨 创造力记忆系统升级版演示完成！")
    print("=" * 80)
    
    return creative_system


if __name__ == "__main__":
    # 运行完整演示
    creative_system = demonstrate_enhanced_creative_system()