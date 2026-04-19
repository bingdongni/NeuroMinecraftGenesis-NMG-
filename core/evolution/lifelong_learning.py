"""
终身学习和自我进化能力实现系统
Lifelong Learning and Self-Evolution System

实现完整的终身学习框架，包含：
1. 持续学习机制和灾难性遗忘避免
2. 元学习和快速适应能力
3. 自我修改和架构优化
4. 知识蒸馏和经验传承
5. 跨任务泛化和迁移学习

Author: AI Assistant
Date: 2025-11-13
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import pickle
import logging
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import os
from datetime import datetime
import math
import random
from abc import ABC, abstractmethod


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """学习状态数据类"""
    task_id: str
    epoch: int
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    loss_history: List[float]
    accuracy_history: List[float]
    knowledge_replay_buffer: Dict[str, Any]
    meta_parameters: Dict[str, Any]
    timestamp: datetime
    task_performance: float
    knowledge_preservation_rate: float


@dataclass
class KnowledgeNode:
    """知识节点表示"""
    concept: str
    importance: float
    connections: List[str]
    strength: float
    last_accessed: datetime
    applications: List[str]
    confidence_score: float


class MemoryBuffer:
    """持续学习记忆缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.importance_weights = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def add_sample(self, sample: Dict[str, Any], importance: float = 1.0):
        """添加样本到缓冲区"""
        self.buffer.append(sample)
        self.importance_weights.append(importance)
        self.timestamps.append(datetime.now())
    
    def sample_replay(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """重要性加权的随机重采样"""
        if len(self.buffer) == 0:
            return []
        
        weights = np.array(self.importance_weights)
        weights = weights / (np.sum(weights) + 1e-8)
        
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)), 
            replace=False,
            p=weights
        )
        
        return [self.buffer[i] for i in indices]
    
    def update_importance(self, sample_idx: int, new_importance: float):
        """更新样本重要性"""
        if 0 <= sample_idx < len(self.importance_weights):
            self.importance_weights[sample_idx] = new_importance


class KnowledgeDistillation:
    """知识蒸馏组件"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_models = {}  # 存储教师模型
        self.distilled_knowledge = {}
        
    def register_teacher(self, task_id: str, model: nn.Module):
        """注册教师模型"""
        self.teacher_models[task_id] = copy.deepcopy(model)
        logger.info(f"注册教师模型: {task_id}")
    
    def distill_knowledge(self, student_model: nn.Module, task_data: Any):
        """从多个教师模型蒸馏知识"""
        distilled_outputs = []
        
        for task_id, teacher_model in self.teacher_models.items():
            teacher_model.eval()
            student_model.eval()
            
            # 计算教师模型软标签
            with torch.no_grad():
                teacher_logits = teacher_model(task_data)
                soft_labels = F.softmax(teacher_logits / self.temperature, dim=1)
                
            # 计算学生模型输出
            student_logits = student_model(task_data)
            student_softmax = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # 知识蒸馏损失
            kd_loss = F.kl_div(
                student_softmax, 
                soft_labels, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            distilled_outputs.append({
                'task_id': task_id,
                'kd_loss': kd_loss.item(),
                'soft_labels': soft_labels.detach()
            })
            
        return distilled_outputs
    
    def save_distilled_knowledge(self, filepath: str):
        """保存蒸馏的知识"""
        knowledge_data = {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'distilled_knowledge': self.distilled_knowledge
        }
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge_data, f)
    
    def load_distilled_knowledge(self, filepath: str):
        """加载蒸馏的知识"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                knowledge_data = pickle.load(f)
            self.temperature = knowledge_data['temperature']
            self.alpha = knowledge_data['alpha']
            self.distilled_knowledge = knowledge_data['distilled_knowledge']


class MetaLearningModule:
    """元学习模块 - MAML实现"""
    
    def __init__(self, base_model: nn.Module, lr_inner: float = 0.01, num_inner_steps: int = 5):
        self.base_model = base_model
        self.lr_inner = lr_inner
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = optim.Adam(base_model.parameters(), lr=1e-3)
        
    def meta_update(self, support_tasks: List[Tuple], query_tasks: List[Tuple]) -> float:
        """元学习更新"""
        meta_loss = 0.0
        
        for (support_x, support_y), (query_x, query_y) in zip(support_tasks, query_tasks):
            # 快速适应（内循环）
            adapted_params = self.fast_adapt(support_x, support_y)
            
            # 在查询集上评估
            query_loss = self.evaluate_on_query(query_x, query_y, adapted_params)
            meta_loss += query_loss
            
        meta_loss /= len(support_tasks)
        
        # 元学习更新（外循环）
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def fast_adapt(self, x_support: torch.Tensor, y_support: torch.Tensor) -> Dict[str, torch.Tensor]:
        """快速适应算法"""
        # 创建模型副本
        adapted_model = copy.deepcopy(self.base_model)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.lr_inner)
        
        # 内循环更新
        for _ in range(self.num_inner_steps):
            adapted_optimizer.zero_grad()
            logits = adapted_model(x_support)
            loss = F.cross_entropy(logits, y_support)
            loss.backward()
            adapted_optimizer.step()
        
        return dict(adapted_model.named_parameters())
    
    def evaluate_on_query(self, x_query: torch.Tensor, y_query: torch.Tensor, 
                         adapted_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """在查询集上评估适应后的模型"""
        with torch.no_grad():
            logits = self.forward_with_params(x_query, adapted_params)
            query_loss = F.cross_entropy(logits, y_query)
        return query_loss
    
    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]):
        """使用指定参数进行前向传播"""
        # 简化的参数替换逻辑
        original_params = {}
        for name, param in self.base_model.named_parameters():
            if name in params:
                original_params[name] = param.data.clone()
                param.data = params[name].data.clone()
        
        logits = self.base_model(x)
        
        # 恢复原始参数
        for name, param in self.base_model.named_parameters():
            if name in original_params:
                param.data = original_params[name]
                
        return logits


class ArchitectureOptimizer:
    """架构优化模块"""
    
    def __init__(self, model: nn.Module, mutation_rate: float = 0.1):
        self.model = model
        self.mutation_rate = mutation_rate
        self.architecture_history = []
        self.performance_history = []
        
    def evaluate_architecture(self, eval_func: Callable[[nn.Module], float]) -> float:
        """评估当前架构性能"""
        performance = eval_func(self.model)
        self.performance_history.append(performance)
        
        # 保存架构信息
        arch_info = self.get_architecture_info()
        self.architecture_history.append(arch_info)
        
        logger.info(f"当前架构性能: {performance:.4f}")
        return performance
    
    def mutate_architecture(self) -> nn.Module:
        """架构变异"""
        new_model = copy.deepcopy(self.model)
        
        # 获取所有模块
        modules = []
        for name, module in new_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.Dropout)):
                modules.append((name, module))
        
        # 随机变异
        for name, module in modules:
            if random.random() < self.mutation_rate:
                if isinstance(module, nn.Linear):
                    self._mutate_linear(new_model, name, module)
                elif isinstance(module, nn.Conv2d):
                    self._mutate_conv2d(new_model, name, module)
                elif isinstance(module, nn.BatchNorm2d):
                    self._mutate_batchnorm(new_model, name, module)
                elif isinstance(module, nn.Dropout):
                    self._mutate_dropout(new_model, name, module)
        
        logger.info("架构变异完成")
        return new_model
    
    def _mutate_linear(self, model: nn.Module, name: str, module: nn.Linear):
        """线性层变异"""
        # 改变隐藏层大小
        new_features = max(16, module.out_features + random.choice([-64, -32, 32, 64]))
        
        # 替换模块
        setattr(
            model, 
            name.split('.')[-1], 
            nn.Linear(module.in_features, new_features, module.bias is not None)
        )
    
    def _mutate_conv2d(self, model: nn.Module, name: str, module: nn.Conv2d):
        """卷积层变异"""
        # 改变通道数或卷积核大小
        new_out_channels = max(module.out_channels + random.choice([-16, -8, 8, 16]), 1)
        
        # 偶尔改变卷积核大小
        kernel_size = module.kernel_size
        if random.random() < 0.3:
            kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
        
        setattr(
            model,
            name.split('.')[-1],
            nn.Conv2d(
                module.in_channels,
                new_out_channels,
                kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None
            )
        )
    
    def _mutate_batchnorm(self, model: nn.Module, name: str, module: nn.BatchNorm2d):
        """批归一化变异"""
        # 微调参数
        module.weight.data *= (1 + random.uniform(-0.1, 0.1))
        module.bias.data += random.uniform(-0.1, 0.1)
    
    def _mutate_dropout(self, model: nn.Module, name: str, module: nn.Dropout):
        """Dropout变异"""
        new_p = min(max(module.p + random.uniform(-0.1, 0.1), 0.0), 0.8)
        setattr(model, name.split('.')[-1], nn.Dropout(new_p))
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """获取架构信息"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        layer_count = defaultdict(int)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                layer_count['linear'] += 1
            elif isinstance(module, nn.Conv2d):
                layer_count['conv2d'] += 1
            elif isinstance(module, nn.BatchNorm2d):
                layer_count['batchnorm'] += 1
            elif isinstance(module, nn.Dropout):
                layer_count['dropout'] += 1
        
        info['layer_counts'] = dict(layer_count)
        return info


class TaskAdapter:
    """任务适配器 - 处理跨任务泛化"""
    
    def __init__(self, feature_extractor: nn.Module):
        self.feature_extractor = feature_extractor
        self.task_adapters = {}
        self.shared_knowledge = {}
        self.task_similarity_matrix = None
        
    def register_task(self, task_id: str, adapter_layers: nn.Module):
        """注册新任务适配器"""
        self.task_adapters[task_id] = adapter_layers
        logger.info(f"注册任务适配器: {task_id}")
    
    def compute_task_similarity(self, task_features: Dict[str, torch.Tensor]):
        """计算任务间相似性"""
        similarities = {}
        
        for task1, features1 in task_features.items():
            similarities[task1] = {}
            for task2, features2 in task_features.items():
                if task1 != task2:
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(
                        features1.mean(dim=0), 
                        features2.mean(dim=0), 
                        dim=0
                    ).item()
                    similarities[task1][task2] = similarity
        
        self.task_similarity_matrix = similarities
        return similarities
    
    def transfer_knowledge(self, source_task: str, target_task: str, 
                          transfer_rate: float = 0.3) -> Dict[str, torch.Tensor]:
        """知识迁移"""
        if source_task not in self.task_adapters or target_task not in self.task_adapters:
            logger.warning(f"任务 {source_task} 或 {target_task} 未注册")
            return {}
        
        source_adapter = self.task_adapters[source_task]
        target_adapter = self.task_adapters[target_task]
        
        transferred_params = {}
        
        # 迁移相关参数
        for name, param in source_adapter.named_parameters():
            if name in dict(target_adapter.named_parameters()):
                # 根据相似性调整迁移率
                transfer_weight = transfer_rate
                if self.task_similarity_matrix:
                    transfer_weight *= (1 + self.task_similarity_matrix.get(source_task, {}).get(target_task, 0))
                
                transferred_params[name] = param.data * transfer_weight
        
        return transferred_params
    
    def adapt_features(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """任务特定的特征适应"""
        # 提取共享特征
        shared_features = self.feature_extractor(x)
        
        # 应用任务特定适配器
        if task_id in self.task_adapters:
            adapted_features = self.task_adapters[task_id](shared_features)
            return adapted_features
        
        return shared_features


class KnowledgeGraph:
    """知识图谱 - 管理概念和关系"""
    
    def __init__(self):
        self.nodes = {}  # 概念节点
        self.edges = defaultdict(list)  # 关系边
        self.concept_importance = {}
        
    def add_concept(self, concept: str, importance: float = 1.0):
        """添加概念节点"""
        self.nodes[concept] = KnowledgeNode(
            concept=concept,
            importance=importance,
            connections=[],
            strength=1.0,
            last_accessed=datetime.now(),
            applications=[],
            confidence_score=1.0
        )
        self.concept_importance[concept] = importance
    
    def add_relationship(self, concept1: str, concept2: str, relationship_type: str, strength: float = 1.0):
        """添加概念间关系"""
        if concept1 in self.nodes and concept2 in self.nodes:
            self.edges[concept1].append({
                'concept': concept2,
                'type': relationship_type,
                'strength': strength
            })
            
            self.nodes[concept1].connections.append(concept2)
            self.nodes[concept1].strength = max(self.nodes[concept1].strength, strength)
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """获取相关概念"""
        related = []
        visited = set()
        
        def dfs(current_concept: str, depth: int):
            if depth > max_depth or current_concept in visited:
                return
                
            visited.add(current_concept)
            related.append(current_concept)
            
            for edge in self.edges.get(current_concept, []):
                dfs(edge['concept'], depth + 1)
        
        dfs(concept, 0)
        return related
    
    def update_concept_strength(self, concept: str, application_count: int = 1):
        """更新概念强度"""
        if concept in self.nodes:
            self.nodes[concept].applications.append(datetime.now().isoformat())
            # 指数衰减的强度更新
            decay_factor = 0.99
            self.nodes[concept].strength *= (1 + application_count * 0.01)
            self.nodes[concept].strength *= decay_factor
            self.nodes[concept].last_accessed = datetime.now()
            
            # 更新置信度分数
            recent_apps = len([app for app in self.nodes[concept].applications 
                             if datetime.fromisoformat(app) > datetime.now()])
            self.nodes[concept].confidence_score = min(1.0, 0.5 + 0.5 * recent_apps / 10)
    
    def prune_low_importance_concepts(self, threshold: float = 0.1):
        """修剪低重要性概念"""
        to_remove = []
        for concept, node in self.nodes.items():
            effective_importance = node.importance * node.confidence_score
            if effective_importance < threshold:
                to_remove.append(concept)
        
        for concept in to_remove:
            del self.nodes[concept]
            if concept in self.edges:
                del self.edges[concept]
            
            # 移除相关边
            for source, edges in self.edges.items():
                self.edges[source] = [e for e in edges if e['concept'] != concept]
        
        logger.info(f"修剪了 {len(to_remove)} 个低重要性概念")


class LifelongLearningSystem:
    """终身学习和自我进化系统主类"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or self._default_config()
        
        # 初始化各个模块
        self.memory_buffer = MemoryBuffer(max_size=self.config['memory_buffer_size'])
        self.knowledge_distillation = KnowledgeDistillation(
            temperature=self.config['kd_temperature'],
            alpha=self.config['kd_alpha']
        )
        self.meta_learning = MetaLearningModule(
            model=model,
            lr_inner=self.config['meta_lr_inner'],
            num_inner_steps=self.config['meta_inner_steps']
        )
        self.architecture_optimizer = ArchitectureOptimizer(
            model=model,
            mutation_rate=self.config['architecture_mutation_rate']
        )
        self.task_adapter = TaskAdapter(feature_extractor=model.feature_extractor if hasattr(model, 'feature_extractor') else model)
        self.knowledge_graph = KnowledgeGraph()
        
        # 学习状态管理
        self.learning_history = []
        self.current_task = None
        self.performance_baseline = {}
        self.knowledge_preservation_threshold = self.config['knowledge_preservation_threshold']
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 启动守护线程
        self._start_background_processes()
        
        logger.info("终身学习系统初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'memory_buffer_size': 10000,
            'kd_temperature': 3.0,
            'kd_alpha': 0.7,
            'meta_lr_inner': 0.01,
            'meta_inner_steps': 5,
            'architecture_mutation_rate': 0.1,
            'knowledge_preservation_threshold': 0.8,
            'replay_batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'save_interval': 100,
            'max_architecture_evolutions': 100,
            'min_task_performance': 0.5
        }
    
    def _start_background_processes(self):
        """启动后台进程"""
        def background_optimizer():
            while True:
                try:
                    self._background_optimization()
                    import time
                    time.sleep(300)  # 5分钟检查一次
                except Exception as e:
                    logger.error(f"后台优化错误: {e}")
        
        background_thread = threading.Thread(target=background_optimizer, daemon=True)
        background_thread.start()
        
        def background_knowledge_pruning():
            while True:
                try:
                    self.knowledge_graph.prune_low_importance_concepts()
                    import time
                    time.sleep(1800)  # 30分钟修剪一次
                except Exception as e:
                    logger.error(f"知识修剪错误: {e}")
        
        pruning_thread = threading.Thread(target=background_knowledge_pruning, daemon=True)
        pruning_thread.start()
    
    def learn_new_task(self, task_id: str, train_loader: DataLoader, 
                      val_loader: DataLoader, eval_func: Callable = None) -> Dict[str, float]:
        """学习新任务"""
        with self.lock:
            self.current_task = task_id
            logger.info(f"开始学习新任务: {task_id}")
            
            results = {}
            
            # 1. 持续学习 - 使用重放缓冲区
            if len(self.memory_buffer.buffer) > 0:
                logger.info("使用经验回放进行持续学习")
                results['continual_learning'] = self._continual_learning_phase(
                    task_id, train_loader, val_loader
                )
            else:
                logger.info("首次学习，无回放经验")
                results['continual_learning'] = self._initial_learning_phase(
                    task_id, train_loader, val_loader
                )
            
            # 2. 元学习适应
            if self.learning_history:
                logger.info("执行元学习适应")
                results['meta_learning'] = self._meta_learning_adaptation(
                    task_id, train_loader
                )
            
            # 3. 知识蒸馏
            if self.learning_history:
                logger.info("执行知识蒸馏")
                results['knowledge_distillation'] = self._knowledge_distillation_phase(
                    task_id, train_loader
                )
            
            # 4. 任务适配和知识迁移
            if self.learning_history:
                logger.info("执行知识迁移")
                results['knowledge_transfer'] = self._knowledge_transfer_phase(
                    task_id, train_loader
                )
            
            # 5. 性能评估和记忆更新
            final_performance = self._evaluate_task_performance(task_id, val_loader)
            results['final_performance'] = final_performance
            
            # 保存学习状态
            self._save_learning_state(task_id, results)
            
            # 更新记忆缓冲区
            self._update_memory_buffer(task_id, val_loader, final_performance)
            
            # 注册教师模型
            self.knowledge_distillation.register_teacher(task_id, self.model)
            
            # 清理和更新
            self._cleanup_old_knowledge()
            
            logger.info(f"任务 {task_id} 学习完成，性能: {final_performance:.4f}")
            return results
    
    def _initial_learning_phase(self, task_id: str, train_loader: DataLoader, 
                              val_loader: DataLoader) -> Dict[str, float]:
        """初始学习阶段"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        metrics = {'loss': [], 'accuracy': []}
        
        for epoch in range(10):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 验证
            accuracy = self._evaluate_model(val_loader)
            avg_loss = epoch_loss / len(train_loader)
            
            metrics['loss'].append(avg_loss)
            metrics['accuracy'].append(accuracy)
            
            logger.info(f"任务 {task_id} - Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return {key: np.mean(values) for key, values in metrics.items()}
    
    def _continual_learning_phase(self, task_id: str, train_loader: DataLoader, 
                                val_loader: DataLoader) -> Dict[str, float]:
        """持续学习阶段 - 避免灾难性遗忘"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # 获取重放样本
        replay_samples = self.memory_buffer.sample_replay(self.config['replay_batch_size'])
        
        metrics = {'loss': [], 'accuracy': []}
        
        for epoch in range(8):  # 持续学习使用更少的epochs
            self.model.train()
            epoch_loss = 0.0
            
            # 混合当前任务和新任务样本
            current_batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                current_loss = F.cross_entropy(output, target)
                
                # 如果有重放样本，添加重放损失
                if replay_samples and current_batch_count < len(replay_samples):
                    replay_sample = replay_samples[current_batch_count % len(replay_samples)]
                    replay_loss = self._compute_replay_loss(replay_sample)
                    total_loss = 0.7 * current_loss + 0.3 * replay_loss
                else:
                    total_loss = current_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                current_batch_count += 1
            
            # 验证
            accuracy = self._evaluate_model(val_loader)
            avg_loss = epoch_loss / len(train_loader)
            
            metrics['loss'].append(avg_loss)
            metrics['accuracy'].append(accuracy)
            
            logger.info(f"持续学习 {task_id} - Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return {key: np.mean(values) for key, values in metrics.items()}
    
    def _compute_replay_loss(self, replay_sample: Dict[str, Any]) -> torch.Tensor:
        """计算重放损失"""
        data = replay_sample.get('data')
        target = replay_sample.get('target')
        
        if data is not None and target is not None:
            with torch.no_grad():
                output = self.model(data)
                return F.cross_entropy(output, target)
        
        return torch.tensor(0.0)
    
    def _meta_learning_adaptation(self, task_id: str, train_loader: DataLoader) -> float:
        """元学习适应"""
        # 构建支持集和查询集
        support_data, support_targets, query_data, query_targets = self._build_meta_tasks(train_loader)
        
        support_tasks = list(zip(support_data, support_targets))
        query_tasks = list(zip(query_data, query_targets))
        
        # 执行元学习更新
        meta_loss = self.meta_learning.meta_update(support_tasks, query_tasks)
        
        logger.info(f"元学习适应完成，Meta Loss: {meta_loss:.4f}")
        return meta_loss
    
    def _build_meta_tasks(self, train_loader: DataLoader) -> Tuple[List, List, List, List]:
        """构建元学习任务"""
        data_list = []
        target_list = []
        
        for batch_data, batch_targets in train_loader:
            data_list.extend(batch_data)
            target_list.extend(batch_targets)
        
        # 随机分割为支持集和查询集
        total_size = len(data_list)
        support_size = int(0.8 * total_size)
        
        indices = torch.randperm(total_size)
        
        support_indices = indices[:support_size]
        query_indices = indices[support_size:]
        
        support_data = [data_list[i] for i in support_indices]
        support_targets = [target_list[i] for i in support_indices]
        query_data = [data_list[i] for i in query_indices]
        query_targets = [target_list[i] for i in query_indices]
        
        return support_data, support_targets, query_data, query_targets
    
    def _knowledge_distillation_phase(self, task_id: str, train_loader: DataLoader) -> Dict[str, float]:
        """知识蒸馏阶段"""
        # 获取小批量数据用于蒸馏
        data_iter = iter(train_loader)
        batch_data, _ = next(data_iter)
        
        # 执行知识蒸馏
        distilled_outputs = self.knowledge_distillation.distill_knowledge(self.model, batch_data)
        
        # 记录蒸馏损失
        distillation_metrics = {
            'kd_loss': np.mean([output['kd_loss'] for output in distilled_outputs]),
            'num_teachers': len(self.knowledge_distillation.teacher_models)
        }
        
        logger.info(f"知识蒸馏完成，平均损失: {distillation_metrics['kd_loss']:.4f}")
        return distillation_metrics
    
    def _knowledge_transfer_phase(self, task_id: str, train_loader: DataLoader) -> Dict[str, Any]:
        """知识迁移阶段"""
        if not self.learning_history:
            return {'transfer_success': False}
        
        # 注册当前任务的适配器
        adapter_layers = self._create_task_adapter()
        self.task_adapter.register_task(task_id, adapter_layers)
        
        # 计算与历史任务的相似性
        current_features = self._extract_task_features(train_loader)
        
        similarity_metrics = {}
        for prev_task_id in self.learning_history.keys():
            if prev_task_id != task_id:
                similarity = self._compute_task_similarity(prev_task_id, current_features)
                if similarity > 0.3:  # 相似性阈值
                    # 执行知识迁移
                    transferred_params = self.task_adapter.transfer_knowledge(
                        prev_task_id, task_id, transfer_rate=0.3
                    )
                    similarity_metrics[prev_task_id] = {
                        'similarity': similarity,
                        'transfer_success': len(transferred_params) > 0
                    }
        
        logger.info(f"知识迁移完成，涉及 {len(similarity_metrics)} 个历史任务")
        return similarity_metrics
    
    def _create_task_adapter(self) -> nn.Module:
        """创建任务适配器"""
        # 简单的适配器层
        class TaskAdapterLayer(nn.Module):
            def __init__(self, input_size: int = 512, adapter_size: int = 128):
                super().__init__()
                self.adapter = nn.Sequential(
                    nn.Linear(input_size, adapter_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(adapter_size, input_size)
                )
            
            def forward(self, x):
                return self.adapter(x)
        
        return TaskAdapterLayer()
    
    def _extract_task_features(self, train_loader: DataLoader) -> torch.Tensor:
        """提取任务特征"""
        features = []
        self.model.eval()
        
        with torch.no_grad():
            for data, _ in train_loader:
                if hasattr(self.model, 'feature_extractor'):
                    batch_features = self.model.feature_extractor(data)
                else:
                    # 如果没有单独的特征提取器，使用模型的中间层
                    x = self.model.conv1(data) if hasattr(self.model, 'conv1') else data
                    batch_features = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
                features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0)
    
    def _compute_task_similarity(self, task_id: str, current_features: torch.Tensor) -> float:
        """计算任务相似性"""
        # 从历史学习状态中获取之前的任务特征
        if task_id in self.learning_history:
            prev_features = self.learning_history[task_id].get('task_features')
            if prev_features is not None:
                return F.cosine_similarity(
                    current_features.mean(dim=0), 
                    prev_features.mean(dim=0), 
                    dim=0
                ).item()
        return 0.0
    
    def _evaluate_task_performance(self, task_id: str, val_loader: DataLoader) -> float:
        """评估任务性能"""
        accuracy = self._evaluate_model(val_loader)
        
        # 更新知识图谱
        self._update_knowledge_graph(task_id, accuracy)
        
        return accuracy
    
    def _update_knowledge_graph(self, task_id: str, performance: float):
        """更新知识图谱"""
        # 添加任务概念
        self.knowledge_graph.add_concept(f"task_{task_id}", importance=performance)
        
        # 更新相关概念的重要性
        for concept in self.knowledge_graph.nodes:
            if f"task_{task_id}" in concept or task_id in concept:
                self.knowledge_graph.update_concept_strength(concept, application_count=1)
    
    def _save_learning_state(self, task_id: str, results: Dict[str, Any]):
        """保存学习状态"""
        learning_state = LearningState(
            task_id=task_id,
            epoch=len(results.get('continual_learning', {}).get('accuracy', [])),
            model_state={name: param.cpu().clone() for name, param in self.model.named_parameters()},
            optimizer_state={},  # 简化版本
            loss_history=results.get('continual_learning', {}).get('loss', []),
            accuracy_history=results.get('continual_learning', {}).get('accuracy', []),
            knowledge_replay_buffer=dict(self.memory_buffer.buffer),
            meta_parameters={},
            timestamp=datetime.now(),
            task_performance=results.get('final_performance', 0.0),
            knowledge_preservation_rate=1.0
        )
        
        self.learning_history[task_id] = learning_state
    
    def _update_memory_buffer(self, task_id: str, val_loader: DataLoader, performance: float):
        """更新记忆缓冲区"""
        importance = min(performance, 1.0)
        
        # 从验证集采样重要样本
        sample_count = min(100, len(val_loader.dataset))
        
        for _ in range(sample_count):
            data, target = val_loader.dataset[random.randint(0, len(val_loader.dataset) - 1)]
            sample = {
                'task_id': task_id,
                'data': data.unsqueeze(0),
                'target': torch.tensor([target]),
                'performance': performance
            }
            
            self.memory_buffer.add_sample(sample, importance=importance)
    
    def _cleanup_old_knowledge(self):
        """清理旧知识"""
        # 移除性能较差的历史任务
        if len(self.learning_history) > 5:  # 保留最多5个任务
            sorted_tasks = sorted(
                self.learning_history.items(),
                key=lambda x: x[1].task_performance,
                reverse=True
            )
            
            tasks_to_remove = sorted_tasks[5:]
            for task_id, _ in tasks_to_remove:
                if task_id in self.learning_history:
                    del self.learning_history[task_id]
                if task_id in self.knowledge_distillation.teacher_models:
                    del self.knowledge_distillation.teacher_models[task_id]
    
    def _background_optimization(self):
        """后台优化"""
        if len(self.learning_history) < 2:
            return
        
        # 定期架构优化
        if len(self.learning_history) % self.config['save_interval'] == 0:
            logger.info("执行后台架构优化")
            self._optimize_architecture()
    
    def _optimize_architecture(self):
        """架构优化"""
        # 评估当前架构
        if hasattr(self, '_eval_function'):
            current_performance = self.architecture_optimizer.evaluate_architecture(self._eval_function)
            
            # 如果性能停滞，尝试架构变异
            if len(self.architecture_optimizer.performance_history) > 2:
                recent_performance = self.architecture_optimizer.performance_history[-3:]
                if max(recent_performance) - min(recent_performance) < 0.01:  # 性能停滞
                    mutated_model = self.architecture_optimizer.mutate_architecture()
                    mutated_performance = self.architecture_optimizer.evaluate_architecture(self._eval_function)
                    
                    if mutated_performance > current_performance:
                        logger.info("架构优化成功，更新模型")
                        self.model = mutated_model
                        self.architecture_optimizer.model = mutated_model
                        self.meta_learning.base_model = mutated_model
    
    def _evaluate_model(self, data_loader: DataLoader) -> float:
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def save_system_state(self, filepath: str):
        """保存系统状态"""
        state = {
            'config': self.config,
            'learning_history': {k: asdict(v) for k, v in self.learning_history.items()},
            'memory_buffer_size': len(self.memory_buffer.buffer),
            'knowledge_graph_nodes': len(self.knowledge_graph.nodes),
            'task_adapters': list(self.task_adapter.task_adapters.keys()),
            'architecture_history': self.architecture_optimizer.architecture_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"系统状态已保存到: {filepath}")
    
    def load_system_state(self, filepath: str):
        """加载系统状态"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.config.update(state.get('config', {}))
            logger.info(f"系统状态已从 {filepath} 加载")
        else:
            logger.warning(f"状态文件不存在: {filepath}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = {
            'total_tasks': len(self.learning_history),
            'total_memory_samples': len(self.memory_buffer.buffer),
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'current_performance': 0.0,
            'average_performance': 0.0,
            'performance_history': self.architecture_optimizer.performance_history,
            'architecture_evolutions': len(self.architecture_optimizer.architecture_history)
        }
        
        if self.learning_history:
            performances = [state.task_performance for state in self.learning_history.values()]
            stats['current_performance'] = performances[-1] if performances else 0.0
            stats['average_performance'] = np.mean(performances)
        
        return stats


# 示例使用和测试函数
def create_simple_model(input_size: int = 784, num_classes: int = 10) -> nn.Module:
    """创建简单的神经网络模型"""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.classifier = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            features = self.feature_extractor(x)
            return self.classifier(features)
    
    return SimpleNet()


def demo_lifelong_learning():
    """演示终身学习系统"""
    print("=== 终身学习和自我进化系统演示 ===\n")
    
    # 创建模型和系统
    model = create_simple_model()
    lifelong_system = LifelongLearningSystem(model)
    
    print("1. 终身学习系统初始化完成")
    print(f"   - 模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"   - 初始配置: {lifelong_system.config}")
    
    # 模拟多个任务的学习过程
    tasks = [
        {'id': 'task_mnist', 'classes': 10, 'samples': 1000},
        {'id': 'task_cifar10', 'classes': 10, 'samples': 800},
        {'id': 'task_fashion_mnist', 'classes': 10, 'samples': 1200}
    ]
    
    learning_results = {}
    
    for i, task in enumerate(tasks):
        print(f"\n2.{i+1} 学习任务: {task['id']}")
        print(f"   - 类别数: {task['classes']}")
        print(f"   - 样本数: {task['samples']}")
        
        # 模拟训练数据加载器
        class MockDataLoader:
            def __init__(self, task_info):
                self.dataset = list(range(task_info['samples']))
                self.task_info = task_info
            
            def __iter__(self):
                for _ in range(5):  # 模拟5个epoch
                    data = torch.randn(32, 784) if 'mnist' in self.task_info['id'] else torch.randn(32, 3, 32, 32)
                    target = torch.randint(0, self.task_info['classes'], (32,))
                    yield data, target
        
        class MockValLoader:
            def __init__(self, task_info):
                self.dataset = list(range(100))
                self.task_info = task_info
            
            def __iter__(self):
                for _ in range(2):
                    data = torch.randn(32, 784) if 'mnist' in self.task_info['id'] else torch.randn(32, 3, 32, 32)
                    target = torch.randint(0, self.task_info['classes'], (32,))
                    yield data, target
        
        train_loader = MockDataLoader(task)
        val_loader = MockValLoader(task)
        
        # 执行学习
        result = lifelong_system.learn_new_task(
            task['id'], train_loader, val_loader
        )
        
        learning_results[task['id']] = result
        print(f"   - 学习完成，性能: {result.get('final_performance', 0):.4f}")
    
    # 获取统计信息
    stats = lifelong_system.get_learning_statistics()
    print(f"\n3. 学习统计:")
    print(f"   - 总任务数: {stats['total_tasks']}")
    print(f"   - 记忆缓冲区大小: {stats['total_memory_samples']}")
    print(f"   - 知识图谱大小: {stats['knowledge_graph_size']}")
    print(f"   - 当前性能: {stats['current_performance']:.4f}")
    print(f"   - 平均性能: {stats['average_performance']:.4f}")
    print(f"   - 架构演化次数: {stats['architecture_evolutions']}")
    
    # 保存系统状态
    save_path = "lifelong_learning_state.pkl"
    lifelong_system.save_system_state(save_path)
    print(f"\n4. 系统状态已保存到: {save_path}")
    
    print("\n=== 演示完成 ===")
    return lifelong_system, learning_results


if __name__ == "__main__":
    # 运行演示
    lifelong_system, results = demo_lifelong_learning()