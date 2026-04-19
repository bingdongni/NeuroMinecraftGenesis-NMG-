"""
多模态感知模块

该模块提供完整的多模态世界模型感知系统，集成了：
- 视觉感知（摄像头+物体识别）
- 音频感知（语音识别）
- 空间感知（点云处理）
- 世界模型动态构建
- 多模态特征融合

主要类：
- MultimodalSensingSystem: 主控制器
- CameraPerception: 视觉感知
- AudioPerception: 音频感知
- SpatialPerception: 空间感知
- WorldModel: 世界模型
- MultimodalFusion: 多模态融合

作者: NeuroMinecraftGenesis
创建时间: 2025-11-13
"""

from .multimodal_sensing import (
    MultimodalSensingSystem,
    CameraPerception,
    AudioPerception,
    SpatialPerception,
    WorldModel,
    MultimodalFusion,
    PerceptionData,
    WorldObject,
    SpatialFeature
)

__all__ = [
    'MultimodalSensingSystem',
    'CameraPerception',
    'AudioPerception',
    'SpatialPerception',
    'WorldModel',
    'MultimodalFusion',
    'PerceptionData',
    'WorldObject',
    'SpatialFeature'
]

__version__ = "1.0.0"
__author__ = "NeuroMinecraftGenesis"