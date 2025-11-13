# 多模态世界模型感知系统

这是一个完整的多模态感知系统，集成了视觉、音频和空间感知能力，能够动态构建世界模型并进行多模态特征融合。

## 功能特性

### 1. 视觉感知模块 (CameraPerception)
- USB摄像头实时捕获
- OpenCV图像预处理
- YOLO物体检测 (可切换到基础OpenCV检测)
- 实时边界框检测和分类

### 2. 音频感知模块 (AudioPerception)
- 实时音频录制
- Whisper语音识别引擎
- 中文语音优先识别
- 静音检测和音频片段处理

### 3. 空间感知模块 (SpatialPerception)
- 激光雷达点云处理 (模拟数据)
- 点云滤波和降噪
- DBSCAN聚类分析
- 空间特征提取（质心、体积、表面积）

### 4. 世界模型 (WorldModel)
- 动态对象跟踪
- 时空关系建模
- 对象生命周期管理
- 置信度动态更新

### 5. 多模态融合 (MultimodalFusion)
- 特征提取和编码
- 模态权重融合
- 特征相似度计算
- 统一的感知表示

## 系统架构

```
MultimodalSensingSystem (主控制器)
├── CameraPerception (视觉感知)
├── AudioPerception (音频感知)
├── SpatialPerception (空间感知)
├── WorldModel (世界模型)
└── MultimodalFusion (多模态融合)
```

## 快速开始

### 基本使用

```python
from core.perception import MultimodalSensingSystem

# 创建系统实例
config = {
    'camera_id': 0,
    'enable_object_detection': True,
    'audio_sample_rate': 16000,
    'num_points': 10000,
    'feature_dim': 512
}

sensing_system = MultimodalSensingSystem(config)

# 启动系统
sensing_system.start_system()

try:
    # 持续获取感知数据
    while True:
        perception = sensing_system.get_latest_perception()
        if perception:
            world_state = perception['world_state']
            fused_features = perception['fused_features']
            
            print(f"检测到 {world_state['num_objects']} 个对象")
            print(f"融合特征维度: {len(fused_features)}")
        
        time.sleep(1)
finally:
    sensing_system.stop_system()
```

### 演示程序

```bash
# 运行完整演示
python core/perception/multimodal_sensing.py
```

## 系统要求

### Python依赖
- opencv-python
- numpy
- open3d
- openai-whisper
- pyaudio
- librosa
- soundfile
- scipy
- scikit-learn
- torch

### 硬件要求
- USB摄像头（可选）
- 麦克风（可选）
- 激光雷达（可选，当前使用模拟数据）

## 配置文件

```python
config = {
    # 摄像头配置
    'camera_id': 0,                    # 摄像头ID
    'enable_object_detection': True,   # 启用物体检测
    
    # 音频配置
    'audio_sample_rate': 16000,        # 采样率
    'audio_chunk_size': 1024,          # 音频块大小
    
    # 点云配置
    'num_points': 20000,               # 点云点数
    
    # 融合配置
    'feature_dim': 512                 # 特征维度
}
```

## 数据流

1. **数据采集**: 各传感器模块独立采集数据
2. **预处理**: 原始数据转换为标准格式
3. **特征提取**: 提取各模态的特征表示
4. **模态融合**: 多模态特征加权融合
5. **世界更新**: 更新世界模型状态
6. **结果输出**: 提供统一的感知结果

## API参考

### MultimodalSensingSystem

**主要方法:**
- `start_system()`: 启动整个系统
- `stop_system()`: 停止系统
- `get_latest_perception()`: 获取最新感知结果
- `get_system_status()`: 获取系统状态

### PerceptionData

**数据结构:**
```python
@dataclass
class PerceptionData:
    timestamp: float          # 时间戳
    modality: str            # 感知模态 ('visual', 'audio', 'spatial')
    data: Dict[str, Any]     # 原始数据
    confidence: float        # 置信度
    metadata: Dict[str, Any] # 元数据
```

## 性能优化

1. **多线程处理**: 各模态独立线程处理
2. **队列缓冲**: 使用队列避免数据阻塞
3. **特征缓存**: 缓存特征计算结果
4. **自适应采样**: 根据系统负载调整采样率

## 扩展开发

### 添加新的感知模态

```python
class CustomPerception:
    def __init__(self):
        self.modality = "custom"
    
    def extract_features(self, data):
        # 实现特征提取逻辑
        return features
```

### 自定义融合策略

```python
class CustomFusion(MultimodalFusion):
    def extract_fused_features(self, perception_data):
        # 实现自定义融合逻辑
        return fused_features
```

## 故障排除

1. **摄像头无法打开**: 检查摄像头ID和权限
2. **音频录制失败**: 检查麦克风权限和采样率设置
3. **模型加载失败**: 检查模型文件路径和依赖安装
4. **性能问题**: 调整缓冲区大小和采样率

## 许可证

MIT License

## 作者

NeuroMinecraftGenesis