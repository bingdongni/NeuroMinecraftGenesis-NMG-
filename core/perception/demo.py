#!/usr/bin/env python3
"""
多模态感知系统演示脚本

该脚本展示如何使用多模态感知系统的核心功能，
包括单模态测试和多模态融合演示。

作者: NeuroMinecraftGenesis
创建时间: 2025-11-13
"""

import sys
import os
import time
import json
import numpy as np
import threading
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from core.perception.multimodal_sensing import (
        MultimodalSensingSystem,
        CameraPerception,
        AudioPerception,
        SpatialPerception,
        WorldModel,
        MultimodalFusion
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请检查模块路径和依赖安装")
    sys.exit(1)


class MultimodalDemo:
    """多模态感知演示"""
    
    def __init__(self):
        self.config = {
            'camera_id': 0,
            'enable_object_detection': True,
            'audio_sample_rate': 16000,
            'num_points': 5000,
            'feature_dim': 256
        }
        self.system = None
        self.is_demo_running = False
    
    def print_header(self, title: str):
        """打印标题"""
        print("\n" + "=" * 60)
        print(f" {title} ".center(60, "="))
        print("=" * 60)
    
    def print_data(self, title: str, data: Dict[str, Any], indent: int = 0):
        """格式化打印数据"""
        prefix = "  " * indent
        print(f"{prefix}{title}:")
        
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}  {key}:")
                self.print_data("", value, indent + 2)
            elif isinstance(value, list) and len(value) > 3:
                print(f"{prefix}  {key}: [{len(value)} items]")
            else:
                print(f"{prefix}  {key}: {value}")
    
    def demo_single_modality(self):
        """演示单模态功能"""
        self.print_header("单模态感知演示")
        
        # 1. 视觉感知演示
        print("\n1. 视觉感知模块演示")
        try:
            camera = CameraPerception(enable_object_detection=False)
            camera.start_capture()
            
            print("摄像头启动中...")
            time.sleep(2)  # 等待摄像头稳定
            
            frame_data = camera.get_latest_frame()
            if frame_data:
                print(f"✓ 获取到一帧图像，时间戳: {frame_data['timestamp']:.2f}")
                print(f"  图像尺寸: 416x416")
                print(f"  检测物体数: {len(frame_data['objects'])}")
            else:
                print("✗ 未能获取图像帧")
            
            camera.stop_capture()
            
        except Exception as e:
            print(f"✗ 视觉感知演示失败: {e}")
        
        # 2. 音频感知演示
        print("\n2. 音频感知模块演示")
        try:
            audio = AudioPerception()
            audio.start_recording()
            
            print("音频录制启动中...")
            time.sleep(2)  # 等待音频稳定
            
            audio_data = audio.get_latest_audio()
            if audio_data:
                print(f"✓ 获取到音频数据，时间戳: {audio_data['timestamp']:.2f}")
                print(f"  音频长度: {len(audio_data['audio'])} 样本")
                print(f"  采样率: {audio_data['sample_rate']} Hz")
                
                # 测试语音识别
                if audio.whisper_model:
                    transcription = audio.transcribe_audio(
                        audio_data['audio'], 
                        audio_data['sample_rate']
                    )
                    print(f"  识别文本: '{transcription['text']}'")
                    print(f"  识别语言: {transcription['language']}")
            else:
                print("✗ 未能获取音频数据")
            
            audio.stop_recording()
            
        except Exception as e:
            print(f"✗ 音频感知演示失败: {e}")
        
        # 3. 空间感知演示
        print("\n3. 空间感知模块演示")
        try:
            spatial = SpatialPerception(num_points=1000)
            
            # 模拟点云数据
            points = spatial._simulate_lidar_data()
            print(f"✓ 生成了 {len(points)} 个点云点")
            
            # 处理点云
            pcd = spatial._process_point_cloud(points)
            print(f"✓ 处理后点云包含 {len(pcd.points)} 个点")
            
            # 提取空间特征
            features = spatial.extract_spatial_features(pcd)
            print(f"✓ 提取到 {len(features)} 个空间特征")
            
            for i, feature in enumerate(features[:3]):  # 只显示前3个
                print(f"  特征 {i+1}:")
                print(f"    质心: {feature.centroid}")
                print(f"    体积: {feature.volume:.2f}")
                print(f"    表面积: {feature.surface_area:.2f}")
        
        except Exception as e:
            print(f"✗ 空间感知演示失败: {e}")
    
    def demo_world_model(self):
        """演示世界模型"""
        self.print_header("世界模型动态构建演示")
        
        try:
            world = WorldModel()
            print("世界模型已初始化")
            
            # 模拟视觉数据输入
            print("\n输入模拟视觉数据...")
            visual_data = {
                'timestamp': time.time(),
                'modality': 'visual',
                'data': {
                    'objects': [
                        {'class': 'person', 'confidence': 0.9, 'bbox': [100, 50, 80, 120]},
                        {'class': 'car', 'confidence': 0.8, 'bbox': [300, 100, 150, 80]}
                    ]
                },
                'confidence': 0.85
            }
            
            from core.perception.multimodal_sensing import PerceptionData
            visual_perception = PerceptionData(**visual_data)
            world.update_world_state([visual_perception])
            
            print(f"✓ 世界中当前有 {len(world.objects)} 个对象")
            
            # 显示世界状态
            current_state = world.get_current_state()
            print("\n当前世界状态:")
            self.print_data("对象", {obj_id: {
                '位置': obj.position.tolist(),
                '置信度': obj.confidence,
                '最后看见': obj.last_seen
            } for obj_id, obj in current_state['objects'].items()})
            
            # 模拟空间数据输入
            print("\n输入模拟空间数据...")
            spatial_data = {
                'timestamp': time.time(),
                'modality': 'spatial',
                'data': {
                    'features': [
                        {
                            'centroid': [200, 100, 50],
                            'volume': 1000.0,
                            'surface_area': 200.0,
                            'bounds': np.array([[0, 0, 0], [400, 200, 100]])
                        }
                    ]
                },
                'confidence': 0.7
            }
            
            # 创建空间特征对象
            from core.perception.multimodal_sensing import SpatialFeature
            feature = SpatialFeature(
                centroid=np.array([200, 100, 50]),
                bounds=np.array([[0, 0, 0], [400, 200, 100]]),
                surface_area=200.0,
                volume=1000.0,
                orientation=np.array([1, 0, 0])
            )
            
            spatial_perception = PerceptionData(
                timestamp=time.time(),
                modality='spatial',
                data={'features': [feature]},
                confidence=0.7
            )
            world.update_world_state([spatial_perception])
            
            print(f"✓ 世界中现在有 {len(world.objects)} 个对象")
            
        except Exception as e:
            print(f"✗ 世界模型演示失败: {e}")
    
    def demo_multimodal_fusion(self):
        """演示多模态融合"""
        self.print_header("多模态融合演示")
        
        try:
            fusion = MultimodalFusion(feature_dim=64)
            print("多模态融合引擎已初始化")
            
            # 创建模拟感知数据
            visual_data = PerceptionData(
                timestamp=time.time(),
                modality='visual',
                data={'frame': np.random.rand(224, 224, 3)},
                confidence=0.8
            )
            
            audio_data = PerceptionData(
                timestamp=time.time(),
                modality='audio',
                data={'audio': np.random.rand(16000), 'sample_rate': 16000},
                confidence=0.7
            )
            
            spatial_data = PerceptionData(
                timestamp=time.time(),
                modality='spatial',
                data={'features': []},
                confidence=0.6
            )
            
            perception_list = [visual_data, audio_data, spatial_data]
            
            # 执行特征融合
            print("\n执行多模态特征融合...")
            fused_features = fusion.extract_fused_features(perception_list)
            
            print(f"✓ 成功提取融合特征")
            print(f"  特征维度: {len(fused_features)}")
            print(f"  特征均值: {np.mean(fused_features):.4f}")
            print(f"  特征标准差: {np.std(fused_features):.4f}")
            
            # 测试特征相似度
            features1 = np.random.rand(64)
            features2 = np.random.rand(64)
            similarity = fusion.compute_similarity(features1, features2)
            
            print(f"\n✓ 特征相似度测试")
            print(f"  相似度: {similarity:.4f}")
            
        except Exception as e:
            print(f"✗ 多模态融合演示失败: {e}")
    
    def demo_integrated_system(self, duration: int = 30):
        """演示集成系统"""
        self.print_header(f"集成系统演示 ({duration}秒)")
        
        try:
            self.system = MultimodalSensingSystem(self.config)
            self.is_demo_running = True
            
            print("启动多模态感知系统...")
            self.system.start_system()
            
            # 创建监控线程
            monitor_thread = threading.Thread(
                target=self._monitor_system,
                args=(duration,)
            )
            monitor_thread.start()
            
            # 等待演示结束
            monitor_thread.join()
            
            print("\n✓ 集成系统演示完成")
            
        except Exception as e:
            print(f"✗ 集成系统演示失败: {e}")
        
        finally:
            if self.system:
                self.system.stop_system()
                self.is_demo_running = False
    
    def _monitor_system(self, duration: int):
        """监控系统和显示结果"""
        start_time = time.time()
        update_interval = 2  # 每2秒更新一次
        
        while (time.time() - start_time) < duration and self.is_demo_running:
            try:
                # 获取最新感知数据
                perception = self.system.get_latest_perception()
                
                if perception:
                    world_state = perception['world_state']
                    stats = perception['stats']
                    fused_features = perception['fused_features']
                    
                    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
                    print(f"检测对象数: {world_state['num_objects']}")
                    print(f"处理帧数: {stats['frame_count']}")
                    print(f"音频段数: {stats['audio_count']}")
                    print(f"点云数: {stats['point_cloud_count']}")
                    print(f"融合次数: {stats['fusion_count']}")
                    print(f"特征维度: {len(fused_features)}")
                    
                    # 显示最新音频
                    if world_state['current_state'].get('last_audio'):
                        audio_text = world_state['current_state']['last_audio']['text']
                        if audio_text:
                            print(f"最新语音: '{audio_text}'")
                    
                    # 显示部分特征值
                    if fused_features:
                        print(f"特征样本: {[f'{x:.3f}' for x in fused_features[:5]]}")
                
                time.sleep(update_interval)
            
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        self.print_header("交互式多模态感知演示")
        
        print("选择演示模式:")
        print("1. 单模态演示")
        print("2. 世界模型演示")
        print("3. 多模态融合演示")
        print("4. 集成系统演示 (30秒)")
        print("5. 完整演示 (所有模式)")
        print("0. 退出")
        
        try:
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == '1':
                self.demo_single_modality()
            elif choice == '2':
                self.demo_world_model()
            elif choice == '3':
                self.demo_multimodal_fusion()
            elif choice == '4':
                duration = input("演示时长 (秒，默认30): ").strip()
                try:
                    duration = int(duration) if duration else 30
                except:
                    duration = 30
                self.demo_integrated_system(duration)
            elif choice == '5':
                print("\n开始完整演示...")
                self.demo_single_modality()
                input("\n按回车继续到世界模型演示...")
                self.demo_world_model()
                input("\n按回车继续到多模态融合演示...")
                self.demo_multimodal_fusion()
                input("\n按回车继续到集成系统演示...")
                self.demo_integrated_system(30)
                print("\n🎉 完整演示结束！")
            elif choice == '0':
                print("退出演示")
                return
            else:
                print("无效选择")
        
        except KeyboardInterrupt:
            print("\n\n演示被用户中断")
        except Exception as e:
            print(f"\n演示执行错误: {e}")
    
    def run_batch_demo(self):
        """运行批量演示"""
        self.print_header("自动批量演示")
        
        print("自动执行所有演示模式...")
        
        self.demo_single_modality()
        time.sleep(2)
        
        self.demo_world_model()
        time.sleep(2)
        
        self.demo_multimodal_fusion()
        time.sleep(2)
        
        self.demo_integrated_system(10)  # 较短的集成演示
        
        print("\n🎉 批量演示完成！")


def main():
    """主函数"""
    try:
        demo = MultimodalDemo()
        
        print("多模态世界模型感知系统演示")
        print("=" * 50)
        print("支持的功能:")
        print("- 视觉感知 (USB摄像头 + 物体识别)")
        print("- 音频感知 (Whisper语音识别)")
        print("- 空间感知 (激光雷达点云处理)")
        print("- 世界模型动态构建")
        print("- 多模态特征融合")
        
        mode = input("\n选择运行模式 (1: 交互式, 2: 批量): ").strip()
        
        if mode == '1':
            demo.run_interactive_demo()
        elif mode == '2':
            demo.run_batch_demo()
        else:
            print("使用交互式模式")
            demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n演示被中断")
    except Exception as e:
        print(f"\n演示执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()