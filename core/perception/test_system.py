#!/usr/bin/env python3
"""
多模态感知系统测试脚本

该脚本用于测试多模态感知系统的各个组件，
验证系统的基本功能和集成效果。

作者: NeuroMinecraftGenesis
创建时间: 2025-11-13
"""

import sys
import os
import time
import json
import traceback
import numpy as np
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from core.perception.multimodal_sensing import (
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
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖包")
    sys.exit(1)


class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time()
        })
        
        status = "✓ 通过" if success else "✗ 失败"
        print(f"[{status}] {test_name}: {message}")
    
    def test_imports(self):
        """测试模块导入"""
        try:
            # 测试基本导入
            import cv2
            import numpy as np
            import open3d
            import whisper
            import pyaudio
            import librosa
            import soundfile
            
            self.log_test("模块导入", True, "所有依赖包导入成功")
        except ImportError as e:
            self.log_test("模块导入", False, f"导入失败: {e}")
            return False
        return True
    
    def test_camera_perception(self):
        """测试摄像头感知模块"""
        try:
            camera = CameraPerception(enable_object_detection=False)
            self.log_test("摄像头感知初始化", True, "摄像头模块初始化成功")
            
            # 测试摄像头连接
            if camera.cap.isOpened():
                self.log_test("摄像头连接", True, "摄像头连接正常")
            else:
                self.log_test("摄像头连接", False, "摄像头连接失败")
            
            camera.cap.release()
            return True
        except Exception as e:
            self.log_test("摄像头感知", False, f"错误: {e}")
            return False
    
    def test_audio_perception(self):
        """测试音频感知模块"""
        try:
            audio = AudioPerception()
            self.log_test("音频感知初始化", True, "音频模块初始化成功")
            
            # 测试Whisper模型加载
            if audio.whisper_model:
                self.log_test("Whisper模型", True, "Whisper模型加载成功")
            else:
                self.log_test("Whisper模型", False, "Whisper模型加载失败")
            
            return True
        except Exception as e:
            self.log_test("音频感知", False, f"错误: {e}")
            return False
    
    def test_spatial_perception(self):
        """测试空间感知模块"""
        try:
            spatial = SpatialPerception(num_points=1000)
            self.log_test("空间感知初始化", True, "空间模块初始化成功")
            
            # 测试点云模拟
            points = spatial._simulate_lidar_data()
            if len(points) > 0:
                self.log_test("点云模拟", True, f"生成了 {len(points)} 个点")
            
            # 测试点云处理
            pcd = spatial._process_point_cloud(points)
            if pcd and len(pcd.points) > 0:
                self.log_test("点云处理", True, f"处理后有 {len(pcd.points)} 个点")
            
            return True
        except Exception as e:
            self.log_test("空间感知", False, f"错误: {e}")
            return False
    
    def test_world_model(self):
        """测试世界模型"""
        try:
            world = WorldModel()
            self.log_test("世界模型初始化", True, "世界模型初始化成功")
            
            # 创建模拟感知数据
            visual_data = PerceptionData(
                timestamp=time.time(),
                modality='visual',
                data={'objects': [
                    {'class': 'test_object', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]}
                ]},
                confidence=0.9
            )
            
            # 更新世界状态
            world.update_world_state([visual_data])
            
            if len(world.objects) > 0:
                self.log_test("世界状态更新", True, f"世界中有 {len(world.objects)} 个对象")
            else:
                self.log_test("世界状态更新", False, "世界状态更新失败")
            
            return True
        except Exception as e:
            self.log_test("世界模型", False, f"错误: {e}")
            return False
    
    def test_multimodal_fusion(self):
        """测试多模态融合"""
        try:
            fusion = MultimodalFusion(feature_dim=64)
            self.log_test("融合引擎初始化", True, "融合引擎初始化成功")
            
            # 创建模拟数据
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
            
            # 测试特征提取
            features = fusion.extract_fused_features([visual_data, audio_data])
            if len(features) == 64:
                self.log_test("特征融合", True, f"成功提取 {len(features)} 维特征")
            else:
                self.log_test("特征融合", False, f"特征维度错误: {len(features)}")
            
            return True
        except Exception as e:
            self.log_test("多模态融合", False, f"错误: {e}")
            return False
    
    def test_system_integration(self):
        """测试系统集成"""
        try:
            config = {
                'camera_id': 0,
                'enable_object_detection': False,  # 关闭物体检测以避免模型加载问题
                'audio_sample_rate': 16000,
                'num_points': 1000,
                'feature_dim': 64
            }
            
            system = MultimodalSensingSystem(config)
            self.log_test("系统集成初始化", True, "系统集成初始化成功")
            
            # 短暂启动系统（2秒）
            print("启动系统集成测试...")
            system.start_system()
            
            # 等待数据采集
            time.sleep(2)
            
            # 尝试获取感知数据
            perception = system.get_latest_perception()
            if perception:
                self.log_test("系统数据流", True, "成功获取感知数据")
            else:
                self.log_test("系统数据流", False, "未能获取感知数据")
            
            # 停止系统
            system.stop_system()
            
            return True
        except Exception as e:
            self.log_test("系统集成", False, f"错误: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 多模态感知系统测试 ===\n")
        
        tests = [
            self.test_imports,
            self.test_camera_perception,
            self.test_audio_perception,
            self.test_spatial_perception,
            self.test_world_model,
            self.test_multimodal_fusion,
            self.test_system_integration
        ]
        
        start_time = time.time()
        
        for test in tests:
            try:
                test()
                print()
            except Exception as e:
                self.log_test("测试执行", False, f"测试异常: {e}")
                print()
        
        # 生成测试报告
        self.generate_report(time.time() - start_time)
    
    def generate_report(self, total_time: float):
        """生成测试报告"""
        print("=== 测试报告 ===")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        print(f"总耗时: {total_time:.2f}秒")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for test in self.test_results:
                if not test['success']:
                    print(f"  - {test['test']}: {test['message']}")
        
        # 保存测试结果
        report = {
            'timestamp': time.time(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests/total_tests,
            'total_time': total_time,
            'test_results': self.test_results
        }
        
        try:
            with open('/workspace/core/perception/test_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n测试报告已保存到: /workspace/core/perception/test_report.json")
        except Exception as e:
            print(f"保存测试报告失败: {e}")
        
        return passed_tests == total_tests


def main():
    """主函数"""
    try:
        tester = SystemTester()
        all_passed = tester.run_all_tests()
        
        if all_passed:
            print("\n🎉 所有测试通过！系统已准备就绪。")
            return 0
        else:
            print("\n⚠️  部分测试失败，请检查错误信息。")
            return 1
    
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())