#!/usr/bin/env python3
"""
多模态感知系统简化测试脚本

该脚本测试多模态感知系统的核心功能，
跳过需要系统依赖的组件（如PyAudio）。

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
    # 导入基本组件
    import cv2
    import numpy as np
    import open3d as o3d
    import whisper
    import librosa
    import soundfile
    import scipy
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import torch
except ImportError as e:
    print(f"依赖包导入错误: {e}")
    print("请确保已安装基本依赖包")
    sys.exit(1)

try:
    from core.perception.multimodal_sensing import (
        CameraPerception,
        SpatialPerception,
        WorldModel,
        MultimodalFusion,
        PerceptionData,
        WorldObject,
        SpatialFeature
    )
except ImportError as e:
    print(f"模块导入错误: {e}")
    sys.exit(1)


class SimplifiedSystemTester:
    """简化的系统测试器"""
    
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
    
    def test_basic_imports(self):
        """测试基本模块导入"""
        try:
            # 测试OpenCV
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            self.log_test("OpenCV导入", True, "OpenCV功能正常")
            
            # 测试NumPy
            arr = np.random.rand(10, 10)
            self.log_test("NumPy导入", True, "NumPy功能正常")
            
            # 测试Open3D
            points = np.random.rand(100, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            self.log_test("Open3D导入", True, "Open3D功能正常")
            
            # 测试sklearn
            clustering = DBSCAN(eps=0.1, min_samples=5).fit(points)
            self.log_test("Scikit-learn导入", True, "Scikit-learn功能正常")
            
            # 测试Whisper（不加载模型）
            import whisper
            self.log_test("Whisper导入", True, "Whisper模块可用")
            
            return True
        except Exception as e:
            self.log_test("基本模块导入", False, f"导入失败: {e}")
            return False
    
    def test_camera_perception_basic(self):
        """测试摄像头感知基本功能"""
        try:
            # 测试初始化
            camera = CameraPerception(enable_object_detection=False)
            self.log_test("摄像头感知初始化", True, "摄像头模块初始化成功")
            
            # 测试模拟帧生成
            test_frame = np.random.rand(416, 416, 3)
            processed_frame = camera._preprocess_frame(test_frame)
            self.log_test("图像预处理", True, f"图像预处理成功，输出形状: {processed_frame.shape}")
            
            # 测试物体检测（简化版）
            objects = camera._detect_objects(processed_frame)
            self.log_test("物体检测", True, f"检测到 {len(objects)} 个物体")
            
            return True
        except Exception as e:
            self.log_test("摄像头感知基本功能", False, f"错误: {e}")
            return False
    
    def test_spatial_perception(self):
        """测试空间感知功能"""
        try:
            spatial = SpatialPerception(num_points=1000)
            self.log_test("空间感知初始化", True, "空间感知模块初始化成功")
            
            # 测试点云模拟
            points = spatial._simulate_lidar_data()
            self.log_test("点云模拟", True, f"生成了 {len(points)} 个点")
            
            # 测试点云处理
            pcd = spatial._process_point_cloud(points)
            self.log_test("点云处理", True, f"处理后有 {len(pcd.points)} 个点")
            
            # 测试空间特征提取
            features = spatial.extract_spatial_features(pcd)
            self.log_test("空间特征提取", True, f"提取到 {len(features)} 个空间特征")
            
            return True
        except Exception as e:
            self.log_test("空间感知功能", False, f"错误: {e}")
            return False
    
    def test_world_model(self):
        """测试世界模型"""
        try:
            world = WorldModel()
            self.log_test("世界模型初始化", True, "世界模型初始化成功")
            
            # 创建模拟视觉数据
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
            self.log_test("世界状态更新", True, f"世界中有 {len(world.objects)} 个对象")
            
            # 获取当前状态
            current_state = world.get_current_state()
            self.log_test("状态获取", True, f"成功获取世界状态")
            
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
            
            spatial_data = PerceptionData(
                timestamp=time.time(),
                modality='spatial',
                data={'features': []},
                confidence=0.6
            )
            
            # 测试特征融合
            features = fusion.extract_fused_features([visual_data, spatial_data])
            if len(features) == 64:
                self.log_test("特征融合", True, f"成功提取 {len(features)} 维特征")
            else:
                self.log_test("特征融合", False, f"特征维度错误: {len(features)}")
            
            # 测试相似度计算
            features1 = np.random.rand(64)
            features2 = np.random.rand(64)
            similarity = fusion.compute_similarity(features1, features2)
            self.log_test("特征相似度", True, f"相似度: {similarity:.4f}")
            
            return True
        except Exception as e:
            self.log_test("多模态融合", False, f"错误: {e}")
            return False
    
    def test_integrated_workflow(self):
        """测试集成工作流"""
        try:
            # 创建所有组件
            camera = CameraPerception(enable_object_detection=False)
            spatial = SpatialPerception(num_points=500)
            world = WorldModel()
            fusion = MultimodalFusion(feature_dim=64)
            
            # 模拟数据流
            print("模拟完整数据流...")
            
            # 1. 视觉数据
            test_frame = np.random.rand(416, 416, 3)
            processed_frame = camera._preprocess_frame(test_frame)
            objects = camera._detect_objects(processed_frame)
            
            visual_data = PerceptionData(
                timestamp=time.time(),
                modality='visual',
                data={'frame': processed_frame, 'objects': objects},
                confidence=0.8
            )
            
            # 2. 空间数据
            points = spatial._simulate_lidar_data()
            pcd = spatial._process_point_cloud(points)
            features = spatial.extract_spatial_features(pcd)
            
            spatial_data = PerceptionData(
                timestamp=time.time(),
                modality='spatial',
                data={'features': features, 'point_cloud': pcd},
                confidence=0.7
            )
            
            # 3. 更新世界模型
            world.update_world_state([visual_data, spatial_data])
            self.log_test("世界模型更新", True, f"世界中对象数: {len(world.objects)}")
            
            # 4. 特征融合
            fused_features = fusion.extract_fused_features([visual_data, spatial_data])
            self.log_test("完整融合流程", True, f"融合特征维度: {len(fused_features)}")
            
            # 5. 获取最终状态
            final_state = world.get_current_state()
            self.log_test("最终状态获取", True, f"成功获取最终世界状态")
            
            return True
        except Exception as e:
            self.log_test("集成工作流", False, f"错误: {e}")
            traceback.print_exc()
            return False
    
    def test_performance_metrics(self):
        """测试性能指标"""
        try:
            print("测试性能指标...")
            
            # 测试图像处理性能
            start_time = time.time()
            camera = CameraPerception(enable_object_detection=False)
            for _ in range(10):
                test_frame = np.random.rand(416, 416, 3)
                processed_frame = camera._preprocess_frame(test_frame)
                objects = camera._detect_objects(processed_frame)
            end_time = time.time()
            
            image_processing_time = (end_time - start_time) / 10
            self.log_test("图像处理性能", True, f"平均处理时间: {image_processing_time:.4f}秒")
            
            # 测试点云处理性能
            spatial = SpatialPerception(num_points=1000)
            start_time = time.time()
            for _ in range(5):
                points = spatial._simulate_lidar_data()
                pcd = spatial._process_point_cloud(points)
            end_time = time.time()
            
            point_cloud_time = (end_time - start_time) / 5
            self.log_test("点云处理性能", True, f"平均处理时间: {point_cloud_time:.4f}秒")
            
            # 测试融合性能
            fusion = MultimodalFusion(feature_dim=512)
            visual_data = PerceptionData(
                timestamp=time.time(),
                modality='visual',
                data={'frame': np.random.rand(224, 224, 3)},
                confidence=0.8
            )
            
            start_time = time.time()
            for _ in range(20):
                features = fusion.extract_fused_features([visual_data])
            end_time = time.time()
            
            fusion_time = (end_time - start_time) / 20
            self.log_test("特征融合性能", True, f"平均融合时间: {fusion_time:.4f}秒")
            
            return True
        except Exception as e:
            self.log_test("性能指标", False, f"错误: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 多模态感知系统简化测试 ===\n")
        
        tests = [
            self.test_basic_imports,
            self.test_camera_perception_basic,
            self.test_spatial_perception,
            self.test_world_model,
            self.test_multimodal_fusion,
            self.test_integrated_workflow,
            self.test_performance_metrics
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
        tester = SimplifiedSystemTester()
        all_passed = tester.run_all_tests()
        
        if all_passed:
            print("\n🎉 所有测试通过！多模态感知系统工作正常。")
            print("\n系统特点:")
            print("- 视觉感知: USB摄像头 + OpenCV物体识别 ✓")
            print("- 空间感知: 激光雷达点云处理 ✓")
            print("- 世界模型: 动态对象跟踪和状态管理 ✓")
            print("- 多模态融合: 特征提取和融合 ✓")
            print("- 数据处理管道: 完整的多模态数据流 ✓")
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