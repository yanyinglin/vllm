#!/usr/bin/env python3
"""
ZeroMQ Pipeline功能测试

测试多进程分布式推理功能
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.offline_pipelining.test_pipeline import (
    test_pipeline,
    test_pipeline_multiprocess,
)


class TestZeroMQPipeline(unittest.TestCase):
    """ZeroMQ Pipeline功能测试"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 检查是否有pipeline目录
        # 如果没有，跳过测试
        cls.pipeline_dir = os.environ.get("TEST_PIPELINE_DIR")
        if not cls.pipeline_dir:
            cls.skip_all = True
            return
        
        cls.pipeline_dir = Path(cls.pipeline_dir)
        if not cls.pipeline_dir.exists():
            cls.skip_all = True
            return
        
        # 检查stage目录
        cls.num_stages = None
        for i in range(10):
            stage_dir = cls.pipeline_dir / f"stage_{i}"
            if stage_dir.exists():
                cls.num_stages = i + 1
            else:
                break
        
        if cls.num_stages is None or cls.num_stages == 0:
            cls.skip_all = True
            return
        
        cls.skip_all = False
        print(f"\n=== Test Setup ===")
        print(f"Pipeline directory: {cls.pipeline_dir}")
        print(f"Number of stages: {cls.num_stages}")
        print(f"==================\n")

    def setUp(self):
        """每个测试前的设置"""
        if hasattr(self.__class__, "skip_all") and self.__class__.skip_all:
            self.skipTest("Pipeline directory not configured")

    def test_single_stage(self):
        """测试单stage推理"""
        if self.num_stages < 1:
            self.skipTest("Need at least 1 stage")
        
        test_input = "Hello, world!"
        print(f"\n--- Test: Single Stage ---")
        print(f"Input: {test_input}")
        
        try:
            output = test_pipeline(
                pipeline_dir=str(self.pipeline_dir),
                num_stages=1,
                test_input=test_input,
                device="cpu",
                use_multiprocess=False,
            )
            print(f"Output: {output}")
            self.assertIsNotNone(output)
            self.assertIsInstance(output, str)
        except Exception as e:
            self.fail(f"Single stage test failed: {e}")

    def test_multi_stage_multiprocess(self):
        """测试多stage多进程推理"""
        if self.num_stages < 2:
            self.skipTest("Need at least 2 stages for multi-stage test")
        
        test_input = "The quick brown fox"
        print(f"\n--- Test: Multi-Stage Multiprocess ---")
        print(f"Input: {test_input}")
        print(f"Number of stages: {self.num_stages}")
        
        try:
            output = test_pipeline_multiprocess(
                pipeline_dir=str(self.pipeline_dir),
                num_stages=self.num_stages,
                test_input=test_input,
                device="cpu",
            )
            print(f"Output: {output}")
            self.assertIsNotNone(output)
            self.assertIsInstance(output, str)
            # 检查输出不是错误消息
            self.assertFalse(output.startswith("ERROR:"))
        except Exception as e:
            self.fail(f"Multi-stage multiprocess test failed: {e}")

    def test_custom_ports(self):
        """测试自定义端口"""
        if self.num_stages < 2:
            self.skipTest("Need at least 2 stages for custom ports test")
        
        # 使用自定义端口范围
        base_port = 6000
        custom_ports = [base_port + i for i in range(self.num_stages - 1)]
        
        test_input = "Test with custom ports"
        print(f"\n--- Test: Custom Ports ---")
        print(f"Input: {test_input}")
        print(f"Ports: {custom_ports}")
        
        try:
            output = test_pipeline_multiprocess(
                pipeline_dir=str(self.pipeline_dir),
                num_stages=self.num_stages,
                test_input=test_input,
                zmq_ports=custom_ports,
                device="cpu",
            )
            print(f"Output: {output}")
            self.assertIsNotNone(output)
            self.assertIsInstance(output, str)
        except Exception as e:
            self.fail(f"Custom ports test failed: {e}")

    def test_different_inputs(self):
        """测试不同输入"""
        if self.num_stages < 2:
            self.skipTest("Need at least 2 stages")
        
        test_inputs = [
            "Hello",
            "What is the meaning of life?",
            "Once upon a time",
        ]
        
        print(f"\n--- Test: Different Inputs ---")
        
        for test_input in test_inputs:
            print(f"\nTesting input: {test_input}")
            try:
                output = test_pipeline_multiprocess(
                    pipeline_dir=str(self.pipeline_dir),
                    num_stages=self.num_stages,
                    test_input=test_input,
                    device="cpu",
                )
                print(f"Output: {output}")
                self.assertIsNotNone(output)
                self.assertIsInstance(output, str)
            except Exception as e:
                self.fail(f"Test failed for input '{test_input}': {e}")


def run_tests():
    """运行测试"""
    # 检查环境变量
    pipeline_dir = os.environ.get("TEST_PIPELINE_DIR")
    if not pipeline_dir:
        print("=" * 60)
        print("WARNING: TEST_PIPELINE_DIR environment variable not set")
        print("=" * 60)
        print("\nTo run tests, set the environment variable:")
        print("  export TEST_PIPELINE_DIR=/path/to/pipeline/export")
        print("\nExample:")
        print("  export TEST_PIPELINE_DIR=/home/yanying/pipeline_export/your_model")
        print("=" * 60)
        return
    
    # 运行测试
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()

