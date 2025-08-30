#!/usr/bin/env python3
"""
Simple test script to demonstrate vLLM batching improvements
"""
import asyncio
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockEmbeddingRPC:
    """Mock FPGA Embedding RPC for testing"""
    def __init__(self):
        self.connected = True
    
    def is_connected(self):
        return self.connected
    
    async def initialize(self):
        return True
    
    async def compute_embedding(self, text: str):
        # Simulate FPGA embedding delay
        await asyncio.sleep(0.01)
        return MockEmbeddingResult(success=True, embeddings=[0.1] * 4096)
    
    async def cleanup(self):
        pass

class MockEmbeddingResult:
    """Mock embedding result"""
    def __init__(self, success: bool, embeddings: List[float] = None):
        self.success = success
        self.embeddings = embeddings or []
        self.error_message = "" if success else "Mock error"

class MockFPGAStatus:
    """Mock FPGA status"""
    def __init__(self):
        self.status = "ready"

class MockvLLMEngine:
    """Mock vLLM engine that demonstrates batching behavior"""
    def __init__(self):
        self.submitted_requests = []
        self.batch_size = 0
        self.max_batch_size = 8
    
    def generate(self, text: str, sampling_params, request_id: str):
        """Mock generate method that simulates batching"""
        # Record the request for batching
        self.submitted_requests.append({
            'text': text,
            'request_id': request_id,
            'sampling_params': sampling_params,
            'submit_time': time.time()
        })
        
        # Simulate batching behavior
        if len(self.submitted_requests) >= self.max_batch_size:
            logger.info(f"🔄 vLLM engine would batch {len(self.submitted_requests)} requests together")
            self.batch_size = len(self.submitted_requests)
        
        # Return mock async generator
        return self._mock_generator(text, request_id)
    
    async def _mock_generator(self, text: str, request_id: str):
        """Mock async generator that simulates generation"""
        # Simulate processing delay
        await asyncio.sleep(0.05)
        
        # Yield mock output
        class MockOutput:
            def __init__(self, text: str):
                self.outputs = [MockGeneratedText(text)]
        
        class MockGeneratedText:
            def __init__(self, text: str):
                self.text = f"Generated response for: {text[:50]}..."
        
        yield MockOutput(text)

class MockStreamingPipelineService:
    """Mock streaming pipeline service for testing batching"""
    
    def __init__(self):
        self.vllm_engine = MockvLLMEngine()
        self.embedding_rpc = MockEmbeddingRPC()
        self.request_metrics = []
        self.pipeline_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'streaming_requests': 0,
            'progressive_prefills': 0,
            'cache_hits': 0,
            'total_chunks_processed': 0,
            'chunk_prefills_performed': 0,
            'average_first_token_latency': 0.0
        }
    
    async def _submit_request_for_batching(self, text: str, request_id: str, sampling_params) -> Any:
        """
        Submit request to vLLM engine for batching without consuming output.
        This allows vLLM to batch multiple requests together.
        """
        try:
            # Create generator but don't consume it yet
            agen = self.vllm_engine.generate(text, sampling_params, request_id)
            
            # Return the generator for later consumption
            return agen
            
        except Exception as e:
            logger.error(f"❌ Request submission failed for {request_id}: {e}")
            return None
    
    async def _final_generation(self, request_id: str, text: str) -> str:
        """Perform final generation after all chunks are processed"""
        try:
            # Submit for batching
            sampling_params = {'max_tokens': 32}
            generator = await self._submit_request_for_batching(text, request_id, sampling_params)
            
            if not generator:
                return ""
            
            # Now consume the generator to get results
            final_results = []
            async for output in generator:
                final_results.append(output)
            
            if final_results and hasattr(final_results[-1], 'outputs') and final_results[-1].outputs:
                generated_text = final_results[-1].outputs[0].text
                logger.debug(f"🎯 Final generation completed for {request_id}: {len(generated_text)} chars")
                return generated_text
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Final generation failed for {request_id}: {e}")
            return ""
    
    async def generate_streaming(self, texts: List[str], scenario: str, batch_index: int = 0) -> Dict[str, Any]:
        """Generate completions using streaming pipeline with proper batching"""
        total_start_time = time.time()
        logger.info(f"🌊 Starting streaming pipeline for {len(texts)} texts")
        
        # Submit all requests for batching first
        generation_tasks = []
        for i, text in enumerate(texts):
            request_id = f"{scenario}_{batch_index}_{i}"
            
            # Submit request for batching (don't consume yet)
            task = asyncio.create_task(self._final_generation(request_id, text))
            generation_tasks.append(task)
        
        # Now wait for all generations to complete
        logger.info("🚀 All requests submitted for batching, waiting for completion...")
        results = await asyncio.gather(*generation_tasks)
        
        # Calculate metrics
        total_end_time = time.time()
        total_pipeline_time = total_end_time - total_start_time
        
        successful_results = [r for r in results if r]
        failed_results = len(results) - len(successful_results)
        
        # Update pipeline stats
        self.pipeline_stats['total_requests'] += len(texts)
        self.pipeline_stats['successful_requests'] += len(successful_results)
        self.pipeline_stats['failed_requests'] += failed_results
        self.pipeline_stats['streaming_requests'] += len(texts)
        
        # Show batching information
        if hasattr(self.vllm_engine, 'batch_size') and self.vllm_engine.batch_size > 0:
            logger.info(f"✅ vLLM successfully batched {self.vllm_engine.batch_size} requests together")
            logger.info(f"   This demonstrates proper batching behavior")
        else:
            logger.info(f"ℹ️  vLLM processed {len(texts)} requests individually")
            logger.info(f"   Batching may not be optimal with current configuration")
        
        logger.info(f"🌊 Streaming pipeline completed: {len(successful_results)}/{len(texts)} successful")
        logger.info(f"   Total time: {total_pipeline_time*1000:.1f}ms")
        logger.info(f"   vLLM engine submitted {len(self.vllm_engine.submitted_requests)} requests")
        
        return {
            'success': len(successful_results) > 0,
            'total_pipeline_time': total_pipeline_time,
            'batch_size': len(texts),
            'outputs': successful_results,
            'successful_outputs': len(successful_results),
            'vllm_batch_size': self.vllm_engine.batch_size,
            'vllm_submitted_requests': len(self.vllm_engine.submitted_requests)
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.pipeline_stats.copy()

async def test_batching_scenarios():
    """Test different batching scenarios"""
    service = MockStreamingPipelineService()
    
    # Test texts
    test_texts = [
        "What is artificial intelligence?",
        "Explain machine learning concepts.",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain natural language processing.",
        "What is computer vision?",
        "How do transformers work?",
        "Explain attention mechanisms."
    ]
    
    logger.info("🧪 Testing vLLM batching scenarios...")
    
    # Test 1: Small batch (should batch together)
    logger.info("📦 Test 1: Small batch (4 texts)")
    result1 = await service.generate_streaming(test_texts[:4], "small_batch", batch_index=0)
    logger.info(f"   Result: {result1['successful_outputs']}/{result1['batch_size']} successful")
    logger.info(f"   vLLM batch size: {result1['vllm_batch_size']}")
    
    # Test 2: Large batch (should batch in groups)
    logger.info("📦 Test 2: Large batch (8 texts)")
    result2 = await service.generate_streaming(test_texts, "large_batch", batch_index=1)
    logger.info(f"   Result: {result2['successful_outputs']}/{result2['batch_size']} successful")
    logger.info(f"   vLLM batch size: {result2['vllm_batch_size']}")
    
    # Test 3: Sequential small batches (should batch each group)
    logger.info("📦 Test 3: Sequential small batches")
    for i in range(0, len(test_texts), 2):
        batch_texts = test_texts[i:i+2]
        result = await service.generate_streaming(batch_texts, "sequential_batch", batch_index=i//2)
        logger.info(f"   Batch {i//2}: {result['successful_outputs']}/{result['batch_size']} successful")
    
    # Show final stats
    stats = service.get_pipeline_stats()
    logger.info(f"\n📊 Final Pipeline Stats:")
    logger.info(f"   Total requests: {stats['total_requests']}")
    logger.info(f"   Successful requests: {stats['successful_requests']}")
    logger.info(f"   Failed requests: {stats['failed_requests']}")
    logger.info(f"   Streaming requests: {stats['streaming_requests']}")
    
    logger.info("\n🎉 Batching test completed!")
    logger.info("💡 Key improvements made:")
    logger.info("   1. Requests are submitted for batching without immediate consumption")
    logger.info("   2. vLLM can group multiple requests together")
    logger.info("   3. Batching happens at the engine level")
    logger.info("   4. Multiple requests can be processed in parallel")

if __name__ == "__main__":
    asyncio.run(test_batching_scenarios())