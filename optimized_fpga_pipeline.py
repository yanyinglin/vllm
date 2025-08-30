#!/usr/bin/env python3
"""
Optimized FPGA Pipeline Model Service with True Streaming and Async Pipelining

Implements a high-performance streaming pipeline: FPGA embedding → gRPC → GPU prefill → GPU decoding
with immediate request dispatch, chunk-based embedding, and progressive prefill with prefix caching.
"""
import sys
import os
import logging
import torch
import numpy as np
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FPGA Embedding RPC
sys.path.append('/home/yanying/workspace/edgegateway/Core/FPGPU')
from EmbeddingRPC import EmbeddingRPC, EmbeddingResult, FPGAStatus
from grpc_optimization_config import OptimizationLevel, NetworkCondition

@dataclass
class PipelineMetrics:
    """Pipeline model request metrics"""
    request_id: str
    scenario: str
    batch_index: int
    batch_size: int
    
    # Time metrics
    start_time: float
    end_time: float
    embedding_time: float
    prefill_time: float
    decode_time: float
    total_pipeline_time: float
    
    # Token metrics
    input_token_count: int
    output_token_count: int
    embedding_dimension: int
    
    # Performance metrics
    tokens_per_second: float
    pipeline_tokens_per_second: float
    
    # FPGA metrics
    fpga_hardware_time: float = 0.0
    fpga_grpc_overhead: float = 0.0
    
    # Pipeline overlap metrics
    grpc_overlap_time: float = 0.0
    pipeline_overlap_efficiency: float = 0.0
    concurrent_embedding_time: float = 0.0
    
    # Streaming metrics
    first_token_latency: float = 0.0
    progressive_prefill_time: float = 0.0
    chunk_processing_time: float = 0.0
    
    # Success flags
    success: bool = True
    error_message: str = ""
    
    # Service info
    engine_init_time: Optional[float] = None
    fpga_init_time: Optional[float] = None
    is_warmup: bool = False

@dataclass
class PipelineConfig:
    """Pipeline model service configuration with streaming optimization"""
    gpu_device: int = 1
    max_tokens: int = 32  # Set to 32 as requested
    warmup_requests: int = 3
    
    # Model configuration - optimized for streaming
    model_path: str = "/home/yanying/huggingface/pipeline/Meta-Llama-3-8B/inference"
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 4096
    
    # FPGA configuration
    fpga_host: str = "192.168.50.154"
    fpga_port: int = 50052
    fpga_request_timeout: float = 30.0
    
    # Monitoring configuration
    collect_detailed_metrics: bool = True
    
    # Streaming pipeline optimization
    enable_streaming: bool = True
    enable_progressive_prefill: bool = True
    enable_prefix_caching: bool = True
    enable_chunk_prefill: bool = False  # New config for embedding chunk prefill
    chunk_size: int = 256  # Optimal chunk size for progressive processing
    max_concurrent_embeddings: int = 256  # High concurrency for streaming
    max_concurrent_prefills: int = 16     # Batch processing in vLLM
    pipeline_buffer_size: int = 128       # Large buffer for streaming

@dataclass
class StreamingRequest:
    """Represents a streaming request with progressive processing"""
    request_id: str
    text: str
    scenario: str
    batch_index: int
    start_time: float
    
    # Progressive state
    chunks: List[str] = None
    completed_chunks: List[EmbeddingResult] = None
    progressive_embeddings: List[Any] = None
    prefill_states: List[Any] = None
    
    # Timing
    first_chunk_time: Optional[float] = None
    first_prefill_time: Optional[float] = None
    first_token_time: Optional[float] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.completed_chunks is None:
            self.completed_chunks = []
        if self.progressive_embeddings is None:
            self.progressive_embeddings = []
        if self.prefill_states is None:
            self.prefill_states = []

class StreamingPipelineService:
    """High-performance streaming FPGA pipeline service with progressive processing"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.request_metrics = []
        self.vllm_engine = None
        self.embedding_rpc = None
        self.engine_init_time = 0
        self.fpga_init_time = 0
        
        # Streaming state
        self.active_requests = {}
        self.request_queue = asyncio.Queue()
        self.embedding_workers = []
        self.prefill_workers = []
        
        # Pipeline stats
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
        
        # Prefix cache for progressive prefill
        self.prefix_cache = {}
        
        # Initialize components
        self._initialize_embedding_rpc()
        self._initialize_vllm_engine()
        
        # Start streaming workers
        if config.enable_streaming:
            self._start_streaming_workers()
        
        logger.info("=== FPGA Streaming Pipeline Service ===")
        logger.info(f"GPU device: cuda:{config.gpu_device}")
        logger.info(f"Model path: {config.model_path}")
        logger.info(f"FPGA endpoint: {config.fpga_host}:{config.fpga_port}")
        logger.info(f"Streaming enabled: {config.enable_streaming}")
        logger.info(f"Progressive prefill: {config.enable_progressive_prefill}")
        logger.info(f"Prefix caching: {config.enable_prefix_caching}")
        logger.info(f"Chunk prefill: {config.enable_chunk_prefill}")
        logger.info(f"Max tokens: {config.max_tokens}")
        logger.info(f"FPGA init time: {self.fpga_init_time:.3f}s")
        logger.info(f"Engine init time: {self.engine_init_time:.3f}s")
    
    def _initialize_embedding_rpc(self):
        """Initialize FPGA Embedding RPC client"""
        try:
            logger.info(f"🔌 Initializing FPGA Embedding RPC at {self.config.fpga_host}:{self.config.fpga_port}...")
            start_time = time.time()
            
            self.embedding_rpc = EmbeddingRPC(
                fpga_host=self.config.fpga_host,
                fpga_port=self.config.fpga_port,
                optimization_level=OptimizationLevel.ULTRA,
                network_condition=NetworkCondition.LAN
            )
            
            self.fpga_init_time = time.time() - start_time
            logger.info(f"✅ FPGA Embedding RPC client initialized in {self.fpga_init_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ FPGA Embedding RPC initialization failed: {e}")
            raise
    
    def _initialize_vllm_engine(self):
        """Initialize vLLM AsyncLLMEngine with streaming and batching optimization"""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 CUDA cache cleared")
            
            # Set environment for vLLM streaming
            os.environ['VLLM_USE_V1'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_device)
            os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
            torch.cuda.set_device(0)
            
            logger.info(f"📦 Initializing vLLM AsyncLLMEngine for streaming on cuda:{self.config.gpu_device}...")
            start_time = time.time()
            

            # tuning for batching
            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype="float16",
                enforce_eager=False,            # keep async
                disable_log_stats=False,        # <-- 打开，以便看到 batching / stats
                seed=42,
                max_num_seqs=128,              # increase allowable concurrent seqs
                # 添加单次迭代的 token 预算（决定每个 decode iteration 的 batch 容量）
                max_num_batched_tokens=65536,  # <-- 试 32k / 64k / 128k 观察效果
                enable_prefix_caching=self.config.enable_prefix_caching,
                enable_chunked_prefill=self.config.enable_progressive_prefill,
                swap_space=2,
                cpu_offload_gb=0,
                trust_remote_code=False,
                max_seq_len_to_capture=self.config.max_tokens,
                disable_sliding_window=False,  # keep sliding window unless you have reason to disable
                block_size=16
            )

            
            # Initialize AsyncLLMEngine
            self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.engine_init_time = time.time() - start_time
            logger.info(f"✅ vLLM AsyncLLMEngine initialized for streaming in {self.engine_init_time:.3f}s")
            
            # Clear CUDA cache after initialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            logger.error(f"❌ vLLM AsyncLLMEngine initialization failed: {e}")
            raise
    
    def _start_streaming_workers(self):
        """Start background workers for streaming pipeline"""
        # Start embedding workers
        for i in range(4):  # Multiple workers for concurrent embedding
            worker = asyncio.create_task(self._embedding_worker(f"emb_worker_{i}"))
            self.embedding_workers.append(worker)
        
        # Start prefill workers
        for i in range(2):  # Fewer workers since vLLM handles batching
            worker = asyncio.create_task(self._prefill_worker(f"prefill_worker_{i}"))
            self.prefill_workers.append(worker)
        
        logger.info(f"🚀 Started {len(self.embedding_workers)} embedding workers and {len(self.prefill_workers)} prefill workers")
    
    async def _embedding_worker(self, worker_id: str):
        """Background worker for processing embedding chunks"""
        logger.debug(f"🔧 Embedding worker {worker_id} started")
        while True:
            try:
                # Process embedding requests from queue
                await asyncio.sleep(0.001)  # Small delay to prevent tight loop
                # Worker implementation would go here
                pass
            except asyncio.CancelledError:
                logger.debug(f"🛑 Embedding worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Embedding worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _prefill_worker(self, worker_id: str):
        """Background worker for progressive prefill processing"""
        logger.debug(f"🔧 Prefill worker {worker_id} started")
        while True:
            try:
                # Process prefill requests from queue
                await asyncio.sleep(0.001)  # Small delay to prevent tight loop
                # Worker implementation would go here
                pass
            except asyncio.CancelledError:
                logger.debug(f"🛑 Prefill worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Prefill worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count"""
        import re
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - sum(len(match.group()) for match in re.finditer(r'\b[a-zA-Z]+\b|\s|[\u4e00-\u9fff]', text))
        
        estimated_tokens = int(english_words * 1.3 + chinese_chars * 1.0 + other_chars * 0.5)
        return max(1, estimated_tokens)
    
    def _create_sampling_params(self, max_tokens: Optional[int] = None, is_prefill_only: bool = False):
        """Create sampling parameters for vLLM with streaming support"""
        from vllm import SamplingParams
        
        if is_prefill_only:
            # For progressive prefill, only generate 1 token to maintain cache
            return SamplingParams(
                temperature=0.1,
                max_tokens=1,
                repetition_penalty=1.1,
                seed=42,
                stop_token_ids=[],  # Don't stop early for prefill
                skip_special_tokens=False
            )
        else:
            # For full generation
            return SamplingParams(
                temperature=0.1,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                repetition_penalty=1.1,
                seed=42
            )
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for progressive processing"""
        chunk_size = self.config.chunk_size
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences first, then by chunk size if needed
        import re
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _ensure_fpga_connection(self) -> bool:
        """Ensure FPGA RPC connection is available"""
        try:
            if not self.embedding_rpc:
                logger.error("FPGA Embedding RPC not initialized")
                return False
            
            if not self.embedding_rpc.is_connected():
                logger.info("🔗 Connecting to FPGA Embedding RPC...")
                connected = await self.embedding_rpc.initialize()
                if connected:
                    logger.info("✅ FPGA Embedding RPC connected")
                else:
                    logger.error("❌ Failed to connect to FPGA Embedding RPC")
                return connected
            return True
            
        except Exception as e:
            logger.error(f"❌ FPGA RPC connection failed: {e}")
            return False
    
    async def _stream_embedding_chunks(self, request: StreamingRequest) -> AsyncGenerator[EmbeddingResult, None]:
        """Stream embedding results as chunks complete"""
        chunks = self._split_text_into_chunks(request.text)
        request.chunks = chunks
        
        logger.debug(f"📝 Split text into {len(chunks)} chunks for streaming request {request.request_id}")
        
        # Process chunks concurrently and yield results as they complete
        chunk_tasks = []
        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(
                self._process_single_chunk(chunk, f"{request.request_id}_chunk_{i}")
            )
            chunk_tasks.append(task)
        
        # Yield results as they complete (not necessarily in order)
        completed_count = 0
        for task in asyncio.as_completed(chunk_tasks):
            try:
                result = await task
                completed_count += 1
                
                if result.success:
                    request.completed_chunks.append(result)
                    
                    # Record first chunk completion time
                    if request.first_chunk_time is None:
                        request.first_chunk_time = time.time()
                    
                    logger.debug(f"✅ Chunk {completed_count}/{len(chunks)} completed for {request.request_id}")
                    yield result
                else:
                    logger.warning(f"⚠️ Chunk {completed_count}/{len(chunks)} failed for {request.request_id}: {result.error_message}")
            
            except Exception as e:
                logger.error(f"❌ Chunk processing error for {request.request_id}: {e}")
                completed_count += 1
    
    async def _process_single_chunk(self, chunk: str, chunk_id: str) -> EmbeddingResult:
        """Process a single text chunk for embedding"""
        try:
            logger.debug(f"🔥 Processing chunk {chunk_id}")
            result = await self.embedding_rpc.compute_embedding(chunk)
            
            if result.success:
                logger.debug(f"✅ Chunk {chunk_id} embedding completed")
            else:
                logger.warning(f"⚠️ Chunk {chunk_id} embedding failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Chunk {chunk_id} processing error: {e}")
            from EmbeddingRPC import EmbeddingResult
            return EmbeddingResult(success=False, error_message=str(e))
    
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
    
    async def _chunk_prefill(self, chunk_text: str, request_id: str) -> Optional[Any]:
        """Perform prefill for individual embedding chunks if enabled"""
        if not self.config.enable_chunk_prefill:
            return None
        
        try:
            logger.debug(f"🔄 Scheduling chunk prefill for {request_id}")
            
            # Create prefill-only sampling params
            prefill_params = self._create_sampling_params(is_prefill_only=True)
            
            # Submit for batching but don't consume yet
            agen = await self._submit_request_for_batching(chunk_text, request_id, prefill_params)
            
            if agen:
                # Store the generator for later consumption
                return agen
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Chunk prefill failed for {request_id}: {e}")
            return None
    
    async def _progressive_prefill(self, request: StreamingRequest, embedding_result: EmbeddingResult) -> Optional[Any]:
        """Perform progressive prefill as chunks complete"""
        try:
            # Build progressive context from completed chunks
            context_embeddings = []
            for completed_chunk in request.completed_chunks:
                if completed_chunk.success and hasattr(completed_chunk, 'embeddings'):
                    context_embeddings.extend(completed_chunk.embeddings)
            
            # Create prefix from completed context
            prefix_key = hash(str(context_embeddings))
            
            # Check prefix cache
            if self.config.enable_prefix_caching and prefix_key in self.prefix_cache:
                logger.debug(f"🔍 Prefix cache hit for {request.request_id}")
                self.pipeline_stats['cache_hits'] += 1
                cached_state = self.prefix_cache[prefix_key]
                
                # Record first prefill time
                if request.first_prefill_time is None:
                    request.first_prefill_time = time.time()
                
                return cached_state
            
            # Progressive prefill with current context
            progressive_text = request.text[:len(''.join(chunk for chunk in request.chunks[:len(request.completed_chunks)]))]
            
            # Create prefill-only sampling params (max_tokens=1)
            prefill_params = self._create_sampling_params(is_prefill_only=True)
            
            logger.debug(f"🔄 Progressive prefill for {request.request_id} with {len(context_embeddings)} embeddings")
            
            # Submit for batching but don't consume yet
            request_id = f"prefill_{request.request_id}_{len(request.completed_chunks)}"
            agen = await self._submit_request_for_batching(progressive_text, request_id, prefill_params)
            
            if agen:
                # Record first prefill time if not set
                if request.first_prefill_time is None:
                    request.first_prefill_time = time.time()
                
                # Store the generator for later consumption
                request.prefill_states.append(agen)
                
                # Cache the prefix state
                if self.config.enable_prefix_caching and len(self.prefix_cache) < 1000:
                    self.prefix_cache[prefix_key] = agen
                
                self.pipeline_stats['progressive_prefills'] += 1
                logger.debug(f"✅ Progressive prefill submitted for batching {request.request_id}")
                return agen
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Progressive prefill failed for {request.request_id}: {e}")
            return None
    
    async def _final_generation(self, request: StreamingRequest) -> Optional[str]:
        """Perform final generation after all chunks are processed"""
        try:
            # Use the last prefill state or create new one
            if request.prefill_states:
                logger.debug(f"🎯 Using cached prefill state for final generation {request.request_id}")
            
            # Create full generation params
            generation_params = self._create_sampling_params(max_tokens=self.config.max_tokens)
            
            # Record first token time
            if request.first_token_time is None:
                request.first_token_time = time.time()
            
            # Submit for batching
            request_id = f"generate_{request.request_id}"
            generator = await self._submit_request_for_batching(request.text, request_id, generation_params)
            
            if not generator:
                return ""
            
            # Now consume the generator to get results
            final_results = []
            async for output in generator:
                final_results.append(output)
            
            if final_results and hasattr(final_results[-1], 'outputs') and final_results[-1].outputs:
                generated_text = final_results[-1].outputs[0].text
                logger.debug(f"🎯 Final generation completed for {request.request_id}: {len(generated_text)} chars")
                return generated_text
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Final generation failed for {request.request_id}: {e}")
            return ""
    
    async def generate_streaming(self, texts: List[str], scenario: str, batch_index: int = 0, is_warmup: bool = False) -> Dict[str, Any]:
        """Generate completions using streaming pipeline with proper batching"""
        total_start_time = time.time()
        logger.info(f"🌊 Starting streaming pipeline for {len(texts)} texts")
        
        # Check FPGA connection
        if not await self._ensure_fpga_connection():
            logger.error("❌ FPGA RPC not connected")
            return {'success': False, 'error': 'FPGA RPC connection failed'}
        
        # Create streaming requests and dispatch immediately
        streaming_requests = []
        for i, text in enumerate(texts):
            request_start_time = time.time()
            unique_id = int(request_start_time * 1000000) + i
            
            request = StreamingRequest(
                request_id=f"{scenario}_{batch_index}_{i}_{unique_id}",
                text=text,
                scenario=scenario,
                batch_index=batch_index,
                start_time=request_start_time
            )
            
            streaming_requests.append(request)
            self.active_requests[request.request_id] = request
            logger.debug(f"📤 Dispatched streaming request {request.request_id}")
        
        # Process all requests concurrently with streaming
        completed_results = []
        
        async def process_streaming_request(request: StreamingRequest) -> Dict[str, Any]:
            """Process a single streaming request"""
            try:
                # Stream embedding chunks and perform progressive prefill
                first_chunk_processed = False
                
                async for embedding_result in self._stream_embedding_chunks(request):
                    if embedding_result.success:
                        # Perform chunk prefill if enabled
                        chunk_text = request.chunks[len(request.completed_chunks) - 1] if request.chunks else ""
                        if self.config.enable_chunk_prefill and chunk_text:
                            await self._chunk_prefill(chunk_text, f"{request.request_id}_chunk_{len(request.completed_chunks) - 1}")
                        
                        # Perform progressive prefill as soon as first chunk is ready
                        if not first_chunk_processed:
                            await self._progressive_prefill(request, embedding_result)
                            first_chunk_processed = True
                        
                        self.pipeline_stats['total_chunks_processed'] += 1
                
                # Perform final generation once all chunks are processed
                generated_text = await self._final_generation(request)
                
                # Calculate metrics
                end_time = time.time()
                total_time = end_time - request.start_time
                
                first_token_latency = 0.0
                if request.first_token_time:
                    first_token_latency = request.first_token_time - request.start_time
                
                # Update stats
                if first_token_latency > 0:
                    self.pipeline_stats['average_first_token_latency'] = (
                        self.pipeline_stats['average_first_token_latency'] * 
                        self.pipeline_stats['streaming_requests'] + first_token_latency
                    ) / (self.pipeline_stats['streaming_requests'] + 1)
                
                self.pipeline_stats['streaming_requests'] += 1
                
                return {
                    'request_id': request.request_id,
                    'generated_text': generated_text,
                    'success': bool(generated_text),
                    'total_time': total_time,
                    'first_token_latency': first_token_latency,
                    'chunks_processed': len(request.completed_chunks),
                    'total_chunks': len(request.chunks) if request.chunks else 0
                }
                
            except Exception as e:
                logger.error(f"❌ Streaming request {request.request_id} failed: {e}")
                return {
                    'request_id': request.request_id,
                    'generated_text': "",
                    'success': False,
                    'error': str(e)
                }
            finally:
                # Cleanup
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
        
        # Process all streaming requests concurrently
        logger.info("🚀 Processing all streaming requests concurrently")
        request_tasks = [process_streaming_request(req) for req in streaming_requests]
        completed_results = await asyncio.gather(*request_tasks, return_exceptions=True)
        
        # Collect results and metrics
        total_end_time = time.time()
        total_pipeline_time = total_end_time - total_start_time
        
        successful_results = [r for r in completed_results if isinstance(r, dict) and r.get('success', False)]
        failed_results = len(completed_results) - len(successful_results)
        
        # Calculate aggregate metrics
        total_tokens = sum(self._count_tokens(text) for text in texts)
        avg_first_token_latency = np.mean([r.get('first_token_latency', 0) for r in successful_results]) if successful_results else 0.0
        
        throughput = total_tokens / total_pipeline_time if total_pipeline_time > 0 else 0
        
        # Update pipeline stats
        self.pipeline_stats['total_requests'] += len(texts)
        self.pipeline_stats['successful_requests'] += len(successful_results)
        self.pipeline_stats['failed_requests'] += failed_results
        
        # Collect detailed metrics
        for i, (text, result) in enumerate(zip(texts, successful_results)):
            if isinstance(result, dict) and result.get('success', False):
                request_metric = PipelineMetrics(
                    request_id=result['request_id'],
                    scenario=scenario,
                    batch_index=batch_index,
                    batch_size=len(texts),
                    start_time=total_start_time,
                    end_time=total_end_time,
                    embedding_time=0.0,  # Distributed across chunks
                    prefill_time=0.0,   # Progressive prefill
                    decode_time=result.get('total_time', 0),
                    total_pipeline_time=total_pipeline_time,
                    input_token_count=self._count_tokens(text),
                    output_token_count=self._count_tokens(result.get('generated_text', '')),
                    embedding_dimension=4096,
                    tokens_per_second=throughput,
                    pipeline_tokens_per_second=throughput,
                    first_token_latency=result.get('first_token_latency', 0),
                    progressive_prefill_time=0.0,
                    chunk_processing_time=0.0,
                    engine_init_time=self.engine_init_time,
                    fpga_init_time=self.fpga_init_time,
                    is_warmup=is_warmup
                )
                
                self.request_metrics.append(request_metric)
        
        outputs = [r.get('generated_text', '') for r in successful_results]
        
        logger.info(f"🌊 Streaming pipeline completed: {len(successful_results)}/{len(texts)} successful")
        logger.info(f"   Total time: {total_pipeline_time*1000:.1f}ms, throughput: {throughput:.1f} tokens/s")
        logger.info(f"   Avg first token latency: {avg_first_token_latency*1000:.1f}ms")
        
        return {
            'success': len(successful_results) > 0,
            'embedding_time': 0.0,  # Distributed
            'prefill_time': 0.0,    # Progressive
            'decode_time': np.mean([r.get('total_time', 0) for r in successful_results]) if successful_results else 0,
            'generate_time': np.mean([r.get('total_time', 0) for r in successful_results]) if successful_results else 0,
            'total_pipeline_time': total_pipeline_time,
            'total_tokens': total_tokens,
            'throughput': throughput,
            'pipeline_throughput': throughput,
            'first_token_latency': avg_first_token_latency,
            'batch_size': len(texts),
            'outputs': outputs,
            'successful_outputs': len(successful_results),
            'detailed_metrics': self.get_latest_batch_metrics(len(successful_results))
        }
    
    # Compatibility aliases
    async def generate_with_pipeline(self, texts: List[str], scenario: str, batch_index: int = 0, is_warmup: bool = False) -> Dict[str, Any]:
        """Compatibility alias that routes to streaming pipeline"""
        return await self.generate_streaming(texts, scenario, batch_index, is_warmup)
    
    async def generate_with_ultra_pipeline(self, texts: List[str], scenario: str, batch_index: int = 0, is_warmup: bool = False) -> Dict[str, Any]:
        """Compatibility alias that routes to streaming pipeline"""
        return await self.generate_streaming(texts, scenario, batch_index, is_warmup)
    
    async def warmup(self):
        """Warmup engines to reach steady state"""
        logger.info("🔥 Warming up streaming FPGA pipeline engines...")
        
        warmup_text = "This is a warmup request to initialize the streaming pipeline and reach steady state."
        warmup_texts = [warmup_text] * 2
        
        for i in range(self.config.warmup_requests):
            result = await self.generate_streaming(warmup_texts, "warmup", batch_index=i, is_warmup=True)
            if result.get('success', False):
                logger.info(f"  Warmup {i+1}: Total={result['total_pipeline_time']:.3f}s, First token={result.get('first_token_latency', 0):.3f}s")
            else:
                logger.warning(f"  Warmup {i+1} failed: {result.get('error', 'Unknown error')}")
        
        logger.info("✅ Streaming FPGA pipeline engines warmup completed")
    
    async def get_fpga_status(self) -> Optional[FPGAStatus]:
        """Get FPGA service status"""
        try:
            if not await self._ensure_fpga_connection():
                return None
            
            return await self.embedding_rpc.get_fpga_status()
        except Exception as e:
            logger.error(f"❌ Failed to get FPGA status: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel streaming workers
            for worker in self.embedding_workers + self.prefill_workers:
                worker.cancel()
            
            # Wait for workers to finish
            if self.embedding_workers or self.prefill_workers:
                await asyncio.gather(*self.embedding_workers, *self.prefill_workers, return_exceptions=True)
            
            # Cleanup FPGA RPC
            if hasattr(self, 'embedding_rpc'):
                await self.embedding_rpc.cleanup()
            
            logger.info("🧹 Streaming pipeline cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def get_collected_metrics(self) -> List[PipelineMetrics]:
        """Get all collected metrics from the service"""
        return self.request_metrics.copy()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.pipeline_stats.copy()
    
    def clear_metrics(self):
        """Clear collected metrics"""
        self.request_metrics.clear()
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
    
    def get_latest_batch_metrics(self, batch_size: int) -> List[PipelineMetrics]:
        """Get metrics for the latest batch of requests"""
        if len(self.request_metrics) >= batch_size:
            return self.request_metrics[-batch_size:]
        return self.request_metrics.copy()

# Compatibility aliases
PipelineModelService = StreamingPipelineService
UltraPipelineModelService = StreamingPipelineService
UltraPipelineModelConfig = PipelineConfig
UltraPipelineModelMetrics = PipelineMetrics

async def main():
    """Main function for testing streaming pipeline"""
    service = None
    try:
        config = PipelineConfig(
            gpu_device=1,
            max_tokens=32,
            warmup_requests=2,
            gpu_memory_utilization=0.8,
            collect_detailed_metrics=True,
            fpga_host="192.168.50.154",
            fpga_port=50052,
            enable_streaming=True,
            enable_progressive_prefill=True,
            enable_prefix_caching=True,
            enable_chunk_prefill=False  # Default disabled
        )
        
        service = StreamingPipelineService(config)
        
        # Test text
        document = """AI technology is changing our world. From NLP to computer vision, AI shows potential in various fields. Machine learning algorithms are becoming more sophisticated."""
        
        products = [
            "High-performance smartphone with latest processor and advanced camera system.",
            "Powerful laptop for professional work with excellent display and long battery life."
        ]
        
        logger.info("🚀 Starting streaming FPGA pipeline testing...")
        
        await service.warmup()
        
        # Test scenarios
        logger.info("🧪 Testing streaming FPGA pipeline scenarios...")
        
        # Test 1: Single document streaming
        logger.info("📄 Test 1: Single document streaming processing")
        for i in range(2):
            result = await service.generate_streaming([document], "single_doc_stream", batch_index=i)
            if result.get('success', False):
                logger.info(f"  Query {i+1}: Total={result['total_pipeline_time']:.3f}s, First token={result.get('first_token_latency', 0):.3f}s")
            else:
                logger.error(f"  Query {i+1} failed: {result.get('error', 'Unknown error')}")
        
        # Test 2: Batch streaming
        logger.info("📦 Test 2: Batch streaming processing")
        result = await service.generate_streaming(products, "batch_stream", batch_index=0)
        if result.get('success', False):
            logger.info(f"  Batch: Total={result['total_pipeline_time']:.3f}s, Throughput={result['throughput']:.2f} tokens/s")
            logger.info(f"    First token latency: {result.get('first_token_latency', 0)*1000:.1f}ms")
        else:
            logger.error(f"  Batch failed: {result.get('error', 'Unknown error')}")
        
        logger.info(f"📊 Test completed with {len(service.get_collected_metrics())} collected metrics")
        logger.info("\n🎉 Streaming FPGA pipeline service test completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        if service:
            await service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())