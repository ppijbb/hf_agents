from vllm import LLM
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "0"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "5"
os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-29"


model  = LLM(
    model = "Gunulhona/Gemma-Ko-Merge",
    # quantization="bitsandbytes",
    # load_format="bitsandbytes",
    max_model_len=4096,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    dtype="float16",
    distributed_executor_backend="ray",
)