import torch
import gc

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("CUDA memory cleaned up")

# 使用示例
cleanup_cuda()