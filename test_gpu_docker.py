#!/usr/bin/env python3
"""
GPU Test Script for Docker Container
"""

def test_gpu_availability():
    """Test if GPU is available in container"""
    try:
        import torch
        print("🐍 Python and PyTorch loaded successfully")
        print(f"📦 PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"🎮 CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"🔢 GPU device count: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"🚀 GPU {i}: {gpu_name}")
                print(f"💾 GPU {i} Memory: {memory_gb:.1f} GB")
            
            # Test tensor operations on GPU
            print("🧪 Testing GPU tensor operations...")
            device = torch.device("cuda")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print(f"✅ GPU tensor operation successful: {z.shape}")
            
            # Test memory allocation
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"📊 GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
            
        else:
            print("❌ No GPU available - will use CPU")
            
        return cuda_available
        
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing GPU availability in Docker container...")
    gpu_ok = test_gpu_availability()
    
    if gpu_ok:
        print("✅ GPU test passed - container ready for GPU acceleration!")
    else:
        print("⚠️  GPU not available - container will use CPU only")