"""
Test script to verify TensorRT, CUDA, and PyCUDA installation
"""

import os
import sys

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        module = __import__(module_name)
        print(f"✅ {module_name} is installed (version: {getattr(module, '__version__', 'unknown')})")
        return module
    except ImportError as e:
        print(f"❌ {module_name} is not installed or had import errors: {e}")
        return None

def check_cuda():
    """Check CUDA installation"""
    print("\n--- Checking CUDA Installation ---")
    
    # Check CUDA environment variables
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✅ CUDA_PATH is set to: {cuda_path}")
    else:
        print("❌ CUDA_PATH environment variable is not set")
    
    # Check PyCUDA
    pycuda = check_module("pycuda.driver")
    if pycuda:
        import pycuda.driver as cuda
        try:
            cuda.init()
            device_count = cuda.Device.count()
            print(f"✅ PyCUDA initialized successfully, detected {device_count} CUDA device(s)")
            
            for i in range(device_count):
                device = cuda.Device(i)
                print(f"  - Device {i}: {device.name()} (Compute Capability: {device.compute_capability()})")
                print(f"    Total Memory: {device.total_memory() / (1024**3):.2f} GB")
        except Exception as e:
            print(f"❌ PyCUDA initialization failed: {e}")
    
    return pycuda is not None

def check_tensorrt():
    """Check TensorRT installation"""
    print("\n--- Checking TensorRT Installation ---")
    
    tensorrt = check_module("tensorrt")
    if tensorrt:
        import tensorrt as trt
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            print(f"✅ TensorRT Builder created successfully")
            
            # Check TensorRT capabilities
            print("TensorRT Capabilities:")
            if hasattr(builder, 'platform_has_fast_fp16'):
                print(f"  - FP16 Support: {builder.platform_has_fast_fp16}")
            if hasattr(builder, 'platform_has_fast_int8'):
                print(f"  - INT8 Support: {builder.platform_has_fast_int8}")
        except Exception as e:
            print(f"❌ TensorRT initialization failed: {e}")
    
    return tensorrt is not None

def check_opencv():
    """Check OpenCV installation"""
    print("\n--- Checking OpenCV Installation ---")
    
    cv2 = check_module("cv2")
    if cv2:
        try:
            # Check if CUDA support is enabled in OpenCV
            build_info = cv2.getBuildInformation()
            if "NVIDIA CUDA" in build_info and "YES" in build_info.split("NVIDIA CUDA")[1].split("\n")[0]:
                print("✅ OpenCV is built with CUDA support")
            else:
                print("⚠️ OpenCV is installed but not built with CUDA support")
        except Exception as e:
            print(f"❌ OpenCV check failed: {e}")
    
    return cv2 is not None

def check_numpy():
    """Check NumPy installation"""
    print("\n--- Checking NumPy Installation ---")
    return check_module("numpy") is not None

def main():
    """Run all checks"""
    print("=" * 60)
    print("TensorRT Environment Test")
    print("=" * 60)
    
    # Basic info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check required modules
    check_numpy()
    check_opencv()
    cuda_ok = check_cuda()
    tensorrt_ok = check_tensorrt()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    if cuda_ok and tensorrt_ok:
        print("✅ All critical components are installed and working!")
        print("You can proceed with running the TensorRT application.")
    else:
        print("❌ There are issues with your installation:")
        if not cuda_ok:
            print("  - CUDA/PyCUDA is not properly installed or configured")
        if not tensorrt_ok:
            print("  - TensorRT is not properly installed or configured")
        print("\nPlease fix these issues before running the TensorRT application.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()