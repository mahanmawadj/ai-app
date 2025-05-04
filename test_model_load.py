"""
Super simple test script just to load a TensorRT model
"""

import os
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Create logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # Use VERBOSE for more detailed logs

def load_engine(engine_path):
    """
    Load a TensorRT engine file
    """
    print(f"Loading TensorRT engine: {engine_path}")
    
    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        if not engine:
            print("Failed to load engine")
            return None
            
        print("Engine loaded successfully")
        return engine
    except Exception as e:
        print(f"Error loading engine: {e}")
        return None

def inspect_engine(engine):
    """
    Inspect a TensorRT engine
    """
    print("\n--- Engine Information ---")
    
    # Print engine properties
    print(f"TensorRT Engine Properties:")
    
    try:
        # In newer TensorRT versions, we use different methods
        # Print engine version
        print(f"TensorRT Version: {trt.__version__}")
        
        # Try different ways to get tensor information
        try:
            print("\nTrying tensor inspection method 1:")
            for i in range(10):  # Try up to 10 tensors
                try:
                    name = engine.get_tensor_name(i)
                    mode = engine.get_tensor_mode(name)
                    shape = engine.get_tensor_shape(name)
                    print(f"  Tensor {i}: {name}, mode={mode}, shape={shape}")
                except:
                    break
        except Exception as e:
            print(f"  Method 1 failed: {e}")
        
        try:
            print("\nTrying tensor inspection method 2:")
            if hasattr(engine, 'num_io_tensors'):
                num_tensors = engine.num_io_tensors
                print(f"  Found {num_tensors} I/O tensors via attribute")
        except Exception as e:
            print(f"  Method 2 failed: {e}")
            
        try:
            print("\nTrying tensor inspection method 3:")
            if hasattr(engine, 'num_bindings'):
                num_bindings = engine.num_bindings
                print(f"  Found {num_bindings} bindings via attribute")
                
                for i in range(num_bindings):
                    name = engine.get_binding_name(i)
                    is_input = engine.binding_is_input(i)
                    shape = engine.get_binding_shape(i)
                    print(f"  Binding {i}: {name}, input={is_input}, shape={shape}")
        except Exception as e:
            print(f"  Method 3 failed: {e}")
            
    except Exception as e:
        print(f"Error inspecting engine: {e}")

def delete_engine_file(engine_path):
    """
    Delete the engine file
    """
    try:
        if os.path.exists(engine_path):
            print(f"Deleting engine file: {engine_path}")
            os.remove(engine_path)
            print("Engine file deleted")
        else:
            print("Engine file does not exist")
    except Exception as e:
        print(f"Error deleting engine file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test TensorRT engine loading')
    parser.add_argument('--model', required=True, help='Path to ONNX or TensorRT engine file')
    parser.add_argument('--delete', action='store_true', help='Delete the engine file after loading')
    
    args = parser.parse_args()
    
    # Check if the model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # If the model is an ONNX file, we need to convert it
    if args.model.endswith('.onnx'):
        print(f"ONNX model provided: {args.model}")
        
        # Get the engine path
        engine_path = os.path.splitext(args.model)[0] + ".engine"
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            print(f"Engine file already exists: {engine_path}")
        else:
            print(f"Engine file does not exist, will create one from ONNX model")
            print("Please use simple_classifier.py to create the engine first")
            return
    else:
        engine_path = args.model
    
    # Load the engine
    engine = load_engine(engine_path)
    
    if engine:
        # Inspect the engine
        inspect_engine(engine)
        
        # Create a context (this is an important test)
        try:
            print("\nCreating execution context...")
            context = engine.create_execution_context()
            print("Context created successfully")
            
            # Clean up
            del context
        except Exception as e:
            print(f"Error creating context: {e}")
    
    # Delete engine file if requested
    if args.delete:
        delete_engine_file(engine_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()