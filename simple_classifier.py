"""
Simple test application to verify TensorRT model loading and inference.
This is a stripped-down version of the full application to help with debugging.
Updated with better label handling for ImageNet models.
"""

import os
import cv2
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import json
import requests
import sys
# print(f"Python executable: {sys.executable}")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Python path: {sys.path}")

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def download_imagenet_labels(output_path='labels/imagenet_labels.json'):
    """
    Download ImageNet class labels and save them to a JSON file.
    
    Args:
        output_path: Path to save the labels JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # URL for ImageNet class labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    
    print(f"Downloading ImageNet labels from {url}")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Parse labels (one per line)
    labels = [line.strip() for line in response.text.splitlines()]
    
    # Create dictionary with class indices as keys and labels as values
    labels_dict = {f"class_{i}": label for i, label in enumerate(labels)}
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    
    print(f"Downloaded {len(labels)} labels and saved to {output_path}")
    return labels_dict

def load_imagenet_labels(labels_path='labels/imagenet_labels.json'):
    """
    Load ImageNet class labels from a JSON file.
    If the file doesn't exist, download the labels.
    
    Args:
        labels_path: Path to the labels JSON file
        
    Returns:
        Dictionary mapping class indices to label names
    """
    abs_path = os.path.abspath(labels_path)
    print(f"Attempting to load labels from: {abs_path}")
    
    if not os.path.exists(labels_path):
        try:
            print(f"Labels file not found, downloading to {labels_path}...")
            return download_imagenet_labels(labels_path)
        except Exception as e:
            print(f"Error downloading labels: {e}")
            return {}
    
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            print(f"Successfully loaded {len(labels)} labels from {labels_path}")
            return labels
    except Exception as e:
        print(f"Error loading labels from {labels_path}: {e}")
        return {}

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for inference"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = (img / 255.0 - mean) / std
    
    # HWC to CHW format
    img = img.transpose((2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img.astype(np.float32)

def build_engine_from_onnx(onnx_path):
    """Build TensorRT engine from ONNX model"""
    print(f"Converting ONNX model to TensorRT engine: {onnx_path}")
    
    # For TensorRT 10+
    # Create NetworkDefinition using ONNX parser
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_flags) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config, \
         trt.Runtime(TRT_LOGGER) as runtime:
        
        # Configure builder
        print("Configuring TensorRT builder...")
        
        # Set FP16 mode if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 mode for faster inference")
        
        # Set memory pool limit 
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
        
        # Load and parse ONNX file
        print(f"Parsing ONNX file: {onnx_path}")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
        
        # Check network inputs and outputs
        print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"Input {i}: {input_tensor.name}, shape={input_tensor.shape}, dtype={input_tensor.dtype}")
        
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")
        
        # Build and serialize engine
        print("Building TensorRT engine... (this may take a while)")
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Deserialize
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        # Save engine to file
        engine_path = os.path.splitext(onnx_path)[0] + ".engine"
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Saved TensorRT engine to: {engine_path}")
        
        return engine

def allocate_buffers(engine):
    """Allocate device buffers for input and output - Updated for TensorRT 10.10.0.31"""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    # In TensorRT 10.10.0.31, use hardcoded tensor names
    # Create execution context
    context = engine.create_execution_context()
    
    # Hardcoded input tensor name for ResNet
    input_name = "x"
    input_shape = engine.get_tensor_shape(input_name)
    input_size = trt.volume(input_shape)
    
    # Hardcoded output tensor name for ResNet
    output_name = "191"
    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape)
    
    # Allocate memory for input
    input_mem = cuda.mem_alloc(input_size * 4)  # 4 bytes for float32
    inputs.append({
        'name': input_name,
        'shape': input_shape,
        'size': input_size,
        'mem': input_mem
    })
    bindings.append(int(input_mem))
    
    # Allocate memory for output
    output_mem = cuda.mem_alloc(output_size * 4)
    outputs.append({
        'name': output_name,
        'shape': output_shape,
        'size': output_size,
        'mem': output_mem
    })
    bindings.append(int(output_mem))
    
    print(f"Allocated memory for input: {input_name}, shape={input_shape}, size={input_size}")
    print(f"Allocated memory for output: {output_name}, shape={output_shape}, size={output_size}")
    
    return inputs, outputs, bindings, stream, context

def run_inference(context, inputs, outputs, bindings, stream, input_data):
    """Run inference with TensorRT engine - Updated for TensorRT 10.10.0.31"""
    # Set input tensor
    input_name = inputs[0]['name']
    
    # Copy input data to device
    cuda.memcpy_htod_async(inputs[0]['mem'], input_data.flatten(), stream)
    
    # Set input tensor address
    context.set_tensor_address(input_name, int(inputs[0]['mem']))
    
    # Set output tensor addresses
    for output in outputs:
        context.set_tensor_address(output['name'], int(output['mem']))
    
    # Run inference
    context.execute_async_v3(stream.handle)
    
    # Create output arrays
    output_data = []
    for output in outputs:
        output_array = np.empty(output['size'], dtype=np.float32)
        cuda.memcpy_dtoh_async(output_array, output['mem'], stream)
        output_data.append(output_array)
    
    # Synchronize the stream
    stream.synchronize()
    
    return output_data

def load_class_names(model_path):
    """Load class names from a file if available, with improved processing"""
    class_file = os.path.splitext(model_path)[0] + '.txt'
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            raw_names = [line.strip() for line in f.readlines()]
        
        # Process ImageNet labels to make them more readable
        processed_names = []
        for name in raw_names:
            # For ImageNet labels, they are often in the format "n01440764 tench, Tinca tinca"
            if ' ' in name:
                # Remove the synset ID and keep only the description
                parts = name.split(' ', 1)
                if len(parts) > 1:
                    name = parts[1]
            processed_names.append(name)
        
        print(f"Loaded {len(processed_names)} class names from {class_file}")
        return processed_names
    
    # Default ImageNet classes as fallback (shortened version)
    print(f"Class names file not found: {class_file}")
    print("Using generic class names as fallback")
    return [f"class_{i}" for i in range(1000)]

def main():
    parser = argparse.ArgumentParser(description='Simple TensorRT ONNX Classifier')
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--image', required=True, help='Path to input image file')
    parser.add_argument('--size', type=int, default=224, help='Input image size (default: 224)')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of TensorRT engine')
    parser.add_argument('--labels', type=str, help='Path to class labels file (if different from model name)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # Load image
    input_data = load_image(args.image, (args.size, args.size))
    print(f"Loaded image with shape: {input_data.shape}")
    
    # Build or load TensorRT engine
    engine_path = os.path.splitext(args.model)[0] + ".engine"
    if os.path.exists(engine_path) and not args.rebuild:
        print(f"Loading existing TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        print(f"Building TensorRT engine from ONNX: {args.model}")
        engine = build_engine_from_onnx(args.model)
    
    # Allocate buffers
    inputs, outputs, bindings, stream, context = allocate_buffers(engine)
    
    labels = load_imagenet_labels()

    # Run inference
    print("Running inference...")
    output_data = run_inference(context, inputs, outputs, bindings, stream, input_data)
    
    # Process output
    output = output_data[0].reshape(outputs[0]['shape'])
    
    # Use custom labels file if provided
    if args.labels:
        class_names = load_class_names(args.labels)
    else:
        class_names = load_class_names(args.model)
    
    # Get top 5 predictions
    top_indices = np.argsort(output[0])[-5:][::-1]
    print("Top 5 predictions:")
    for i, idx in enumerate(top_indices):
        class_id = int(idx)
        prob = float(output[0][idx]) * 100  # Convert to percentage
        # Try to get the label from our downloaded labels
        label = labels.get(f"class_{class_id}")
        if label:
            print(f"{i+1}. {label}: {prob:.2f}%")
        else:
            print(f"{i+1}. class_{class_id}: {prob:.2f}%")
    
    print(f"\nPrediction complete for: {args.image}")
    # Use the label from our downloaded labels for the top prediction if available
    top_label = labels.get(f"class_{top_indices[0]}", class_names[top_indices[0]])
    print(f"Top prediction: {top_label} ({output[0][top_indices[0]]:.4f})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")