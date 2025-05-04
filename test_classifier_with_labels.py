"""
Test script for image classification with proper label display
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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
    
    return img.astype(np.float32), cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_engine(engine_path):
    """Load TensorRT engine"""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Allocate device buffers for input and output"""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    # Input tensor (hardcoded for ResNet)
    input_name = "x"
    input_shape = engine.get_tensor_shape(input_name)
    input_size = trt.volume(input_shape)
    
    # Output tensor (hardcoded for ResNet)
    output_name = "191"
    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape)
    
    # Allocate GPU memory for input
    input_mem = cuda.mem_alloc(input_size * 4)  # 4 bytes for float32
    inputs.append({
        'name': input_name,
        'shape': input_shape,
        'size': input_size,
        'mem': input_mem
    })
    bindings.append(int(input_mem))
    
    # Allocate GPU memory for output
    output_mem = cuda.mem_alloc(output_size * 4)
    outputs.append({
        'name': output_name,
        'shape': output_shape,
        'size': output_size,
        'mem': output_mem
    })
    bindings.append(int(output_mem))
    
    return inputs, outputs, bindings, stream

def run_inference(engine, inputs, outputs, bindings, stream, input_data):
    """Run inference with TensorRT engine"""
    # Create execution context
    context = engine.create_execution_context()
    
    # Copy input data to device
    cuda.memcpy_htod_async(inputs[0]['mem'], input_data.flatten(), stream)
    
    # Set input tensor address
    context.set_tensor_address(inputs[0]['name'], int(inputs[0]['mem']))
    
    # Set output tensor address
    context.set_tensor_address(outputs[0]['name'], int(outputs[0]['mem']))
    
    # Run inference
    context.execute_async_v3(stream.handle)
    
    # Create output array
    output_data = np.empty(outputs[0]['size'], dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, outputs[0]['mem'], stream)
    
    # Synchronize the stream
    stream.synchronize()
    
    return output_data

def load_class_names(class_file):
    """Load class names from a file"""
    if not os.path.exists(class_file):
        print(f"Class names file not found: {class_file}")
        return [f"class_{i}" for i in range(1000)]
    
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Process ImageNet labels to make them more readable
    processed_names = []
    for name in class_names:
        # For ImageNet labels, they are often in the format "n01440764 tench, Tinca tinca"
        if ' ' in name:
            # Remove the synset ID and keep only the description
            parts = name.split(' ', 1)
            if len(parts) > 1:
                name = parts[1]
        processed_names.append(name)
    
    return processed_names

def display_results(image, results, class_names):
    """Display the classification results"""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Create bar chart of top 5 predictions
    plt.subplot(1, 2, 2)
    labels = [class_names[idx] for idx, _ in results]
    values = [score for _, score in results]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.xlabel('Probability')
    plt.title('Top 5 Predictions')
    
    plt.tight_layout()
    plt.savefig("classification_result.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Image Classification')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--labels', help='Path to class labels')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Image file not found: {args.image}")
        return
    
    # Determine model path
    model_path = args.model
    if model_path.endswith('.onnx'):
        engine_path = os.path.splitext(model_path)[0] + '.engine'
        if not os.path.exists(engine_path):
            print(f"Engine file not found: {engine_path}")
            print("Please run simple_classifier.py first to create the engine")
            return
    else:
        engine_path = model_path
    
    # Determine labels path
    if args.labels:
        class_file = args.labels
    else:
        class_file = os.path.splitext(model_path)[0] + '.txt'
    
    # Load class names
    print(f"Loading class names from {class_file}")
    class_names = load_class_names(class_file)
    print(f"Loaded {len(class_names)} class names")
    
    # Load engine
    print(f"Loading TensorRT engine: {engine_path}")
    engine = load_engine(engine_path)
    
    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    input_data, original_image = load_image(args.image)
    
    # Run inference
    print("Running inference...")
    output_data = run_inference(engine, inputs, outputs, bindings, stream, input_data)
    
    # Apply softmax to get probabilities
    exp_scores = np.exp(output_data - np.max(output_data))
    probs = exp_scores / exp_scores.sum()
    
    # Get top 5 predictions
    top_indices = np.argsort(probs)[-5:][::-1]
    
    # Format results
    results = [(idx, float(probs[idx]) * 100) for idx in top_indices]
    
    # Print results
    print("\nTop 5 predictions:")
    for i, (idx, prob) in enumerate(results):
        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        print(f"{i+1}. {class_name}: {prob:.2f}%")
    
    # Display results
    display_results(original_image, results, class_names)
    
    print("\nDone!")

if __name__ == "__main__":
    main()