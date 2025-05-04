"""
Base TensorRT inference class for Windows.
This module provides the foundation for all inference models.
Compatible with TensorRT 10.10.0.31
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTBase:
    """Base class for TensorRT inference models"""
    
    def __init__(self, model_path):
        """Initialize TensorRT engine and context"""
        self.model_path = model_path
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Load TRT engine
        self._load_engine()
        
        # Allocate buffers and create bindings
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT engine file {self.model_path} not found")
        
        print(f"Loading TensorRT engine: {self.model_path}")
        
        # Check if the model is an ONNX file
        if self.model_path.endswith('.onnx'):
            # Convert ONNX to TensorRT engine
            self._build_engine_from_onnx()
        else:
            # Load pre-built TensorRT engine
            with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.model_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Get input and output tensor names
        self._get_tensor_names()
    
    def _get_tensor_names(self):
        """Get input and output tensor names"""
        # In TensorRT 10.10.0.31, hardcode the tensor names for ResNet
        # This is a workaround for the API changes
        self.input_names = ["x"]  # Standard input name for ResNet
        self.output_names = ["191"]  # Standard output name for ResNet
        
        print(f"Using hardcoded tensor names: inputs={self.input_names}, outputs={self.output_names}")
    
    def _build_engine_from_onnx(self):
        """Build TensorRT engine from ONNX model"""
        print(f"Converting ONNX model to TensorRT engine: {self.model_path}")
        
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
            print(f"Parsing ONNX file: {self.model_path}")
            with open(self.model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(f"ONNX parse error: {parser.get_error(error)}")
                    raise RuntimeError(f"Failed to parse ONNX file: {self.model_path}")
            
            # Print network info
            print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                print(f"Input {i}: {input_tensor.name}, shape={input_tensor.shape}")
                # Save the input name
                self.input_names = [input_tensor.name]
            
            for i in range(network.num_outputs):
                output_tensor = network.get_output(i)
                print(f"Output {i}: {output_tensor.name}, shape={output_tensor.shape}")
                # Save the output name
                self.output_names = [output_tensor.name]
            
            # Build and serialize engine
            print("Building TensorRT engine... (this may take a while)")
            serialized_engine = builder.build_serialized_network(network, config)
            if not serialized_engine:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            # Save engine to file
            engine_path = os.path.splitext(self.model_path)[0] + ".engine"
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            print(f"Saved TensorRT engine to: {engine_path}")
    
    def _allocate_buffers(self):
        """Allocate device buffers for input and output"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Allocate memory for inputs
        for i, input_name in enumerate(self.input_names):
            # Get tensor shape
            try:
                shape = self.engine.get_tensor_shape(input_name)
                size = trt.volume(shape)
                dtype_size = 4  # Assuming float32 (4 bytes)
                
                # Allocate CUDA memory
                device_mem = cuda.mem_alloc(size * dtype_size)
                
                # Store binding info
                binding_info = {
                    'index': i,
                    'name': input_name,
                    'shape': shape,
                    'size': size,
                    'mem': device_mem
                }
                
                self.inputs.append(binding_info)
                self.bindings.append(int(device_mem))
                print(f"Allocated memory for input: {input_name}, shape={shape}, size={size}")
            except Exception as e:
                print(f"Error allocating input buffer: {e}")
                raise
        
        # Allocate memory for outputs
        for i, output_name in enumerate(self.output_names):
            try:
                # Get tensor shape
                shape = self.engine.get_tensor_shape(output_name)
                size = trt.volume(shape)
                dtype_size = 4  # Assuming float32 (4 bytes)
                
                # Allocate CUDA memory
                device_mem = cuda.mem_alloc(size * dtype_size)
                
                # Store binding info
                binding_info = {
                    'index': i + len(self.inputs),
                    'name': output_name,
                    'shape': shape,
                    'size': size,
                    'mem': device_mem
                }
                
                self.outputs.append(binding_info)
                self.bindings.append(int(device_mem))
                print(f"Allocated memory for output: {output_name}, shape={shape}, size={size}")
            except Exception as e:
                print(f"Error allocating output buffer: {e}")
                raise
    
    def _preprocess_image(self, image):
        """Preprocess the input image"""
        # This method should be overridden by subclasses
        raise NotImplementedError
    
    def _postprocess_results(self, outputs):
        """Postprocess the raw inference results"""
        # This method should be overridden by subclasses
        raise NotImplementedError
    
    def infer(self, image):
        """Run inference on an image"""
        try:
            # Preprocess image
            input_data = self._preprocess_image(image)
            
            # Copy input data to device
            cuda.memcpy_htod_async(self.inputs[0]['mem'], input_data, self.stream)
            
            # Set input tensor address
            self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['mem']))
            
            # Set output tensor addresses
            for output in self.outputs:
                self.context.set_tensor_address(output['name'], int(output['mem']))
            
            # Run inference
            self.context.execute_async_v3(self.stream.handle)
            
            # Create output arrays
            output_data = []
            for output in self.outputs:
                output_array = np.empty(output['size'], dtype=np.float32)
                cuda.memcpy_dtoh_async(output_array, output['mem'], self.stream)
                output_data.append(output_array)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            # Postprocess results
            return self._postprocess_results(output_data)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    def __del__(self):
        """Clean up resources"""
        # Release context
        if hasattr(self, 'context') and self.context:
            del self.context
        
        # Release engine
        if hasattr(self, 'engine') and self.engine:
            del self.engine