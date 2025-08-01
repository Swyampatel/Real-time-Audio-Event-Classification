import tensorflow as tf
import numpy as np
import time
import os
from pathlib import Path
import json

class ModelOptimizer:
    """
    Optimize trained models for real-time inference
    """
    
    def __init__(self, model_path):
        """
        Initialize model optimizer
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model
        """
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"Model loaded successfully!")
            print(f"Model size: {self.get_model_size()} MB")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def get_model_size(self):
        """
        Get model size in MB
        """
        if self.model is None:
            return 0
        
        # Save to temporary file to get size
        temp_path = "temp_model.h5"
        self.model.save(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return round(size_mb, 2)
    
    def quantize_model(self, output_path, quantization_type='dynamic'):
        """
        Apply quantization to reduce model size and improve inference speed
        
        Args:
            output_path: Path to save quantized model
            quantization_type: 'dynamic', 'int8', or 'float16'
        """
        if self.model is None:
            print("No model loaded!")
            return None
        
        print(f"Applying {quantization_type} quantization...")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantization_type == 'dynamic':
            # Dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif quantization_type == 'int8':
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif quantization_type == 'float16':
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        try:
            quantized_model = converter.convert()
            
            # Save quantized model
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            # Get quantized model size
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            original_size = self.get_model_size()
            compression_ratio = original_size / quantized_size
            
            print(f"Quantization complete!")
            print(f"Original size: {original_size:.2f} MB")
            print(f"Quantized size: {quantized_size:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            
            return output_path
            
        except Exception as e:
            print(f"Error during quantization: {e}")
            return None
    
    def _representative_dataset(self):
        """
        Generate representative dataset for int8 quantization
        """
        # Generate random data matching input shape
        input_shape = self.model.input_shape[1:]  # Remove batch dimension
        
        for _ in range(100):
            # Generate random mel spectrogram-like data
            data = np.random.randn(1, *input_shape).astype(np.float32)
            # Normalize to typical mel spectrogram range
            data = (data - np.mean(data)) / np.std(data)
            data = np.clip(data, -80, 0)  # Typical dB range for mel spectrograms
            yield [data]
    
    def prune_model(self, output_path, target_sparsity=0.5):
        """
        Apply magnitude-based pruning to reduce model complexity
        
        Args:
            output_path: Path to save pruned model
            target_sparsity: Target sparsity level (0.0 to 1.0)
        """
        if self.model is None:
            print("No model loaded!")
            return None
        
        try:
            import tensorflow_model_optimization as tfmot
            
            print(f"Applying magnitude-based pruning (sparsity: {target_sparsity})...")
            
            # Define pruning parameters
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            # Apply pruning to the model
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, **pruning_params
            )
            
            # Compile the model
            model_for_pruning.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fine-tune with pruning (using dummy data)
            dummy_x = np.random.randn(100, *self.model.input_shape[1:])
            dummy_y = np.random.randint(0, 10, (100, 10))
            dummy_y = tf.keras.utils.to_categorical(dummy_y, 10)
            
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_logs')
            ]
            
            model_for_pruning.fit(
                dummy_x, dummy_y,
                batch_size=32,
                epochs=5,
                callbacks=callbacks,
                verbose=1
            )
            
            # Remove pruning wrappers
            model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
            
            # Save pruned model
            model_for_export.save(output_path)
            
            print(f"Pruning complete! Saved to: {output_path}")
            return output_path
            
        except ImportError:
            print("TensorFlow Model Optimization not installed!")
            print("Install with: pip install tensorflow-model-optimization")
            return None
        except Exception as e:
            print(f"Error during pruning: {e}")
            return None
    
    def optimize_for_inference(self, output_dir):
        """
        Apply multiple optimization techniques
        
        Args:
            output_dir: Directory to save optimized models
        """
        if self.model is None:
            print("No model loaded!")
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Original model benchmark
        original_time = self.benchmark_model(self.model)
        original_size = self.get_model_size()
        
        results['original'] = {
            'path': self.model_path,
            'size_mb': original_size,
            'inference_time_ms': original_time,
            'fps': 1000 / original_time if original_time > 0 else 0
        }
        
        print(f"\nOriginal model benchmark:")
        print(f"  Size: {original_size:.2f} MB")
        print(f"  Inference time: {original_time:.2f} ms")
        print(f"  FPS: {results['original']['fps']:.1f}")
        
        # Dynamic quantization
        print("\n" + "="*50)
        print("DYNAMIC QUANTIZATION")
        print("="*50)
        
        dynamic_path = os.path.join(output_dir, "model_dynamic_quantized.tflite")
        if self.quantize_model(dynamic_path, 'dynamic'):
            dynamic_time = self.benchmark_tflite_model(dynamic_path)
            dynamic_size = os.path.getsize(dynamic_path) / (1024 * 1024)
            
            results['dynamic_quantized'] = {
                'path': dynamic_path,
                'size_mb': dynamic_size,
                'inference_time_ms': dynamic_time,
                'fps': 1000 / dynamic_time if dynamic_time > 0 else 0,
                'size_reduction': original_size / dynamic_size,
                'speedup': original_time / dynamic_time if dynamic_time > 0 else 1
            }
        
        # Float16 quantization
        print("\n" + "="*50)
        print("FLOAT16 QUANTIZATION")
        print("="*50)
        
        float16_path = os.path.join(output_dir, "model_float16_quantized.tflite")
        if self.quantize_model(float16_path, 'float16'):
            float16_time = self.benchmark_tflite_model(float16_path)
            float16_size = os.path.getsize(float16_path) / (1024 * 1024)
            
            results['float16_quantized'] = {
                'path': float16_path,
                'size_mb': float16_size,
                'inference_time_ms': float16_time,
                'fps': 1000 / float16_time if float16_time > 0 else 0,
                'size_reduction': original_size / float16_size,
                'speedup': original_time / float16_time if float16_time > 0 else 1
            }
        
        # Model pruning
        print("\n" + "="*50)
        print("MODEL PRUNING")
        print("="*50)
        
        pruned_path = os.path.join(output_dir, "model_pruned.keras")
        if self.prune_model(pruned_path, target_sparsity=0.5):
            # Load pruned model and benchmark
            pruned_model = tf.keras.models.load_model(pruned_path)
            pruned_time = self.benchmark_model(pruned_model)
            
            # Get pruned model size
            pruned_size = os.path.getsize(pruned_path) / (1024 * 1024)
            
            results['pruned'] = {
                'path': pruned_path,
                'size_mb': pruned_size,
                'inference_time_ms': pruned_time,
                'fps': 1000 / pruned_time if pruned_time > 0 else 0,
                'size_reduction': original_size / pruned_size,
                'speedup': original_time / pruned_time if pruned_time > 0 else 1
            }
        
        # Save results
        results_path = os.path.join(output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n" + "="*50)
        print("OPTIMIZATION SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Size: {result['size_mb']:.2f} MB")
            print(f"  Inference: {result['inference_time_ms']:.2f} ms")
            print(f"  FPS: {result['fps']:.1f}")
            if 'size_reduction' in result:
                print(f"  Size reduction: {result['size_reduction']:.2f}x")
                print(f"  Speed improvement: {result['speedup']:.2f}x")
        
        return results
    
    def benchmark_model(self, model, num_runs=100):
        """
        Benchmark Keras model inference time
        
        Args:
            model: Keras model to benchmark
            num_runs: Number of inference runs
        
        Returns:
            Average inference time in milliseconds
        """
        # Generate test data
        input_shape = model.input_shape[1:]
        test_data = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = model.predict(test_data, verbose=0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.predict(test_data, verbose=0)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        return avg_time_ms
    
    def benchmark_tflite_model(self, tflite_path, num_runs=100):
        """
        Benchmark TensorFlow Lite model inference time
        
        Args:
            tflite_path: Path to TFLite model
            num_runs: Number of inference runs
        
        Returns:
            Average inference time in milliseconds
        """
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Generate test data
            input_shape = input_details[0]['shape'][1:]  # Remove batch dimension
            test_data = np.random.randn(1, *input_shape).astype(input_details[0]['dtype'])
            
            # Warm up
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], test_data)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                interpreter.set_tensor(input_details[0]['index'], test_data)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / num_runs) * 1000
            return avg_time_ms
            
        except Exception as e:
            print(f"Error benchmarking TFLite model: {e}")
            return float('inf')
    
    def create_inference_wrapper(self, optimized_model_path, output_path):
        """
        Create a wrapper class for optimized model inference
        
        Args:
            optimized_model_path: Path to optimized model
            output_path: Path to save wrapper script
        """
        wrapper_code = f'''
import tensorflow as tf
import numpy as np
from audio_preprocessing import AudioPreprocessor

class OptimizedAudioClassifier:
    """
    Optimized wrapper for real-time audio classification
    """
    
    def __init__(self, model_path="{optimized_model_path}"):
        self.model_path = model_path
        self.preprocessor = AudioPreprocessor(sample_rate=22050)
        self.class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load optimized model"""
        if self.model_path.endswith('.tflite'):
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.is_tflite = True
        else:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.is_tflite = False
    
    def predict(self, audio_data):
        """
        Make prediction on audio data
        
        Args:
            audio_data: Raw audio signal
        
        Returns:
            probabilities: Class probabilities
            predicted_class: Predicted class name
            confidence: Prediction confidence
        """
        # Preprocess audio
        mel_spec = self.preprocessor.compute_mel_spectrogram(audio_data)
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
        
        # Make prediction
        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], 
                                      mel_spec.astype(self.input_details[0]['dtype']))
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            probabilities = self.model.predict(mel_spec, verbose=0)[0]
        
        # Get results
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return probabilities, predicted_class, confidence
    
    def predict_streaming(self, audio_stream, chunk_duration=4.0, overlap=0.5):
        """
        Process streaming audio
        
        Args:
            audio_stream: Stream of audio data
            chunk_duration: Duration of audio chunks (seconds)  
            overlap: Overlap between chunks (0-1)
        
        Yields:
            prediction results
        """
        sample_rate = 22050
        chunk_size = int(chunk_duration * sample_rate)
        hop_size = int(chunk_size * (1 - overlap))
        
        audio_buffer = np.array([])
        
        for audio_chunk in audio_stream:
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])
            
            while len(audio_buffer) >= chunk_size:
                # Extract chunk
                chunk = audio_buffer[:chunk_size]
                
                # Make prediction
                probs, pred_class, confidence = self.predict(chunk)
                
                yield {{
                    'probabilities': probs,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'timestamp': time.time()
                }}
                
                # Move buffer
                audio_buffer = audio_buffer[hop_size:]
'''
        
        with open(output_path, 'w') as f:
            f.write(wrapper_code)
        
        print(f"Inference wrapper saved to: {output_path}")


def main():
    """
    Main optimization function
    """
    # Model path (update this to your trained model)
    model_path = "../models/best_cnn_lstm_model.keras"
    output_dir = "../models/optimized"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Create optimizer
    print("Audio Model Optimization")
    print("=" * 50)
    
    optimizer = ModelOptimizer(model_path)
    
    # Run optimization
    results = optimizer.optimize_for_inference(output_dir)
    
    # Create inference wrapper for best performing model
    best_model = 'dynamic_quantized'  # Usually the best balance of speed/accuracy
    if best_model in results:
        wrapper_path = os.path.join(output_dir, "optimized_classifier.py")
        optimizer.create_inference_wrapper(results[best_model]['path'], wrapper_path)
    
    print(f"\nOptimization complete! Results saved to: {output_dir}")
    print("Use the optimized models for faster real-time inference.")


if __name__ == "__main__":
    main()