import numpy as np
import sounddevice as sd
import tensorflow as tf
import threading
import queue
import time
from collections import deque
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import json

from audio_preprocessing import AudioPreprocessor

class RealTimeAudioClassifier:
    def __init__(self, model_path, sample_rate=22050, chunk_duration=4.0, overlap=0.5):
        """
        Initialize real-time audio classifier
        
        Args:
            model_path: Path to trained model
            sample_rate: Audio sample rate
            chunk_duration: Duration of audio chunks for classification (seconds)
            overlap: Overlap between consecutive chunks (0-1)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
        # Calculate chunk parameters
        self.chunk_size = int(chunk_duration * sample_rate)
        self.hop_size = int(self.chunk_size * (1 - overlap))
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.model = None
        self.audio_buffer = deque(maxlen=self.chunk_size * 3)  # Buffer for incoming audio
        self.prediction_queue = queue.Queue()
        self.is_recording = False
        
        # Class names for UrbanSound8K
        self.class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
        
        # Security alert mapping
        self.security_alerts = {
            'gun_shot': {'level': 'CRITICAL', 'color': 'red', 'message': '🚨 GUNSHOT DETECTED'},
            'glass_breaking': {'level': 'HIGH', 'color': 'orange', 'message': '🔔 Glass Breaking'},
            'dog_bark': {'level': 'MEDIUM', 'color': 'yellow', 'message': '🐕 Dog Barking'},
            'car_horn': {'level': 'LOW', 'color': 'blue', 'message': '🚗 Car Horn'},
            'siren': {'level': 'HIGH', 'color': 'orange', 'message': '🚨 Emergency Siren'},
            'jackhammer': {'level': 'LOW', 'color': 'gray', 'message': '🔨 Construction Noise'},
            'drilling': {'level': 'LOW', 'color': 'gray', 'message': '🔧 Drilling'}
        }
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)
        self.fps = 0
        
        self.load_model()
        
    def load_model(self):
        """
        Load trained model
        """
        try:
            print(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully!")
            
            # Compile for inference
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Warm up model with dummy input
            dummy_input = np.random.randn(1, 128, 173, 1)
            _ = self.model.predict(dummy_input, verbose=0)
            print("Model warmed up!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio stream
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio data to buffer
        audio_chunk = indata[:, 0]  # Use only first channel
        self.audio_buffer.extend(audio_chunk)
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process audio chunk and make prediction
        
        Args:
            audio_chunk: Audio data chunk
        
        Returns:
            prediction: Class probabilities
            confidence: Maximum confidence
            predicted_class: Predicted class name
        """
        start_time = time.time()
        
        try:
            # Normalize audio
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Extract mel spectrogram
            mel_spec = self.preprocessor.compute_mel_spectrogram(audio_chunk)
            
            # Add batch and channel dimensions
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = self.model.predict(mel_spec, verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(prediction)
            confidence = prediction[predicted_idx]
            predicted_class = self.class_names[predicted_idx]
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return prediction, confidence, predicted_class
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return None, 0.0, "unknown"
    
    def smooth_predictions(self, prediction):
        """
        Smooth predictions using temporal averaging
        
        Args:
            prediction: Current prediction probabilities
        
        Returns:
            smoothed_prediction: Smoothed probabilities
        """
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) < 2:
            return prediction
        
        # Exponential moving average
        weights = np.exp(np.linspace(-1, 0, len(self.prediction_history)))
        weights = weights / np.sum(weights)
        
        smoothed = np.average(list(self.prediction_history), axis=0, weights=weights)
        return smoothed
    
    def get_security_alert(self, predicted_class, confidence, threshold=0.4):
        """
        Generate security alert based on prediction
        
        Args:
            predicted_class: Predicted class name
            confidence: Prediction confidence
            threshold: Minimum confidence for alert
        
        Returns:
            alert_info: Alert information dictionary
        """
        if confidence < threshold:
            return None
        
        if predicted_class in self.security_alerts:
            alert = self.security_alerts[predicted_class].copy()
            alert['class'] = predicted_class
            alert['confidence'] = confidence
            alert['timestamp'] = time.time()
            return alert
        
        return None
    
    def inference_loop(self):
        """
        Main inference loop running in separate thread
        """
        while self.is_recording:
            if len(self.audio_buffer) >= self.chunk_size:
                # Extract audio chunk
                audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_size:])
                
                # Process chunk
                prediction, confidence, predicted_class = self.process_audio_chunk(audio_chunk)
                
                if prediction is not None:
                    # Smooth predictions
                    smoothed_prediction = self.smooth_predictions(prediction)
                    smoothed_confidence = np.max(smoothed_prediction)
                    smoothed_class = self.class_names[np.argmax(smoothed_prediction)]
                    
                    # Check for security alerts
                    alert = self.get_security_alert(smoothed_class, smoothed_confidence)
                    
                    # Put result in queue
                    result = {
                        'prediction': smoothed_prediction,
                        'confidence': smoothed_confidence,
                        'class': smoothed_class,
                        'alert': alert,
                        'timestamp': time.time(),
                        'audio_chunk': audio_chunk[-1000:]  # Last 1000 samples for visualization
                    }
                    
                    try:
                        self.prediction_queue.put(result, timeout=0.01)
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Update FPS
                if len(self.inference_times) > 0:
                    avg_inference_time = np.mean(list(self.inference_times))
                    self.fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                
                # Wait before next inference
                time.sleep(0.1)  # 10 FPS
    
    def start_recording(self):
        """
        Start real-time audio recording and classification
        """
        if self.model is None:
            print("No model loaded!")
            return
        
        print("Starting real-time audio classification...")
        self.is_recording = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype=np.float32
            ):
                print("Recording started. Press Ctrl+C to stop.")
                print(f"Sample rate: {self.sample_rate} Hz")
                print(f"Chunk duration: {self.chunk_duration} seconds")
                print(f"Overlap: {self.overlap * 100}%")
                
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping recording...")
        except Exception as e:
            print(f"Error in audio stream: {e}")
        finally:
            self.stop_recording()
    
    def stop_recording(self):
        """
        Stop recording and inference
        """
        self.is_recording = False
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join(timeout=1.0)
        print("Recording stopped.")
    
    def get_latest_prediction(self):
        """
        Get latest prediction from queue
        
        Returns:
            result: Latest prediction result or None
        """
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None


class RealTimeVisualizationGUI:
    """
    GUI for real-time visualization of audio classification
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Real-Time Audio Event Classification - Smart Security")
        self.root.geometry("1200x800")
        self.root.configure(bg='black')
        
        self.setup_gui()
        self.latest_result = None
        self.alert_history = deque(maxlen=20)
        
        # Start GUI update loop
        self.update_gui()
    
    def setup_gui(self):
        """
        Setup GUI components
        """
        # Title
        title_label = tk.Label(
            self.root,
            text="🛡️ SMART SECURITY SYSTEM - AUDIO EVENT DETECTION",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="black"
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="black")
        main_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Left panel - Current detection
        left_frame = tk.Frame(main_frame, bg="black", relief="ridge", bd=2)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Current detection label
        tk.Label(left_frame, text="CURRENT DETECTION", font=("Arial", 16, "bold"),
                fg="cyan", bg="black").pack(pady=10)
        
        # Status display
        self.status_frame = tk.Frame(left_frame, bg="black")
        self.status_frame.pack(pady=20)
        
        self.class_label = tk.Label(
            self.status_frame,
            text="LISTENING...",
            font=("Arial", 24, "bold"),
            fg="green",
            bg="black"
        )
        self.class_label.pack(pady=10)
        
        self.confidence_label = tk.Label(
            self.status_frame,
            text="Confidence: ---%",
            font=("Arial", 16),
            fg="white",
            bg="black"
        )
        self.confidence_label.pack(pady=5)
        
        # Progress bar for confidence
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.confidence_var,
            maximum=100,
            length=300
        )
        self.confidence_bar.pack(pady=10)
        
        # Performance metrics
        self.fps_label = tk.Label(
            left_frame,
            text="FPS: --",
            font=("Arial", 12),
            fg="gray",
            bg="black"
        )
        self.fps_label.pack(side="bottom", pady=5)
        
        # Right panel - Alerts and history
        right_frame = tk.Frame(main_frame, bg="black", relief="ridge", bd=2)
        right_frame.pack(side="right", fill="both", expand=True)
        
        # Alert panel
        tk.Label(right_frame, text="SECURITY ALERTS", font=("Arial", 16, "bold"),
                fg="red", bg="black").pack(pady=10)
        
        # Current alert
        self.alert_frame = tk.Frame(right_frame, bg="black")
        self.alert_frame.pack(pady=10, padx=10, fill="x")
        
        self.alert_label = tk.Label(
            self.alert_frame,
            text="No alerts",
            font=("Arial", 14),
            fg="gray",
            bg="black"
        )
        self.alert_label.pack()
        
        # Alert history
        tk.Label(right_frame, text="RECENT ALERTS", font=("Arial", 12, "bold"),
                fg="orange", bg="black").pack(pady=(20, 5))
        
        self.history_frame = tk.Frame(right_frame, bg="black")
        self.history_frame.pack(fill="both", expand=True, padx=10)
        
        # Scrollable text for history
        self.history_text = tk.Text(
            self.history_frame,
            bg="black",
            fg="white",
            font=("Courier", 10),
            height=15,
            state="disabled"
        )
        scrollbar = tk.Scrollbar(self.history_frame, command=self.history_text.yview)
        self.history_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.history_text.pack(side="left", fill="both", expand=True)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="black")
        button_frame.pack(side="bottom", pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="START MONITORING",
            font=("Arial", 14, "bold"),
            bg="green",
            fg="white",
            command=self.start_monitoring,
            width=15
        )
        self.start_button.pack(side="left", padx=10)
        
        self.stop_button = tk.Button(
            button_frame,
            text="STOP MONITORING",
            font=("Arial", 14, "bold"),
            bg="red",
            fg="white",
            command=self.stop_monitoring,
            width=15,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)
    
    def start_monitoring(self):
        """
        Start audio monitoring
        """
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start classifier in separate thread
        self.monitor_thread = threading.Thread(target=self.classifier.start_recording)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """
        Stop audio monitoring
        """
        self.classifier.stop_recording()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        self.class_label.config(text="STOPPED", fg="red")
        self.confidence_label.config(text="Confidence: ---%")
        self.confidence_var.set(0)
        self.alert_label.config(text="No alerts", fg="gray")
    
    def update_gui(self):
        """
        Update GUI with latest predictions
        """
        # Get latest result
        result = self.classifier.get_latest_prediction()
        if result:
            self.latest_result = result
            
            # Update main display
            class_name = result['class'].replace('_', ' ').title()
            confidence = result['confidence'] * 100
            
            self.class_label.config(text=class_name)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.confidence_var.set(confidence)
            
            # Color coding based on confidence
            if confidence > 80:
                color = "lime"
            elif confidence > 60:
                color = "yellow"
            else:
                color = "orange"
            self.class_label.config(fg=color)
            
            # Handle alerts
            if result['alert']:
                alert = result['alert']
                alert_text = f"{alert['message']} ({confidence:.1f}%)"
                self.alert_label.config(text=alert_text, fg=alert['color'])
                
                # Add to history
                timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                history_entry = f"[{timestamp}] {alert['level']}: {alert_text}\n"
                
                self.history_text.config(state="normal")
                self.history_text.insert("1.0", history_entry)
                self.history_text.config(state="disabled")
                
                # Keep only recent entries
                lines = self.history_text.get("1.0", "end").split('\n')
                if len(lines) > 100:
                    self.history_text.config(state="normal")
                    self.history_text.delete("50.0", "end")
                    self.history_text.config(state="disabled")
            else:
                self.alert_label.config(text="No alerts", fg="gray")
        
        # Update FPS
        if hasattr(self.classifier, 'fps'):
            self.fps_label.config(text=f"FPS: {self.classifier.fps:.1f}")
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def run(self):
        """
        Run the GUI
        """
        self.root.mainloop()


def main():
    """
    Main function for real-time audio classification
    """
    # Model path (update this to your trained model)
    model_path = "../models/best_simple_model.keras"  # Using your trained model
    
    # Create classifier
    print("Initializing real-time audio classifier...")
    classifier = RealTimeAudioClassifier(model_path)
    
    # Create and run GUI
    print("Starting GUI...")
    gui = RealTimeVisualizationGUI(classifier)
    gui.run()


if __name__ == "__main__":
    main()