from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import librosa
import base64
import io
import wave
import threading
import time
from collections import deque
import json
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_preprocessing import AudioPreprocessor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

class WebAudioClassifier:
    def __init__(self, model_path):
        """
        Initialize web-based audio classifier
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = AudioPreprocessor(sample_rate=22050)
        
        # Class names for UrbanSound8K
        self.class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
        
        # Security alert mapping
        self.security_alerts = {
            'gun_shot': {'level': 'CRITICAL', 'color': '#ff0000', 'message': '🚨 GUNSHOT DETECTED'},
            'dog_bark': {'level': 'MEDIUM', 'color': '#ffa500', 'message': '🐕 Dog Barking'},
            'car_horn': {'level': 'LOW', 'color': '#0080ff', 'message': '🚗 Car Horn'},
            'siren': {'level': 'HIGH', 'color': '#ff4500', 'message': '🚨 Emergency Siren'},
            'jackhammer': {'level': 'LOW', 'color': '#808080', 'message': '🔨 Construction Noise'},
            'drilling': {'level': 'LOW', 'color': '#808080', 'message': '🔧 Drilling'}
        }
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        self.alert_history = deque(maxlen=50)
        
        self.load_model()
    
    def load_model(self):
        """
        Load trained model
        """
        try:
            print(f"Loading model from {self.model_path}")
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                print("Model loaded successfully!")
                
                # Warm up model
                dummy_input = np.random.randn(1, 128, 173, 1)
                _ = self.model.predict(dummy_input, verbose=0)
                print("Model warmed up!")
            else:
                print(f"Model file not found: {self.model_path}")
                # Create a dummy model for demonstration
                self.create_dummy_model()
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """
        Create a dummy model for demonstration purposes
        """
        print("Creating dummy model for demonstration...")
        from models import AudioEventClassifier
        
        classifier = AudioEventClassifier(
            input_shape=(128, 173, 1),
            num_classes=len(self.class_names),
            model_type='pure_cnn'
        )
        self.model = classifier.build_model()
        print("Dummy model created!")
    
    def process_audio_data(self, audio_data, sample_rate=22050):
        """
        Process audio data and make prediction
        
        Args:
            audio_data: Raw audio data
            sample_rate: Sample rate of audio
        
        Returns:
            result: Prediction result dictionary
        """
        try:
            # Ensure audio is the right length (4 seconds)
            target_length = int(4.0 * sample_rate)
            if len(audio_data) < target_length:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            else:
                # Truncate
                audio_data = audio_data[:target_length]
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract mel spectrogram
            mel_spec = self.preprocessor.compute_mel_spectrogram(audio_data)
            
            # Add batch and channel dimensions
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = self.model.predict(mel_spec, verbose=0)[0]
            
            # Smooth predictions
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > 1:
                weights = np.exp(np.linspace(-1, 0, len(self.prediction_history)))
                weights = weights / np.sum(weights)
                smoothed_prediction = np.average(list(self.prediction_history), axis=0, weights=weights)
            else:
                smoothed_prediction = prediction
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(smoothed_prediction)
            confidence = smoothed_prediction[predicted_idx]
            predicted_class = self.class_names[predicted_idx]
            
            # Create result
            result = {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {name: float(prob) for name, prob in zip(self.class_names, smoothed_prediction)},
                'timestamp': time.time()
            }
            
            # Check for security alerts
            alert = self.get_security_alert(predicted_class, confidence)
            if alert:
                result['alert'] = alert
                self.alert_history.append(alert)
            
            return result
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return {
                'class': 'unknown',
                'confidence': 0.0,
                'probabilities': {name: 0.0 for name in self.class_names},
                'error': str(e),
                'timestamp': time.time()
            }
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process a single audio chunk and return prediction
        
        Args:
            audio_chunk: Raw audio data chunk
        
        Returns:
            prediction: Array of class probabilities
        """
        try:
            # Normalize
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Extract mel spectrogram
            mel_spec = self.preprocessor.compute_mel_spectrogram(audio_chunk)
            
            # Add batch and channel dimensions
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = self.model.predict(mel_spec, verbose=0)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return np.zeros(len(self.class_names))
    
    def get_security_alert(self, predicted_class, confidence, threshold=0.4):
        """
        Generate security alert based on prediction
        """
        if confidence < threshold:
            return None
        
        if predicted_class in self.security_alerts:
            alert = self.security_alerts[predicted_class].copy()
            alert['class'] = predicted_class
            alert['confidence'] = float(confidence)
            alert['timestamp'] = time.time()
            return alert
        
        return None
    
    def get_alert_history(self):
        """
        Get recent alert history
        """
        return list(self.alert_history)


# Initialize classifier
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_simple_model.keras')
classifier = WebAudioClassifier(model_path)

@app.route('/')
def index():
    """
    Main page
    """
    return render_template('index.html')

@app.route('/api/test_audio_file', methods=['POST'])
def test_audio_file():
    """
    Test an uploaded audio file for alerts
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Read audio data with better error handling
        try:
            # Reset file pointer
            audio_file.seek(0)
            
            # Try multiple approaches for different file formats
            try:
                # First try with librosa (works best with WAV, FLAC)
                audio_data, sample_rate = librosa.load(audio_file, sr=22050)
            except Exception as e1:
                # Reset file pointer
                audio_file.seek(0)
                
                try:
                    # Try with pydub for MP3, M4A support
                    from pydub import AudioSegment
                    
                    # Load with pydub
                    audio_segment = AudioSegment.from_file(audio_file)
                    
                    # Convert to mono and resample to 22050 Hz
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(22050)
                    
                    # Convert to numpy array
                    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    
                    # Normalize to [-1, 1] range
                    if audio_segment.sample_width == 2:  # 16-bit
                        audio_data = audio_data / (2**15)
                    elif audio_segment.sample_width == 3:  # 24-bit
                        audio_data = audio_data / (2**23)
                    elif audio_segment.sample_width == 4:  # 32-bit
                        audio_data = audio_data / (2**31)
                    else:  # 8-bit or other
                        audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
                    
                    sample_rate = 22050
                    
                except Exception as e2:
                    # Reset file pointer for final attempt
                    audio_file.seek(0)
                    
                    try:
                        # Last resort: try with soundfile
                        import soundfile as sf
                        audio_data, original_sr = sf.read(audio_file)
                        
                        # Resample if needed
                        if original_sr != 22050:
                            import scipy.signal
                            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * 22050 / original_sr))
                        
                        # Ensure mono
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        sample_rate = 22050
                        
                    except Exception as e3:
                        return jsonify({
                            'error': f'Could not read audio file. Please try converting to WAV format first. '
                                   f'Errors: librosa({str(e1)}), pydub({str(e2)}), soundfile({str(e3)})'
                        }), 400
                        
        except Exception as e:
            return jsonify({'error': f'Unexpected error reading audio file: {str(e)}'}), 400
        
        # Process full audio in chunks
        chunk_duration = 4.0  # seconds
        hop_duration = 1.0    # seconds (reduced overlap for clearer timestamps)
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(hop_duration * sample_rate)
        
        results = []
        alerts_found = []
        
        total_duration = len(audio_data) / sample_rate
        
        # Process audio in sliding windows
        for i in range(0, len(audio_data) - chunk_samples + 1, hop_samples):
            chunk = audio_data[i:i + chunk_samples]
            
            # Get prediction
            prediction = classifier.process_audio_chunk(chunk)
            predicted_class = classifier.class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            # Check for alerts
            alert = classifier.get_security_alert(predicted_class, confidence)
            
            start_time = i / sample_rate
            end_time = min((i + chunk_samples) / sample_rate, total_duration)
            
            result = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'class': predicted_class,
                'confidence': confidence,
                'alert': alert
            }
            results.append(result)
            
            if alert:
                alerts_found.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'alert': alert,
                    'class': predicted_class,
                    'confidence': confidence
                })
        
        return jsonify({
            'success': True,
            'filename': audio_file.filename,
            'duration': len(audio_data) / sample_rate,
            'total_predictions': len(results),
            'alerts_found': alerts_found,
            'all_results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Read audio data
        audio_data, sample_rate = librosa.load(audio_file, sr=22050)
        
        # Process audio
        result = classifier.process_audio_data(audio_data, sample_rate)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('audio_data')
def handle_audio_data(data):
    """
    Handle real-time audio data from browser
    """
    try:
        print(f"Received audio data")
        
        # Decode base64 audio data
        audio_data = base64.b64decode(data['audio'])
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        print(f"Audio array shape: {audio_array.shape}, duration: {len(audio_array)/22050:.2f}s")
        
        # Process audio
        result = classifier.process_audio_data(audio_array)
        
        print(f"Prediction: {result['class']} ({result['confidence']:.2%})")
        
        # Emit result back to client
        emit('prediction', result)
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        emit('error', {'message': str(e)})

@socketio.on('get_alert_history')
def handle_get_alert_history():
    """
    Get alert history
    """
    history = classifier.get_alert_history()
    emit('alert_history', {'alerts': history})

@socketio.on('connect')
def handle_connect():
    """
    Handle client connection
    """
    print('Client connected')
    emit('status', {'message': 'Connected to audio classification server'})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client disconnection
    """
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Flask-SocketIO server...")
    print(f"Server will run on port {port}")
    socketio.run(app, debug=False, host='0.0.0.0', port=port)