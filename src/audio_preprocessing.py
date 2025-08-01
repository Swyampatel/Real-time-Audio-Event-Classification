import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import os

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512, n_mfcc=20):
        """
        Initialize audio preprocessor with configurable parameters
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        
    def load_audio(self, file_path, duration=4.0):
        """
        Load audio file and normalize to fixed duration
        
        Args:
            file_path: Path to audio file
            duration: Target duration in seconds
        
        Returns:
            audio: Normalized audio signal
            sr: Sample rate
        """
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            
            # Pad or truncate to exact duration
            target_length = int(duration * self.sample_rate)
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                # Truncate
                audio = audio[:target_length]
                
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def compute_mel_spectrogram(self, audio):
        """
        Compute mel spectrogram from audio signal
        
        Args:
            audio: Audio signal
        
        Returns:
            mel_spec_db: Mel spectrogram in dB scale
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def compute_mfcc(self, audio):
        """
        Compute MFCC features from audio signal
        
        Args:
            audio: Audio signal
        
        Returns:
            mfcc: MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def compute_spectral_features(self, audio):
        """
        Compute various spectral features
        
        Args:
            audio: Audio signal
        
        Returns:
            features: Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return features
    
    def extract_features(self, audio):
        """
        Extract all features from audio
        
        Args:
            audio: Audio signal
        
        Returns:
            feature_dict: Dictionary containing all extracted features
        """
        feature_dict = {}
        
        # Mel spectrogram
        feature_dict['mel_spectrogram'] = self.compute_mel_spectrogram(audio)
        
        # MFCC
        feature_dict['mfcc'] = self.compute_mfcc(audio)
        
        # Spectral features
        spectral_features = self.compute_spectral_features(audio)
        feature_dict.update(spectral_features)
        
        # Delta and delta-delta MFCC
        feature_dict['mfcc_delta'] = librosa.feature.delta(feature_dict['mfcc'])
        feature_dict['mfcc_delta2'] = librosa.feature.delta(feature_dict['mfcc'], order=2)
        
        return feature_dict
    
    def visualize_features(self, audio, features, save_path=None):
        """
        Visualize extracted features
        
        Args:
            audio: Audio signal
            features: Extracted features dictionary
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Waveform
        axes[0, 0].plot(np.linspace(0, len(audio)/self.sample_rate, len(audio)), audio)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Mel spectrogram
        img1 = librosa.display.specshow(
            features['mel_spectrogram'],
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Mel Spectrogram')
        fig.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        # MFCC
        img2 = librosa.display.specshow(
            features['mfcc'],
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('MFCC')
        axes[1, 0].set_ylabel('MFCC Coefficients')
        fig.colorbar(img2, ax=axes[1, 0])
        
        # Spectral centroid
        time_frames = np.arange(features['spectral_centroid'].shape[1])
        axes[1, 1].semilogy(time_frames, features['spectral_centroid'][0])
        axes[1, 1].set_title('Spectral Centroid')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Hz')
        
        # Chroma features
        img3 = librosa.display.specshow(
            features['chroma'],
            x_axis='time',
            y_axis='chroma',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=axes[2, 0]
        )
        axes[2, 0].set_title('Chromagram')
        fig.colorbar(img3, ax=axes[2, 0])
        
        # Zero crossing rate
        axes[2, 1].plot(features['zero_crossing_rate'][0])
        axes[2, 1].set_title('Zero Crossing Rate')
        axes[2, 1].set_xlabel('Frame')
        axes[2, 1].set_ylabel('Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_batch(self, file_paths, return_audio=False):
        """
        Process a batch of audio files
        
        Args:
            file_paths: List of audio file paths
            return_audio: Whether to return raw audio along with features
        
        Returns:
            batch_features: List of feature dictionaries
            batch_audio: List of audio signals (if return_audio=True)
        """
        batch_features = []
        batch_audio = []
        
        for file_path in file_paths:
            audio, sr = self.load_audio(file_path)
            if audio is not None:
                features = self.extract_features(audio)
                batch_features.append(features)
                if return_audio:
                    batch_audio.append(audio)
        
        if return_audio:
            return batch_features, batch_audio
        return batch_features


def demo_preprocessing():
    """
    Demonstrate preprocessing capabilities
    """
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Create demo signal (simulating glass breaking - sharp transient)
    duration = 4.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))
    
    # Simulate glass breaking sound (sharp attack with high frequencies)
    envelope = np.exp(-5 * t) * (t < 0.5)
    noise = np.random.randn(len(t)) * 0.3
    harmonic = np.sum([np.sin(2 * np.pi * f * t) / (i+1) 
                       for i, f in enumerate(range(2000, 8000, 500))], axis=0)
    audio = envelope * (harmonic + noise)
    audio = audio / np.max(np.abs(audio))
    
    # Extract features
    features = preprocessor.extract_features(audio)
    
    # Visualize
    preprocessor.visualize_features(audio, features, save_path='../data/preprocessing_demo.png')
    
    print("Preprocessing demo complete!")
    print(f"Mel spectrogram shape: {features['mel_spectrogram'].shape}")
    print(f"MFCC shape: {features['mfcc'].shape}")
    
    return preprocessor, features


if __name__ == "__main__":
    print("Audio Preprocessing Module")
    print("=" * 50)
    
    # Run demo
    preprocessor, features = demo_preprocessing()
    
    print("\nFeature shapes:")
    for name, feat in features.items():
        if isinstance(feat, np.ndarray):
            print(f"  {name}: {feat.shape}")