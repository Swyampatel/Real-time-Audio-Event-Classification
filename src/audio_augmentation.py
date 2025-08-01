import numpy as np
import librosa
import random
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Normalize, AddBackgroundNoise, TimeMask,
    FrequencyMask, ClippingDistortion, AddGaussianSNR
)

class AudioAugmentor:
    def __init__(self, sample_rate=22050):
        """
        Initialize audio augmentation pipeline
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
        # Define augmentation transforms
        self.light_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            Normalize(p=1.0)
        ])
        
        self.medium_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.3),
            TimeMask(min_band_part=0.01, max_band_part=0.05, p=0.2),
            FrequencyMask(min_frequency_band=0.01, max_frequency_band=0.1, p=0.2),
            Normalize(p=1.0)
        ])
        
        self.heavy_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.6),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.2, max_shift=0.2, p=0.4),
            TimeMask(min_band_part=0.01, max_band_part=0.1, p=0.3),
            FrequencyMask(min_frequency_band=0.01, max_frequency_band=0.15, p=0.3),
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=0.2),
            AddGaussianSNR(min_snr_db=5, max_snr_db=20, p=0.3),
            Normalize(p=1.0)
        ])
    
    def time_stretch(self, audio, rate=None):
        """
        Apply time stretching to audio
        
        Args:
            audio: Input audio signal
            rate: Stretch rate (None for random)
        
        Returns:
            Stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        # Ensure same length as original
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
        
        return stretched
    
    def pitch_shift(self, audio, n_steps=None):
        """
        Apply pitch shifting to audio
        
        Args:
            audio: Input audio signal
            n_steps: Number of semitones to shift (None for random)
        
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = np.random.uniform(-4, 4)
        
        shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
        
        return shifted
    
    def add_noise(self, audio, noise_level=None):
        """
        Add Gaussian noise to audio
        
        Args:
            audio: Input audio signal
            noise_level: Noise amplitude (None for random)
        
        Returns:
            Noisy audio
        """
        if noise_level is None:
            noise_level = np.random.uniform(0.001, 0.02)
        
        noise = np.random.randn(len(audio)) * noise_level
        noisy_audio = audio + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio
    
    def add_background_noise(self, audio, background_audio, snr_db=None):
        """
        Add background noise at specified SNR
        
        Args:
            audio: Input audio signal
            background_audio: Background noise signal
            snr_db: Signal-to-noise ratio in dB (None for random)
        
        Returns:
            Audio with background noise
        """
        if snr_db is None:
            snr_db = np.random.uniform(5, 20)
        
        # Ensure background audio is same length
        if len(background_audio) > len(audio):
            start_idx = np.random.randint(0, len(background_audio) - len(audio))
            background_audio = background_audio[start_idx:start_idx + len(audio)]
        else:
            background_audio = np.tile(background_audio, 
                                     int(np.ceil(len(audio) / len(background_audio))))[:len(audio)]
        
        # Calculate scaling factor for desired SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(background_audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Mix signals
        mixed = audio + scale * background_audio
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed
    
    def time_mask(self, audio, mask_ratio=None):
        """
        Apply time masking (zeroing out portions of audio)
        
        Args:
            audio: Input audio signal
            mask_ratio: Ratio of audio to mask (None for random)
        
        Returns:
            Masked audio
        """
        if mask_ratio is None:
            mask_ratio = np.random.uniform(0.01, 0.1)
        
        masked_audio = audio.copy()
        mask_length = int(len(audio) * mask_ratio)
        
        # Apply multiple masks
        num_masks = np.random.randint(1, 4)
        for _ in range(num_masks):
            start_idx = np.random.randint(0, len(audio) - mask_length)
            masked_audio[start_idx:start_idx + mask_length] = 0
        
        return masked_audio
    
    def augment(self, audio, augmentation_type='medium'):
        """
        Apply augmentation based on type
        
        Args:
            audio: Input audio signal
            augmentation_type: 'light', 'medium', or 'heavy'
        
        Returns:
            Augmented audio
        """
        if augmentation_type == 'light':
            return self.light_augment(samples=audio, sample_rate=self.sample_rate)
        elif augmentation_type == 'medium':
            return self.medium_augment(samples=audio, sample_rate=self.sample_rate)
        elif augmentation_type == 'heavy':
            return self.heavy_augment(samples=audio, sample_rate=self.sample_rate)
        else:
            return audio
    
    def mixup(self, audio1, audio2, label1, label2, alpha=0.2):
        """
        Apply mixup augmentation
        
        Args:
            audio1, audio2: Audio signals to mix
            label1, label2: Corresponding labels
            alpha: Beta distribution parameter
        
        Returns:
            mixed_audio: Mixed audio signal
            mixed_label: Mixed label
        """
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        
        # Mix labels (for soft labels)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_audio, mixed_label
    
    def spec_augment(self, mel_spectrogram, time_mask_param=20, freq_mask_param=10):
        """
        Apply SpecAugment to mel spectrogram
        
        Args:
            mel_spectrogram: Input mel spectrogram
            time_mask_param: Maximum time mask length
            freq_mask_param: Maximum frequency mask length
        
        Returns:
            Augmented spectrogram
        """
        augmented = mel_spectrogram.copy()
        
        # Time masking
        num_time_masks = np.random.randint(0, 3)
        for _ in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_param, augmented.shape[1]))
            t0 = np.random.randint(0, augmented.shape[1] - t)
            augmented[:, t0:t0+t] = 0
        
        # Frequency masking
        num_freq_masks = np.random.randint(0, 3)
        for _ in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_param, augmented.shape[0]))
            f0 = np.random.randint(0, augmented.shape[0] - f)
            augmented[f0:f0+f, :] = 0
        
        return augmented
    
    def create_augmented_batch(self, audio_batch, labels_batch, augment_ratio=0.5):
        """
        Create augmented batch with original and augmented samples
        
        Args:
            audio_batch: Batch of audio signals
            labels_batch: Corresponding labels
            augment_ratio: Ratio of samples to augment
        
        Returns:
            augmented_batch: Combined original and augmented audio
            augmented_labels: Corresponding labels
        """
        batch_size = len(audio_batch)
        num_augment = int(batch_size * augment_ratio)
        
        augmented_audio = []
        augmented_labels = []
        
        # Keep original samples
        augmented_audio.extend(audio_batch)
        augmented_labels.extend(labels_batch)
        
        # Add augmented samples
        indices = np.random.choice(batch_size, num_augment, replace=True)
        for idx in indices:
            aug_type = np.random.choice(['light', 'medium', 'heavy'], p=[0.5, 0.3, 0.2])
            aug_audio = self.augment(audio_batch[idx], aug_type)
            augmented_audio.append(aug_audio)
            augmented_labels.append(labels_batch[idx])
        
        # Add mixup samples
        num_mixup = num_augment // 2
        for _ in range(num_mixup):
            idx1, idx2 = np.random.choice(batch_size, 2, replace=False)
            mixed_audio, mixed_label = self.mixup(
                audio_batch[idx1], audio_batch[idx2],
                labels_batch[idx1], labels_batch[idx2]
            )
            augmented_audio.append(mixed_audio)
            augmented_labels.append(mixed_label)
        
        return np.array(augmented_audio), np.array(augmented_labels)


def demo_augmentation():
    """
    Demonstrate augmentation techniques
    """
    import matplotlib.pyplot as plt
    from audio_preprocessing import AudioPreprocessor
    
    # Create augmentor and preprocessor
    augmentor = AudioAugmentor()
    preprocessor = AudioPreprocessor()
    
    # Create demo audio
    duration = 4.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))
    
    # Simulate dog bark (rhythmic bursts)
    bark_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    audio = np.zeros_like(t)
    for bark_t in bark_times:
        burst = np.exp(-50 * (t - bark_t) ** 2) * np.sin(2 * np.pi * 800 * t)
        audio += burst
    
    audio = audio / np.max(np.abs(audio))
    
    # Apply different augmentations
    aug_light = augmentor.augment(audio, 'light')
    aug_medium = augmentor.augment(audio, 'medium')
    aug_heavy = augmentor.augment(audio, 'heavy')
    
    # Visualize
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    # Original
    mel_orig = preprocessor.compute_mel_spectrogram(audio)
    axes[0, 0].plot(audio)
    axes[0, 0].set_title('Original Audio')
    librosa.display.specshow(mel_orig, ax=axes[0, 1], x_axis='time', y_axis='mel')
    axes[0, 1].set_title('Original Mel Spectrogram')
    
    # Light augmentation
    mel_light = preprocessor.compute_mel_spectrogram(aug_light)
    axes[1, 0].plot(aug_light)
    axes[1, 0].set_title('Light Augmentation')
    librosa.display.specshow(mel_light, ax=axes[1, 1], x_axis='time', y_axis='mel')
    axes[1, 1].set_title('Light Aug Mel Spectrogram')
    
    # Medium augmentation
    mel_medium = preprocessor.compute_mel_spectrogram(aug_medium)
    axes[2, 0].plot(aug_medium)
    axes[2, 0].set_title('Medium Augmentation')
    librosa.display.specshow(mel_medium, ax=axes[2, 1], x_axis='time', y_axis='mel')
    axes[2, 1].set_title('Medium Aug Mel Spectrogram')
    
    # Heavy augmentation
    mel_heavy = preprocessor.compute_mel_spectrogram(aug_heavy)
    axes[3, 0].plot(aug_heavy)
    axes[3, 0].set_title('Heavy Augmentation')
    librosa.display.specshow(mel_heavy, ax=axes[3, 1], x_axis='time', y_axis='mel')
    axes[3, 1].set_title('Heavy Aug Mel Spectrogram')
    
    plt.tight_layout()
    plt.savefig('../data/augmentation_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Augmentation demo complete!")
    print("Visualization saved to: ../data/augmentation_demo.png")


if __name__ == "__main__":
    print("Audio Augmentation Module")
    print("=" * 50)
    
    # Run demo
    demo_augmentation()