import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback
import numpy as np

class AudioEventClassifier:
    def __init__(self, input_shape=(128, 173, 1), num_classes=10, model_type='cnn_lstm'):
        """
        Initialize audio event classifier
        
        Args:
            input_shape: Shape of input mel spectrogram (n_mels, time_steps, channels)
            num_classes: Number of output classes
            model_type: Type of model architecture
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        
    def build_cnn_lstm_model(self):
        """
        Build CNN + LSTM hybrid model for temporal audio patterns
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Feature Extraction
        # Block 1
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 4
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Reshape for LSTM
        x = layers.Reshape((1, 256))(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_pure_cnn_model(self):
        """
        Build pure CNN model (faster for real-time)
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Inception-style blocks for multi-scale feature extraction
        def inception_block(x, filters):
            # 1x1 convolution
            conv1x1 = layers.Conv2D(filters//4, (1, 1), padding='same')(x)
            conv1x1 = layers.BatchNormalization()(conv1x1)
            conv1x1 = layers.ReLU()(conv1x1)
            
            # 3x3 convolution
            conv3x3 = layers.Conv2D(filters//2, (1, 1), padding='same')(x)
            conv3x3 = layers.BatchNormalization()(conv3x3)
            conv3x3 = layers.ReLU()(conv3x3)
            conv3x3 = layers.Conv2D(filters//2, (3, 3), padding='same')(conv3x3)
            conv3x3 = layers.BatchNormalization()(conv3x3)
            conv3x3 = layers.ReLU()(conv3x3)
            
            # 5x5 convolution
            conv5x5 = layers.Conv2D(filters//4, (1, 1), padding='same')(x)
            conv5x5 = layers.BatchNormalization()(conv5x5)
            conv5x5 = layers.ReLU()(conv5x5)
            conv5x5 = layers.Conv2D(filters//4, (5, 5), padding='same')(conv5x5)
            conv5x5 = layers.BatchNormalization()(conv5x5)
            conv5x5 = layers.ReLU()(conv5x5)
            
            # Concatenate
            output = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
            return output
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Inception blocks
        x = inception_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = inception_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = inception_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_attention_model(self):
        """
        Build CNN with attention mechanism
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN backbone
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Self-attention mechanism
        # Reshape for attention
        shape = tf.shape(x)
        h, w, c = shape[1], shape[2], shape[3]
        x_reshaped = layers.Reshape((-1, 256))(x)
        
        # Query, Key, Value projections
        query = layers.Dense(128)(x_reshaped)
        key = layers.Dense(128)(x_reshaped)
        value = layers.Dense(256)(x_reshaped)
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(128, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention
        attended = tf.matmul(attention_weights, value)
        
        # Reshape back
        x = layers.Reshape((h, w, 256))(attended)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model(self):
        """
        Build model based on specified type
        """
        if self.model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        elif self.model_type == 'pure_cnn':
            self.model = self.build_pure_cnn_model()
        elif self.model_type == 'attention':
            self.model = self.build_attention_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss
        """
        if self.model is None:
            self.build_model()
        
        # Use label smoothing for better generalization
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        # Adam optimizer with learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for addressing class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Compute focal loss
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = y_true * (1 - y_pred) ** self.gamma + (1 - y_true) * y_pred ** self.gamma
        focal_loss = -alpha_factor * focal_weight * (y_true * tf.math.log(y_pred) + 
                                                      (1 - y_true) * tf.math.log(1 - y_pred))
        
        return tf.reduce_mean(focal_loss)


class PerClassAccuracy(Callback):
    """
    Callback to monitor per-class accuracy during training
    """
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
    
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        predictions = self.model.predict(x_val)
        
        # Convert to class indices
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate per-class accuracy
        print("\nPer-class accuracy:")
        for i, class_name in enumerate(self.class_names):
            mask = y_true == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == i)
                print(f"  {class_name}: {class_acc:.3f}")


def create_model_ensemble(input_shape, num_classes):
    """
    Create an ensemble of models for better accuracy
    """
    models = []
    
    # Model 1: CNN-LSTM
    classifier1 = AudioEventClassifier(input_shape, num_classes, 'cnn_lstm')
    model1 = classifier1.build_model()
    models.append(model1)
    
    # Model 2: Pure CNN
    classifier2 = AudioEventClassifier(input_shape, num_classes, 'pure_cnn')
    model2 = classifier2.build_model()
    models.append(model2)
    
    # Model 3: Attention
    classifier3 = AudioEventClassifier(input_shape, num_classes, 'attention')
    model3 = classifier3.build_model()
    models.append(model3)
    
    # Ensemble
    inputs = layers.Input(shape=input_shape)
    outputs = []
    for model in models:
        outputs.append(model(inputs))
    
    # Average predictions
    averaged = layers.Average()(outputs)
    ensemble_model = models.Model(inputs=inputs, outputs=averaged)
    
    return ensemble_model


def demo_model():
    """
    Demonstrate model architectures
    """
    print("Model Architecture Demo")
    print("=" * 50)
    
    # Create different model architectures
    input_shape = (128, 173, 1)  # (n_mels, time_steps, channels)
    num_classes = 10
    
    # CNN-LSTM model
    print("\n1. CNN-LSTM Hybrid Model:")
    classifier1 = AudioEventClassifier(input_shape, num_classes, 'cnn_lstm')
    model1 = classifier1.build_model()
    print(f"Total parameters: {model1.count_params():,}")
    
    # Pure CNN model
    print("\n2. Pure CNN Model (Inception-style):")
    classifier2 = AudioEventClassifier(input_shape, num_classes, 'pure_cnn')
    model2 = classifier2.build_model()
    print(f"Total parameters: {model2.count_params():,}")
    
    # Attention model
    print("\n3. CNN with Attention Model:")
    classifier3 = AudioEventClassifier(input_shape, num_classes, 'attention')
    model3 = classifier3.build_model()
    print(f"Total parameters: {model3.count_params():,}")
    
    # Save model architectures
    for i, model in enumerate([model1, model2, model3], 1):
        tf.keras.utils.plot_model(
            model,
            to_file=f'../models/model_architecture_{i}.png',
            show_shapes=True,
            show_layer_names=True
        )
    
    print("\nModel architectures saved to ../models/")
    
    return classifier1, classifier2, classifier3


if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('../models', exist_ok=True)
    
    # Run demo
    demo_model()