# 🎵 Audio Security Monitor

An AI-powered audio event classification system that detects and identifies various sound events with a focus on security threat detection. Built with deep learning using TensorFlow and deployed with a modern web interface.

## 🚀 Features

- **Intelligent Audio Analysis**: Advanced ML model trained on UrbanSound8K dataset
- **Security Threat Detection**: Specialized detection for gunshots, sirens, and other security-related sounds
- **Multi-format Support**: Handles MP3, WAV, M4A, FLAC, and OGG audio files
- **Instant Analysis**: Upload and analyze audio files with confidence scoring
- **Modern Web Interface**: Clean, responsive React frontend with dark theme
- **Cloud Deployment**: Backend on Railway, Frontend on Vercel

## 🎯 Detected Sound Classes

The system can identify 10 different urban sound categories:

- 🔫 **Gun Shot** (Critical Alert)
- 🚨 **Siren** (High Alert) 
- 🐕 **Dog Bark** (Medium Alert)
- 🚗 **Car Horn** (Low Alert)
- 🔨 **Jackhammer** (Construction)
- 🔧 **Drilling** (Construction)
- ❄️ **Air Conditioner** (Environmental)
- 🚛 **Engine Idling** (Traffic)
- 👶 **Children Playing** (Human Activity)
- 🎵 **Street Music** (Entertainment)

## 🏗️ Architecture

### Backend (Python/Flask)
- **Framework**: Flask with modern API design
- **ML Model**: TensorFlow/Keras CNN optimized for audio classification
- **Audio Processing**: Librosa, PyDub, SoundFile for comprehensive format support
- **Deployment**: Railway.app with automatic scaling

### Frontend (React)
- **Framework**: React 18 with modern hooks and components
- **Styling**: CSS3 with custom variables, gradients, and smooth animations
- **File Upload**: Drag & drop interface with real-time progress indication
- **Deployment**: Vercel with global CDN

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 16+
- Git

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/swyampatel/Real-time-Audio-Event-Classification.git
cd Real-time-Audio-Event-Classification

# Create virtual environment
python -m venv audio_env
source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
python web/app.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🚀 Deployment

### Backend (Railway)
1. Connect your GitHub repository to Railway
2. Set environment variables:
   - `PYTHON_VERSION=3.11.9`
3. Deploy with start command: `gunicorn --bind 0.0.0.0:$PORT --chdir web app:app`

### Frontend (Vercel)
1. Connect repository to Vercel
2. Set root directory to `frontend`
3. Add environment variable:
   - `REACT_APP_BACKEND_URL=https://your-railway-app.up.railway.app`

## 📊 Model Performance

The audio classification model achieves:
- **Overall Accuracy**: 85%+
- **Security Alert Precision**: 90%+
- **Processing Time**: <2 seconds per 4-second audio clip
- **Supported Sample Rates**: Automatic resampling to 22050 Hz

## 🔧 API Endpoints

### `POST /api/test_audio_file`
Upload and analyze audio files for security threats.

**Request**: Multipart form data with audio file
**Response**:
```json
{
  "success": true,
  "filename": "audio.mp3",
  "duration": 10.5,
  "total_predictions": 8,
  "alerts_found": [
    {
      "start_time": 2.0,
      "end_time": 6.0,
      "alert": {
        "level": "CRITICAL",
        "message": "🚨 GUNSHOT DETECTED",
        "confidence": 0.92
      }
    }
  ],
  "all_results": [...]
}
```

## 🎨 User Interface

### Main Dashboard
- **File Upload Area**: Intuitive drag & drop or click to select audio files
- **Analysis Results**: Real-time display of detected events with confidence scores
- **Security Alerts**: Highlighted critical findings with severity levels
- **Confidence Visualization**: Interactive bars showing classification certainty
- **Prediction Timeline**: Detailed temporal analysis of audio segments

### Security Alert Levels
- 🔴 **Critical**: Immediate security threats (gunshots)
- 🟠 **High**: Emergency services (sirens)
- 🟡 **Medium**: Disturbances (dog barking)
- 🔵 **Low**: General urban sounds (car horns)

## 🔍 Technical Details

### Audio Processing Pipeline
1. **Input Validation**: Comprehensive file format and size checking
2. **Preprocessing**: Resampling to 22050 Hz, mono conversion
3. **Feature Extraction**: Mel-spectrogram computation (128 bands)
4. **Model Inference**: CNN prediction with confidence scoring
5. **Post-processing**: Sliding window analysis with temporal overlap
6. **Alert Generation**: Security-focused threat detection and classification

### Model Architecture
- **Input**: Mel-spectrograms (128 x 173 x 1)
- **Architecture**: Deep Convolutional Neural Network
- **Training Dataset**: UrbanSound8K (8,732 audio samples)
- **Optimization**: Adam optimizer with categorical crossentropy loss
- **Regularization**: Dropout layers and batch normalization

## 📁 Project Structure

```
Real-time-Audio-Event-Classification/
├── 📂 frontend/              # React web application
│   ├── 📂 src/
│   │   ├── 📂 components/    # Modular React components
│   │   ├── App.js           # Main application component
│   │   └── index.js         # Application entry point
│   ├── package.json         # Frontend dependencies
│   └── .env.production      # Production environment variables
├── 📂 web/                  # Flask backend server
│   ├── app.py              # Main Flask application with API routes
│   └── 📂 templates/       # HTML templates (legacy support)
├── 📂 src/                  # Core Python modules
│   ├── audio_preprocessing.py  # Audio processing utilities
│   ├── models.py              # Neural network architectures
│   └── realtime_inference.py  # Inference engine
├── 📂 models/               # Trained ML models
│   └── best_simple_model.keras  # Production-ready model
├── requirements.txt         # Python package dependencies
├── render.yaml             # Railway deployment configuration
└── README.md               # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Guidelines
- Follow PEP 8 for Python code formatting
- Use ESLint configuration for JavaScript code quality
- Write descriptive commit messages with proper scope
- Add comprehensive tests for new features
- Update documentation for API changes

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Swyam Patel**
- 📧 Email: Patelswyam80@gmail.com
- 🐙 GitHub: [@swyampatel](https://github.com/swyampatel)
- 🌐 LinkedIn: [Swyam Patel](https://linkedin.com/in/swyampatel)

## 🙏 Acknowledgments

- **UrbanSound8K Dataset**: J. Salamon, C. Jacoby and J. P. Bello for providing high-quality urban sound samples
- **TensorFlow Team**: For the powerful and accessible machine learning framework
- **Librosa Developers**: For excellent audio processing capabilities and documentation
- **React Community**: For the amazing frontend framework and ecosystem
- **Open Source Community**: For the countless libraries and tools that made this project possible

## 🔮 Future Enhancements

- [ ] Real-time audio streaming analysis with WebRTC
- [ ] Mobile application for iOS and Android
- [ ] Custom model training interface for specific use cases
- [ ] Multi-language support for global accessibility
- [ ] Advanced alert customization and filtering
- [ ] Integration with existing security systems and APIs
- [ ] Batch processing capabilities for multiple files
- [ ] Advanced audio visualization and spectral analysis
- [ ] Machine learning model versioning and A/B testing
- [ ] Comprehensive analytics dashboard with reporting

## 🛡️ Security & Privacy

### Important Notes
- This system is designed for **legitimate security monitoring purposes only**
- Ensure compliance with local privacy laws and regulations
- Always inform individuals when audio monitoring is active
- Data is processed securely and not stored permanently
- All audio analysis happens on secure cloud infrastructure

### Data Handling
- Audio files are processed in memory and not permanently stored
- No personal information is collected or retained
- All communications use HTTPS encryption
- Model predictions are stateless and privacy-preserving

## 📞 Support & Contact

For questions, issues, or collaboration opportunities:

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/swyampatel/Real-time-Audio-Event-Classification/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/swyampatel/Real-time-Audio-Event-Classification/discussions)
- **📧 Direct Contact**: Patelswyam80@gmail.com
- **🚀 Feature Requests**: Open an issue with the "enhancement" label

## 🎯 Use Cases

### Security Applications
- **Residential Security**: Home monitoring systems with sound-based alerts
- **Commercial Buildings**: Office and retail space monitoring
- **Public Spaces**: Parks, schools, and community area surveillance
- **Emergency Response**: Automatic detection of gunshots and distress signals

### Smart City Integration
- **Urban Planning**: Traffic noise analysis and city soundscape monitoring
- **Emergency Services**: Faster response to acoustic emergency signals
- **Environmental Monitoring**: Construction noise and pollution tracking
- **Public Safety**: Proactive threat detection in public spaces

---

**⚡ Built with passion for audio AI and security applications**

*🚀 Ready to deploy? This comprehensive system can be up and running in under 30 minutes!*

*Last updated: August 2025 | Version 2.0*