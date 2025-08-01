import React, { useState, useRef } from 'react';
import './FileUpload.css';

const FileUpload = ({ onFileUpload, backendUrl }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('audio/')) {
        handleFileSelect(file);
      } else {
        alert('Please drop an audio file');
      }
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      // Shake animation
      const uploadSection = document.querySelector('.upload-section');
      uploadSection.style.animation = 'shake 0.5s ease-in-out';
      setTimeout(() => {
        uploadSection.style.animation = '';
      }, 500);
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = await onFileUpload(selectedFile);
      setResults(result);
    } catch (error) {
      console.error('Error analyzing file:', error);
      alert('Error analyzing file: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (seconds) => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}:${remainingSeconds.toFixed(1).padStart(4, '0')}`;
    }
  };

  return (
    <div className="file-upload-section">
      <div 
        className={`upload-section ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="upload-container">
          <div className="upload-icon">
            <i className="fas fa-file-audio"></i>
          </div>
          
          <div className="upload-text">
            <div className="upload-title">Test Audio Files</div>
            <div className="upload-subtitle">Upload audio files to analyze for security threats</div>
          </div>
          
          <div className="upload-buttons">
            <input 
              type="file" 
              ref={fileInputRef}
              accept="audio/*" 
              className="upload-input"
              onChange={handleFileInputChange}
            />
            <label 
              htmlFor="audioFile" 
              className={`upload-label ${selectedFile ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <i className={`fas ${selectedFile ? 'fa-check-circle' : 'fa-folder-open'}`}></i>
              <span>
                {selectedFile 
                  ? (selectedFile.name.length > 25 
                      ? selectedFile.name.substring(0, 22) + '...' 
                      : selectedFile.name)
                  : 'Choose File'
                }
              </span>
            </label>
            <button 
              className={`analyze-btn ${isAnalyzing ? 'loading' : ''}`}
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <div className="loading-spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-brain"></i>
                  <span>Analyze Audio</span>
                </>
              )}
            </button>
          </div>
          
          <div className="supported-formats">
            <span className="format-tag">MP3</span>
            <span className="format-tag">WAV</span>
            <span className="format-tag">M4A</span>
            <span className="format-tag">FLAC</span>
            <span className="format-tag">OGG</span>
          </div>
        </div>
      </div>

      {results && (
        <div className="upload-results">
          {results.error ? (
            <div className="error-message">
              Error: {results.error}
            </div>
          ) : (
            <div className="results-content">
              <div className="results-summary">
                <div className="summary-grid">
                  <div className="summary-item">
                    <div className="summary-value">{results.filename}</div>
                    <div className="summary-label">Filename</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-value">{results.duration.toFixed(2)}s</div>
                    <div className="summary-label">Duration</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-value">{results.total_predictions}</div>
                    <div className="summary-label">Predictions</div>
                  </div>
                </div>
              </div>

              {results.alerts_found.length > 0 ? (
                <div className="alerts-section">
                  <div className="alerts-header">
                    <strong>🚨 Security Alerts Detected:</strong>
                  </div>
                  {results.alerts_found.map((item, index) => {
                    const alert = item.alert;
                    const startTime = formatTime(item.start_time);
                    const endTime = formatTime(item.end_time);
                    return (
                      <div key={index} className={`alert-item alert-${alert.level.toLowerCase()}`}>
                        <div className="alert-content">
                          <div className="alert-info">
                            <strong>{alert.message}</strong>
                            <div className="alert-time-range">
                              Detected in: {startTime} - {endTime}
                            </div>
                          </div>
                          <div className="alert-confidence">
                            <div className="confidence-value">
                              {Math.round(item.confidence * 100)}%
                            </div>
                            <div className="confidence-label">confidence</div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="no-threats">
                  <i className="fas fa-check-circle"></i>
                  <div>No security threats detected</div>
                </div>
              )}

              <details className="all-predictions">
                <summary>View All Predictions</summary>
                <div className="predictions-list">
                  {results.all_results.map((pred, index) => {
                    const isAlert = pred.alert ? '🚨' : '';
                    const startTime = formatTime(pred.start_time);
                    const endTime = formatTime(pred.end_time);
                    const confidence = Math.round(pred.confidence * 100);
                    const confidenceColor = confidence > 70 ? 'var(--success)' : 
                                           confidence > 40 ? 'var(--warning)' : 'var(--text-secondary)';
                    
                    return (
                      <div key={index} className="prediction-item">
                        <div className="prediction-info">
                          <div className="prediction-class">
                            {pred.class.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                          <div className="prediction-time">
                            {startTime} - {endTime}
                          </div>
                        </div>
                        <div className="prediction-confidence">
                          <span style={{ color: confidenceColor, fontWeight: '600' }}>
                            {confidence}% {isAlert}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUpload;