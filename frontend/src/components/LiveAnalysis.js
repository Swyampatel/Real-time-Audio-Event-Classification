import React, { useRef, useEffect } from 'react';
import './LiveAnalysis.css';

const LiveAnalysis = ({ currentPrediction, isRecording, onStartRecording, onStopRecording, metrics }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Initial flat line
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Animate waveform when recording
    if (isRecording) {
      let phase = 0;
      const animate = () => {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
        ctx.beginPath();
        
        for (let x = 0; x < canvas.width; x += 5) {
          const y = canvas.height / 2 + Math.sin((x + phase) * 0.02) * 30 * Math.random();
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        
        ctx.stroke();
        phase += 5;
        animationRef.current = requestAnimationFrame(animate);
      };
      animate();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRecording]);

  const getClassEmoji = (className) => {
    const emojis = {
      'air_conditioner': '❄️',
      'car_horn': '🚗',
      'children_playing': '👶',
      'dog_bark': '🐕',
      'drilling': '🔧',
      'engine_idling': '🚛',
      'gun_shot': '🔫',
      'jackhammer': '🔨',
      'siren': '🚨',
      'street_music': '🎵'
    };
    return emojis[className] || '🔊';
  };

  const formatClassName = (className) => {
    if (className === 'Ready to Monitor' || className === 'Monitoring...') {
      return className;
    }
    return className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const displayClass = currentPrediction.class;
  const confidence = Math.round(currentPrediction.confidence * 100);

  return (
    <div className="card visualizer-card">
      <div className="card-header">
        <div className="card-icon">
          <i className="fas fa-waveform"></i>
        </div>
        <h2 className="card-title">Live Audio Analysis</h2>
      </div>
      
      <div className="current-detection">
        <div className="detection-class">
          {displayClass !== 'Ready to Monitor' && displayClass !== 'Monitoring...' && getClassEmoji(displayClass)} {formatClassName(displayClass)}
        </div>
        <div className="confidence-meter">
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${confidence}%` }}
            ></div>
          </div>
          <div className="confidence-text">{confidence}%</div>
        </div>
      </div>

      <div className="btn-group">
        <button 
          className="btn btn-primary" 
          onClick={onStartRecording}
          disabled={isRecording}
        >
          <i className="fas fa-play"></i>
          Start Monitoring
        </button>
        <button 
          className="btn btn-danger" 
          onClick={onStopRecording}
          disabled={!isRecording}
        >
          <i className="fas fa-stop"></i>
          Stop Monitoring
        </button>
      </div>

      <div className="waveform-container">
        <canvas ref={canvasRef} className="waveform-canvas"></canvas>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{metrics.fps}</div>
          <div className="metric-label">FPS</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{metrics.totalDetections}</div>
          <div className="metric-label">Detections</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{metrics.totalAlerts}</div>
          <div className="metric-label">Alerts</div>
        </div>
      </div>
    </div>
  );
};

export default LiveAnalysis;