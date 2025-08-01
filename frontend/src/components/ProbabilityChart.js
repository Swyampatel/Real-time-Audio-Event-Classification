import React from 'react';
import './ProbabilityChart.css';

const ProbabilityChart = ({ probabilities = {}, currentClass, confidence }) => {
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

  const getClassColor = (className) => {
    const colors = {
      'gun_shot': 'var(--danger)',
      'siren': 'var(--warning)',
      'dog_bark': '#fbbf24',
      'car_horn': 'var(--accent-primary)',
      'default': 'var(--accent-secondary)'
    };
    return colors[className] || colors.default;
  };

  const formatClassName = (className) => {
    return className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Sort probabilities and show top 5
  const sortedProbs = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon probability-icon">
          <i className="fas fa-chart-bar"></i>
        </div>
        <h2 className="card-title">Last Analysis Results</h2>
      </div>
      
      {currentClass && (
        <div className="current-result">
          <div className="result-class">
            {getClassEmoji(currentClass)} {formatClassName(currentClass)}
          </div>
          <div className="result-confidence">
            {Math.round(confidence * 100)}% confidence
          </div>
        </div>
      )}
      
      <div className="probability-bars">
        {sortedProbs.length === 0 ? (
          <div className="no-data">
            <p>Upload an audio file to see analysis results</p>
          </div>
        ) : (
          sortedProbs.map(([className, prob]) => {
            const percentage = Math.round(prob * 100);
            return (
              <div key={className} className="prob-item">
                <div className="prob-label">
                  {getClassEmoji(className)} {formatClassName(className)}
                </div>
                <div className="prob-bar">
                  <div 
                    className="prob-fill" 
                    style={{ 
                      width: `${percentage}%`, 
                      background: getClassColor(className) 
                    }}
                  ></div>
                </div>
                <div className="prob-value">{percentage}%</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default ProbabilityChart;