import React from 'react';
import './SecurityAlerts.css';

const SecurityAlerts = ({ alerts }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon alert-icon">
          <i className="fas fa-exclamation-triangle"></i>
        </div>
        <h2 className="card-title">Security Alerts</h2>
      </div>
      
      <div className="alert-list">
        {alerts.length === 0 ? (
          <div className="no-alerts">
            <i className="fas fa-shield-check"></i>
            <p>No security alerts detected</p>
          </div>
        ) : (
          alerts.map((alert, index) => (
            <div 
              key={`${alert.timestamp}-${index}`} 
              className={`alert-item alert-${alert.level.toLowerCase()}`}
            >
              <div className="alert-content">
                <div className="alert-info">
                  <strong>{alert.message}</strong>
                  <div className="alert-time">
                    {formatTime(alert.timestamp)}
                  </div>
                </div>
                <div className="alert-confidence">
                  <div className="confidence-value">
                    {Math.round(alert.confidence * 100)}%
                  </div>
                  <div className="confidence-label">confidence</div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default SecurityAlerts;