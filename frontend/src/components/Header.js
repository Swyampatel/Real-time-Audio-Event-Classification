import React from 'react';
import './Header.css';

const Header = ({ isConnected, isRecording }) => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <div className="logo-icon">
            <i className="fas fa-shield-alt"></i>
          </div>
          <div className="logo-text">
            <h1>Audio Security Monitor</h1>
            <p>Real-time threat detection system</p>
          </div>
        </div>
        <div className="status-badge">
          <div className={`status-indicator ${isConnected ? 'connected' : ''} ${isRecording ? 'recording' : ''}`}></div>
          <span>{isConnected ? (isRecording ? 'Recording' : 'Connected') : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
};

export default Header;