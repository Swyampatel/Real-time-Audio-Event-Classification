import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <div className="logo-icon">
            <i className="fas fa-shield-alt"></i>
          </div>
          <div className="logo-text">
            <h1>Audio Security Monitor</h1>
            <p>AI-powered audio threat detection</p>
          </div>
        </div>
        <div className="status-badge">
          <div className="status-indicator connected"></div>
          <span>Ready for Analysis</span>
        </div>
      </div>
    </header>
  );
};

export default Header;