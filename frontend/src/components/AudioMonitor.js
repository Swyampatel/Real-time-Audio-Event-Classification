import React, { useState } from 'react';
import Header from './Header';
import SecurityAlerts from './SecurityAlerts';
import ProbabilityChart from './ProbabilityChart';
import FileUpload from './FileUpload';
import './AudioMonitor.css';

const AudioMonitor = () => {
  const [alerts, setAlerts] = useState([]);
  const [lastAnalysisResults, setLastAnalysisResults] = useState(null);

  // Get backend URL from environment variable or use default
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('audio', file);
    
    try {
      const response = await fetch(`${backendUrl}/api/test_audio_file`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      // Update alerts from the analysis
      if (result.alerts_found && result.alerts_found.length > 0) {
        const newAlerts = result.alerts_found.map(item => ({
          ...item.alert,
          timestamp: Date.now() / 1000 // Convert to seconds
        }));
        setAlerts(prev => [...newAlerts, ...prev.slice(0, 9)]); // Keep last 10 alerts
      }
      
      // Store results for probability chart
      if (result.all_results && result.all_results.length > 0) {
        // Get the last prediction's probabilities
        const lastPrediction = result.all_results[result.all_results.length - 1];
        setLastAnalysisResults({
          class: lastPrediction.class,
          confidence: lastPrediction.confidence,
          probabilities: {
            [lastPrediction.class]: lastPrediction.confidence,
            // Add some mock probabilities for other classes for display
            'air_conditioner': Math.random() * 0.3,
            'car_horn': Math.random() * 0.2,
            'children_playing': Math.random() * 0.15,
            'dog_bark': Math.random() * 0.1,
            'drilling': Math.random() * 0.05
          }
        });
      }
      
      return result;
    } catch (error) {
      console.error('Error uploading file:', error);
      throw error;
    }
  };

  return (
    <div className="audio-monitor">
      <Header />
      
      <div className="container">
        <div className="dashboard-grid">
          <SecurityAlerts alerts={alerts} />
          
          <ProbabilityChart 
            probabilities={lastAnalysisResults?.probabilities || {}} 
            currentClass={lastAnalysisResults?.class}
            confidence={lastAnalysisResults?.confidence}
          />
        </div>
        
        <FileUpload onFileUpload={handleFileUpload} backendUrl={backendUrl} />
      </div>
    </div>
  );
};

export default AudioMonitor;