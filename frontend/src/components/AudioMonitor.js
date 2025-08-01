import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import Header from './Header';
import LiveAnalysis from './LiveAnalysis';
import SecurityAlerts from './SecurityAlerts';
import ProbabilityChart from './ProbabilityChart';
import FileUpload from './FileUpload';
import './AudioMonitor.css';

const AudioMonitor = () => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState({
    class: 'Ready to Monitor',
    confidence: 0,
    probabilities: {}
  });
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState({
    totalDetections: 0,
    totalAlerts: 0,
    fps: '--'
  });

  const mediaRecorderRef = useRef(null);
  const audioStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const lastUpdateRef = useRef(Date.now());

  // Get backend URL from environment variable or use default
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(backendUrl, {
      transports: ['websocket', 'polling']
    });

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    });

    newSocket.on('prediction', (data) => {
      handlePrediction(data);
    });

    newSocket.on('error', (error) => {
      console.error('Socket error:', error);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [backendUrl]);

  const handlePrediction = (data) => {
    setCurrentPrediction(data);
    
    // Update metrics
    setMetrics(prev => ({
      ...prev,
      totalDetections: prev.totalDetections + 1,
      fps: calculateFPS()
    }));

    // Handle alerts
    if (data.alert) {
      setAlerts(prev => [data.alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
      setMetrics(prev => ({
        ...prev,
        totalAlerts: prev.totalAlerts + 1
      }));
    }
  };

  const calculateFPS = () => {
    const now = Date.now();
    const timeDiff = (now - lastUpdateRef.current) / 1000;
    lastUpdateRef.current = now;
    return timeDiff > 0 ? (1 / timeDiff).toFixed(1) : '--';
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 22050,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false
        } 
      });

      audioStreamRef.current = stream;
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      
      setIsRecording(true);
      processAudioStream();
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Microphone access denied or not available');
    }
  };

  const processAudioStream = () => {
    if (!audioStreamRef.current || !socket) return;

    const recordingLength = 4000; // 4 seconds
    const chunks = [];

    mediaRecorderRef.current = new MediaRecorder(audioStreamRef.current);
    
    mediaRecorderRef.current.ondataavailable = (event) => {
      chunks.push(event.data);
    };

    mediaRecorderRef.current.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      await convertAndSendAudio(blob);
      
      // Continue recording if still active
      if (isRecording) {
        setTimeout(() => processAudioStream(), 1000);
      }
    };

    // Record for 4 seconds
    mediaRecorderRef.current.start();
    setTimeout(() => {
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    }, recordingLength);
  };

  const convertAndSendAudio = async (blob) => {
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      
      // Get audio data (mono)
      const audioData = audioBuffer.getChannelData(0);
      
      // Resample to 22050 Hz if needed
      let resampledData = audioData;
      if (audioBuffer.sampleRate !== 22050) {
        const ratio = 22050 / audioBuffer.sampleRate;
        const newLength = Math.floor(audioData.length * ratio);
        resampledData = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
          const srcIndex = i / ratio;
          const srcIndexFloor = Math.floor(srcIndex);
          const srcIndexCeil = Math.ceil(srcIndex);
          const fraction = srcIndex - srcIndexFloor;
          
          if (srcIndexCeil < audioData.length) {
            resampledData[i] = audioData[srcIndexFloor] * (1 - fraction) + 
                              audioData[srcIndexCeil] * fraction;
          } else {
            resampledData[i] = audioData[srcIndexFloor];
          }
        }
      }
      
      // Convert to base64
      const float32Array = new Float32Array(resampledData);
      const uint8Array = new Uint8Array(float32Array.buffer);
      
      let binary = '';
      const chunkSize = 8192;
      for (let i = 0; i < uint8Array.length; i += chunkSize) {
        const chunk = uint8Array.slice(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
      }
      const base64Audio = btoa(binary);
      
      // Send to server
      socket.emit('audio_data', { audio: base64Audio });
    } catch (error) {
      console.error('Error processing audio:', error);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('audio', file);
    
    try {
      const response = await fetch(`${backendUrl}/api/test_audio_file`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error uploading file:', error);
      throw error;
    }
  };

  return (
    <div className="audio-monitor">
      <Header isConnected={isConnected} isRecording={isRecording} />
      
      <div className="container">
        <div className="dashboard-grid">
          <LiveAnalysis
            currentPrediction={currentPrediction}
            isRecording={isRecording}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            metrics={metrics}
          />
          
          <SecurityAlerts alerts={alerts} />
          
          <ProbabilityChart probabilities={currentPrediction.probabilities} />
        </div>
        
        <FileUpload onFileUpload={handleFileUpload} backendUrl={backendUrl} />
      </div>
    </div>
  );
};

export default AudioMonitor;