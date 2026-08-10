import React, { useState } from 'react';
import WebcamStreamer from './components/WebcamStreamer';
import StatusBadge from './components/StatusBadge';
import AnalyticsCard from './components/AnalyticsCard';

export default function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isMirrored, setIsMirrored] = useState(false);
  const [predictionData, setPredictionData] = useState(null);

  const handleStart = () => setIsStreaming(true);
  const handleStop = () => {
    setIsStreaming(false);
    setPredictionData(null);
  };
  const toggleMirror = () => setIsMirrored(!isMirrored);

  return (
    <div>
      {/* Header */}
      <header className="app-header">
        <div className="logo-group">
          <div className="logo-badge">
            <span style={{ fontSize: '1.5rem' }}>😷</span>
          </div>
          <div>
            <h1 className="app-title">FACE MASK DETECTION SYSTEM</h1>
            <p style={{ fontSize: '0.8rem', color: '#9CA3AF' }}>Real-time COVID Safety Verification System</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: isStreaming ? '#00FF87' : '#9CA3AF',
            boxShadow: isStreaming ? '0 0 10px #00FF87' : 'none'
          }} />
          <span style={{ fontSize: '0.85rem', color: '#D1D5DB', fontWeight: 600 }}>
            {isStreaming ? 'SYSTEM LIVE' : 'SYSTEM READY'}
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Video Camera Stream */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="glass-panel" style={{ padding: '1rem' }}>
            <WebcamStreamer
              isStreaming={isStreaming}
              isMirrored={isMirrored}
              onPredictionUpdate={setPredictionData}
            />
          </div>

          {/* Action Control Bar */}
          <div className="glass-panel" style={{ padding: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              {!isStreaming ? (
                <button className="btn-primary" onClick={handleStart}>
                  ▶ Start Camera
                </button>
              ) : (
                <button className="btn-danger" onClick={handleStop}>
                  ⏹ Stop Camera
                </button>
              )}

              <button className="btn-secondary" onClick={toggleMirror}>
                🔄 {isMirrored ? 'Disable Mirror' : 'Enable Mirror'}
              </button>
            </div>

            <span style={{ fontSize: '0.85rem', color: '#9CA3AF' }}>
              60 FPS Camera Stream + PyTorch Deep Learning
            </span>
          </div>
        </section>

        {/* Right Column: Status & Analytics Sidebar */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <StatusBadge
            status={predictionData?.overall_status}
            totalFaces={predictionData?.total_faces || 0}
            maskedCount={predictionData?.masked_count || 0}
            unmaskedCount={predictionData?.unmasked_count || 0}
            faces={predictionData?.faces}
          />

          <AnalyticsCard data={predictionData} />
        </aside>
      </main>
    </div>
  );
}
