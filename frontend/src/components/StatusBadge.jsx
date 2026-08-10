import React from 'react';

export default function StatusBadge({ status, totalFaces, maskedCount, unmaskedCount, faces }) {
  let badgeClass = 'no-face';
  let icon = '🔍';
  let title = 'NO FACE DETECTED';
  let desc = 'Align your face in front of the video camera...';
  let color = '#9CA3AF';
  let confidenceText = '';

  if (totalFaces > 0) {
    if (unmaskedCount > 0) {
      badgeClass = 'unsafe';
      icon = '🚨';
      title = '⚠ UNSAFE PERSON DETECTED';
      desc = `ALERT: ${unmaskedCount} unmasked face(s) detected. Please wear a mask!`;
      color = '#FF2E54';
    } else if (status === 'CHECKING') {
      badgeClass = 'no-face';
      icon = '🔍';
      title = '🔍 ANALYZING...';
      desc = 'Stabilizing live prediction metrics...';
      color = '#FACC15';
    } else {
      badgeClass = 'safe';
      icon = '✅';
      title = '✓ ALL SAFE';
      desc = `STATUS: All ${totalFaces} detected face(s) are wearing masks!`;
      color = '#00FF87';
    }

    if (faces && faces.length > 0) {
      const topFace = faces[0];
      const confPercent = (topFace.confidence * 100).toFixed(1);
      confidenceText = `SMOOTHED CONFIDENCE: ${confPercent}%`;
    }
  }

  return (
    <div className={`status-badge ${badgeClass}`}>
      <div style={{ fontSize: '2.5rem', marginBottom: '0.25rem' }}>{icon}</div>
      <div className="status-text" style={{ color }}>{title}</div>
      <div style={{ fontSize: '0.85rem', color: '#D1D5DB', marginTop: '0.4rem' }}>{desc}</div>
      
      {confidenceText && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.35rem 0.75rem',
          background: 'rgba(0,0,0,0.3)',
          borderRadius: '20px',
          display: 'inline-block',
          fontSize: '0.8rem',
          fontWeight: 700,
          color: color,
          letterSpacing: '0.5px'
        }}>
          {confidenceText}
        </div>
      )}
    </div>
  );
}
