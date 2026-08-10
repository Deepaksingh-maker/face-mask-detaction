import React from 'react';

export default function AnalyticsCard({ data }) {
  const total = data?.total_faces || 0;
  const masked = data?.masked_count || 0;
  const unmasked = data?.unmasked_count || 0;

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <h3 style={{ fontSize: '0.95rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: '#9CA3AF', marginBottom: '1rem' }}>
        📊 Live Camera Analytics
      </h3>

      <div className="stat-row">
        <span style={{ color: '#D1D5DB' }}>Total Faces Detected</span>
        <span className="stat-value" style={{ color: '#F3F4F6' }}>{total}</span>
      </div>

      <div className="stat-row">
        <span style={{ color: '#D1D5DB' }}>Masked (Safe)</span>
        <span className="stat-value" style={{ color: '#00FF87' }}>{masked}</span>
      </div>

      <div className="stat-row">
        <span style={{ color: '#D1D5DB' }}>Unmasked (Unsafe)</span>
        <span className="stat-value" style={{ color: '#FF2E54' }}>{unmasked}</span>
      </div>

      <div className="stat-row" style={{ borderBottom: 'none' }}>
        <span style={{ color: '#D1D5DB' }}>AI Model Core</span>
        <span className="stat-value" style={{ color: '#3B82F6', fontSize: '0.85rem' }}>PyTorch / Edge CNN</span>
      </div>
    </div>
  );
}
