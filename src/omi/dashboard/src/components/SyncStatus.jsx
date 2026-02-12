import React, { useEffect, useState } from 'react';
import './SyncStatus.css';

/**
 * SyncStatus Component
 *
 * Displays distributed sync status including topology, instances, and lag metrics.
 * Shows real-time synchronization state for multi-instance deployments.
 */
const SyncStatus = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStatus();
    // Auto-refresh every 5 seconds
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/sync/status');

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      if (!data) {
        throw new Error('Invalid sync status data received');
      }

      setStatus(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return 'Invalid date';
    }
  };

  const formatLag = (lagSeconds) => {
    if (lagSeconds === null || lagSeconds === undefined) return 'N/A';
    if (lagSeconds < 1) return '< 1s';
    if (lagSeconds < 60) return `${lagSeconds.toFixed(1)}s`;
    const minutes = Math.floor(lagSeconds / 60);
    const seconds = Math.floor(lagSeconds % 60);
    return `${minutes}m ${seconds}s`;
  };

  const getStateColor = (state) => {
    const stateMap = {
      'idle': '#94a3b8',      // slate
      'syncing': '#3b82f6',   // blue
      'connected': '#10b981', // green
      'error': '#ef4444',     // red
      'partitioned': '#f59e0b' // amber
    };
    return stateMap[state?.toLowerCase()] || '#94a3b8';
  };

  const getStateIcon = (state) => {
    const iconMap = {
      'idle': '⏸️',
      'syncing': '🔄',
      'connected': '✅',
      'error': '❌',
      'partitioned': '⚠️'
    };
    return iconMap[state?.toLowerCase()] || '❓';
  };

  if (loading && !status) {
    return (
      <div className="sync-container">
        <div className="sync-loading">
          <div className="spinner"></div>
          <p>Loading sync status...</p>
        </div>
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="sync-container">
        <div className="sync-error">
          <h3>Failed to load sync status</h3>
          <p>{error}</p>
          <button onClick={loadStatus}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="sync-container">
      <div className="sync-header">
        <div className="sync-summary">
          <div className="summary-card">
            <div className="summary-icon">{getStateIcon(status.state)}</div>
            <div className="summary-content">
              <div className="summary-value" style={{ color: getStateColor(status.state) }}>
                {status.state || 'Unknown'}
              </div>
              <div className="summary-label">Sync State</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">🏛️</div>
            <div className="summary-content">
              <div className="summary-value">{status.topology || 'Unknown'}</div>
              <div className="summary-label">Topology</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">{status.is_leader ? '👑' : '📡'}</div>
            <div className="summary-content">
              <div className="summary-value">{status.is_leader ? 'Leader' : 'Follower'}</div>
              <div className="summary-label">Role</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">⏱️</div>
            <div className="summary-content">
              <div className="summary-value">{formatLag(status.lag_seconds)}</div>
              <div className="summary-label">Sync Lag</div>
            </div>
          </div>
        </div>

        <button className="refresh-btn" onClick={loadStatus} title="Refresh sync status">
          🔄
        </button>
      </div>

      <div className="sync-details">
        <div className="detail-section">
          <h3 className="section-title">Instance Information</h3>
          <div className="detail-grid">
            <div className="detail-item">
              <span className="detail-label">Instance ID:</span>
              <span className="detail-value">{status.instance_id || 'N/A'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Last Sync:</span>
              <span className="detail-value">{formatTimestamp(status.last_sync)}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Sync Operations:</span>
              <span className="detail-value">{status.sync_count || 0}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Error Count:</span>
              <span className="detail-value" style={{ color: status.error_count > 0 ? '#ef4444' : 'inherit' }}>
                {status.error_count || 0}
              </span>
            </div>
          </div>

          {status.last_error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              <span className="error-text">{status.last_error}</span>
            </div>
          )}
        </div>

        <div className="detail-section">
          <h3 className="section-title">Network Status</h3>
          <div className="detail-grid">
            <div className="detail-item">
              <span className="detail-label">Registered Instances:</span>
              <span className="detail-value">{status.registered_instances || 0}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Healthy Instances:</span>
              <span className="detail-value" style={{
                color: status.healthy_instances === status.registered_instances ? '#10b981' : '#f59e0b'
              }}>
                {status.healthy_instances || 0}
              </span>
            </div>
            <div className="detail-item health-indicator">
              <span className="detail-label">Network Health:</span>
              <div className="health-bar">
                <div
                  className="health-fill"
                  style={{
                    width: `${status.registered_instances > 0
                      ? (status.healthy_instances / status.registered_instances) * 100
                      : 0}%`,
                    backgroundColor: status.healthy_instances === status.registered_instances ? '#10b981' : '#f59e0b'
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {status.topology_info && Object.keys(status.topology_info).length > 0 && (
          <div className="detail-section">
            <h3 className="section-title">Topology Details</h3>
            <div className="topology-info">
              <pre className="topology-json">
                {JSON.stringify(status.topology_info, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SyncStatus;
