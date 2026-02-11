import React, { useEffect, useState } from 'react';
import { fetchBeliefs } from '../api/client';
import './BeliefNetwork.css';

/**
 * BeliefNetwork Component
 *
 * Displays the belief network with confidence visualization.
 * High confidence beliefs are shown in green, low confidence in red.
 * Includes sorting, filtering, and detailed belief information.
 */
const BeliefNetwork = () => {
  const [beliefs, setBeliefs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('last_updated');
  const [sortDir, setSortDir] = useState('desc');

  useEffect(() => {
    loadBeliefs();
  }, [sortBy, sortDir]);

  const loadBeliefs = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await fetchBeliefs({
        limit: 100,
        order_by: sortBy,
        order_dir: sortDir
      });

      if (!data || !data.beliefs) {
        throw new Error('Invalid beliefs data received');
      }

      setBeliefs(data.beliefs);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleSortChange = (field) => {
    if (sortBy === field) {
      setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) {
      return 'high';
    } else if (confidence >= 0.4) {
      return 'medium';
    } else {
      return 'low';
    }
  };

  const getConfidenceLabel = (confidence) => {
    const level = getConfidenceColor(confidence);
    return {
      high: 'High Confidence',
      medium: 'Medium Confidence',
      low: 'Low Confidence'
    }[level];
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) {
      return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    } else if (diffDays < 7) {
      return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  if (loading) {
    return (
      <div className="belief-container">
        <div className="belief-loading">
          <div className="spinner"></div>
          <p>Loading beliefs...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="belief-container">
        <div className="belief-error">
          <h3>Failed to load beliefs</h3>
          <p>{error}</p>
          <button onClick={loadBeliefs}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="belief-container">
      <div className="belief-header">
        <div className="belief-stats">
          <span className="stat-item">
            <strong>{beliefs.length}</strong> beliefs
          </span>
          <span className="stat-separator">•</span>
          <span className="stat-item">
            <strong>{beliefs.filter((b) => b.confidence >= 0.7).length}</strong> high confidence
          </span>
          <span className="stat-separator">•</span>
          <span className="stat-item">
            <strong>{beliefs.filter((b) => b.confidence < 0.4).length}</strong> low confidence
          </span>
        </div>

        <div className="belief-controls">
          <select
            className="sort-select"
            value={sortBy}
            onChange={(e) => handleSortChange(e.target.value)}
            aria-label="Sort beliefs by"
          >
            <option value="last_updated">Last Updated</option>
            <option value="confidence">Confidence</option>
            <option value="created_at">Created</option>
            <option value="evidence_count">Evidence Count</option>
          </select>
          <button
            className="sort-dir-btn"
            onClick={() => setSortDir(sortDir === 'desc' ? 'asc' : 'desc')}
            title={sortDir === 'desc' ? 'Sort descending' : 'Sort ascending'}
          >
            {sortDir === 'desc' ? '↓' : '↑'}
          </button>
          <button className="refresh-btn" onClick={loadBeliefs} title="Refresh">
            ⟲
          </button>
        </div>
      </div>

      <div className="belief-legend">
        <span className="legend-title">Confidence Levels:</span>
        <div className="legend-item">
          <span className="confidence-badge confidence-high">≥ 0.7</span>
          <span className="legend-label">High (Green)</span>
        </div>
        <div className="legend-item">
          <span className="confidence-badge confidence-medium">0.4 - 0.69</span>
          <span className="legend-label">Medium (Yellow)</span>
        </div>
        <div className="legend-item">
          <span className="confidence-badge confidence-low">&lt; 0.4</span>
          <span className="legend-label">Low (Red)</span>
        </div>
      </div>

      {beliefs.length === 0 ? (
        <div className="belief-empty">
          <span className="empty-icon">💡</span>
          <p>No beliefs found in the network</p>
        </div>
      ) : (
        <div className="belief-list">
          {beliefs.map((belief) => (
            <div key={belief.id} className="belief-item">
              <div className="belief-item-header">
                <div className="belief-title">
                  <span className={`confidence-indicator confidence-${getConfidenceColor(belief.confidence)}`}></span>
                  <span className="belief-id">{belief.id}</span>
                </div>
                <div className="belief-confidence">
                  <span className={`confidence-badge confidence-${getConfidenceColor(belief.confidence)}`}>
                    {(belief.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="confidence-label">{getConfidenceLabel(belief.confidence)}</span>
                </div>
              </div>

              <div className="belief-content">
                {belief.content}
              </div>

              <div className="belief-meta">
                <div className="meta-row">
                  <span className="meta-item">
                    <span className="meta-icon">📊</span>
                    <span className="meta-label">Evidence:</span>
                    <strong>{belief.supporting_evidence || 0}</strong> supporting,
                    <strong>{belief.contradicting_evidence || 0}</strong> contradicting
                  </span>
                </div>
                <div className="meta-row">
                  <span className="meta-item">
                    <span className="meta-icon">📅</span>
                    <span className="meta-label">Created:</span>
                    {formatDate(belief.created_at)}
                  </span>
                  <span className="meta-separator">•</span>
                  <span className="meta-item">
                    <span className="meta-label">Updated:</span>
                    {formatDate(belief.last_updated)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BeliefNetwork;
