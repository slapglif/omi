import React, { useEffect, useState } from 'react';
import './VersionTimeline.css';

/**
 * VersionTimeline Component
 *
 * Displays chronological timeline of memory versions (CREATE/UPDATE/DELETE operations).
 * Fetches data from /api/v1/dashboard/versions/timeline endpoint.
 * Shows operation type, timestamp, content preview, and supports date range filtering.
 */
const VersionTimeline = () => {
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortDir, setSortDir] = useState('desc');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [operationType, setOperationType] = useState('');

  useEffect(() => {
    fetchVersions();
  }, [sortDir, dateFrom, dateTo, operationType]);

  const fetchVersions = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams({
        limit: '100',
        offset: '0'
      });

      if (dateFrom) {
        params.append('date_from', dateFrom);
      }

      if (dateTo) {
        params.append('date_to', dateTo);
      }

      if (operationType) {
        params.append('operation_type', operationType);
      }

      const url = `/api/v1/dashboard/versions/timeline?${params.toString()}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch versions: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      const versionsList = data.versions || [];
      setVersions(versionsList);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchVersions();
  };

  const handleClearFilters = () => {
    setDateFrom('');
    setDateTo('');
    setOperationType('');
  };

  const toggleSortDirection = () => {
    setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown time';

    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffSecs < 60) {
      return `${diffSecs} second${diffSecs !== 1 ? 's' : ''} ago`;
    } else if (diffMins < 60) {
      return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleString();
    }
  };

  const getOperationIcon = (operationType) => {
    const icons = {
      'CREATE': '➕',
      'UPDATE': '✏️',
      'DELETE': '🗑️',
      'default': '📝'
    };
    return icons[operationType] || icons.default;
  };

  const getOperationColor = (operationType) => {
    if (operationType === 'CREATE') return 'create';
    if (operationType === 'UPDATE') return 'update';
    if (operationType === 'DELETE') return 'delete';
    return 'default';
  };

  const truncateContent = (content, maxLength = 150) => {
    if (!content) return '';
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  const getSortedVersions = () => {
    const sorted = [...versions];
    sorted.sort((a, b) => {
      const timeA = new Date(a.created_at).getTime();
      const timeB = new Date(b.created_at).getTime();
      return sortDir === 'desc' ? timeB - timeA : timeA - timeB;
    });
    return sorted;
  };

  if (loading) {
    return (
      <div className="version-timeline-container">
        <div className="version-timeline-loading">
          <div className="spinner"></div>
          <p>Loading version history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="version-timeline-container">
        <div className="version-timeline-error">
          <h3>Failed to load version history</h3>
          <p>{error}</p>
          <button onClick={handleRefresh}>Retry</button>
        </div>
      </div>
    );
  }

  const sortedVersions = getSortedVersions();

  return (
    <div className="version-timeline-container">
      <div className="version-timeline-header">
        <div className="version-timeline-stats">
          <span className="stat-item">
            <strong>{versions.length}</strong> versions
          </span>
        </div>

        <div className="version-timeline-filters">
          <input
            type="date"
            className="date-filter"
            placeholder="From date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            title="Filter from date"
          />
          <input
            type="date"
            className="date-filter"
            placeholder="To date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            title="Filter to date"
          />
          <select
            className="operation-filter"
            value={operationType}
            onChange={(e) => setOperationType(e.target.value)}
            title="Filter by operation"
          >
            <option value="">All operations</option>
            <option value="CREATE">CREATE</option>
            <option value="UPDATE">UPDATE</option>
            <option value="DELETE">DELETE</option>
          </select>
          {(dateFrom || dateTo || operationType) && (
            <button
              className="clear-filters-btn"
              onClick={handleClearFilters}
              title="Clear filters"
            >
              ✕
            </button>
          )}
        </div>

        <div className="version-timeline-controls">
          <button
            className="sort-dir-btn"
            onClick={toggleSortDirection}
            title={sortDir === 'desc' ? 'Newest first' : 'Oldest first'}
          >
            {sortDir === 'desc' ? '↓' : '↑'}
          </button>
          <button className="refresh-btn" onClick={handleRefresh} title="Refresh">
            ⟲
          </button>
        </div>
      </div>

      {versions.length === 0 ? (
        <div className="version-timeline-empty">
          <span className="empty-icon">📜</span>
          <p>No version history found.</p>
          <p className="empty-hint">Memory versions will appear here as memories are created and updated.</p>
        </div>
      ) : (
        <div className="version-timeline-list">
          {sortedVersions.map((version, idx) => (
            <div key={`${version.version_id}-${idx}`} className={`version-item version-${getOperationColor(version.operation_type)}`}>
              <div className="version-icon">{getOperationIcon(version.operation_type)}</div>

              <div className="version-content">
                <div className="version-header">
                  <span className="version-operation">{version.operation_type}</span>
                  <span className="version-timestamp">{formatTimestamp(version.created_at)}</span>
                </div>

                <div className="version-details">
                  <div className="detail-item">
                    <span className="detail-label">Memory ID:</span>
                    <span className="detail-value memory-id">{version.memory_id}</span>
                  </div>

                  <div className="detail-item">
                    <span className="detail-label">Version:</span>
                    <span className="detail-value">v{version.version_number}</span>
                  </div>

                  {version.content && (
                    <div className="detail-item version-content-preview">
                      <span className="detail-label">Content:</span>
                      <span className="detail-value">{truncateContent(version.content)}</span>
                    </div>
                  )}

                  {version.previous_version_id && (
                    <div className="detail-item">
                      <span className="detail-label">Previous:</span>
                      <span className="detail-value version-link">{version.previous_version_id.substring(0, 8)}...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default VersionTimeline;
