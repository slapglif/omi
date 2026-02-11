import React, { useEffect, useState, useRef } from 'react';
import './SessionTimeline.css';

/**
 * SessionTimeline Component
 *
 * Displays chronological timeline of memory operations using SSE (Server-Sent Events).
 * Connects to /api/v1/events endpoint for real-time event streaming.
 * Shows event type, timestamp, and event details.
 */
const SessionTimeline = () => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);
  const [sortDir, setSortDir] = useState('desc'); // desc = newest first
  const eventSourceRef = useRef(null);

  useEffect(() => {
    connectToEventStream();

    // Cleanup on unmount
    return () => {
      disconnectFromEventStream();
    };
  }, []);

  const connectToEventStream = () => {
    try {
      setLoading(true);
      setError(null);

      // Create EventSource connection to SSE endpoint
      const eventSource = new EventSource('/api/v1/events');
      eventSourceRef.current = eventSource;

      // Handle connection open
      eventSource.onopen = () => {
        setConnected(true);
        setLoading(false);
      };

      // Handle incoming messages
      eventSource.onmessage = (e) => {
        try {
          const eventData = JSON.parse(e.data);

          // Skip connection message
          if (eventData.type === 'connected') {
            return;
          }

          // Add new event to the list
          setEvents((prevEvents) => {
            // Add event with unique key and ensure no duplicates
            const eventWithKey = {
              ...eventData,
              key: `${eventData.event_type}-${eventData.timestamp}-${Date.now()}`
            };
            return [eventWithKey, ...prevEvents];
          });
        } catch (err) {
          console.error('Failed to parse event data:', err);
        }
      };

      // Handle errors
      eventSource.onerror = (err) => {
        console.error('EventSource error:', err);
        setConnected(false);

        // If connection fails initially, show error
        if (loading) {
          setError('Failed to connect to event stream');
          setLoading(false);
        }

        // EventSource will automatically try to reconnect
      };
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const disconnectFromEventStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setConnected(false);
    }
  };

  const handleReconnect = () => {
    disconnectFromEventStream();
    setEvents([]);
    connectToEventStream();
  };

  const handleClearEvents = () => {
    setEvents([]);
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

  const getEventIcon = (eventType) => {
    const icons = {
      'memory.stored': '💾',
      'memory.recalled': '🔍',
      'belief.updated': '💡',
      'belief.contradiction_detected': '⚠️',
      'session.started': '▶️',
      'session.ended': '⏹️',
      'default': '📝'
    };
    return icons[eventType] || icons.default;
  };

  const getEventColor = (eventType) => {
    if (eventType?.includes('contradiction')) return 'warning';
    if (eventType?.includes('session')) return 'session';
    if (eventType?.includes('belief')) return 'belief';
    if (eventType?.includes('memory')) return 'memory';
    return 'default';
  };

  const renderEventDetails = (event) => {
    // Extract relevant details from event payload
    const details = [];

    if (event.memory_id) {
      details.push({ label: 'Memory ID', value: event.memory_id });
    }

    if (event.belief_id) {
      details.push({ label: 'Belief ID', value: event.belief_id });
    }

    if (event.content) {
      details.push({ label: 'Content', value: event.content });
    }

    if (event.memory_type) {
      details.push({ label: 'Type', value: event.memory_type });
    }

    if (event.confidence !== undefined) {
      details.push({ label: 'Confidence', value: `${(event.confidence * 100).toFixed(1)}%` });
    }

    if (event.relevance_score !== undefined) {
      details.push({ label: 'Relevance', value: `${(event.relevance_score * 100).toFixed(1)}%` });
    }

    if (event.results_count !== undefined) {
      details.push({ label: 'Results', value: event.results_count });
    }

    return details;
  };

  const getSortedEvents = () => {
    const sorted = [...events];
    sorted.sort((a, b) => {
      const timeA = new Date(a.timestamp).getTime();
      const timeB = new Date(b.timestamp).getTime();
      return sortDir === 'desc' ? timeB - timeA : timeA - timeB;
    });
    return sorted;
  };

  if (loading) {
    return (
      <div className="timeline-container">
        <div className="timeline-loading">
          <div className="spinner"></div>
          <p>Connecting to event stream...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="timeline-container">
        <div className="timeline-error">
          <h3>Failed to connect to event stream</h3>
          <p>{error}</p>
          <button onClick={handleReconnect}>Reconnect</button>
        </div>
      </div>
    );
  }

  const sortedEvents = getSortedEvents();

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <div className="timeline-stats">
          <span className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-indicator"></span>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="stat-separator">•</span>
          <span className="stat-item">
            <strong>{events.length}</strong> events
          </span>
        </div>

        <div className="timeline-controls">
          <button
            className="sort-dir-btn"
            onClick={toggleSortDirection}
            title={sortDir === 'desc' ? 'Newest first' : 'Oldest first'}
          >
            {sortDir === 'desc' ? '↓' : '↑'}
          </button>
          <button className="clear-btn" onClick={handleClearEvents} title="Clear events">
            🗑️
          </button>
          <button className="refresh-btn" onClick={handleReconnect} title="Reconnect">
            ⟲
          </button>
        </div>
      </div>

      {events.length === 0 ? (
        <div className="timeline-empty">
          <span className="empty-icon">📅</span>
          <p>No events yet. Waiting for memory operations...</p>
          <p className="empty-hint">Events will appear here in real-time as they occur.</p>
        </div>
      ) : (
        <div className="timeline-list">
          {sortedEvents.map((event) => (
            <div key={event.key} className={`timeline-event event-${getEventColor(event.event_type)}`}>
              <div className="event-icon">{getEventIcon(event.event_type)}</div>

              <div className="event-content">
                <div className="event-header">
                  <span className="event-type">{event.event_type || 'unknown'}</span>
                  <span className="event-timestamp">{formatTimestamp(event.timestamp)}</span>
                </div>

                {renderEventDetails(event).length > 0 && (
                  <div className="event-details">
                    {renderEventDetails(event).map((detail, idx) => (
                      <div key={idx} className="detail-item">
                        <span className="detail-label">{detail.label}:</span>
                        <span className="detail-value">{detail.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SessionTimeline;
