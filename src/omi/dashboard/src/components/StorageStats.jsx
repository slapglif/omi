import React, { useEffect, useState } from 'react';
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';
import { fetchStats } from '../api/client';
import './StorageStats.css';

// Register Chart.js components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/**
 * StorageStats Component
 *
 * Displays database storage statistics with charts.
 * Shows memory counts, edge counts, type distributions, and edge distributions.
 */
const StorageStats = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await fetchStats();

      if (!data) {
        throw new Error('Invalid stats data received');
      }

      setStats(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Chart colors
  const typeColors = {
    fact: '#10b981',      // green
    experience: '#3b82f6', // blue
    belief: '#a855f7',     // purple
    decision: '#f59e0b'    // amber
  };

  const edgeColors = {
    SUPPORTS: '#10b981',     // green
    CONTRADICTS: '#ef4444',  // red
    RELATED_TO: '#3b82f6',   // blue
    DEPENDS_ON: '#f59e0b',   // amber
    POSTED: '#8b5cf6',       // violet
    DISCUSSED: '#ec4899'     // pink
  };

  // Prepare chart data for memory type distribution
  const getTypeChartData = () => {
    if (!stats || !stats.type_distribution) {
      return null;
    }

    const types = Object.keys(stats.type_distribution);
    const counts = Object.values(stats.type_distribution);

    return {
      labels: types.map((t) => t.charAt(0).toUpperCase() + t.slice(1)),
      datasets: [
        {
          label: 'Memory Count',
          data: counts,
          backgroundColor: types.map((t) => typeColors[t] || '#94a3b8'),
          borderColor: '#ffffff',
          borderWidth: 2
        }
      ]
    };
  };

  // Prepare chart data for edge type distribution
  const getEdgeChartData = () => {
    if (!stats || !stats.edge_distribution) {
      return null;
    }

    const types = Object.keys(stats.edge_distribution);
    const counts = Object.values(stats.edge_distribution);

    return {
      labels: types,
      datasets: [
        {
          label: 'Edge Count',
          data: counts,
          backgroundColor: types.map((t) => edgeColors[t] || '#94a3b8'),
          borderColor: '#ffffff',
          borderWidth: 2
        }
      ]
    };
  };

  // Chart options
  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 15,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const label = context.label || '';
            const value = context.parsed || 0;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            return `Count: ${context.parsed.y}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0
        }
      }
    }
  };

  if (loading) {
    return (
      <div className="stats-container">
        <div className="stats-loading">
          <div className="spinner"></div>
          <p>Loading statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="stats-container">
        <div className="stats-error">
          <h3>Failed to load statistics</h3>
          <p>{error}</p>
          <button onClick={loadStats}>Retry</button>
        </div>
      </div>
    );
  }

  const typeChartData = getTypeChartData();
  const edgeChartData = getEdgeChartData();

  return (
    <div className="stats-container">
      <div className="stats-header">
        <div className="stats-summary">
          <div className="summary-card">
            <div className="summary-icon">🧠</div>
            <div className="summary-content">
              <div className="summary-value">{stats.memory_count || 0}</div>
              <div className="summary-label">Total Memories</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">🔗</div>
            <div className="summary-content">
              <div className="summary-value">{stats.edge_count || 0}</div>
              <div className="summary-label">Total Edges</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <div className="summary-value">
                {stats.type_distribution ? Object.keys(stats.type_distribution).length : 0}
              </div>
              <div className="summary-label">Memory Types</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">⚡</div>
            <div className="summary-content">
              <div className="summary-value">
                {stats.edge_distribution ? Object.keys(stats.edge_distribution).length : 0}
              </div>
              <div className="summary-label">Edge Types</div>
            </div>
          </div>
        </div>

        <button className="refresh-btn" onClick={loadStats} title="Refresh statistics">
          ⟲
        </button>
      </div>

      <div className="stats-charts">
        <div className="chart-section">
          <h3 className="chart-title">Memory Type Distribution</h3>
          {typeChartData && Object.keys(stats.type_distribution).length > 0 ? (
            <div className="chart-wrapper">
              <Pie data={typeChartData} options={pieChartOptions} />
            </div>
          ) : (
            <div className="chart-empty">
              <span className="empty-icon">📊</span>
              <p>No memory data available</p>
            </div>
          )}
          <div className="chart-legend">
            {stats.type_distribution &&
              Object.entries(stats.type_distribution).map(([type, count]) => (
                <div key={type} className="legend-item">
                  <span
                    className="legend-color"
                    style={{ backgroundColor: typeColors[type] || '#94a3b8' }}
                  ></span>
                  <span className="legend-label">
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </span>
                  <span className="legend-value">{count}</span>
                </div>
              ))}
          </div>
        </div>

        <div className="chart-section">
          <h3 className="chart-title">Edge Type Distribution</h3>
          {edgeChartData && Object.keys(stats.edge_distribution).length > 0 ? (
            <div className="chart-wrapper">
              <Bar data={edgeChartData} options={barChartOptions} />
            </div>
          ) : (
            <div className="chart-empty">
              <span className="empty-icon">🔗</span>
              <p>No edge data available</p>
            </div>
          )}
          <div className="chart-legend">
            {stats.edge_distribution &&
              Object.entries(stats.edge_distribution).map(([type, count]) => (
                <div key={type} className="legend-item">
                  <span
                    className="legend-color"
                    style={{ backgroundColor: edgeColors[type] || '#94a3b8' }}
                  ></span>
                  <span className="legend-label">{type}</span>
                  <span className="legend-value">{count}</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StorageStats;
