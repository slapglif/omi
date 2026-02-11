import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { fetchGraph } from '../api/client';
import SearchBar from './SearchBar';
import './GraphVisualization.css';

// Register the fcose layout extension
cytoscape.use(fcose);

const GraphVisualization = () => {
  const cyRef = useRef(null);
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ nodeCount: 0, edgeCount: 0 });
  const [highlightedNodes, setHighlightedNodes] = useState([]);

  // Color palette for memory types
  const memoryTypeColors = {
    fact: '#10b981',      // green
    experience: '#3b82f6', // blue
    belief: '#8b5cf6',     // purple
    decision: '#f59e0b'    // amber
  };

  // Edge type styles
  const edgeTypeStyles = {
    SUPPORTS: { color: '#10b981', style: 'solid' },
    CONTRADICTS: { color: '#ef4444', style: 'dashed' },
    RELATED_TO: { color: '#6b7280', style: 'solid' },
    DEPENDS_ON: { color: '#3b82f6', style: 'dotted' },
    POSTED: { color: '#8b5cf6', style: 'solid' },
    DISCUSSED: { color: '#f59e0b', style: 'solid' }
  };

  useEffect(() => {
    const loadGraphData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch graph data from API
        const data = await fetchGraph({ limit: 200 });

        if (!data || !data.nodes || !data.edges) {
          throw new Error('Invalid graph data received');
        }

        // Transform nodes for Cytoscape
        const nodes = data.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.content.substring(0, 50) + (node.content.length > 50 ? '...' : ''),
            fullContent: node.content,
            type: node.memory_type,
            created: node.created_at,
            accessed: node.last_accessed,
            accessCount: node.access_count
          }
        }));

        // Transform edges for Cytoscape
        const edges = data.edges.map((edge, index) => ({
          data: {
            id: edge.id || `edge-${index}`,
            source: edge.source_id,
            target: edge.target_id,
            type: edge.edge_type,
            strength: edge.strength,
            created: edge.created_at
          }
        }));

        // Update stats
        setStats({
          nodeCount: nodes.length,
          edgeCount: edges.length
        });

        // Initialize Cytoscape
        if (containerRef.current) {
          // Destroy existing instance if any
          if (cyRef.current) {
            cyRef.current.destroy();
          }

          cyRef.current = cytoscape({
            container: containerRef.current,
            elements: [...nodes, ...edges],
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': (ele) => {
                    const type = ele.data('type');
                    return memoryTypeColors[type] || '#6b7280';
                  },
                  'label': 'data(label)',
                  'color': '#1e293b',
                  'text-outline-color': '#ffffff',
                  'text-outline-width': 2,
                  'font-size': '10px',
                  'width': 30,
                  'height': 30,
                  'text-valign': 'bottom',
                  'text-halign': 'center',
                  'text-margin-y': 5
                }
              },
              {
                selector: 'node:selected',
                style: {
                  'border-width': 3,
                  'border-color': '#2563eb',
                  'width': 35,
                  'height': 35
                }
              },
              {
                selector: 'node.highlighted',
                style: {
                  'border-width': 4,
                  'border-color': '#f59e0b',
                  'width': 40,
                  'height': 40,
                  'z-index': 999,
                  'box-shadow': '0 0 20px rgba(245, 158, 11, 0.6)'
                }
              },
              {
                selector: 'node.dimmed',
                style: {
                  'opacity': 0.2
                }
              },
              {
                selector: 'edge',
                style: {
                  'width': (ele) => {
                    const strength = ele.data('strength') || 0.5;
                    return Math.max(1, strength * 3);
                  },
                  'line-color': (ele) => {
                    const type = ele.data('type');
                    return edgeTypeStyles[type]?.color || '#6b7280';
                  },
                  'line-style': (ele) => {
                    const type = ele.data('type');
                    return edgeTypeStyles[type]?.style || 'solid';
                  },
                  'target-arrow-color': (ele) => {
                    const type = ele.data('type');
                    return edgeTypeStyles[type]?.color || '#6b7280';
                  },
                  'target-arrow-shape': 'triangle',
                  'curve-style': 'bezier',
                  'opacity': 0.6
                }
              },
              {
                selector: 'edge:selected',
                style: {
                  'opacity': 1,
                  'width': 3
                }
              }
            ],
            layout: {
              name: 'fcose',
              quality: 'default',
              animate: true,
              animationDuration: 1000,
              randomize: false,
              fit: true,
              padding: 30,
              nodeDimensionsIncludeLabels: true,
              // Force-directed layout parameters
              idealEdgeLength: 100,
              edgeElasticity: 0.45,
              nestingFactor: 0.1,
              gravity: 0.25,
              numIter: 2500,
              tile: true,
              tilingPaddingVertical: 10,
              tilingPaddingHorizontal: 10,
              gravityRangeCompound: 1.5,
              gravityCompound: 1.0,
              gravityRange: 3.8,
              initialEnergyOnIncremental: 0.3
            },
            minZoom: 0.1,
            maxZoom: 3,
            wheelSensitivity: 0.2
          });

          // Add tooltips on node hover
          cyRef.current.on('mouseover', 'node', (event) => {
            const node = event.target;
            const data = node.data();

            // Create tooltip
            node.style({
              'font-size': '12px',
              'text-outline-width': 3
            });
          });

          cyRef.current.on('mouseout', 'node', (event) => {
            const node = event.target;
            node.style({
              'font-size': '10px',
              'text-outline-width': 2
            });
          });

          // Add click handler to show full content
          cyRef.current.on('tap', 'node', (event) => {
            const node = event.target;
            const data = node.data();

            // You can enhance this later with a modal or side panel
            console.log('Node clicked:', {
              id: data.id,
              type: data.type,
              content: data.fullContent,
              created: data.created,
              accessed: data.accessed,
              accessCount: data.accessCount
            });
          });

          // Add edge click handler
          cyRef.current.on('tap', 'edge', (event) => {
            const edge = event.target;
            const data = edge.data();

            console.log('Edge clicked:', {
              id: data.id,
              type: data.type,
              strength: data.strength,
              source: data.source,
              target: data.target
            });
          });
        }

        setLoading(false);
      } catch (err) {
        console.error('Failed to load graph:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    loadGraphData();

    // Cleanup on unmount
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, []);

  // Fit graph to viewport
  const handleFit = () => {
    if (cyRef.current) {
      cyRef.current.fit(null, 30);
    }
  };

  // Reset zoom
  const handleResetZoom = () => {
    if (cyRef.current) {
      cyRef.current.zoom(1);
      cyRef.current.center();
    }
  };

  // Re-run layout
  const handleRelayout = () => {
    if (cyRef.current) {
      const layout = cyRef.current.layout({
        name: 'fcose',
        quality: 'default',
        animate: true,
        animationDuration: 1000,
        randomize: true,
        fit: true,
        padding: 30
      });
      layout.run();
    }
  };

  // Handle search results
  const handleSearchResults = (matchedIds, results) => {
    if (!cyRef.current) return;

    setHighlightedNodes(matchedIds);

    // Remove previous highlighting
    cyRef.current.nodes().removeClass('highlighted dimmed');

    if (matchedIds.length > 0) {
      // Highlight matched nodes
      matchedIds.forEach((id) => {
        const node = cyRef.current.getElementById(id);
        if (node) {
          node.addClass('highlighted');
        }
      });

      // Dim non-matched nodes
      cyRef.current.nodes().filter((node) => !matchedIds.includes(node.id())).addClass('dimmed');

      // Fit view to highlighted nodes
      const highlightedCollection = cyRef.current.collection();
      matchedIds.forEach((id) => {
        const node = cyRef.current.getElementById(id);
        if (node) {
          highlightedCollection.merge(node);
        }
      });

      if (highlightedCollection.length > 0) {
        cyRef.current.fit(highlightedCollection, 100);
      }
    }
  };

  // Clear search highlighting
  const handleClearSearch = () => {
    if (!cyRef.current) return;

    setHighlightedNodes([]);
    cyRef.current.nodes().removeClass('highlighted dimmed');
    cyRef.current.fit(null, 30);
  };

  if (loading) {
    return (
      <div className="graph-container">
        <div className="graph-loading">
          <div className="spinner"></div>
          <p>Loading memory graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-container">
        <div className="graph-error">
          <h3>Failed to load graph</h3>
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-container">
      <div className="graph-header">
        <div className="graph-stats">
          <span className="stat-item">
            <strong>{stats.nodeCount}</strong> nodes
          </span>
          <span className="stat-separator">•</span>
          <span className="stat-item">
            <strong>{stats.edgeCount}</strong> edges
          </span>
          {highlightedNodes.length > 0 && (
            <>
              <span className="stat-separator">•</span>
              <span className="stat-item stat-highlighted">
                <strong>{highlightedNodes.length}</strong> highlighted
              </span>
            </>
          )}
        </div>

        <div className="graph-controls">
          <button onClick={handleFit} className="control-btn" title="Fit to screen">
            🔍 Fit
          </button>
          <button onClick={handleResetZoom} className="control-btn" title="Reset zoom">
            ⟲ Reset
          </button>
          <button onClick={handleRelayout} className="control-btn" title="Re-run layout">
            ⚡ Relayout
          </button>
        </div>
      </div>

      <div className="graph-search-section">
        <SearchBar
          onSearchResults={handleSearchResults}
          onClearSearch={handleClearSearch}
        />
      </div>

      <div className="graph-legend">
        <div className="legend-section">
          <span className="legend-title">Memory Types:</span>
          {Object.entries(memoryTypeColors).map(([type, color]) => (
            <div key={type} className="legend-item">
              <span className="legend-color" style={{ backgroundColor: color }}></span>
              <span className="legend-label">{type}</span>
            </div>
          ))}
        </div>

        <div className="legend-section">
          <span className="legend-title">Edge Types:</span>
          {Object.entries(edgeTypeStyles).map(([type, style]) => (
            <div key={type} className="legend-item">
              <span
                className={`legend-line legend-line-${style.style}`}
                style={{ backgroundColor: style.color }}
              ></span>
              <span className="legend-label">{type.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
      </div>

      <div ref={containerRef} className="graph-canvas"></div>
    </div>
  );
};

export default GraphVisualization;
