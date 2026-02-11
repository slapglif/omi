import React, { useState } from 'react';
import GraphVisualization from './components/GraphVisualization';
import BeliefNetwork from './components/BeliefNetwork';

function App() {
  const [activeTab, setActiveTab] = useState('graph');

  const tabs = [
    { id: 'graph', label: 'Graph', icon: '🕸️' },
    { id: 'beliefs', label: 'Beliefs', icon: '💡' },
    { id: 'stats', label: 'Stats', icon: '📊' },
    { id: 'timeline', label: 'Timeline', icon: '📅' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'graph':
        return (
          <div className="tab-content">
            <h2>Memory Graph</h2>
            <GraphVisualization />
          </div>
        );
      case 'beliefs':
        return (
          <div className="tab-content">
            <h2>Belief Network</h2>
            <BeliefNetwork />
          </div>
        );
      case 'stats':
        return (
          <div className="tab-content">
            <h2>Storage Statistics</h2>
            <p>Storage statistics and charts will be implemented here.</p>
          </div>
        );
      case 'timeline':
        return (
          <div className="tab-content">
            <h2>Session Timeline</h2>
            <p>Session timeline will be implemented here.</p>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>OMI Memory Dashboard</h1>
          <p className="subtitle">Visualize and explore AI agent memory graphs</p>
        </div>
      </header>

      <nav className="tab-navigation">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <main className="main-content">
        {renderTabContent()}
      </main>

      <footer className="app-footer">
        <p>OMI (OpenClaw Memory Infrastructure) v0.1.0</p>
      </footer>
    </div>
  );
}

export default App;
