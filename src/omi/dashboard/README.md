# OMI Dashboard

Web-based visualization dashboard for exploring OMI memory graphs, belief networks, and storage statistics.

## Overview

This is a React + Vite application that provides an interactive UI for the OMI memory system. The dashboard includes:

- **Graph Visualization**: Force-directed graph using Cytoscape.js showing memory nodes and relationship edges
- **Search**: Semantic search with real-time highlighting of matching nodes
- **Belief Network**: Confidence-based visualization of agent beliefs with color coding
- **Storage Stats**: Charts showing memory and edge type distributions
- **Session Timeline**: Real-time event streaming via Server-Sent Events (SSE)

## Tech Stack

- **Framework**: React 18.2.0
- **Build Tool**: Vite 5.0.12
- **Graph Visualization**: Cytoscape.js 3.28.1 with fcose layout
- **Charts**: Chart.js 4.4.1 via react-chartjs-2
- **API Communication**: Fetch API with SSE (EventSource)

## Development Setup

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

This starts the Vite dev server at http://localhost:5173

The dev server proxies `/api` requests to `http://localhost:8420` (the OMI backend).

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

**Expected output:**
- `dist/index.html` - Entry point
- `dist/assets/*.js` - Bundled JavaScript (code-split by vendor)
- `dist/assets/*.css` - Bundled styles
- Target bundle size: < 500KB total

### Preview Production Build

```bash
npm run preview
```

This serves the production build locally for testing.

## Project Structure

```
src/omi/dashboard/
├── src/
│   ├── main.jsx              # React entry point
│   ├── App.jsx               # Main application with tab navigation
│   ├── App.css               # Application-wide styles
│   ├── api/
│   │   └── client.js         # API client for backend communication
│   └── components/
│       ├── GraphVisualization.jsx  # Cytoscape graph component
│       ├── GraphVisualization.css
│       ├── SearchBar.jsx           # Semantic search component
│       ├── SearchBar.css
│       ├── BeliefNetwork.jsx       # Beliefs with confidence visualization
│       ├── BeliefNetwork.css
│       ├── StorageStats.jsx        # Statistics charts
│       ├── StorageStats.css
│       ├── SessionTimeline.jsx     # Real-time event timeline
│       └── SessionTimeline.css
├── index.html                # HTML entry point
├── vite.config.js           # Vite configuration
├── package.json             # Dependencies and scripts
└── README.md                # This file
```

## API Endpoints

The dashboard consumes these backend API endpoints:

- `GET /api/v1/dashboard/graph` - Memory nodes and edges
- `GET /api/v1/dashboard/memories` - Filtered memory list
- `GET /api/v1/dashboard/edges` - Relationship edges
- `GET /api/v1/dashboard/beliefs` - Belief network data
- `GET /api/v1/dashboard/stats` - Storage statistics
- `GET /api/v1/dashboard/search?q={query}` - Semantic search
- `GET /api/v1/events` - SSE event stream for real-time updates

All endpoints return JSON except `/api/v1/events` which uses Server-Sent Events.

## Integration with OMI Backend

The built dashboard is served by the OMI FastAPI server:

```bash
# Start the OMI server with dashboard
omi serve --dashboard

# Dashboard available at:
# http://localhost:8420/dashboard
```

The FastAPI server (configured in `src/omi/rest_api.py`) mounts the `dist/` directory as static files at the `/dashboard` route.

## Color Coding

### Memory Types (Graph Nodes)
- **Fact**: Green (#10b981)
- **Experience**: Blue (#3b82f6)
- **Belief**: Purple (#8b5cf6)
- **Decision**: Amber (#f59e0b)

### Edge Types (Graph Relationships)
- **SUPPORTS**: Green solid line
- **CONTRADICTS**: Red dashed line
- **RELATED_TO**: Gray solid line
- **DEPENDS_ON**: Blue dotted line
- **POSTED**: Purple solid line
- **DISCUSSED**: Amber solid line

### Belief Confidence Levels
- **High (≥ 0.7)**: Green
- **Medium (0.4 - 0.69)**: Yellow
- **Low (< 0.4)**: Red

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Modern browsers with ES modules support required.

## Troubleshooting

### Dashboard won't load in browser
- Ensure you've run `npm run build` to create the `dist/` directory
- Check that `omi serve --dashboard` shows "Dashboard UI available at /dashboard"
- Verify no console errors in browser developer tools

### Graph visualization not rendering
- Check browser console for Cytoscape.js errors
- Ensure backend API is running and returning data at `/api/v1/dashboard/graph`
- Try the "Relayout" button to trigger a fresh layout

### Search not working
- Verify the backend `/api/v1/dashboard/search` endpoint is accessible
- Check that embeddings are configured (requires NIM or Ollama)
- Look for API errors in browser console

### Timeline not receiving events
- Verify SSE connection at `/api/v1/events` in Network tab
- Check for CORS issues if running dev server separately
- Ensure backend EventBus is emitting events

## Performance

- Initial bundle size: ~300-400KB (minified + gzipped)
- Code splitting: React, graph (Cytoscape), and chart (Chart.js) are separate chunks
- Graph performance: Tested with up to 1000 nodes without significant lag
- SSE overhead: Minimal (<1KB per event)

## License

Same as OMI parent project.
