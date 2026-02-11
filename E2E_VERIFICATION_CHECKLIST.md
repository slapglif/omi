# End-to-End Verification Checklist for Web Dashboard

**Task:** Subtask-5-2 - End-to-end verification: serve command → dashboard loads → all features work

**Date:** 2026-02-11

## Prerequisites

### 1. Frontend Build (from subtask-5-1)
⚠️ **REQUIRED:** Frontend must be built before testing

```bash
cd src/omi/dashboard
npm install
npm run build
```

**Verify:**
- [ ] `src/omi/dashboard/dist/` directory exists
- [ ] `src/omi/dashboard/dist/index.html` exists
- [ ] `src/omi/dashboard/dist/assets/` contains JS and CSS files

### 2. OMI Initialization
Ensure OMI is initialized with some test data:

```bash
# Initialize OMI
omi init

# Store some test memories
omi store "Python is a programming language" --type fact
omi store "I learned about FastAPI today" --type experience
omi store "REST APIs should follow semantic versioning" --type belief
omi store "Use React for the dashboard UI" --type decision

# Create some edges (relationships)
# This happens automatically through belief updates and related memories
```

---

## Phase 1: Server Startup Verification

### Test 1.1: Server Starts Successfully
```bash
omi serve --dashboard
```

**Expected Output:**
```
Starting OMI REST API Server...
  Host: 0.0.0.0
  Port: 8420
  Dashboard: enabled
  Reload: disabled

Dashboard available at http://localhost:8420/dashboard

INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Dashboard static files mounted from .../src/omi/dashboard/dist
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8420 (Press CTRL+C to quit)
```

**Checklist:**
- [ ] Server starts without errors
- [ ] Port 8420 is bound successfully
- [ ] "Dashboard static files mounted" message appears
- [ ] No Python tracebacks or errors

### Test 1.2: Verify CLI Help
```bash
omi serve --help
```

**Checklist:**
- [ ] `--dashboard/--no-dashboard` flag listed
- [ ] `--host` option shown (default: 0.0.0.0)
- [ ] `--port` option shown (default: 8420)
- [ ] `--reload` flag shown

### Test 1.3: API Root Endpoint
```bash
curl http://localhost:8420/ | jq
```

**Expected Response:**
```json
{
  "service": "OMI REST API",
  "version": "1.0.0",
  "endpoints": {
    "/dashboard": "Web dashboard for memory exploration (if built)",
    "/api/v1/events": "SSE endpoint for real-time event streaming",
    "/api/v1/dashboard/memories": "Retrieve memories with filters and pagination",
    ...
  }
}
```

**Checklist:**
- [ ] Returns 200 OK
- [ ] JSON response includes all dashboard endpoints
- [ ] "/dashboard" endpoint listed

---

## Phase 2: Backend API Verification

### Test 2.1: Dashboard API Endpoints
Test each API endpoint responds correctly:

```bash
# Get memories
curl http://localhost:8420/api/v1/dashboard/memories?limit=10 | jq

# Get edges
curl http://localhost:8420/api/v1/dashboard/edges | jq

# Get graph (memories + edges)
curl http://localhost:8420/api/v1/dashboard/graph?limit=20 | jq

# Get beliefs
curl http://localhost:8420/api/v1/dashboard/beliefs | jq

# Get stats
curl http://localhost:8420/api/v1/dashboard/stats | jq

# Search memories
curl "http://localhost:8420/api/v1/dashboard/search?q=python&limit=5" | jq
```

**Checklist:**
- [ ] All endpoints return 200 OK
- [ ] `/memories` returns `{memories: [], total_count: N, limit: N, offset: 0}`
- [ ] `/edges` returns `{edges: [], total_count: N, limit: N, offset: 0}`
- [ ] `/graph` returns `{nodes: [], edges: []}`
- [ ] `/beliefs` returns `{beliefs: [], total_count: N}`
- [ ] `/stats` returns `{memory_count: N, edge_count: N, type_distribution: {...}, edge_distribution: {...}}`
- [ ] `/search` returns `{query: "python", results: [...]}`
- [ ] No 500 errors or tracebacks

### Test 2.2: SSE Event Stream
```bash
curl -N http://localhost:8420/api/v1/events
```

**Expected:** Stream stays open, shows keepalive pings every 30 seconds

**Checklist:**
- [ ] Connection stays open (doesn't close immediately)
- [ ] Receives keepalive events: `data: {"type": "keepalive", "timestamp": "..."}`
- [ ] Press Ctrl+C to close - no errors

---

## Phase 3: Dashboard UI Verification

### Test 3.1: Dashboard Loads
**Browser:** Open http://localhost:8420/dashboard

**Checklist:**
- [ ] Dashboard HTML page loads
- [ ] No 404 errors in browser console
- [ ] CSS loads correctly (page is styled, not plain HTML)
- [ ] JavaScript loads without errors
- [ ] React app renders
- [ ] Tab navigation visible (Graph, Beliefs, Stats, Timeline)

### Test 3.2: Browser Console Check
**Open browser DevTools (F12) → Console tab**

**Checklist:**
- [ ] No JavaScript errors
- [ ] No 404 errors for assets
- [ ] No CORS errors
- [ ] API requests show in Network tab (if DevTools open)

---

## Phase 4: Feature-by-Feature Verification

### Feature 4.1: Graph Visualization Tab

**Steps:**
1. Click "Graph" tab (should be active by default)
2. Wait for graph to load

**Checklist:**
- [ ] Graph visualization renders
- [ ] Cytoscape.js canvas appears
- [ ] Memory nodes visible as circles
- [ ] Nodes are color-coded by type:
  - [ ] Green = fact
  - [ ] Blue = experience
  - [ ] Purple = belief
  - [ ] Amber = decision
- [ ] Edges (lines) connecting nodes visible
- [ ] Edge styles vary by type (solid, dashed, dotted)
- [ ] Graph controls visible:
  - [ ] "Fit to Screen" button
  - [ ] "Reset Zoom" button
  - [ ] "Relayout" button
- [ ] Graph stats header shows: "X nodes, Y edges"
- [ ] Legend shows memory types and edge types
- [ ] Can zoom with mouse wheel
- [ ] Can pan by dragging
- [ ] Can click nodes (highlights node)
- [ ] Controls work as expected

### Feature 4.2: Search Functionality

**Steps:**
1. Ensure Graph tab is active
2. Find search bar at top of graph section
3. Type "python" in search bar
4. Wait for results (debounced 500ms)

**Checklist:**
- [ ] Search bar visible above graph
- [ ] Typing triggers search after short delay
- [ ] Loading spinner appears during search
- [ ] Search results list appears below search bar
- [ ] Results show:
  - [ ] Memory content preview
  - [ ] Relevance score (e.g., "85% match")
  - [ ] Memory type badge (color-coded)
  - [ ] Creation date
  - [ ] Access count
- [ ] Matched nodes highlighted in graph:
  - [ ] Highlighted nodes have amber border
  - [ ] Non-matched nodes dimmed (faded)
  - [ ] Graph viewport fits to highlighted nodes
- [ ] Click on search result focuses on specific node
- [ ] Click "×" (clear) button resets highlighting
- [ ] Stats header shows "X highlighted" when search active

### Feature 4.3: Beliefs Tab

**Steps:**
1. Click "Beliefs" tab
2. Wait for beliefs to load

**Checklist:**
- [ ] Beliefs tab content appears
- [ ] Beliefs listed as cards
- [ ] Each belief card shows:
  - [ ] Belief content
  - [ ] Confidence indicator dot (colored)
  - [ ] Confidence badge (percentage)
  - [ ] Evidence counts (supporting/contradicting)
  - [ ] Created/updated timestamps
- [ ] Confidence colors correct:
  - [ ] Green dot/badge for high confidence (≥0.7)
  - [ ] Yellow dot/badge for medium confidence (0.4-0.69)
  - [ ] Red dot/badge for low confidence (<0.4)
- [ ] Statistics header shows:
  - [ ] Total beliefs count
  - [ ] High confidence count
  - [ ] Low confidence count
- [ ] Sort controls visible:
  - [ ] "Sort by" dropdown (confidence, updated, created, evidence)
  - [ ] Sort direction toggle (↑/↓)
- [ ] "Refresh" button works
- [ ] Sorting changes belief order
- [ ] Responsive layout (try resizing window)

### Feature 4.4: Stats Tab

**Steps:**
1. Click "Stats" tab
2. Wait for statistics to load

**Checklist:**
- [ ] Stats tab content appears
- [ ] Summary cards visible:
  - [ ] Total Memories card (with count)
  - [ ] Total Edges card (with count)
  - [ ] Memory Types card (with count)
  - [ ] Edge Types card (with count)
- [ ] Memory Type Distribution (Pie Chart):
  - [ ] Chart renders correctly
  - [ ] Colors match memory types (green, blue, purple, amber)
  - [ ] Legend shows all memory types
  - [ ] Hover shows percentage and count
- [ ] Edge Type Distribution (Bar Chart):
  - [ ] Chart renders correctly
  - [ ] Bars color-coded by edge type
  - [ ] Legend shows all edge types
  - [ ] Hover shows count
- [ ] "Refresh" button works
- [ ] Charts responsive (try resizing window)
- [ ] No chart rendering errors

### Feature 4.5: Timeline Tab (SSE Events)

**Steps:**
1. Click "Timeline" tab
2. Wait for connection to establish

**Checklist:**
- [ ] Timeline tab content appears
- [ ] Connection status indicator visible
- [ ] Status shows "Connected" with green pulse animation
- [ ] Timeline is empty initially (if no recent events)
- [ ] Empty state message: "Waiting for memory operations..."

**Trigger events:** In a separate terminal, run:
```bash
# Store a new memory
omi store "Testing the timeline feature" --type experience

# Recall memories
omi recall "python" --limit 3
```

**Back to browser:**
- [ ] New events appear in real-time (no page refresh needed)
- [ ] Events show:
  - [ ] Event type icon (💾, 🔍, etc.)
  - [ ] Event timestamp (relative time)
  - [ ] Event details (memory ID, content, etc.)
  - [ ] Color-coded left border by event type
- [ ] Event types color-coded:
  - [ ] Blue border = memory events
  - [ ] Purple border = belief events
  - [ ] Green border = session events
  - [ ] Amber border = warning events
- [ ] Controls work:
  - [ ] Sort direction toggle (newest first ↓ / oldest first ↑)
  - [ ] Clear events button (🗑️) clears list
  - [ ] Reconnect button (⟲) re-establishes connection
- [ ] Timeline scrollable if many events
- [ ] No connection errors

---

## Phase 5: Responsive Design Check

### Test 5.1: Desktop View
- [ ] Dashboard looks good at 1920x1080
- [ ] All features accessible
- [ ] No horizontal scrollbars (except timeline)
- [ ] Layout uses full width appropriately

### Test 5.2: Tablet View
**Resize browser to ~768px width**

- [ ] Layout adapts to smaller width
- [ ] Tab navigation still visible and usable
- [ ] Graph controls still accessible
- [ ] Beliefs cards stack vertically
- [ ] Stats cards adjust to 2-column or 1-column grid
- [ ] Charts remain readable

### Test 5.3: Mobile View
**Resize browser to ~375px width**

- [ ] All tabs accessible
- [ ] Graph remains interactive
- [ ] Search bar adapts to narrow width
- [ ] Beliefs cards full width
- [ ] Stats cards single column
- [ ] Charts responsive and readable
- [ ] Timeline scrollable

---

## Phase 6: Performance & Polish

### Test 6.1: Loading States
- [ ] Components show loading spinners while fetching data
- [ ] No "flash of unstyled content"
- [ ] Smooth transitions between loading and loaded states

### Test 6.2: Error Handling
**Test error states (if possible):**
- [ ] Stop the server, verify UI shows connection errors
- [ ] Error states have retry buttons
- [ ] Error messages are user-friendly

### Test 6.3: Browser Compatibility
Test in multiple browsers:
- [ ] Chrome/Chromium (primary target)
- [ ] Firefox
- [ ] Safari (if on macOS)
- [ ] Edge

---

## Phase 7: Final Checks

### Test 7.1: No Console Errors
**Open browser console (F12) and check all tabs:**
- [ ] No JavaScript errors in any tab
- [ ] No failed network requests
- [ ] No CORS issues
- [ ] No React warnings

### Test 7.2: Memory Leaks
**Open DevTools → Performance → Memory:**
- [ ] Switch between tabs multiple times
- [ ] Memory usage stays reasonable
- [ ] No continuous memory growth
- [ ] EventSource connections properly cleaned up

### Test 7.3: Accessibility
- [ ] Can navigate dashboard with keyboard (Tab key)
- [ ] Focus indicators visible
- [ ] Buttons have hover states
- [ ] Color contrast sufficient for readability

---

## Summary Checklist

### Code Quality
- [x] Backend Python syntax valid (dashboard_api.py, rest_api.py, cli/serve.py)
- [x] Frontend components exist (11 component files)
- [x] package.json dependencies correct
- [ ] Frontend built (dist/ directory exists) ← **MANUAL STEP REQUIRED**

### Server
- [ ] `omi serve --dashboard` starts successfully
- [ ] Listens on port 8420
- [ ] No Python errors or warnings
- [ ] Dashboard static files mounted

### API Endpoints (7 endpoints)
- [ ] GET /api/v1/dashboard/memories
- [ ] GET /api/v1/dashboard/edges
- [ ] GET /api/v1/dashboard/graph
- [ ] GET /api/v1/dashboard/beliefs
- [ ] GET /api/v1/dashboard/stats
- [ ] GET /api/v1/dashboard/search
- [ ] GET /api/v1/events (SSE)

### Dashboard UI (4 tabs)
- [ ] Graph tab with Cytoscape.js visualization
- [ ] Beliefs tab with confidence color coding
- [ ] Stats tab with Chart.js charts
- [ ] Timeline tab with SSE real-time events

### Features (5 major features)
- [ ] Graph visualization (nodes, edges, layout, colors)
- [ ] Semantic search with highlighting
- [ ] Belief network with confidence visualization
- [ ] Storage statistics with charts
- [ ] Session timeline with real-time SSE

### Acceptance Criteria (from spec.md)
- [ ] omi serve --dashboard opens web UI at http://localhost:8420/dashboard
- [ ] Graph visualization shows memory nodes with relationship edges (force-directed layout)
- [ ] Belief network view shows confidence levels with color coding (green=high, red=low)
- [ ] Search bar performs semantic recall and highlights results in graph
- [ ] Storage statistics dashboard shows tier sizes, memory counts, and trends
- [ ] Session timeline view shows chronological memory operations
- [ ] Dashboard works on read-only REST API (no write operations from UI by default)

---

## Automated Verification Script

You can also run this automated check script:

```bash
#!/bin/bash
# e2e_verify.sh - Automated checks for what's possible without browser

echo "=== OMI Dashboard E2E Verification ==="
echo

# Check 1: Frontend built
echo "1. Checking frontend build..."
if [ -d "src/omi/dashboard/dist" ]; then
    echo "   ✓ Frontend dist/ directory exists"
    if [ -f "src/omi/dashboard/dist/index.html" ]; then
        echo "   ✓ index.html found"
    else
        echo "   ✗ index.html missing!"
        exit 1
    fi
else
    echo "   ✗ Frontend not built! Run: cd src/omi/dashboard && npm run build"
    exit 1
fi

# Check 2: Python modules syntax
echo "2. Checking Python syntax..."
python3 -m py_compile src/omi/dashboard_api.py && echo "   ✓ dashboard_api.py"
python3 -m py_compile src/omi/rest_api.py && echo "   ✓ rest_api.py"
python3 -m py_compile src/omi/cli/serve.py && echo "   ✓ cli/serve.py"

# Check 3: Frontend components
echo "3. Checking frontend components..."
components=(
    "src/omi/dashboard/src/components/GraphVisualization.jsx"
    "src/omi/dashboard/src/components/SearchBar.jsx"
    "src/omi/dashboard/src/components/BeliefNetwork.jsx"
    "src/omi/dashboard/src/components/StorageStats.jsx"
    "src/omi/dashboard/src/components/SessionTimeline.jsx"
)
for comp in "${components[@]}"; do
    if [ -f "$comp" ]; then
        echo "   ✓ $(basename $comp)"
    else
        echo "   ✗ Missing: $comp"
        exit 1
    fi
done

# Check 4: API client
echo "4. Checking API client..."
if [ -f "src/omi/dashboard/src/api/client.js" ]; then
    echo "   ✓ API client exists"
else
    echo "   ✗ API client missing!"
    exit 1
fi

echo
echo "=== Automated Checks Complete ==="
echo
echo "Next steps:"
echo "1. Start the server: omi serve --dashboard"
echo "2. Open browser: http://localhost:8420/dashboard"
echo "3. Follow the manual verification checklist in E2E_VERIFICATION_CHECKLIST.md"
echo
```

Save as `e2e_verify.sh`, make executable with `chmod +x e2e_verify.sh`, and run it.

---

## Troubleshooting

### Issue: Dashboard shows 404
**Solution:** Frontend not built. Run `cd src/omi/dashboard && npm run build`

### Issue: API returns 500 errors
**Solution:** Check OMI is initialized: `omi init`

### Issue: Graph shows no nodes
**Solution:** No memories in database. Store some: `omi store "test" --type fact`

### Issue: Timeline shows no events
**Solution:** Trigger events by storing/recalling memories in a separate terminal

### Issue: Port 8420 already in use
**Solution:** Use different port: `omi serve --dashboard --port 8421`

### Issue: CORS errors in browser
**Solution:** Vite proxy should handle this. Check vite.config.js has correct proxy config.

---

## Completion Criteria

This subtask (subtask-5-2) is complete when:

- [x] All backend code syntax valid
- [x] All frontend components exist
- [ ] Frontend successfully built (dist/ directory)
- [ ] Server starts without errors
- [ ] Dashboard loads in browser
- [ ] All 4 tabs render correctly
- [ ] All 5 major features work
- [ ] No console errors
- [ ] All acceptance criteria met

**Status:** ⏳ Ready for manual verification (frontend build required first)

---

**Last Updated:** 2026-02-11
**Subtask:** subtask-5-2
**Next:** subtask-5-3 (Update README documentation)
