#!/bin/bash
# e2e_verify.sh - Automated E2E verification checks for OMI Dashboard
# This script verifies what's possible without a browser

set -e

echo "================================================================"
echo "   OMI Dashboard End-to-End Verification Script"
echo "================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Function to print success
success() {
    echo -e "   ${GREEN}✓${NC} $1"
}

# Function to print failure
failure() {
    echo -e "   ${RED}✗${NC} $1"
    FAILED=1
}

# Function to print warning
warning() {
    echo -e "   ${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Check 1: Frontend Build
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Frontend Build Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "src/omi/dashboard/dist" ]; then
    success "Frontend dist/ directory exists"

    if [ -f "src/omi/dashboard/dist/index.html" ]; then
        success "index.html found"
    else
        failure "index.html missing in dist/"
    fi

    if [ -d "src/omi/dashboard/dist/assets" ]; then
        success "assets/ directory exists"

        # Count JS and CSS files
        JS_COUNT=$(find src/omi/dashboard/dist/assets -name "*.js" 2>/dev/null | wc -l)
        CSS_COUNT=$(find src/omi/dashboard/dist/assets -name "*.css" 2>/dev/null | wc -l)

        if [ $JS_COUNT -gt 0 ]; then
            success "Found $JS_COUNT JavaScript bundle(s)"
        else
            failure "No JavaScript bundles found"
        fi

        if [ $CSS_COUNT -gt 0 ]; then
            success "Found $CSS_COUNT CSS bundle(s)"
        else
            failure "No CSS bundles found"
        fi
    else
        failure "assets/ directory missing"
    fi
else
    failure "Frontend not built!"
    echo
    echo "   To build the frontend, run:"
    echo "   cd src/omi/dashboard"
    echo "   npm install"
    echo "   npm run build"
    echo
    exit 1
fi

echo

# ============================================================================
# Check 2: Backend Python Code
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Backend Python Code Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check dashboard_api.py
if python3 -m py_compile src/omi/dashboard_api.py 2>/dev/null; then
    success "dashboard_api.py syntax valid"
else
    failure "dashboard_api.py has syntax errors"
fi

# Check rest_api.py
if python3 -m py_compile src/omi/rest_api.py 2>/dev/null; then
    success "rest_api.py syntax valid"
else
    failure "rest_api.py has syntax errors"
fi

# Check cli/serve.py
if python3 -m py_compile src/omi/cli/serve.py 2>/dev/null; then
    success "cli/serve.py syntax valid"
else
    failure "cli/serve.py has syntax errors"
fi

echo

# ============================================================================
# Check 3: Frontend Components
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Frontend Components Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check React components
components=(
    "src/omi/dashboard/src/App.jsx:Main Application"
    "src/omi/dashboard/src/components/GraphVisualization.jsx:Graph Visualization"
    "src/omi/dashboard/src/components/GraphVisualization.css:Graph Styles"
    "src/omi/dashboard/src/components/SearchBar.jsx:Search Bar"
    "src/omi/dashboard/src/components/SearchBar.css:Search Styles"
    "src/omi/dashboard/src/components/BeliefNetwork.jsx:Belief Network"
    "src/omi/dashboard/src/components/BeliefNetwork.css:Belief Styles"
    "src/omi/dashboard/src/components/StorageStats.jsx:Storage Stats"
    "src/omi/dashboard/src/components/StorageStats.css:Stats Styles"
    "src/omi/dashboard/src/components/SessionTimeline.jsx:Session Timeline"
    "src/omi/dashboard/src/components/SessionTimeline.css:Timeline Styles"
)

for item in "${components[@]}"; do
    IFS=':' read -r file name <<< "$item"
    if [ -f "$file" ]; then
        success "$name"
    else
        failure "$name missing: $file"
    fi
done

echo

# ============================================================================
# Check 4: API Client & Config
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Frontend Configuration Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check API client
if [ -f "src/omi/dashboard/src/api/client.js" ]; then
    success "API client exists"

    # Verify API client has required functions
    if grep -q "fetchGraph" "src/omi/dashboard/src/api/client.js"; then
        success "fetchGraph() function present"
    else
        failure "fetchGraph() function missing"
    fi

    if grep -q "searchMemories" "src/omi/dashboard/src/api/client.js"; then
        success "searchMemories() function present"
    else
        failure "searchMemories() function missing"
    fi
else
    failure "API client missing!"
fi

# Check Vite config
if [ -f "src/omi/dashboard/vite.config.js" ]; then
    success "Vite config exists"

    # Check for API proxy configuration
    if grep -q "proxy" "src/omi/dashboard/vite.config.js"; then
        success "API proxy configured"
    else
        warning "API proxy not found in vite.config.js"
    fi
else
    failure "Vite config missing!"
fi

# Check package.json
if [ -f "src/omi/dashboard/package.json" ]; then
    success "package.json exists"

    # Check for required dependencies
    deps=("react" "cytoscape" "chart.js")
    for dep in "${deps[@]}"; do
        if grep -q "\"$dep\"" "src/omi/dashboard/package.json"; then
            success "$dep dependency listed"
        else
            failure "$dep dependency missing"
        fi
    done
else
    failure "package.json missing!"
fi

echo

# ============================================================================
# Check 5: Documentation
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Documentation Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "E2E_VERIFICATION_CHECKLIST.md" ]; then
    success "E2E Verification Checklist exists"
else
    warning "E2E Verification Checklist not found"
fi

echo

# ============================================================================
# Summary
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All automated checks passed!${NC}"
    echo
    echo "Next steps for manual verification:"
    echo
    echo "1. Ensure OMI is initialized:"
    echo "   omi init"
    echo
    echo "2. Create some test memories:"
    echo "   omi store \"Python is a programming language\" --type fact"
    echo "   omi store \"I learned about FastAPI today\" --type experience"
    echo
    echo "3. Start the server:"
    echo "   omi serve --dashboard"
    echo
    echo "4. Open in browser:"
    echo "   http://localhost:8420/dashboard"
    echo
    echo "5. Follow the manual checklist:"
    echo "   E2E_VERIFICATION_CHECKLIST.md"
    echo
    echo "6. Test all features:"
    echo "   - Graph visualization (force-directed layout)"
    echo "   - Search with semantic recall"
    echo "   - Beliefs with confidence colors"
    echo "   - Storage statistics with charts"
    echo "   - Timeline with SSE real-time events"
    echo
else
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo
    echo "Please fix the issues above before proceeding."
    echo
    exit 1
fi

echo "================================================================"
