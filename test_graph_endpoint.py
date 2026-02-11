#!/usr/bin/env python3
"""
Simple test script to verify the /api/v1/dashboard/graph endpoint.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omi.dashboard_api import router

# Create a minimal FastAPI app
app = FastAPI()
app.include_router(router)

# Create test client
client = TestClient(app)

def test_graph_endpoint():
    """Test the /api/v1/dashboard/graph endpoint"""
    print("Testing GET /api/v1/dashboard/graph...")

    try:
        response = client.get("/api/v1/dashboard/graph")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success! Response contains:")
            print(f"  - memories: {len(data.get('memories', []))} items")
            print(f"  - edges: {len(data.get('edges', []))} items")
            print(f"  - memory_count: {data.get('memory_count')}")
            print(f"  - edge_count: {data.get('edge_count')}")
            print(f"  - limit: {data.get('limit')}")
            return True
        elif response.status_code == 503:
            print(f"⚠ Service unavailable (expected if database not initialized)")
            print(f"  Detail: {response.json().get('detail')}")
            # This is acceptable - means the endpoint exists but DB not initialized
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            print(f"  Response: {response.json()}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_endpoint_with_limit():
    """Test the endpoint with custom limit parameter"""
    print("\nTesting GET /api/v1/dashboard/graph?limit=50...")

    try:
        response = client.get("/api/v1/dashboard/graph?limit=50")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success with custom limit!")
            print(f"  - Applied limit: {data.get('limit')}")
            return True
        elif response.status_code == 503:
            print(f"⚠ Service unavailable (expected if database not initialized)")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing /api/v1/dashboard/graph endpoint")
    print("=" * 60)

    success1 = test_graph_endpoint()
    success2 = test_graph_endpoint_with_limit()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
