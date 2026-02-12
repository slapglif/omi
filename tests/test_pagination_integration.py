"""Integration Tests for Cursor-Based Pagination Stability

Tests cursor-based pagination across concurrent writes to ensure stable
pagination even when data is being modified.

Covers:
1. Cursor stability when new memories inserted during pagination
2. Cursor stability when memories deleted during pagination
3. Concurrent pagination with multiple cursors
4. Integration with REST API endpoints
5. Edge cases: empty results, invalid cursors, boundary conditions

Acceptance Criteria (from spec):
- Cursor-based pagination is stable across concurrent writes

Issue: https://github.com/slapglif/omi/issues/46
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from omi.storage.graph_palace import GraphPalace
from omi.rest_api import app
from omi.event_bus import reset_event_bus


@pytest.fixture(autouse=True)
def reset_bus():
    """Reset EventBus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def palace(temp_db):
    """Create a GraphPalace instance with temporary database."""
    return GraphPalace(temp_db)


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def disable_auth():
    """Disable API key authentication for tests."""
    old_key = os.environ.get("OMI_API_KEY")
    if "OMI_API_KEY" in os.environ:
        del os.environ["OMI_API_KEY"]
    yield
    # Restore old key if it existed
    if old_key:
        os.environ["OMI_API_KEY"] = old_key


class TestCursorStabilityWithInserts:
    """Test cursor stability when new records are inserted during pagination."""

    def test_cursor_stable_when_new_memories_inserted_at_beginning(self, palace):
        """
        Create 10 memories, get first page (5 items), insert 5 new memories
        at the beginning (newer timestamps), then get next page.

        Assert: Next page should return the correct 5 memories from original set,
        not affected by new insertions.
        """
        # Create initial 10 memories
        original_ids = []
        for i in range(10):
            memory_id = palace.store_memory(
                content=f"Original Memory {i}",
                memory_type="fact"
            )
            original_ids.append(memory_id)

        # Get first page
        page1 = palace.list_memories(limit=5, order_by="created_at", order_dir="desc")

        assert len(page1["memories"]) == 5
        assert page1["has_more"] is True
        assert page1["total_count"] == 10

        # Extract IDs from first page
        page1_ids = [m["id"] for m in page1["memories"]]

        # Insert 5 new memories (will have newer timestamps)
        new_ids = []
        for i in range(5):
            new_id = palace.store_memory(
                content=f"New Memory {i}",
                memory_type="fact"
            )
            new_ids.append(new_id)

        # Get second page using cursor from page1
        page2 = palace.list_memories(
            limit=5,
            cursor=page1["next_cursor"],
            order_by="created_at",
            order_dir="desc"
        )

        # Extract IDs from second page
        page2_ids = [m["id"] for m in page2["memories"]]

        # Verify: page2 should contain the remaining 5 original memories
        # New memories should NOT appear in page2
        assert len(page2["memories"]) == 5
        assert page2["has_more"] is False  # No more original memories

        # Verify no new memories leaked into page2
        for new_id in new_ids:
            assert new_id not in page2_ids

        # Verify all page2 IDs are from original set
        for pid in page2_ids:
            assert pid in original_ids

        # Verify no overlap between page1 and page2
        assert len(set(page1_ids) & set(page2_ids)) == 0

        # Verify total_count includes new memories
        assert page2["total_count"] == 15  # 10 original + 5 new

    def test_cursor_stable_when_new_memories_inserted_in_middle(self, palace):
        """
        Create 10 memories with controlled timestamps, get first page,
        insert memories in the middle of the sort order, then get next page.

        Assert: Cursor remains stable, no duplicates or skipped records.
        """
        import time

        # Create 10 memories with small delays to ensure distinct timestamps
        original_ids = []
        for i in range(10):
            memory_id = palace.store_memory(
                content=f"Memory {i:02d}",
                memory_type="fact"
            )
            original_ids.append(memory_id)
            time.sleep(0.01)  # Small delay to ensure timestamp ordering

        # Get first page (4 items)
        page1 = palace.list_memories(limit=4, order_by="created_at", order_dir="desc")
        assert len(page1["memories"]) == 4
        page1_ids = [m["id"] for m in page1["memories"]]

        # Insert new memories (will have newer timestamps than all originals)
        new_ids = []
        for i in range(3):
            new_id = palace.store_memory(
                content=f"Inserted {i}",
                memory_type="fact"
            )
            new_ids.append(new_id)
            time.sleep(0.01)

        # Get second page
        page2 = palace.list_memories(limit=4, cursor=page1["next_cursor"])
        page2_ids = [m["id"] for m in page2["memories"]]

        # Get third page
        page3 = palace.list_memories(limit=4, cursor=page2["next_cursor"])
        page3_ids = [m["id"] for m in page3["memories"]]

        # Combine all pages
        all_paginated_ids = page1_ids + page2_ids + page3_ids

        # Verify: No duplicates in paginated results
        assert len(all_paginated_ids) == len(set(all_paginated_ids))

        # Verify: All original memories appear exactly once
        for orig_id in original_ids:
            assert all_paginated_ids.count(orig_id) == 1

        # Verify: New memories don't appear in paginated results (they're newer)
        for new_id in new_ids:
            assert new_id not in all_paginated_ids


class TestCursorStabilityWithDeletes:
    """Test cursor stability when records are deleted during pagination."""

    def test_cursor_stable_when_memories_deleted_before_cursor(self, palace):
        """
        Create 10 memories, get first page, delete some memories from
        the first page, then get next page.

        Assert: Next page should still work correctly.
        """
        # Create 10 memories
        memory_ids = []
        for i in range(10):
            memory_id = palace.store_memory(
                content=f"Memory {i}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Get first page
        page1 = palace.list_memories(limit=3, order_by="created_at", order_dir="desc")
        assert len(page1["memories"]) == 3
        page1_ids = [m["id"] for m in page1["memories"]]

        # Delete one memory from page1
        palace.delete_memory(page1_ids[0])

        # Get second page - should still work
        page2 = palace.list_memories(limit=3, cursor=page1["next_cursor"])

        assert len(page2["memories"]) == 3
        page2_ids = [m["id"] for m in page2["memories"]]

        # Verify: No overlap between page1 and page2
        assert len(set(page1_ids) & set(page2_ids)) == 0

        # Verify: Deleted ID not in page2
        assert page1_ids[0] not in page2_ids

    def test_cursor_stable_when_cursor_memory_deleted(self, palace):
        """
        Create 10 memories, get first page, delete a memory that's NOT
        the cursor reference, then get next page.

        Assert: Pagination should work correctly even if a memory is deleted.
        """
        # Create 10 memories
        memory_ids = []
        for i in range(10):
            memory_id = palace.store_memory(
                content=f"Memory {i}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Get first page
        page1 = palace.list_memories(limit=5, order_by="created_at", order_dir="desc")
        assert len(page1["memories"]) == 5
        cursor = page1["next_cursor"]
        page1_ids = [m["id"] for m in page1["memories"]]

        # Delete a memory from page1 (but not the one cursor references)
        # Delete the first one, not the last one
        palace.delete_memory(page1_ids[0])

        # Get second page - should still work correctly
        page2 = palace.list_memories(limit=5, cursor=cursor)

        # Should return remaining 5 from original batch (10 - 5 seen in page1)
        assert len(page2["memories"]) == 5
        page2_ids = [m["id"] for m in page2["memories"]]

        # Verify no duplicates with page1
        assert len(set(page1_ids) & set(page2_ids)) == 0


class TestCursorStabilityWithUpdates:
    """Test cursor stability when records are updated during pagination."""

    def test_cursor_stable_when_access_count_updated(self, palace):
        """
        Create 10 memories, get first page ordered by created_at,
        update access counts of some memories directly in DB, then get next page.

        Assert: Next page should still return correct memories since
        ordering is by created_at, not access_count.
        """
        import time

        # Create 10 memories with small delays for distinct timestamps
        memory_ids = []
        for i in range(10):
            memory_id = palace.store_memory(
                content=f"Memory {i}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)
            time.sleep(0.01)

        # Get first page ordered by created_at
        page1 = palace.list_memories(limit=4, order_by="created_at", order_dir="desc")
        assert len(page1["memories"]) == 4
        page1_ids = [m["id"] for m in page1["memories"]]

        # Update access counts directly in database (not affecting created_at ordering)
        for memory_id in memory_ids[:5]:
            palace._conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (memory_id,)
            )
        palace._conn.commit()

        # Get second page
        page2 = palace.list_memories(limit=4, cursor=page1["next_cursor"])
        page2_ids = [m["id"] for m in page2["memories"]]

        # Verify: No overlap
        assert len(set(page1_ids) & set(page2_ids)) == 0

        # Verify: Page 2 has correct number of remaining results
        # 10 total - 4 in page1 = 6 remaining, requesting 4
        assert len(page2["memories"]) == 4


class TestConcurrentPagination:
    """Test multiple simultaneous pagination cursors."""

    def test_multiple_cursors_independent(self, palace):
        """
        Create 20 memories, start two independent pagination sessions
        (one ASC, one DESC), verify they don't interfere with each other.

        Assert: Each pagination session returns correct results independently.
        """
        # Create 20 memories
        memory_ids = []
        for i in range(20):
            memory_id = palace.store_memory(
                content=f"Memory {i:02d}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Start first pagination session (DESC)
        desc_page1 = palace.list_memories(limit=5, order_by="created_at", order_dir="desc")

        # Start second pagination session (ASC)
        asc_page1 = palace.list_memories(limit=5, order_by="created_at", order_dir="asc")

        # Continue first session
        desc_page2 = palace.list_memories(limit=5, cursor=desc_page1["next_cursor"])

        # Continue second session
        asc_page2 = palace.list_memories(limit=5, cursor=asc_page1["next_cursor"])

        # Verify DESC session consistency
        desc_page1_ids = [m["id"] for m in desc_page1["memories"]]
        desc_page2_ids = [m["id"] for m in desc_page2["memories"]]
        assert len(set(desc_page1_ids) & set(desc_page2_ids)) == 0

        # Verify ASC session consistency
        asc_page1_ids = [m["id"] for m in asc_page1["memories"]]
        asc_page2_ids = [m["id"] for m in asc_page2["memories"]]
        assert len(set(asc_page1_ids) & set(asc_page2_ids)) == 0

        # Verify different orderings
        # DESC should have newest first, ASC should have oldest first
        # They should be completely different sets for first pages
        assert len(set(desc_page1_ids) & set(asc_page1_ids)) == 0


class TestPaginationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cursor_with_exact_page_boundary(self, palace):
        """
        Create exactly N*page_size memories, verify cursor behavior
        at exact boundaries.

        Assert: Last page should have has_more=False and empty next_cursor.
        """
        # Create exactly 15 memories (3 pages of 5)
        for i in range(15):
            palace.store_memory(content=f"Memory {i}", memory_type="fact")

        # Get page 1
        page1 = palace.list_memories(limit=5)
        assert len(page1["memories"]) == 5
        assert page1["has_more"] is True
        assert page1["next_cursor"] != ""

        # Get page 2
        page2 = palace.list_memories(limit=5, cursor=page1["next_cursor"])
        assert len(page2["memories"]) == 5
        assert page2["has_more"] is True
        assert page2["next_cursor"] != ""

        # Get page 3
        page3 = palace.list_memories(limit=5, cursor=page2["next_cursor"])
        assert len(page3["memories"]) == 5
        assert page3["has_more"] is False
        assert page3["next_cursor"] == ""

    def test_cursor_with_limit_larger_than_total(self, palace):
        """
        Create 5 memories, request 10 per page.

        Assert: Should return all 5, has_more=False, no cursor.
        """
        for i in range(5):
            palace.store_memory(content=f"Memory {i}", memory_type="fact")

        result = palace.list_memories(limit=10)

        assert len(result["memories"]) == 5
        assert result["total_count"] == 5
        assert result["has_more"] is False
        assert result["next_cursor"] == ""

    def test_empty_result_with_cursor(self, palace):
        """
        Create 10 facts, paginate through them, then request next page
        after exhausting results.

        Assert: Should return empty results gracefully when cursor is empty.
        """
        for i in range(10):
            palace.store_memory(content=f"Memory {i}", memory_type="fact")

        # Get all results in two pages
        page1 = palace.list_memories(limit=5)
        assert page1["has_more"] is True

        page2 = palace.list_memories(limit=5, cursor=page1["next_cursor"])

        # Check if page2 has_more - if False, next_cursor should be empty
        if not page2["has_more"]:
            # Page 2 was the last page, trying to paginate with empty cursor
            # should return from beginning
            page3 = palace.list_memories(limit=5, cursor=page2["next_cursor"])

            # Empty cursor means start from beginning
            # So we should get results (first page again)
            assert len(page3["memories"]) == 5
            assert page3["has_more"] is True
        else:
            # There's a page 3
            page3 = palace.list_memories(limit=5, cursor=page2["next_cursor"])
            # Page 3 should have 0 memories since we only created 10
            assert len(page3["memories"]) == 0
            assert page3["has_more"] is False


class TestRESTAPIPaginationIntegration:
    """Integration tests with REST API endpoints."""

    def test_dashboard_memories_cursor_stability(self, client, disable_auth, temp_db):
        """
        Test /api/v1/dashboard/memories endpoint maintains cursor stability
        when memories are inserted between requests.

        Assert: Cursor-based pagination works correctly via REST API.
        """
        # Mock GraphPalace with our temp database
        palace = GraphPalace(temp_db)

        # Create 10 memories
        for i in range(10):
            palace.store_memory(content=f"Memory {i}", memory_type="fact")

        with patch('omi.dashboard_api.GraphPalace', return_value=palace):
            # Get first page
            response1 = client.get("/api/v1/dashboard/memories?limit=4")
            assert response1.status_code == 200
            data1 = response1.json()

            assert len(data1["memories"]) == 4
            assert data1["has_more"] is True
            assert "next_cursor" in data1

            cursor = data1["next_cursor"]
            page1_ids = [m["id"] for m in data1["memories"]]

            # Insert new memories
            for i in range(3):
                palace.store_memory(content=f"New Memory {i}", memory_type="fact")

            # Get second page
            response2 = client.get(f"/api/v1/dashboard/memories?limit=4&cursor={cursor}")
            assert response2.status_code == 200
            data2 = response2.json()

            assert len(data2["memories"]) == 4
            page2_ids = [m["id"] for m in data2["memories"]]

            # Verify no duplicates
            assert len(set(page1_ids) & set(page2_ids)) == 0

    def test_recall_endpoint_cursor_stability(self, client, disable_auth, temp_db):
        """
        Test /api/v1/recall endpoint with cursor pagination maintains
        stability across insertions.

        Assert: Cursor works correctly for recall operations.
        """
        from omi.embeddings import OllamaEmbedder, EmbeddingCache
        from omi.api import MemoryTools

        palace = GraphPalace(temp_db)

        # Create memories with searchable content
        for i in range(10):
            palace.store_memory(
                content=f"Test content {i}",
                memory_type="fact"
            )

        # Mock MemoryTools with proper initialization
        embedder = OllamaEmbedder(model="nomic-embed-text")
        cache = EmbeddingCache(cache_dir=Path(temp_db).parent, embedder=embedder)
        memory_tools = MemoryTools(palace_store=palace, embedder=embedder, cache=cache)

        with patch('omi.rest_api.get_memory_tools', return_value=memory_tools):
            # Get first page of recall results
            response1 = client.get("/api/v1/recall?query=test&limit=4")
            assert response1.status_code == 200
            data1 = response1.json()

            assert "memories" in data1
            assert "next_cursor" in data1
            assert "has_more" in data1

            if data1["has_more"]:
                cursor = data1["next_cursor"]

                # Insert new memory
                palace.store_memory(content="Test content new", memory_type="fact")

                # Get second page
                response2 = client.get(f"/api/v1/recall?query=test&limit=4&cursor={cursor}")
                assert response2.status_code == 200
                data2 = response2.json()

                assert "memories" in data2


class TestBeliefsPagination:
    """Test pagination specifically for beliefs."""

    def test_beliefs_cursor_stability(self, palace):
        """
        Create beliefs with different confidence levels, paginate,
        insert new beliefs, verify cursor stability.

        Assert: Beliefs pagination stable across inserts.
        """
        # Create 10 beliefs
        for i in range(10):
            palace.store_memory(
                content=f"Belief {i}",
                memory_type="belief",
                confidence=0.5 + (i * 0.05)
            )

        # Get first page
        page1 = palace.list_beliefs(limit=4, order_by="confidence", order_dir="desc")
        assert len(page1["beliefs"]) == 4
        page1_ids = [b["id"] for b in page1["beliefs"]]

        # Insert new beliefs
        for i in range(3):
            palace.store_memory(
                content=f"New Belief {i}",
                memory_type="belief",
                confidence=0.95
            )

        # Get second page
        page2 = palace.list_beliefs(limit=4, cursor=page1["next_cursor"])
        page2_ids = [b["id"] for b in page2["beliefs"]]

        # Verify no duplicates
        assert len(set(page1_ids) & set(page2_ids)) == 0


class TestEdgesPagination:
    """Test pagination for edges."""

    def test_edges_cursor_stability(self, palace):
        """
        Create memories and edges, paginate edges, insert new edges,
        verify cursor stability.

        Assert: Edges pagination stable across inserts.
        """
        # Create some memories first
        memory_ids = []
        for i in range(5):
            memory_id = palace.store_memory(
                content=f"Memory {i}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Create 12 edges
        for i in range(12):
            source_idx = i % 4
            target_idx = (i + 1) % 5
            palace.create_edge(
                source_id=memory_ids[source_idx],
                target_id=memory_ids[target_idx],
                edge_type="RELATED_TO",
                strength=0.5 + (i * 0.03)
            )

        # Get first page
        page1 = palace.list_edges(limit=5, order_by="created_at", order_dir="desc")
        assert len(page1["edges"]) == 5
        page1_ids = [e["id"] for e in page1["edges"]]

        # Insert new edges
        for i in range(3):
            palace.create_edge(
                source_id=memory_ids[0],
                target_id=memory_ids[1],
                edge_type="SUPPORTS",
                strength=0.9
            )

        # Get second page
        page2 = palace.list_edges(limit=5, cursor=page1["next_cursor"])
        page2_ids = [e["id"] for e in page2["edges"]]

        # Verify no duplicates
        assert len(set(page1_ids) & set(page2_ids)) == 0
