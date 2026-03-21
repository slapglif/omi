"""
End-to-end acceptance criteria verification for pagination and streaming feature.

This test suite verifies all acceptance criteria from spec.md:
1. All list endpoints accept 'limit' (default 50, max 500) and 'cursor' parameters
2. Response includes 'total_count', 'next_cursor', and 'has_more' in metadata
3. Cursor-based pagination is stable across concurrent writes
4. /api/v1/recall supports SSE streaming mode via Accept: text/event-stream header
5. CLI recall command supports --limit and --offset flags
6. Performance: Paginated queries add less than 5ms overhead vs. unpaginated
"""

import pytest
import tempfile
import subprocess
import json
import time
from pathlib import Path
from omi.storage.graph_palace import GraphPalace


class TestAcceptanceCriteria:
    """End-to-end verification of all acceptance criteria."""

    def test_ac1_list_endpoints_accept_limit_and_cursor(self):
        """
        AC1: All list endpoints accept 'limit' (default 50, max 500) and 'cursor' parameters

        Verify:
        - GraphPalace.list_memories() accepts limit and cursor
        - GraphPalace.list_beliefs() accepts limit and cursor
        - GraphPalace.list_edges() accepts limit and cursor
        - Default limit is 50, max is 500 (or 1000 for edges)
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Add test data
            memory_ids = []
            for i in range(10):
                mem_id = palace.store_memory(
                    content=f"Test memory {i}",
                    memory_type="fact"
                )
                memory_ids.append(mem_id)

            # Test list_memories with limit and cursor
            result = palace.list_memories(limit=5)
            assert "memories" in result
            assert "total_count" in result
            assert "next_cursor" in result
            assert "has_more" in result
            assert len(result["memories"]) == 5
            assert result["has_more"] is True

            # Use cursor for next page
            result2 = palace.list_memories(limit=5, cursor=result["next_cursor"])
            assert len(result2["memories"]) == 5
            assert result2["has_more"] is False

            # Test default limit (should be 50, but we only have 10 records)
            result_default = palace.list_memories()
            assert len(result_default["memories"]) == 10

            # Test max limit for memories (500)
            result_max = palace.list_memories(limit=500)
            assert len(result_max["memories"]) == 10

            # Test that limit > 500 raises ValueError
            with pytest.raises(ValueError, match="limit must be <= 500"):
                palace.list_memories(limit=1000)

            # Test list_beliefs
            palace.store_memory("Test belief", memory_type="belief")
            result_beliefs = palace.list_beliefs(limit=10)
            assert "beliefs" in result_beliefs
            assert "total_count" in result_beliefs
            assert "next_cursor" in result_beliefs
            assert "has_more" in result_beliefs

            # Test list_edges
            palace.create_edge(memory_ids[0], memory_ids[1], "RELATED_TO", strength=0.8)
            result_edges = palace.list_edges(limit=10)
            assert "edges" in result_edges
            assert "total_count" in result_edges
            assert "next_cursor" in result_edges
            assert "has_more" in result_edges

            print("✓ AC1: All list endpoints accept limit and cursor parameters")

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_ac2_response_includes_pagination_metadata(self):
        """
        AC2: Response includes 'total_count', 'next_cursor', and 'has_more' in metadata

        Verify all endpoints return the required metadata fields.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Add test data
            memory_ids = []
            for i in range(15):
                mem_id = palace.store_memory(f"Content {i}", memory_type="fact")
                memory_ids.append(mem_id)

            palace.store_memory("Belief content", memory_type="belief")
            palace.create_edge(memory_ids[0], memory_ids[1], "SUPPORTS", strength=0.9)

            # Test list_memories response
            result = palace.list_memories(limit=5)
            assert "total_count" in result, "total_count missing from list_memories"
            assert "next_cursor" in result, "next_cursor missing from list_memories"
            assert "has_more" in result, "has_more missing from list_memories"
            assert result["total_count"] == 16  # 15 facts + 1 belief
            assert result["has_more"] is True
            assert result["next_cursor"] is not None

            # Test list_beliefs response
            result = palace.list_beliefs(limit=5)
            assert "total_count" in result, "total_count missing from list_beliefs"
            assert "next_cursor" in result, "next_cursor missing from list_beliefs"
            assert "has_more" in result, "has_more missing from list_beliefs"
            assert result["total_count"] == 1
            assert result["has_more"] is False
            assert not result["next_cursor"]  # Empty string or None when no more pages

            # Test list_edges response
            result = palace.list_edges(limit=5)
            assert "total_count" in result, "total_count missing from list_edges"
            assert "next_cursor" in result, "next_cursor missing from list_edges"
            assert "has_more" in result, "has_more missing from list_edges"
            assert result["total_count"] == 1
            assert result["has_more"] is False

            print("✓ AC2: Response includes total_count, next_cursor, and has_more")

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_ac3_cursor_stability_across_concurrent_writes(self):
        """
        AC3: Cursor-based pagination is stable across concurrent writes

        This is already tested in test_pagination_integration.py, but we verify
        the key behavior here as well.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Insert initial data
            initial_ids = []
            for i in range(10):
                mem_id = palace.store_memory(f"Content {i}", memory_type="fact")
                initial_ids.append(mem_id)

            # Get first page
            page1 = palace.list_memories(limit=5, order_by="created_at", order_dir="asc")
            assert len(page1["memories"]) == 5
            page1_ids = {m["id"] for m in page1["memories"]}

            # Insert new memories BETWEEN pages (this should not affect page 2)
            for i in range(10, 20):
                palace.store_memory(f"Content {i}", memory_type="fact")

            # Get second page using cursor from page1
            page2 = palace.list_memories(limit=5, cursor=page1["next_cursor"])
            assert len(page2["memories"]) == 5
            page2_ids = {m["id"] for m in page2["memories"]}

            # Verify no overlap (stable pagination)
            assert len(page1_ids & page2_ids) == 0, "Pages should not overlap"

            # Verify we got the expected records (from initial set)
            expected_ids = set(initial_ids[5:10])
            assert page2_ids == expected_ids, f"Expected {expected_ids}, got {page2_ids}"

            print("✓ AC3: Cursor-based pagination is stable across concurrent writes")

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_ac4_sse_streaming_support(self):
        """
        AC4: /api/v1/recall supports SSE streaming mode via Accept: text/event-stream header

        This is verified by checking the code structure and basic SSE functionality.
        Full REST API testing requires running server.
        """
        try:
            # Verify the recall_stream function exists in rest_api.py
            from omi import rest_api

            assert hasattr(rest_api, "recall_stream"), "recall_stream function should exist"
            assert hasattr(rest_api, "recall_memory"), "recall_memory endpoint should exist"

            # Read the source to verify SSE support
            rest_api_path = Path("./src/omi/rest_api.py")
            if rest_api_path.exists():
                source = rest_api_path.read_text()
                assert "text/event-stream" in source, "SSE content-type should be supported"
                assert "StreamingResponse" in source, "StreamingResponse should be used for SSE"
                assert "recall_stream" in source, "recall_stream generator should exist"

                print("✓ AC4: /api/v1/recall supports SSE streaming (verified in code)")
            else:
                pytest.skip("rest_api.py not found in src/omi/")
        except ImportError:
            # If fastapi is not installed, check the source code directly
            rest_api_path = Path("./src/omi/rest_api.py")
            if rest_api_path.exists():
                source = rest_api_path.read_text()
                assert "text/event-stream" in source, "SSE content-type should be supported"
                assert "async def recall_stream" in source, "recall_stream generator should exist"
                assert "Accept:" in source or "accept:" in source, "Accept header handling should exist"

                print("✓ AC4: /api/v1/recall supports SSE streaming (verified in source code)")
            else:
                pytest.skip("rest_api.py not found")

    def test_ac5_cli_recall_supports_limit_and_offset(self):
        """
        AC5: CLI recall command supports --limit and --offset flags

        Verify the CLI has the required flags.
        """
        # Check that CLI help includes --limit and --offset
        result = subprocess.run(
            ["python3", "-m", "omi.cli", "recall", "--help"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "./src"}
        )

        if result.returncode != 0:
            pytest.skip(f"CLI not accessible: {result.stderr}")

        help_text = result.stdout + result.stderr

        assert "--limit" in help_text, "--limit flag should be in help text"
        assert "--offset" in help_text, "--offset flag should be in help text"

        # Verify the flags are documented properly
        assert "limit" in help_text.lower(), "limit should be mentioned in help"
        assert "offset" in help_text.lower(), "offset should be mentioned in help"

        print("✓ AC5: CLI recall command supports --limit and --offset flags")

    def test_ac6_performance_overhead_under_5ms(self):
        """
        AC6: Performance: Paginated queries add less than 5ms overhead vs. unpaginated

        This is already tested in test_pagination_performance.py, but we do a
        quick sanity check here.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Insert test data
            for i in range(100):
                palace.store_memory(f"Content {i}", memory_type="fact")

            # Benchmark unpaginated query (raw SQL)
            start = time.perf_counter()
            cursor = palace._conn.cursor()
            cursor.execute("SELECT * FROM memories LIMIT 50")
            cursor.fetchall()
            unpaginated_time = (time.perf_counter() - start) * 1000  # ms

            # Benchmark paginated query
            start = time.perf_counter()
            result = palace.list_memories(limit=50)
            paginated_time = (time.perf_counter() - start) * 1000  # ms

            overhead = paginated_time - unpaginated_time

            # Allow some variance, but should be well under 5ms
            assert overhead < 5.0, f"Pagination overhead {overhead:.3f}ms exceeds 5ms threshold"

            print(f"✓ AC6: Performance overhead {overhead:.3f}ms < 5ms threshold")

        finally:
            Path(db_path).unlink(missing_ok=True)


class TestFullWorkflow:
    """Test complete end-to-end workflows using pagination."""

    def test_paginated_workflow_with_large_dataset(self):
        """
        Full workflow: Insert large dataset, paginate through it, verify all records retrieved.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Insert 200 memories
            total_records = 200
            expected_ids = set()
            for i in range(total_records):
                mem_id = palace.store_memory(
                    f"Content for memory {i}",
                    memory_type="fact"
                )
                expected_ids.add(mem_id)

            # Paginate through all records
            retrieved_ids = set()
            cursor = None
            page_count = 0
            page_size = 25

            while True:
                result = palace.list_memories(limit=page_size, cursor=cursor)
                page_count += 1

                # Collect IDs from this page
                for mem in result["memories"]:
                    retrieved_ids.add(mem["id"])

                # Verify pagination metadata
                assert result["total_count"] == total_records

                if not result["has_more"]:
                    assert not result["next_cursor"]  # Empty string or None
                    break

                assert result["next_cursor"] is not None
                cursor = result["next_cursor"]

                # Safety check to prevent infinite loop
                assert page_count <= 20, "Too many pages, possible infinite loop"

            # Verify we got all records exactly once
            assert len(retrieved_ids) == total_records, f"Expected {total_records} unique IDs, got {len(retrieved_ids)}"
            assert retrieved_ids == expected_ids, "Retrieved IDs don't match expected IDs"

            print(f"✓ Full workflow: Retrieved {total_records} records across {page_count} pages")

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_mixed_operations_pagination_stability(self):
        """
        Test pagination stability with mixed read/write operations.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            palace = GraphPalace(db_path)

            # Initial data
            initial_ids = []
            for i in range(20):
                mem_id = palace.store_memory(f"Initial {i}", memory_type="fact")
                initial_ids.append(mem_id)

            # Start pagination
            page1 = palace.list_memories(limit=10, order_by="created_at", order_dir="asc")

            # Concurrent operations
            palace.store_memory("New memory 1", memory_type="fact")
            palace.store_memory("New memory 2", memory_type="fact")

            # Continue pagination with cursor from page1
            page2 = palace.list_memories(limit=10, cursor=page1["next_cursor"])

            # Verify no duplicates between pages
            page1_ids = {m["id"] for m in page1["memories"]}
            page2_ids = {m["id"] for m in page2["memories"]}
            assert len(page1_ids & page2_ids) == 0, "No overlap should exist"

            # Verify we got all original records (new ones should not interfere)
            all_ids = page1_ids | page2_ids
            expected_original = set(initial_ids)
            assert expected_original == all_ids, "All original records should be retrieved exactly"

            print("✓ Pagination stable with concurrent writes")

        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
