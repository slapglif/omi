"""
test_ann_accuracy.py - Accuracy Validation for ANN Vector Index

Validates that ANN (HNSW) search accuracy meets the 95% recall@10 requirement
compared to exact brute-force search.

Recall@10 = |ANN_top10 ∩ Exact_top10| / 10

This validates the core accuracy requirement from:
Issue: https://github.com/slapglif/omi/issues/[ANN-index]
Spec: Acceptance Criteria - Accuracy: ANN recall@10 is within 95% of exact brute-force results
"""
import uuid
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import numpy as np

from omi.storage.ann_index import ANNIndex


def generate_synthetic_embeddings(count: int, dim: int = 1024) -> List[Tuple[str, List[float]]]:
    """
    Generate synthetic embeddings for accuracy testing.

    Creates random normalized vectors that simulate real embeddings.

    Args:
        count: Number of embeddings to generate
        dim: Embedding dimension (default: 1024 for NIM)

    Returns:
        List of (memory_id, embedding) tuples
    """
    embeddings = []
    for i in range(count):
        # Generate random unit vector (normalized)
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit length
        memory_id = str(uuid.uuid4())
        embeddings.append((memory_id, vec.tolist()))

    return embeddings


def populate_ann_index(ann_index: ANNIndex, embeddings: List[Tuple[str, List[float]]]) -> None:
    """
    Populate ANN index with test embeddings.

    Uses rebuild_from_embeddings() for better index quality with batch construction.

    Args:
        ann_index: ANNIndex instance
        embeddings: List of (memory_id, embedding) tuples
    """
    ann_index.rebuild_from_embeddings(embeddings)


def get_exact_top_k(embeddings: List[Tuple[str, List[float]]],
                     query_embedding: List[float],
                     k: int = 10) -> List[str]:
    """
    Compute exact top-k nearest neighbors using brute-force cosine similarity.

    Args:
        embeddings: All embeddings in the dataset
        query_embedding: Query vector
        k: Number of neighbors to return

    Returns:
        List of memory IDs in order of similarity (highest first)
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return []

    similarities = []
    for memory_id, embedding in embeddings:
        emb_vec = np.array(embedding, dtype=np.float32)
        emb_norm = np.linalg.norm(emb_vec)

        if emb_norm > 0:
            # Cosine similarity
            similarity = np.dot(query_vec, emb_vec) / (query_norm * emb_norm)
            similarities.append((memory_id, float(similarity)))

    # Sort by similarity (descending) and return top-k IDs
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [memory_id for memory_id, _ in similarities[:k]]


def calculate_recall_at_k(ann_results: List[str], exact_results: List[str], k: int = 10) -> float:
    """
    Calculate recall@k: what fraction of exact top-k results are in ANN top-k.

    Args:
        ann_results: Memory IDs from ANN search (top-k)
        exact_results: Memory IDs from exact search (top-k)
        k: Number of results to consider

    Returns:
        Recall@k as a percentage (0-100)
    """
    ann_set = set(ann_results[:k])
    exact_set = set(exact_results[:k])

    if not exact_set:
        return 100.0  # Trivial case

    overlap = ann_set.intersection(exact_set)
    return (len(overlap) / k) * 100.0


class TestANNAccuracy:
    """
    Accuracy validation tests for ANN vector index.

    Compares ANN (HNSW) search results vs exact brute-force search
    to verify >= 95% recall@10 accuracy requirement.
    """

    def test_ann_accuracy_1000_memories(self):
        """
        Test ANN accuracy on 1000 memories with multiple queries.

        Validates that recall@10 >= 95% compared to exact brute-force search.
        Uses 1000 memories for faster testing while still being statistically meaningful.
        """
        num_memories = 1000
        num_queries = 50
        k = 10
        target_recall = 95.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "accuracy_test.db"

            # Generate synthetic embeddings
            embeddings = generate_synthetic_embeddings(num_memories, dim=1024)

            # Create ANN index directly (no recency weighting)
            ann_index = ANNIndex(str(db_path), dim=1024, enable_persistence=False)
            populate_ann_index(ann_index, embeddings)

            # Generate random query embeddings
            query_embeddings = []
            for _ in range(num_queries):
                vec = np.random.randn(1024).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                query_embeddings.append(vec.tolist())

            # Compare ANN vs Exact for each query
            recall_scores = []

            for query_emb in query_embeddings:
                # Get ANN results (direct from ANNIndex, no recency weighting)
                ann_results = ann_index.search(query_emb, k=k)
                ann_ids = [memory_id for memory_id, _ in ann_results]

                # Get exact results (brute-force)
                exact_ids = get_exact_top_k(embeddings, query_emb, k=k)

                # Calculate recall@k
                recall = calculate_recall_at_k(ann_ids, exact_ids, k=k)
                recall_scores.append(recall)

            # Calculate average recall@10
            avg_recall = np.mean(recall_scores)
            min_recall = np.min(recall_scores)
            max_recall = np.max(recall_scores)

            print(f"\n{'='*70}")
            print("ANN ACCURACY VALIDATION (Recall@10)")
            print(f"{'='*70}")
            print(f"\nDataset: {num_memories:,} memories with 1024-dim embeddings")
            print(f"Queries: {num_queries} random queries")
            print(f"k: {k} (top-{k} results)")
            print(f"\n{'-'*70}")
            print("Recall@10 Results:")
            print(f"{'-'*70}")
            print(f"  Average recall@10:    {avg_recall:.2f}%")
            print(f"  Min recall@10:        {min_recall:.2f}%")
            print(f"  Max recall@10:        {max_recall:.2f}%")
            print(f"  Target:               >= {target_recall}%")
            print(f"\n  Status:               {'✓ PASS' if avg_recall >= target_recall else '✗ FAIL'}")
            print(f"{'='*70}\n")

            # Assert accuracy requirement
            assert avg_recall >= target_recall, (
                f"ANN average recall@10 is {avg_recall:.2f}%, "
                f"below {target_recall}% target"
            )


    def test_ann_accuracy_with_768_dims(self):
        """
        Test ANN accuracy with 768-dimensional embeddings (Ollama).

        Validates that the 95% recall@10 requirement holds for both
        common embedding dimensions (768 for Ollama, 1024 for NIM).
        """
        num_memories = 1000
        num_queries = 50
        k = 10
        target_recall = 95.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "accuracy_test_768.db"

            # Generate 768-dim synthetic embeddings
            embeddings = generate_synthetic_embeddings(num_memories, dim=768)

            # Create ANN index directly
            ann_index = ANNIndex(str(db_path), dim=768, enable_persistence=False)
            populate_ann_index(ann_index, embeddings)

            # Generate random query embeddings
            query_embeddings = []
            for _ in range(num_queries):
                vec = np.random.randn(768).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                query_embeddings.append(vec.tolist())

            # Compare ANN vs Exact for each query
            recall_scores = []

            for query_emb in query_embeddings:
                # Get ANN results
                ann_results = ann_index.search(query_emb, k=k)
                ann_ids = [memory_id for memory_id, _ in ann_results]

                # Get exact results
                exact_ids = get_exact_top_k(embeddings, query_emb, k=k)

                # Calculate recall@k
                recall = calculate_recall_at_k(ann_ids, exact_ids, k=k)
                recall_scores.append(recall)

            # Calculate average recall@10
            avg_recall = np.mean(recall_scores)

            print(f"\nResults for 768-dim embeddings ({num_memories} memories):")
            print(f"  Average recall@10:  {avg_recall:.2f}%")
            print(f"  Target:             >= {target_recall}%")

            # Assert accuracy requirement
            assert avg_recall >= target_recall, (
                f"ANN average recall@10 is {avg_recall:.2f}%, "
                f"below {target_recall}% target"
            )

    def test_ann_accuracy_edge_cases(self):
        """
        Test ANN accuracy with edge cases.

        Tests:
        - Small dataset (k=10 with only 50 memories)
        - Queries at varying similarity levels
        """
        num_memories = 50
        num_queries = 20
        k = 10  # Top-10 from 50 memories
        target_recall = 95.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "accuracy_test_edge.db"

            # Generate synthetic embeddings
            embeddings = generate_synthetic_embeddings(num_memories, dim=1024)

            # Create ANN index directly
            ann_index = ANNIndex(str(db_path), dim=1024, enable_persistence=False)
            populate_ann_index(ann_index, embeddings)

            # Generate query embeddings with varying similarity
            query_embeddings = []

            # Add some queries that are similar to dataset items
            for i in range(num_queries // 2):
                # Take an existing embedding and add small noise
                base_emb = np.array(embeddings[i % len(embeddings)][1], dtype=np.float32)
                noise = np.random.randn(1024).astype(np.float32) * 0.1
                vec = base_emb + noise
                vec = vec / np.linalg.norm(vec)
                query_embeddings.append(vec.tolist())

            # Add some random queries
            for _ in range(num_queries - (num_queries // 2)):
                vec = np.random.randn(1024).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                query_embeddings.append(vec.tolist())

            # Compare ANN vs Exact for each query
            recall_scores = []

            for query_emb in query_embeddings:
                # Get ANN results
                ann_results = ann_index.search(query_emb, k=k)
                ann_ids = [memory_id for memory_id, _ in ann_results]

                # Get exact results
                exact_ids = get_exact_top_k(embeddings, query_emb, k=k)

                # Calculate recall@k
                recall = calculate_recall_at_k(ann_ids, exact_ids, k=k)
                recall_scores.append(recall)

            # Calculate average recall@10
            avg_recall = np.mean(recall_scores)

            print(f"\nResults for edge cases (small dataset, {num_memories} memories):")
            print(f"  Average recall@10:  {avg_recall:.2f}%")
            print(f"  Target:             >= {target_recall}%")

            # Assert accuracy requirement
            assert avg_recall >= target_recall, (
                f"ANN average recall@10 is {avg_recall:.2f}%, "
                f"below {target_recall}% target"
            )


if __name__ == "__main__":
    """
    Run accuracy validation as standalone script for development.

    Usage: python tests/test_ann_accuracy.py
    """
    print("Running ANN Accuracy Validation...")
    print("This will take a few seconds...\n")

    test = TestANNAccuracy()
    test.test_ann_accuracy_1000_memories()
    print("\n✓ All accuracy tests passed!")
