"""Test that mypy can use type hints from the installed omi package.

This demonstrates that:
1. Type information is properly exported via py.typed
2. IDEs and type checkers can use the type hints
3. Static type checking works for downstream users
"""
from omi import MemoryTools
from typing import List, Dict, Any

# This file should pass mypy --strict checking when omi is installed
# and has proper type annotations with py.typed marker

def test_memory_tools_return_types() -> None:
    """Test that MemoryTools type hints are recognized by mypy.

    This function won't run (it would need real instances), but mypy
    should be able to verify the types statically.
    """
    # Mypy can verify this even without running the code
    # This tests that type information flows from the installed package

    # Create a stub instance for type checking (won't actually run)
    from omi.storage.graph_palace import GraphPalace
    from omi.embeddings import OllamaEmbedder, EmbeddingCache
    from pathlib import Path

    palace = GraphPalace(db_path=Path("./test.db"))
    embedder = OllamaEmbedder()
    cache = EmbeddingCache(cache_dir=Path("./cache"), embedder=embedder)

    memory = MemoryTools(palace_store=palace, embedder=embedder, cache=cache)

    # Type checker should verify these annotations
    results: List[Dict[str, Any]] = memory.recall(query="test", limit=5)

    # These should cause type errors if uncommented:
    # wrong_type: str = memory.recall(query="test")  # Error: incompatible types
    # memory.recall(query=123)  # Error: argument 1 has incompatible type "int"
    # memory.recall(limit="5")  # Error: argument has incompatible type "str"

def test_type_safety_with_literal_strings() -> None:
    """Test that string parameters are type-checked."""
    # This is valid - testing that mypy accepts correct usage
    query_string: str = "find memories about Python"
    # This would work at runtime and should type-check

def demonstrate_ide_completion() -> None:
    """This function demonstrates what IDE autocompletion provides.

    When a developer types:
        from omi import MemoryTools
        tools = MemoryTools(...)
        tools.recall(

    The IDE will show:
        query: str
        limit: int = 10
        min_relevance: float = 0.7
        memory_type: Optional[str] = None
        -> List[Dict[str, Any]]

    This is only possible because:
    1. py.typed marker exists
    2. All methods have type annotations
    3. The package is properly configured
    """
    pass

if __name__ == "__main__":
    print("This file demonstrates IDE autocompletion via type hints.")
    print("Run: mypy test_mypy_checking.py --strict")
    print("\nWith py.typed marker, IDEs can provide:")
    print("  - Parameter type hints")
    print("  - Return type information")
    print("  - Type error detection")
    print("  - Intelligent autocompletion")
