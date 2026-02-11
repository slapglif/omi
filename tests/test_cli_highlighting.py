"""Tests for CLI term highlighting functionality.

Tests the highlight_terms() function and its integration with the recall command.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


class TestHighlightTerms:
    """Tests for the highlight_terms() utility function."""

    def test_single_word_highlighting(self):
        """Test highlighting a single word in text."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "This is a test query"
        query = "test"
        result = highlight_terms(text, query)

        # Result should contain ANSI escape codes for yellow/bold
        assert "\x1b[" in result  # Contains ANSI codes
        assert "test" in result.lower()  # Original text preserved

    def test_multi_word_highlighting(self):
        """Test highlighting multiple words from query."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "Python has excellent libraries for data science"
        query = "Python libraries"
        result = highlight_terms(text, query)

        # Both terms should be highlighted
        assert "\x1b[" in result
        # Count ANSI escape sequences (each highlight adds escape codes)
        assert result.count("\x1b[") >= 2

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "Authentication Bug Fixed"
        query = "authentication bug"
        result = highlight_terms(text, query)

        # Should highlight despite different case
        assert "\x1b[" in result
        assert "Authentication" in result
        assert "Bug" in result

    def test_empty_query(self):
        """Test that empty query returns unchanged text."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "Some content here"
        query = ""
        result = highlight_terms(text, query)

        # Should return original text without modification
        assert result == text
        assert "\x1b[" not in result

    def test_empty_text(self):
        """Test that empty text returns empty string."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = ""
        query = "test"
        result = highlight_terms(text, query)

        assert result == ""

    def test_no_matches(self):
        """Test text with no matching terms."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "Some random content"
        query = "nonexistent"
        result = highlight_terms(text, query)

        # Should return original text unchanged
        assert result == text

    def test_special_characters_in_query(self):
        """Test that special regex characters in query are escaped."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "File path: /usr/local/bin"
        query = "/usr/local"
        result = highlight_terms(text, query)

        # Should not raise regex error
        assert "/usr/local" in result

    def test_overlapping_terms(self):
        """Test highlighting with overlapping or repeated terms."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "test testing test"
        query = "test"
        result = highlight_terms(text, query)

        # Should highlight all occurrences of 'test'
        # Note: 'testing' contains 'test' so it will be partially highlighted
        assert "\x1b[" in result

    def test_preserves_original_text_content(self):
        """Test that highlighting preserves the original text content."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms
        import re

        text = "Fixed authentication bug in login flow"
        query = "authentication login"
        result = highlight_terms(text, query)

        # Strip ANSI codes to check original text is preserved
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned = ansi_escape.sub('', result)
        assert cleaned == text

    def test_whitespace_in_query(self):
        """Test handling of extra whitespace in query."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import highlight_terms

        text = "Python programming language"
        query = "  Python   programming  "
        result = highlight_terms(text, query)

        # Should handle extra whitespace gracefully
        assert "\x1b[" in result
        assert "Python" in result
        assert "programming" in result


class TestRecallCommandHighlighting:
    """Tests for term highlighting in the recall command output."""

    def test_recall_highlights_terms(self):
        """Test that recall command highlights search terms in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.full_text_search with test data
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Fixed authentication bug in login system"
            mock_mem.memory_type = "experience"
            mock_mem.confidence = 0.95
            mock_mem.created_at = None
            mock_results = [mock_mem]

            with patch.object(GraphPalace, 'full_text_search', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    # Enable color output
                    result = runner.invoke(cli, ["recall", "authentication"], color=True)

            # Check that output contains highlighted content
            assert result.exit_code == 0
            # Should contain ANSI escape codes for highlighting
            assert "\x1b[" in result.output
            assert "authentication" in result.output.lower()

    def test_recall_json_output_no_highlighting(self):
        """Test that JSON output does not contain ANSI escape codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.full_text_search
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Fixed authentication bug"
            mock_mem.memory_type = "experience"
            mock_mem.confidence = 0.95
            mock_mem.created_at = None
            mock_results = [mock_mem]

            with patch.object(GraphPalace, 'full_text_search', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "authentication", "--json-output"])

            # Check that JSON output has no ANSI codes
            assert result.exit_code == 0
            # Should not contain ANSI escape codes
            assert "\x1b[" not in result.output
            # But should contain the content
            assert "authentication" in result.output.lower()
            # Should be valid JSON
            assert '"content":' in result.output

    def test_recall_highlights_multiple_terms(self):
        """Test that recall highlights multiple search terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.full_text_search
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Python has excellent libraries for data science"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = None
            mock_mem.created_at = None
            mock_results = [mock_mem]

            with patch.object(GraphPalace, 'full_text_search', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    # Enable color output
                    result = runner.invoke(cli, ["recall", "Python libraries"], color=True)

            # Check that both terms are in output
            assert result.exit_code == 0
            assert "Python" in result.output
            assert "libraries" in result.output
            # Should contain ANSI escape codes
            assert "\x1b[" in result.output

    def test_recall_highlighting_with_truncation(self):
        """Test that highlighting works correctly with content truncation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock with content longer than 80 chars
            long_content = (
                "This is a very long piece of content that exceeds eighty characters "
                "and should be truncated by the display logic with authentication term"
            )
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = long_content
            mock_mem.memory_type = "experience"
            mock_mem.confidence = None
            mock_mem.created_at = None
            mock_results = [mock_mem]

            with patch.object(GraphPalace, 'full_text_search', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "content"])

            # Should handle truncation properly
            assert result.exit_code == 0
            # Output should contain ellipsis for truncation
            assert "..." in result.output

    def test_recall_no_results_no_highlighting(self):
        """Test recall with no results doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock with empty results
            with patch.object(GraphPalace, 'full_text_search', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "nonexistent"])

            # Should handle empty results gracefully
            assert result.exit_code == 0
            assert "0 found" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
