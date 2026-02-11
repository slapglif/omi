import React, { useState, useEffect, useCallback } from 'react';
import { searchMemories } from '../api/client';
import './SearchBar.css';

/**
 * SearchBar Component
 *
 * Provides semantic search functionality with debouncing and result highlighting.
 * Calls the backend /api/v1/dashboard/search endpoint and returns matching memory IDs
 * to the parent component for graph highlighting.
 */
const SearchBar = ({ onSearchResults, onClearSearch }) => {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);

  // Debounce search to avoid excessive API calls
  useEffect(() => {
    // Clear results if query is empty
    if (!query || query.trim().length === 0) {
      setResults([]);
      setError(null);
      setShowResults(false);
      if (onClearSearch) {
        onClearSearch();
      }
      return;
    }

    // Debounce: wait 500ms after user stops typing
    const timeoutId = setTimeout(() => {
      performSearch(query);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [query]);

  const performSearch = async (searchQuery) => {
    try {
      setSearching(true);
      setError(null);

      // Call search API with min_relevance threshold
      const response = await searchMemories(searchQuery, {
        limit: 20,
        min_relevance: 0.3
      });

      setResults(response.results || []);
      setShowResults(true);

      // Notify parent component of matching node IDs for highlighting
      if (onSearchResults) {
        const matchedIds = response.results.map((result) => result.id);
        onSearchResults(matchedIds, response.results);
      }
    } catch (err) {
      setError(err.message);
      setResults([]);
      setShowResults(false);
    } finally {
      setSearching(false);
    }
  };

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleClear = () => {
    setQuery('');
    setResults([]);
    setError(null);
    setShowResults(false);
    if (onClearSearch) {
      onClearSearch();
    }
  };

  const handleResultClick = (result) => {
    // Notify parent to focus on specific node
    if (onSearchResults) {
      onSearchResults([result.id], [result]);
    }
  };

  return (
    <div className="search-bar-container">
      <div className="search-input-wrapper">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="search-input"
          placeholder="Search memories semantically..."
          value={query}
          onChange={handleInputChange}
          aria-label="Search memories"
        />
        {searching && <span className="search-loading">⏳</span>}
        {query && !searching && (
          <button
            className="search-clear-btn"
            onClick={handleClear}
            aria-label="Clear search"
            title="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {error && (
        <div className="search-error">
          <span className="error-icon">⚠️</span>
          <span className="error-text">{error}</span>
        </div>
      )}

      {showResults && results.length > 0 && (
        <div className="search-results">
          <div className="search-results-header">
            <span className="results-count">
              Found <strong>{results.length}</strong> matching memories
            </span>
          </div>
          <div className="search-results-list">
            {results.map((result) => (
              <div
                key={result.id}
                className="search-result-item"
                onClick={() => handleResultClick(result)}
              >
                <div className="result-header">
                  <span className={`result-type result-type-${result.memory_type}`}>
                    {result.memory_type}
                  </span>
                  <span className="result-relevance">
                    {(result.relevance_score * 100).toFixed(0)}% match
                  </span>
                </div>
                <div className="result-content">
                  {result.content.substring(0, 120)}
                  {result.content.length > 120 ? '...' : ''}
                </div>
                <div className="result-meta">
                  <span className="meta-item">
                    Created: {new Date(result.created_at).toLocaleDateString()}
                  </span>
                  <span className="meta-separator">•</span>
                  <span className="meta-item">
                    Accessed: {result.access_count || 0}x
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {showResults && results.length === 0 && !searching && (
        <div className="search-no-results">
          <span className="no-results-icon">🔍</span>
          <span className="no-results-text">
            No memories found matching "<strong>{query}</strong>"
          </span>
        </div>
      )}
    </div>
  );
};

export default SearchBar;
