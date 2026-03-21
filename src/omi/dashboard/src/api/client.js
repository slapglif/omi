/**
 * API Client for OMI Dashboard
 *
 * Provides functions to communicate with the backend dashboard API endpoints.
 * All functions return promises and include proper error handling.
 *
 * Base URL: /api/v1/dashboard (proxied to backend by Vite in development)
 */

const BASE_URL = '/api/v1/dashboard';

/**
 * Helper function to build URL with query parameters
 * @param {string} endpoint - The API endpoint path
 * @param {Object} params - Query parameters object
 * @returns {string} Complete URL with query string
 */
function buildUrl(endpoint, params = {}) {
  const url = new URL(endpoint, window.location.origin);

  // Add query parameters
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      url.searchParams.append(key, value);
    }
  });

  return url.toString();
}

/**
 * Helper function to handle API responses
 * @param {Response} response - Fetch API response object
 * @returns {Promise<Object>} Parsed JSON data
 * @throws {Error} If response is not ok
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;

    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch (e) {
      // Could not parse error as JSON, use default message
    }

    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Fetch complete graph data (memories + edges) in one call
 *
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of nodes and edges to return (default: 100, max: 1000)
 * @returns {Promise<Object>} Graph data with memories and edges
 * @throws {Error} If request fails
 */
export async function fetchGraph(options = {}) {
  const { limit = 100 } = options;

  const url = buildUrl(`${BASE_URL}/graph`, { limit });
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch memories with optional filters and pagination
 *
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of memories to return (default: 100, max: 1000)
 * @param {number} options.offset - Number of memories to skip (default: 0)
 * @param {string} options.memory_type - Filter by type (fact, experience, belief, decision)
 * @param {string} options.order_by - Field to order by (created_at, access_count, last_accessed)
 * @param {string} options.order_dir - Order direction (asc, desc)
 * @returns {Promise<Object>} Memories data with pagination info
 * @throws {Error} If request fails
 */
export async function fetchMemories(options = {}) {
  const {
    limit = 100,
    offset = 0,
    memory_type = null,
    order_by = 'created_at',
    order_dir = 'desc'
  } = options;

  const params = {
    limit,
    offset,
    order_by,
    order_dir
  };

  if (memory_type) {
    params.memory_type = memory_type;
  }

  const url = buildUrl(`${BASE_URL}/memories`, params);
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch relationship edges with optional filters and pagination
 *
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of edges to return (default: 100, max: 1000)
 * @param {number} options.offset - Number of edges to skip (default: 0)
 * @param {string} options.edge_type - Filter by type (SUPPORTS, CONTRADICTS, RELATED_TO, DEPENDS_ON, POSTED, DISCUSSED)
 * @param {string} options.order_by - Field to order by (created_at, strength)
 * @param {string} options.order_dir - Order direction (asc, desc)
 * @returns {Promise<Object>} Edges data with pagination info
 * @throws {Error} If request fails
 */
export async function fetchEdges(options = {}) {
  const {
    limit = 100,
    offset = 0,
    edge_type = null,
    order_by = 'created_at',
    order_dir = 'desc'
  } = options;

  const params = {
    limit,
    offset,
    order_by,
    order_dir
  };

  if (edge_type) {
    params.edge_type = edge_type;
  }

  const url = buildUrl(`${BASE_URL}/edges`, params);
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch beliefs from the belief network
 *
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of beliefs to return (default: 100, max: 1000)
 * @param {number} options.offset - Number of beliefs to skip (default: 0)
 * @param {string} options.order_by - Field to order by (confidence, created_at, last_updated, evidence_count)
 * @param {string} options.order_dir - Order direction (asc, desc)
 * @returns {Promise<Object>} Beliefs data with pagination info
 * @throws {Error} If request fails
 */
export async function fetchBeliefs(options = {}) {
  const {
    limit = 100,
    offset = 0,
    order_by = 'last_updated',
    order_dir = 'desc'
  } = options;

  const params = {
    limit,
    offset,
    order_by,
    order_dir
  };

  const url = buildUrl(`${BASE_URL}/beliefs`, params);
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch database storage statistics
 *
 * @returns {Promise<Object>} Statistics including memory count, edge count, and type distributions
 * @throws {Error} If request fails
 */
export async function fetchStats() {
  const url = `${BASE_URL}/stats`;
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Perform semantic search for memories
 *
 * @param {string} query - Search query text (required)
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of results to return (default: 10, max: 100)
 * @param {number} options.min_relevance - Minimum relevance threshold (default: 0.5, range: 0.0-1.0)
 * @returns {Promise<Object>} Search results with relevance scores
 * @throws {Error} If request fails or query is empty
 */
export async function searchMemories(query, options = {}) {
  if (!query || query.trim().length === 0) {
    throw new Error('Search query cannot be empty');
  }

  const {
    limit = 10,
    min_relevance = 0.5
  } = options;

  const params = {
    q: query.trim(),
    limit,
    min_relevance
  };

  const url = buildUrl(`${BASE_URL}/search`, params);
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch version timeline (list of snapshots)
 *
 * @param {Object} options - Query options
 * @param {number} options.limit - Maximum number of versions to return (default: 50, max: 500)
 * @param {number} options.offset - Number of versions to skip (default: 0)
 * @param {string} options.order_dir - Order direction (asc, desc) (default: desc)
 * @returns {Promise<Object>} Version timeline data with pagination info
 * @throws {Error} If request fails
 */
export async function getVersionTimeline(options = {}) {
  const {
    limit = 50,
    offset = 0,
    order_dir = 'desc'
  } = options;

  const params = {
    limit,
    offset,
    order_dir
  };

  const url = buildUrl(`${BASE_URL}/versions`, params);
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Fetch details of a specific version/snapshot
 *
 * @param {string} versionId - The version ID to fetch
 * @returns {Promise<Object>} Version details including metadata and changes
 * @throws {Error} If request fails or version not found
 */
export async function getVersionDetails(versionId) {
  if (!versionId || versionId.trim().length === 0) {
    throw new Error('Version ID cannot be empty');
  }

  const url = `${BASE_URL}/versions/${encodeURIComponent(versionId)}`;
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Restore memory state to a specific version
 *
 * @param {string} versionId - The version ID to restore to
 * @param {Object} options - Restore options
 * @param {boolean} options.dry_run - Preview changes without applying (default: false)
 * @returns {Promise<Object>} Restore operation result
 * @throws {Error} If request fails or version not found
 */
export async function restoreToVersion(versionId, options = {}) {
  if (!versionId || versionId.trim().length === 0) {
    throw new Error('Version ID cannot be empty');
  }

  const { dry_run = false } = options;

  const url = `${BASE_URL}/versions/${encodeURIComponent(versionId)}/restore`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ dry_run })
  });
  return handleResponse(response);
}

/**
 * Check dashboard API health
 *
 * @returns {Promise<Object>} Health status
 * @throws {Error} If request fails
 */
export async function checkHealth() {
  const url = `${BASE_URL}/health`;
  const response = await fetch(url);
  return handleResponse(response);
}

// Export all functions as named exports
export default {
  fetchGraph,
  fetchMemories,
  fetchEdges,
  fetchBeliefs,
  fetchStats,
  searchMemories,
  getVersionTimeline,
  getVersionDetails,
  restoreToVersion,
  checkHealth
};
