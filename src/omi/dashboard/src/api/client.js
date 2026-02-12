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

/**
 * Fetch distributed sync status
 *
 * @returns {Promise<Object>} Sync status including topology, lag metrics, and instance list
 * @throws {Error} If request fails
 */
export async function fetchSyncStatus() {
  const url = '/api/sync/status';
  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Start incremental sync (real-time event-based synchronization)
 *
 * @returns {Promise<Object>} Operation result with status and message
 * @throws {Error} If request fails
 */
export async function startIncrementalSync() {
  const url = '/api/sync/incremental/start';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  });
  return handleResponse(response);
}

/**
 * Stop incremental sync
 *
 * @returns {Promise<Object>} Operation result with status and message
 * @throws {Error} If request fails
 */
export async function stopIncrementalSync() {
  const url = '/api/sync/incremental/stop';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  });
  return handleResponse(response);
}

/**
 * Import memory snapshot from another OMI instance (bulk sync)
 *
 * @param {string} instanceId - Source instance ID
 * @param {string} endpoint - Source instance network endpoint (URL)
 * @returns {Promise<Object>} Sync result with success status and message
 * @throws {Error} If request fails or parameters are invalid
 */
export async function bulkSyncFrom(instanceId, endpoint) {
  if (!instanceId || !endpoint) {
    throw new Error('Both instanceId and endpoint are required for bulk sync');
  }

  const url = '/api/sync/bulk/from';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      instance_id: instanceId,
      endpoint: endpoint
    })
  });
  return handleResponse(response);
}

/**
 * Export memory snapshot to another OMI instance (bulk sync)
 *
 * @param {string} instanceId - Target instance ID
 * @param {string} endpoint - Target instance network endpoint (URL)
 * @returns {Promise<Object>} Sync result with success status and message
 * @throws {Error} If request fails or parameters are invalid
 */
export async function bulkSyncTo(instanceId, endpoint) {
  if (!instanceId || !endpoint) {
    throw new Error('Both instanceId and endpoint are required for bulk sync');
  }

  const url = '/api/sync/bulk/to';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      instance_id: instanceId,
      endpoint: endpoint
    })
  });
  return handleResponse(response);
}

/**
 * Register an OMI instance to the sync cluster
 *
 * @param {string} instanceId - Unique identifier for the instance
 * @param {string} endpoint - Network endpoint (URL), optional
 * @returns {Promise<Object>} Registration result
 * @throws {Error} If request fails or instanceId is invalid
 */
export async function registerInstance(instanceId, endpoint = null) {
  if (!instanceId) {
    throw new Error('instanceId is required to register an instance');
  }

  const url = '/api/sync/instances/register';
  const body = { instance_id: instanceId };
  if (endpoint) {
    body.endpoint = endpoint;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  return handleResponse(response);
}

/**
 * Remove an OMI instance from the sync cluster
 *
 * @param {string} instanceId - ID of instance to unregister
 * @returns {Promise<Object>} Unregistration result
 * @throws {Error} If request fails or instanceId is invalid
 */
export async function unregisterInstance(instanceId) {
  if (!instanceId) {
    throw new Error('instanceId is required to unregister an instance');
  }

  const url = `/api/sync/instances/${instanceId}`;
  const response = await fetch(url, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json'
    }
  });
  return handleResponse(response);
}

/**
 * Reconcile memory stores after network partition with conflict resolution
 *
 * @param {string} instanceId - ID of instance to reconcile with
 * @returns {Promise<Object>} Reconciliation result
 * @throws {Error} If request fails or instanceId is invalid
 */
export async function reconcilePartition(instanceId) {
  if (!instanceId) {
    throw new Error('instanceId is required to reconcile partition');
  }

  const url = '/api/sync/reconcile';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      instance_id: instanceId
    })
  });
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
  checkHealth,
  fetchSyncStatus,
  startIncrementalSync,
  stopIncrementalSync,
  bulkSyncFrom,
  bulkSyncTo,
  registerInstance,
  unregisterInstance,
  reconcilePartition
};
