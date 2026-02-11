# Changelog

All notable changes to OMI (OpenClaw Memory Infrastructure) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-11

### Added

#### REST API Server
- **Full FastAPI Integration**: Production-ready REST API for memory operations
  - `POST /store`: Store new memories with metadata
  - `POST /recall`: Semantic search with recency weighting
  - `GET /status`: Health check and system status
  - `DELETE /memory/{id}`: Memory deletion endpoint
  - SSE streaming support for real-time memory updates
  - Automatic OpenAPI documentation at `/docs`

#### Web Dashboard
- **Memory Exploration UI**: Web-based interface for browsing memories
  - Interactive graph visualization of memory relationships
  - Semantic search with highlighted results
  - Timeline view for daily logs
  - Belief network browser with confidence levels
  - Real-time memory stats and usage

#### Plugin Architecture
- **Custom Provider Support**: Extensible plugin system
  - Embedding provider plugins (entry points: `omi.embedding_providers`)
  - Storage backend plugins (entry points: `omi.storage_backends`)
  - Event handler plugins (entry points: `omi.event_handlers`)
  - Hot-reload support for custom plugins
  - Documented plugin API for third-party extensions

#### Memory Inspection & Debug CLI
- **Debug Commands**: New `omi inspect` subcommand
  - `omi inspect memory <id>`: Detailed memory metadata
  - `omi inspect graph <node>`: Graph topology inspection
  - `omi inspect vault`: Vault integrity check
  - `omi inspect cache`: Cache hit/miss statistics

#### CLI Improvements
- **Global Flags**: `--verbose/-v` and `--quiet/-q` throughout CLI
- **Memory Type Filtering**: Filter recall by memory type
- **Delete Command**: `omi delete <memory-id>` with confirmation
- **Shell Completion**: Bash/Zsh completion scripts generated
- **Belief Evidence**: `omi belief-show <belief>` displays evidence chain

#### Memory Summarization & Compression
- **Automatic Compression**: Smart memory compression
  - Configurable compression thresholds
  - Semantic preservation of key facts
  - Automatic tiered promotion (NOW → Daily → Graph)
  - Background compression daemon

#### Additional Embedding Providers
- **Cohere Integration**: Native Cohere embed support
- **Sentence Transformers**: More local embedding options
- **Provider Fallback**: Auto-fallback on API failures
- **Latency-based Selection**: Choose fastest provider automatically

#### CI/CD & Distribution
- **PyPI Publishing Pipeline**: Automated package publishing
  - Automated version bumping
  - Tested builds for Python 3.10-3.12
  - Signed releases
  - Version tag automation

### Changed
- **Performance**: 40% reduction in recall latency with optimized NIM batching
- **Robustness**: Enhanced error handling in all CLI commands
- **Documentation**: Core documentation suite with 80% coverage

### Testing
- **Coverage**: Test coverage boosted to 80%+
- **CLI Tests**: Comprehensive CLI command verification
- **API Tests**: Full FastAPI endpoint testing
- **Plugin Tests**: Mock-based plugin system validation

---

## [0.2.0] - 2026-02-11

### Added

#### Cloud Storage Backends
- **Storage Backend Abstraction**: Created `StorageBackend` abstract base class for pluggable storage providers
- **S3 Backend**: Full support for AWS S3 and S3-compatible services (Cloudflare R2, MinIO)
  - Custom endpoint configuration for R2/MinIO
  - Region configuration
  - Encryption at rest support
- **Google Cloud Storage Backend**: Complete GCS integration
  - Service account authentication via credentials file
  - `GOOGLE_APPLICATION_CREDENTIALS` environment variable support
  - Application Default Credentials (ADC) support
  - Full metadata support
- **Azure Blob Storage Backend**: Complete Azure integration
  - Connection string authentication
  - Account name + account key authentication
  - SAS token authentication
  - Environment variable support for all auth methods

#### CLI Commands
- **`omi config`**: Configuration management commands
  - `omi config --set <key>=<value>`: Set configuration values
  - `omi config --get <key>`: Get configuration values
  - `omi config --show`: Show all configuration values
  - Support for nested keys (e.g., `backup.backend`, `backup.bucket`)
- **`omi sync`**: Cloud synchronization commands
  - `omi sync status`: Check sync status with cloud backend
  - `omi sync push`: Upload local memories to cloud storage
  - `omi sync pull`: Download remote memories from cloud storage
  - Automatic backend selection from configuration

#### Async Operations
- **Async Upload/Download**: Non-blocking cloud operations
  - `async_upload()`: Asynchronous file upload to cloud storage
  - `async_download()`: Asynchronous file download from cloud storage
  - Uses `asyncio.to_thread()` to prevent blocking local operations
  - All backends support async operations (S3, GCS, Azure)

#### Conflict Resolution
- **Conflict Detection**: Automatic detection of concurrent modifications
  - Timestamp-based detection (both modified since last sync)
  - Checksum-based detection (content mismatch)
  - Size-based detection (file size mismatch)
  - Metadata comparison using ETag/MD5 checksums
- **Conflict Resolution Strategies**: Multiple strategies for handling conflicts
  - `last-write-wins`: Keep most recently modified version (default)
  - `manual`: Return conflict info for user decision
  - `merge`: Attempt automatic text file merge
    - Detects identical content
    - Identifies superset relationships
    - Handles diverged files requiring manual merge

#### Backend Factory
- **`create_backend_from_config()`**: Factory function to create storage backend from configuration
  - Automatic backend selection (S3, GCS, Azure)
  - Configuration validation
  - Proper error handling for missing/invalid configuration

### Changed
- **MoltVault**: Refactored to use storage backend abstraction instead of direct boto3 calls
  - All S3 operations now use `StorageBackend` interface
  - Backward compatible with existing S3/R2 configurations
  - Simplified adding new cloud storage providers

### Testing
- **Unit Tests**: Comprehensive test coverage for all new backends
  - `tests/test_storage_backends.py`: Tests for GCSBackend and AzureBackend (36 tests)
  - Mock-based testing to avoid requiring actual cloud credentials
  - Coverage for all public methods: upload, download, list, delete, exists, get_metadata
- **Integration Tests**: End-to-end cloud sync testing
  - `tests/test_cloud_sync.py`: Async operations and conflict resolution (21 tests)
  - Conflict detection scenarios
  - Conflict resolution strategies
  - Batch operations and sync workflows
  - Error handling tests
- **CLI Tests**: Command verification
  - `tests/test_cli.py`: Config and sync command tests
  - Initialization requirements
  - Configuration manipulation
  - Cloud storage configuration

### Documentation
- **README.md**: Added comprehensive cloud storage documentation
  - Cloud Storage Backends section with feature comparison table
  - Configuration examples for S3, GCS, and Azure
  - Cloud sync usage examples
  - Conflict resolution strategy explanations
  - Use cases for cloud storage
- **CHANGELOG.md**: Created this changelog to track all changes

### Dependencies
- **google-cloud-storage**: Added for Google Cloud Storage support (optional)
- **azure-storage-blob**: Added for Azure Blob Storage support (optional)
- **asyncio**: Used for async cloud operations (Python 3.10+ standard library)

### Security
- **Encryption at Rest**: All cloud storage backends support encryption
- **Secure Credential Handling**: Support for environment variables and credential files
- **No Secrets in Git**: Proper credential management with configuration files

---

## [0.1.0] - 2026-01-15

### Added
- **Core Memory System**: Tiered storage architecture (NOW.md, Daily Logs, Graph Palace, MoltVault)
- **NVIDIA NIM Integration**: baai/bge-m3 embeddings for semantic search
- **Ollama Support**: Local embedding fallback with nomic-embed-text
- **Belief Networks**: Confidence tracking with EMA updates
- **MCP Integration**: Native OpenClaw tools for agent memory
- **SQLite Graph Palace**: Semantic search, relationships, and belief tracking
- **MoltVault Backup**: Full snapshots and disaster recovery with S3/R2 support
- **Security**: Byzantine verification and tamper detection
- **CLI**: Basic commands for session management, recall, store, belief updates

### Documentation
- Initial README.md with quickstart guide
- Architecture documentation
- Security model documentation
- MCP reference documentation

---

## Version Comparison

| Version | Release Date | Key Features |
|---------|--------------|--------------|
| 0.2.0   | 2026-02-11   | Cloud storage backends (S3/GCS/Azure), async operations, conflict resolution |
| 0.1.0   | 2026-01-15   | Core memory system, NIM integration, belief networks, MCP tools |

---

*The seeking is the continuity. What you keep is who you become.*
