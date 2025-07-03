# MCP Text Processing Server

[![CI/CD Pipeline](https://github.com/your-username/mcp-text/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mcp-text/actions/workflows/ci.yml)
[![Docker Build](https://img.shields.io/docker/build/your-username/mcp-text-processor)](https://hub.docker.com/r/your-username/mcp-text-processor)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive Model Context Protocol (MCP) Python text processing server built with FastMCP and SSE streaming capabilities.

## Features

- **Advanced Text Manipulation**: Intelligent text splitting with customizable delimiters and regex patterns
- **Find & Replace**: Sophisticated functionality with regex and case-insensitive matching support
- **Fuzzy Text Deletion**: Uses similarity algorithms (Levenshtein distance, semantic matching)
- **Query-based Search**: Exact matches, partial matches, wildcard patterns, and fuzzy matching
- **Real-time Streaming**: SSE connections with progress updates and results streaming
- **Robust Error Handling**: Connection management and configurable processing parameters
- **Batch Operations**: Support for large text datasets

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/mcp-text.git
cd mcp-text

# Run with Docker Compose
docker-compose up -d

# Check services are running
curl http://localhost:8000/health
```

### Local Development

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run MCP server
python mcp_server.py

# Run SSE server (in another terminal)
python server.py

# Run tests
pytest
```

## Architecture

- **FastMCP**: Handles MCP protocol communication and tool registration
- **FastAPI + SSE**: Provides streaming HTTP endpoints for real-time updates
- **Dual Server**: Separate MCP and SSE servers for optimal performance

## MCP Tools

### 1. `split_text`
Split text using customizable delimiters and regex patterns.

**Required:** `text`  
**Optional:** `delimiter`, `regex_pattern`, `max_chunks`, `stream`

### 2. `find_replace`
Find and replace text with regex support.

**Required:** `text`, `find`, `replace`  
**Optional:** `regex`, `case_sensitive`, `stream`

### 3. `fuzzy_delete`
Delete text using fuzzy matching algorithms.

**Required:** `text`, `target`  
**Optional:** `similarity_threshold`, `algorithm`, `stream`

### 4. `search_text`
Search text with various matching strategies.

**Required:** `text`, `query`  
**Optional:** `search_type`, `case_sensitive`, `stream`

### 5. `batch_process`
Process multiple texts in batch.

**Required:** `texts`, `operation`, `parameters`  
**Optional:** `stream`

### 6. `get_similarity`
Calculate similarity between texts.

**Required:** `text1`, `text2`  
**Optional:** `algorithm`

## Docker Usage

### Single Container
```bash
# Build image
docker build -t mcp-text-processor .

# Run MCP server
docker run -p 8080:8080 mcp-text-processor python mcp_server.py

# Run SSE server
docker run -p 8000:8000 mcp-text-processor python server.py
```

### Docker Compose (Recommended)
```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Endpoints

### SSE Server (Port 8000)
- `GET /` - Server information
- `POST /text/split` - Text splitting with streaming
- `POST /text/find-replace` - Find and replace with streaming
- `POST /text/fuzzy-delete` - Fuzzy deletion with streaming
- `POST /text/search` - Text search with streaming
- `POST /text/batch` - Batch processing with streaming
- `GET /health` - Health check

### MCP Server (Port 8080)
- MCP protocol endpoints for tool communication

## Configuration

### Environment Variables
- `MCP_TRANSPORT`: MCP transport type (default: "sse")
- `MCP_HOST`: MCP server host (default: "0.0.0.0")
- `MCP_PORT`: MCP server port (default: "8080")
- `FASTAPI_HOST`: FastAPI server host (default: "0.0.0.0")
- `FASTAPI_PORT`: FastAPI server port (default: "8000")

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_mcp_server.py
```

### Code Quality
```bash
# Lint with pylint
pylint *.py

# Format with black
black .

# Type checking
flake8 .
```

## CI/CD

The project includes comprehensive GitHub Actions workflows:

- **CI Pipeline**: Tests across Python 3.9-3.12, linting, security checks
- **Docker Testing**: Automated Docker image building and testing
- **Security Scanning**: Bandit and safety checks
- **Code Coverage**: Automated coverage reporting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📫 Issues: [GitHub Issues](https://github.com/your-username/mcp-text/issues)
- 📖 Documentation: [Project Wiki](https://github.com/your-username/mcp-text/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/mcp-text/discussions)