# MCP Text Processing Server

A comprehensive Model Context Protocol (MCP) Python text processing server built with FastMCP and SSE streaming capabilities.

## Features

- **Advanced Text Manipulation**: Intelligent text splitting with customizable delimiters and regex patterns
- **Find & Replace**: Sophisticated functionality with regex and case-insensitive matching support
- **Fuzzy Text Deletion**: Uses similarity algorithms (Levenshtein distance, semantic matching)
- **Query-based Search**: Exact matches, partial matches, wildcard patterns, and fuzzy matching
- **Real-time Streaming**: SSE connections with progress updates and results streaming
- **Robust Error Handling**: Connection management and configurable processing parameters
- **Batch Operations**: Support for large text datasets

## Development Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run MCP server
python mcp_server.py

# Run FastAPI server with SSE
python server.py

# Run tests
pytest

# Lint
flake8 .
black .
```

## Project Structure

```
mcp-text/
├── mcp_server.py      # Main MCP server implementation (FastMCP)
├── server.py          # FastAPI server with SSE streaming
├── text_processor.py  # Core text processing logic
├── sse_handler.py     # Server-Sent Events handling
├── requirements.txt   # Python dependencies
├── tests/            # Test suite
└── CLAUDE.md         # This file
```

## Architecture

- **FastMCP**: Handles MCP protocol communication and tool registration
- **FastAPI + SSE**: Provides streaming HTTP endpoints for real-time updates
- **SSE Client**: Connects to streaming endpoints for progress updates

## Dependencies

- FastMCP for MCP protocol implementation
- FastAPI for SSE streaming endpoints
- Text processing libraries (regex, difflib for fuzzy matching)
- Sentence transformers for semantic similarity
- Testing framework (pytest)