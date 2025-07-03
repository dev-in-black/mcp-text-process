#!/usr/bin/env python3
"""
FastAPI Server with SSE Streaming for MCP Text Processing
Companion server to the FastMCP text processing server
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from text_processor import TextProcessor
from sse_handler import SSEHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessRequest(BaseModel):
    """Base request model for text processing operations"""
    text: str = Field(..., description="Input text to process")
    stream: bool = Field(default=True, description="Enable streaming response")


class SplitRequest(TextProcessRequest):
    """Request model for text splitting operations"""
    delimiter: Optional[str] = Field(default=None, description="Custom delimiter")
    regex_pattern: Optional[str] = Field(default=None, description="Regex pattern for splitting")
    max_chunks: Optional[int] = Field(default=None, description="Maximum number of chunks")


class FindReplaceRequest(TextProcessRequest):
    """Request model for find and replace operations"""
    find: str = Field(..., description="Text to find")
    replace: str = Field(..., description="Replacement text")
    regex: bool = Field(default=False, description="Use regex matching")
    case_sensitive: bool = Field(default=True, description="Case sensitive matching")


class FuzzyDeleteRequest(TextProcessRequest):
    """Request model for fuzzy deletion operations"""
    target: str = Field(..., description="Text to delete")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold (0-1)")
    algorithm: str = Field(default="levenshtein", description="Similarity algorithm")


class SearchRequest(TextProcessRequest):
    """Request model for search operations"""
    query: str = Field(..., description="Search query")
    search_type: str = Field(default="exact", description="Search type: exact, partial, wildcard, fuzzy")
    case_sensitive: bool = Field(default=True, description="Case sensitive search")


class BatchProcessRequest(BaseModel):
    """Request model for batch processing operations"""
    texts: List[str] = Field(..., description="List of texts to process")
    operation: str = Field(..., description="Operation type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    stream: bool = Field(default=True, description="Enable streaming response")


# Global instances
text_processor = TextProcessor()
sse_handler = SSEHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting FastAPI SSE Server for MCP Text Processing")
    yield
    logger.info("Shutting down FastAPI SSE Server")


# Create FastAPI app
app = FastAPI(
    title="MCP Text Processing SSE Server",
    description="SSE streaming server for MCP text processing operations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": "MCP Text Processing SSE Server",
        "version": "1.0.0",
        "description": "SSE streaming server for MCP text processing operations",
        "companion_to": "FastMCP Text Processing Server",
        "endpoints": {
            "split": "/text/split",
            "find_replace": "/text/find-replace",
            "fuzzy_delete": "/text/fuzzy-delete",
            "search": "/text/search",
            "batch": "/text/batch"
        },
        "note": "This server provides SSE streaming capabilities for the main MCP server"
    }


@app.post("/text/split")
async def split_text(request: SplitRequest):
    """Split text with customizable delimiters and regex patterns"""
    try:
        if request.stream:
            return EventSourceResponse(
                sse_handler.stream_split(
                    request.text,
                    delimiter=request.delimiter,
                    regex_pattern=request.regex_pattern,
                    max_chunks=request.max_chunks
                )
            )
        else:
            result = await text_processor.split_text(
                request.text,
                delimiter=request.delimiter,
                regex_pattern=request.regex_pattern,
                max_chunks=request.max_chunks
            )
            return {"result": result}
    except Exception as e:
        logger.error(f"Error in split_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/find-replace")
async def find_replace(request: FindReplaceRequest):
    """Find and replace text with regex support"""
    try:
        if request.stream:
            return EventSourceResponse(
                sse_handler.stream_find_replace(
                    request.text,
                    request.find,
                    request.replace,
                    regex=request.regex,
                    case_sensitive=request.case_sensitive
                )
            )
        else:
            result = await text_processor.find_replace(
                request.text,
                request.find,
                request.replace,
                regex=request.regex,
                case_sensitive=request.case_sensitive
            )
            return {"result": result}
    except Exception as e:
        logger.error(f"Error in find_replace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/fuzzy-delete")
async def fuzzy_delete(request: FuzzyDeleteRequest):
    """Delete text using fuzzy matching algorithms"""
    try:
        if request.stream:
            return EventSourceResponse(
                sse_handler.stream_fuzzy_delete(
                    request.text,
                    request.target,
                    similarity_threshold=request.similarity_threshold,
                    algorithm=request.algorithm
                )
            )
        else:
            result = await text_processor.fuzzy_delete(
                request.text,
                request.target,
                similarity_threshold=request.similarity_threshold,
                algorithm=request.algorithm
            )
            return {"result": result}
    except Exception as e:
        logger.error(f"Error in fuzzy_delete: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/search")
async def search_text(request: SearchRequest):
    """Search text with various matching algorithms"""
    try:
        if request.stream:
            return EventSourceResponse(
                sse_handler.stream_search(
                    request.text,
                    request.query,
                    search_type=request.search_type,
                    case_sensitive=request.case_sensitive
                )
            )
        else:
            result = await text_processor.search(
                request.text,
                request.query,
                search_type=request.search_type,
                case_sensitive=request.case_sensitive
            )
            return {"result": result}
    except Exception as e:
        logger.error(f"Error in search_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/batch")
async def batch_process(request: BatchProcessRequest):
    """Process multiple texts in batch with streaming support"""
    try:
        if request.stream:
            return EventSourceResponse(
                sse_handler.stream_batch_process(
                    request.texts,
                    request.operation,
                    request.parameters
                )
            )
        else:
            result = await text_processor.batch_process(
                request.texts,
                request.operation,
                request.parameters
            )
            return {"result": result}
    except Exception as e:
        logger.error(f"Error in batch_process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )