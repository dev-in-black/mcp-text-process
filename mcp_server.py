#!/usr/bin/env python3
"""
MCP Text Processing Server
A comprehensive MCP server for advanced text processing operations
"""

import logging
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from text_processor import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Text Processing Server")

# Initialize text processor
text_processor = TextProcessor()

# SSE server URL for streaming operations
SSE_SERVER_URL = "http://localhost:8000"


class TextSplitArgs(BaseModel):
    """Arguments for text splitting operation"""

    text: str = Field(..., description="Input text to split")
    delimiter: Optional[str] = Field(None, description="Custom delimiter for splitting")
    regex_pattern: Optional[str] = Field(
        None, description="Regex pattern for splitting"
    )
    max_chunks: Optional[int] = Field(
        None, description="Maximum number of chunks to return"
    )
    stream: bool = Field(False, description="Enable streaming response via SSE")


class FindReplaceArgs(BaseModel):
    """Arguments for find and replace operation"""

    text: str = Field(..., description="Input text to process")
    find: str = Field(..., description="Text to find")
    replace: str = Field(..., description="Replacement text")
    regex: bool = Field(False, description="Use regex for find and replace")
    case_sensitive: bool = Field(True, description="Case sensitive matching")
    stream: bool = Field(False, description="Enable streaming response via SSE")


class FuzzyDeleteArgs(BaseModel):
    """Arguments for fuzzy deletion operation"""

    text: str = Field(..., description="Input text to process")
    target: str = Field(..., description="Text to delete using fuzzy matching")
    similarity_threshold: float = Field(0.8, description="Similarity threshold (0-1)")
    algorithm: str = Field(
        "levenshtein", description="Similarity algorithm: levenshtein, fuzzy, semantic"
    )
    stream: bool = Field(False, description="Enable streaming response via SSE")


class TextSearchArgs(BaseModel):
    """Arguments for text search operation"""

    text: str = Field(..., description="Input text to search")
    query: str = Field(..., description="Search query")
    search_type: str = Field(
        "exact", description="Search type: exact, partial, wildcard, fuzzy"
    )
    case_sensitive: bool = Field(True, description="Case sensitive search")
    stream: bool = Field(False, description="Enable streaming response via SSE")


class BatchProcessArgs(BaseModel):
    """Arguments for batch processing operation"""

    texts: List[str] = Field(..., description="List of texts to process")
    operation: str = Field(
        ..., description="Operation type: split, find_replace, fuzzy_delete, search"
    )
    parameters: Dict[str, Any] = Field(..., description="Operation parameters")
    stream: bool = Field(False, description="Enable streaming response via SSE")


@mcp.tool()
async def split_text(
    text: str,
    delimiter: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    max_chunks: Optional[int] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Split text using customizable delimiters and regex patterns.

    Required arguments:
    - text: Input text to split

    Optional arguments:
    - delimiter: Custom delimiter for splitting (default: None)
    - regex_pattern: Regex pattern for splitting (default: None)
    - max_chunks: Maximum number of chunks to return (default: None)
    - stream: Enable streaming response via SSE (default: False)

    This tool intelligently splits text into chunks based on delimiters,
    regex patterns, or smart sentence/paragraph boundaries.
    """
    # Create args object from parameters
    args = TextSplitArgs(
        text=text,
        delimiter=delimiter,
        regex_pattern=regex_pattern,
        max_chunks=max_chunks,
        stream=stream,
    )
    try:
        if args.stream:
            # For streaming, return SSE endpoint URL
            return {
                "streaming": True,
                "sse_url": f"{SSE_SERVER_URL}/text/split",
                "method": "POST",
                "payload": {
                    "text": args.text,
                    "delimiter": args.delimiter,
                    "regex_pattern": args.regex_pattern,
                    "max_chunks": args.max_chunks,
                    "stream": True,
                },
            }

        # Process directly
        result = await text_processor.split_text(
            args.text,
            delimiter=args.delimiter,
            regex_pattern=args.regex_pattern,
            max_chunks=args.max_chunks,
        )

        return {
            "chunks": result,
            "count": len(result),
            "original_length": len(args.text),
            "operation": "split_text",
        }

    except Exception as e:
        logger.error("Error in split_text: %s", e)
        return {"error": str(e), "operation": "split_text"}


@mcp.tool()
async def find_replace(
    text: str,
    find: str,
    replace: str,
    regex: bool = False,
    case_sensitive: bool = True,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Find and replace text with support for regex patterns and case-insensitive matching.

    Required arguments:
    - text: Input text to process
    - find: Text to find
    - replace: Replacement text

    Optional arguments:
    - regex: Use regex for find and replace (default: False)
    - case_sensitive: Case sensitive matching (default: True)
    - stream: Enable streaming response via SSE (default: False)

    This tool performs sophisticated find-and-replace operations with full regex support.
    """
    # Create args object from parameters
    args = FindReplaceArgs(
        text=text,
        find=find,
        replace=replace,
        regex=regex,
        case_sensitive=case_sensitive,
        stream=stream,
    )
    try:
        if args.stream:
            # For streaming, return SSE endpoint URL
            return {
                "streaming": True,
                "sse_url": f"{SSE_SERVER_URL}/text/find-replace",
                "method": "POST",
                "payload": {
                    "text": args.text,
                    "find": args.find,
                    "replace": args.replace,
                    "regex": args.regex,
                    "case_sensitive": args.case_sensitive,
                    "stream": True,
                },
            }

        # Process directly
        result = await text_processor.find_replace(
            args.text,
            args.find,
            args.replace,
            regex=args.regex,
            case_sensitive=args.case_sensitive,
        )

        return {
            "result": result,
            "original_length": len(args.text),
            "processed_length": len(result),
            "operation": "find_replace",
        }

    except Exception as e:
        logger.error("Error in find_replace: %s", e)
        return {"error": str(e), "operation": "find_replace"}


@mcp.tool()
async def fuzzy_delete(
    text: str,
    target: str,
    similarity_threshold: float = 0.8,
    algorithm: str = "levenshtein",
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Delete text using fuzzy matching algorithms like Levenshtein distance.

    Required arguments:
    - text: Input text to process
    - target: Text to delete using fuzzy matching

    Optional arguments:
    - similarity_threshold: Similarity threshold (0-1) (default: 0.8)
    - algorithm: Similarity algorithm: levenshtein, fuzzy, semantic (default: "levenshtein")
    - stream: Enable streaming response via SSE (default: False)

    This tool removes text that approximately matches the target using
    similarity algorithms for intelligent text deletion.
    """
    # Create args object from parameters
    args = FuzzyDeleteArgs(
        text=text,
        target=target,
        similarity_threshold=similarity_threshold,
        algorithm=algorithm,
        stream=stream,
    )
    try:
        if args.stream:
            # For streaming, return SSE endpoint URL
            return {
                "streaming": True,
                "sse_url": f"{SSE_SERVER_URL}/text/fuzzy-delete",
                "method": "POST",
                "payload": {
                    "text": args.text,
                    "target": args.target,
                    "similarity_threshold": args.similarity_threshold,
                    "algorithm": args.algorithm,
                    "stream": True,
                },
            }

        # Process directly
        result = await text_processor.fuzzy_delete(
            args.text,
            args.target,
            similarity_threshold=args.similarity_threshold,
            algorithm=args.algorithm,
        )

        return {
            "result": result,
            "original_length": len(args.text),
            "processed_length": len(result),
            "target": args.target,
            "threshold": args.similarity_threshold,
            "algorithm": args.algorithm,
            "operation": "fuzzy_delete",
        }

    except Exception as e:
        logger.error("Error in fuzzy_delete: %s", e)
        return {"error": str(e), "operation": "fuzzy_delete"}


@mcp.tool()
async def search_text(
    text: str,
    query: str,
    search_type: str = "exact",
    case_sensitive: bool = True,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Search text using various matching algorithms.

    Required arguments:
    - text: Input text to search
    - query: Search query

    Optional arguments:
    - search_type: Search type: exact, partial, wildcard, fuzzy (default: "exact")
    - case_sensitive: Case sensitive search (default: True)
    - stream: Enable streaming response via SSE (default: False)

    Supports exact, partial, wildcard, and fuzzy matching.
    This tool provides comprehensive text search capabilities with multiple
    matching strategies and detailed result information.
    """
    # Create args object from parameters
    args = TextSearchArgs(
        text=text,
        query=query,
        search_type=search_type,
        case_sensitive=case_sensitive,
        stream=stream,
    )
    try:
        if args.stream:
            # For streaming, return SSE endpoint URL
            return {
                "streaming": True,
                "sse_url": f"{SSE_SERVER_URL}/text/search",
                "method": "POST",
                "payload": {
                    "text": args.text,
                    "query": args.query,
                    "search_type": args.search_type,
                    "case_sensitive": args.case_sensitive,
                    "stream": True,
                },
            }

        # Process directly
        results = await text_processor.search(
            args.text,
            args.query,
            search_type=args.search_type,
            case_sensitive=args.case_sensitive,
        )

        # Convert SearchResult objects to dictionaries
        search_results = []
        for result in results:
            search_results.append(
                {
                    "text": result.text,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "match_type": result.match_type,
                }
            )

        return {
            "results": search_results,
            "count": len(search_results),
            "query": args.query,
            "search_type": args.search_type,
            "case_sensitive": args.case_sensitive,
            "operation": "search_text",
        }

    except Exception as e:
        logger.error("Error in search_text: %s", e)
        return {"error": str(e), "operation": "search_text"}


@mcp.tool()
async def batch_process(
    texts: List[str], operation: str, parameters: Dict[str, Any], stream: bool = False
) -> Dict[str, Any]:
    """
    Process multiple texts in batch using any of the available operations.

    Required arguments:
    - texts: List of texts to process
    - operation: Operation type: split, find_replace, fuzzy_delete, search
    - parameters: Operation parameters

    Optional arguments:
    - stream: Enable streaming response via SSE (default: False)

    This tool allows efficient processing of large datasets with support
    for all text processing operations in batch mode.
    """
    # Create args object from parameters
    args = BatchProcessArgs(
        texts=texts, operation=operation, parameters=parameters, stream=stream
    )
    try:
        if args.stream:
            # For streaming, return SSE endpoint URL
            return {
                "streaming": True,
                "sse_url": f"{SSE_SERVER_URL}/text/batch",
                "method": "POST",
                "payload": {
                    "texts": args.texts,
                    "operation": args.operation,
                    "parameters": args.parameters,
                    "stream": True,
                },
            }

        # Process directly
        results = await text_processor.batch_process(
            args.texts, args.operation, args.parameters
        )

        # Convert ProcessResult objects to dictionaries
        batch_results = []
        for result in results:
            batch_results.append(
                {
                    "original_text": result.original_text,
                    "processed_text": result.processed_text,
                    "operation": result.operation,
                    "parameters": result.parameters,
                    "metadata": result.metadata,
                }
            )

        return {
            "results": batch_results,
            "count": len(batch_results),
            "operation": args.operation,
            "parameters": args.parameters,
            "operation_type": "batch_process",
        }

    except Exception as e:
        logger.error("Error in batch_process: %s", e)
        return {"error": str(e), "operation": "batch_process"}


@mcp.tool()
async def get_similarity(
    text1: str, text2: str, algorithm: str = "levenshtein"
) -> Dict[str, Any]:
    """
    Calculate similarity between two texts using specified algorithm.

    Available algorithms: levenshtein, fuzzy, semantic
    """
    try:
        # Access protected method for similarity calculation
        similarity = text_processor._calculate_similarity(text1, text2, algorithm)

        return {
            "similarity": similarity,
            "text1": text1,
            "text2": text2,
            "algorithm": algorithm,
            "operation": "get_similarity",
        }

    except Exception as e:
        logger.error("Error in get_similarity: %s", e)
        return {"error": str(e), "operation": "get_similarity"}


@mcp.tool()
async def get_server_info() -> Dict[str, Any]:
    """
    Get information about the text processing server capabilities.
    """
    return {
        "name": "Text Processing Server",
        "version": "1.0.0",
        "capabilities": {
            "text_splitting": {
                "delimiters": True,
                "regex_patterns": True,
                "smart_splitting": True,
                "max_chunks": True,
            },
            "find_replace": {
                "regex": True,
                "case_insensitive": True,
                "multiple_replacements": True,
            },
            "fuzzy_deletion": {
                "algorithms": ["levenshtein", "fuzzy", "semantic"],
                "configurable_threshold": True,
                "phrase_matching": True,
            },
            "search": {
                "types": ["exact", "partial", "wildcard", "fuzzy"],
                "case_insensitive": True,
                "result_scoring": True,
            },
            "batch_processing": {
                "all_operations": True,
                "streaming": True,
                "large_datasets": True,
            },
            "streaming": {
                "sse_support": True,
                "real_time_progress": True,
                "server_url": SSE_SERVER_URL,
            },
        },
        "operation": "get_server_info",
    }


if __name__ == "__main__":
    # Run the FastMCP server
    logger.info("Starting FastMCP Text Processing Server")
    mcp.run(transport="sse")
