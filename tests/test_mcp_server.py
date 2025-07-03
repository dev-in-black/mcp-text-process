"""
Tests for FastMCP Text Processing Server
"""

import pytest
import asyncio
from mcp.client.sse import sse_client
from mcp_server import text_processor


class TestMCPServer:
    """Test FastMCP server tools"""

    @pytest.fixture
    async def mcp_client(self):
        """Create MCP client for testing"""
        # For testing, we'll test the underlying functions directly
        # In real tests, you'd connect to the actual MCP server
        return None

    @pytest.mark.asyncio
    async def test_split_text_tool(self):
        """Test split_text MCP tool"""
        from mcp_server import split_text, TextSplitArgs

        args = TextSplitArgs(
            text="This is a test. Another sentence! Final question?",
            delimiter=None,
            regex_pattern=None,
            max_chunks=None,
            stream=False,
        )

        result = await split_text(args)

        assert "chunks" in result
        assert "count" in result
        assert result["count"] == len(result["chunks"])
        assert result["operation"] == "split_text"

    @pytest.mark.asyncio
    async def test_find_replace_tool(self):
        """Test find_replace MCP tool"""
        from mcp_server import find_replace, FindReplaceArgs

        args = FindReplaceArgs(
            text="hello world hello universe",
            find="hello",
            replace="hi",
            regex=False,
            case_sensitive=False,
            stream=False,
        )

        result = await find_replace(args)

        assert "result" in result
        assert "hi world" in result["result"]
        assert result["operation"] == "find_replace"

    @pytest.mark.asyncio
    async def test_fuzzy_delete_tool(self):
        """Test fuzzy_delete MCP tool"""
        from mcp_server import fuzzy_delete, FuzzyDeleteArgs

        args = FuzzyDeleteArgs(
            text="apple banana orange apple grape",
            target="apple",
            similarity_threshold=1.0,
            algorithm="levenshtein",
            stream=False,
        )

        result = await fuzzy_delete(args)

        assert "result" in result
        assert "apple" not in result["result"]
        assert result["operation"] == "fuzzy_delete"

    @pytest.mark.asyncio
    async def test_search_text_tool(self):
        """Test search_text MCP tool"""
        from mcp_server import search_text, TextSearchArgs

        args = TextSearchArgs(
            text="The quick brown fox jumps over the lazy dog",
            query="fox",
            search_type="exact",
            case_sensitive=True,
            stream=False,
        )

        result = await search_text(args)

        assert "results" in result
        assert "count" in result
        assert result["count"] == 1
        assert result["results"][0]["text"] == "fox"
        assert result["operation"] == "search_text"

    @pytest.mark.asyncio
    async def test_batch_process_tool(self):
        """Test batch_process MCP tool"""
        from mcp_server import batch_process, BatchProcessArgs

        args = BatchProcessArgs(
            texts=["hello world", "hello universe"],
            operation="find_replace",
            parameters={"find": "hello", "replace": "hi", "case_sensitive": False},
            stream=False,
        )

        result = await batch_process(args)

        assert "results" in result
        assert "count" in result
        assert result["count"] == 2
        assert result["operation_type"] == "batch_process"

    @pytest.mark.asyncio
    async def test_get_similarity_tool(self):
        """Test get_similarity MCP tool"""
        from mcp_server import get_similarity

        result = await get_similarity("apple", "aple", "levenshtein")

        assert "similarity" in result
        assert "algorithm" in result
        assert result["algorithm"] == "levenshtein"
        assert result["operation"] == "get_similarity"
        assert 0.0 <= result["similarity"] <= 1.0

    @pytest.mark.asyncio
    async def test_get_server_info_tool(self):
        """Test get_server_info MCP tool"""
        from mcp_server import get_server_info

        result = await get_server_info()

        assert "name" in result
        assert "version" in result
        assert "capabilities" in result
        assert result["operation"] == "get_server_info"

        # Check capabilities structure
        capabilities = result["capabilities"]
        assert "text_splitting" in capabilities
        assert "find_replace" in capabilities
        assert "fuzzy_deletion" in capabilities
        assert "search" in capabilities
        assert "batch_processing" in capabilities
        assert "streaming" in capabilities

    @pytest.mark.asyncio
    async def test_streaming_enabled_responses(self):
        """Test tools with streaming enabled"""
        from mcp_server import split_text, TextSplitArgs

        args = TextSplitArgs(text="This is a test.", stream=True)

        result = await split_text(args)

        assert result["streaming"] is True
        assert "sse_url" in result
        assert "method" in result
        assert "payload" in result
        assert result["method"] == "POST"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in MCP tools"""
        from mcp_server import find_replace, FindReplaceArgs

        # Test with invalid regex
        args = FindReplaceArgs(
            text="test text",
            find="[invalid",
            replace="replace",
            regex=True,
            stream=False,
        )

        result = await find_replace(args)

        assert "error" in result
        assert result["operation"] == "find_replace"

    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty text in MCP tools"""
        from mcp_server import split_text, TextSplitArgs

        args = TextSplitArgs(text="", stream=False)

        result = await split_text(args)

        assert "chunks" in result
        assert result["chunks"] == []
        assert result["count"] == 0


class TestMCPIntegration:
    """Integration tests for MCP server"""

    @pytest.mark.asyncio
    async def test_text_processor_integration(self):
        """Test that MCP tools integrate properly with text processor"""
        # Test that text processor is properly initialized
        assert text_processor is not None

        # Test basic text processing
        result = await text_processor.split_text("test text")
        assert isinstance(result, list)

        result = await text_processor.find_replace("test", "test", "replaced")
        assert result == "replaced"

    def test_pydantic_models(self):
        """Test Pydantic model validation"""
        from mcp_server import TextSplitArgs, FindReplaceArgs, FuzzyDeleteArgs

        # Test TextSplitArgs
        args = TextSplitArgs(text="test")
        assert args.text == "test"
        assert args.stream is False

        # Test FindReplaceArgs
        args = FindReplaceArgs(text="test", find="test", replace="replaced")
        assert args.text == "test"
        assert args.find == "test"
        assert args.replace == "replaced"

        # Test FuzzyDeleteArgs
        args = FuzzyDeleteArgs(text="test", target="test")
        assert args.text == "test"
        assert args.target == "test"
        assert args.similarity_threshold == 0.8
        assert args.algorithm == "levenshtein"


if __name__ == "__main__":
    pytest.main([__file__])
