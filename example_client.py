#!/usr/bin/env python3
"""
Example client for testing FastMCP Text Processing Server
"""

import asyncio
import json
from typing import Dict, Any
import httpx
from fastmcp.client import FastMCPClient


async def test_mcp_tools():
    """Test MCP text processing tools"""
    
    # Connect to FastMCP server
    client = FastMCPClient("stdio://python mcp_server.py")
    
    try:
        # Initialize connection
        await client.connect()
        
        print("🔗 Connected to FastMCP Text Processing Server")
        
        # Test text splitting
        print("\n📝 Testing text splitting...")
        split_result = await client.call_tool("split_text", {
            "text": "This is a test. Another sentence! Final question?",
            "stream": False
        })
        print(f"Split result: {split_result}")
        
        # Test find and replace
        print("\n🔍 Testing find and replace...")
        replace_result = await client.call_tool("find_replace", {
            "text": "Hello world, hello universe",
            "find": "hello",
            "replace": "hi",
            "case_sensitive": False,
            "stream": False
        })
        print(f"Replace result: {replace_result}")
        
        # Test fuzzy deletion
        print("\n🎯 Testing fuzzy deletion...")
        fuzzy_result = await client.call_tool("fuzzy_delete", {
            "text": "apple banana orange apple grape",
            "target": "apple",
            "similarity_threshold": 1.0,
            "algorithm": "levenshtein",
            "stream": False
        })
        print(f"Fuzzy delete result: {fuzzy_result}")
        
        # Test search
        print("\n🔍 Testing text search...")
        search_result = await client.call_tool("search_text", {
            "text": "The quick brown fox jumps over the lazy dog",
            "query": "fox",
            "search_type": "exact",
            "case_sensitive": True,
            "stream": False
        })
        print(f"Search result: {search_result}")
        
        # Test similarity calculation
        print("\n📊 Testing similarity calculation...")
        similarity_result = await client.call_tool("get_similarity", {
            "text1": "apple",
            "text2": "aple",
            "algorithm": "levenshtein"
        })
        print(f"Similarity result: {similarity_result}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        batch_result = await client.call_tool("batch_process", {
            "texts": ["hello world", "hello universe", "hello galaxy"],
            "operation": "find_replace",
            "parameters": {
                "find": "hello",
                "replace": "hi",
                "case_sensitive": False
            },
            "stream": False
        })
        print(f"Batch result: {batch_result}")
        
        # Test server info
        print("\n📋 Testing server info...")
        info_result = await client.call_tool("get_server_info")
        print(f"Server info: {json.dumps(info_result, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await client.close()


async def test_sse_streaming():
    """Test SSE streaming endpoints"""
    
    print("\n🌊 Testing SSE streaming...")
    
    async with httpx.AsyncClient() as client:
        # Test streaming text splitting
        payload = {
            "text": "This is a long text that will be split into multiple chunks. " * 10,
            "delimiter": ".",
            "stream": True
        }
        
        try:
            async with client.stream("POST", "http://localhost:8000/text/split", json=payload) as response:
                if response.status_code == 200:
                    print("📡 Streaming text split results:")
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            print(f"  📨 {chunk.strip()}")
                else:
                    print(f"❌ HTTP Error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            print("💡 Make sure the FastAPI server is running: python server.py")


async def test_mcp_with_streaming():
    """Test MCP tools with streaming enabled"""
    
    client = FastMCPClient("stdio://python mcp_server.py")
    
    try:
        await client.connect()
        
        print("\n🌊 Testing MCP tools with streaming enabled...")
        
        # Test text splitting with streaming
        split_result = await client.call_tool("split_text", {
            "text": "This is a test. Another sentence! Final question?",
            "stream": True
        })
        print(f"Stream split result: {split_result}")
        
        # The result will contain SSE endpoint information
        if split_result.get("streaming"):
            print(f"📡 SSE Endpoint: {split_result['sse_url']}")
            print(f"📨 Payload: {split_result['payload']}")
            
            # You can use this information to connect to the SSE endpoint
            # with an SSE client library
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await client.close()


async def main():
    """Run all tests"""
    
    print("🚀 FastMCP Text Processing Server - Example Client")
    print("=" * 50)
    
    # Test MCP tools
    await test_mcp_tools()
    
    # Test SSE streaming
    await test_sse_streaming()
    
    # Test MCP with streaming
    await test_mcp_with_streaming()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())