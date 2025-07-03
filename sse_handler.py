"""
Server-Sent Events (SSE) Handler for Real-time Text Processing
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List
from datetime import datetime

from text_processor import TextProcessor

logger = logging.getLogger(__name__)


class SSEHandler:
    """Handler for Server-Sent Events streaming"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    async def _send_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format and send SSE event"""
        event_data = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        return f"data: {json.dumps(event_data)}\n\n"
    
    async def _send_progress(self, current: int, total: int, message: str = "") -> str:
        """Send progress update event"""
        return await self._send_event("progress", {
            "current": current,
            "total": total,
            "percentage": (current / total) * 100 if total > 0 else 0,
            "message": message
        })
    
    async def _send_error(self, error: str) -> str:
        """Send error event"""
        return await self._send_event("error", {"error": error})
    
    async def _send_result(self, result: Any) -> str:
        """Send result event"""
        return await self._send_event("result", {"result": result})
    
    async def _send_completion(self) -> str:
        """Send completion event"""
        return await self._send_event("complete", {"message": "Processing completed"})
    
    async def stream_split(
        self,
        text: str,
        delimiter: str = None,
        regex_pattern: str = None,
        max_chunks: int = None
    ) -> AsyncGenerator[str, None]:
        """Stream text splitting operation"""
        
        try:
            yield await self._send_progress(0, 1, "Starting text splitting...")
            
            # Process the text
            chunks = await self.text_processor.split_text(
                text, delimiter=delimiter, regex_pattern=regex_pattern, max_chunks=max_chunks
            )
            
            yield await self._send_progress(1, 1, f"Split into {len(chunks)} chunks")
            
            # Stream chunks one by one
            for i, chunk in enumerate(chunks):
                yield await self._send_event("chunk", {
                    "index": i,
                    "chunk": chunk,
                    "length": len(chunk)
                })
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            yield await self._send_result(chunks)
            yield await self._send_completion()
            
        except Exception as e:
            logger.error(f"Error in stream_split: {e}")
            yield await self._send_error(str(e))
    
    async def stream_find_replace(
        self,
        text: str,
        find: str,
        replace: str,
        regex: bool = False,
        case_sensitive: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream find and replace operation"""
        
        try:
            yield await self._send_progress(0, 1, "Starting find and replace...")
            
            # Count matches first
            import re
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                matches = list(re.finditer(find, text, flags=flags))
            else:
                pattern = find if case_sensitive else find.lower()
                search_text = text if case_sensitive else text.lower()
                matches = []
                start = 0
                while True:
                    pos = search_text.find(pattern, start)
                    if pos == -1:
                        break
                    matches.append(type('Match', (), {'start': lambda: pos, 'end': lambda: pos + len(find)})())
                    start = pos + 1
            
            yield await self._send_event("matches_found", {
                "count": len(matches),
                "matches": [{"start": m.start(), "end": m.end()} for m in matches[:10]]  # Limit to first 10
            })
            
            # Perform replacement
            result = await self.text_processor.find_replace(
                text, find, replace, regex=regex, case_sensitive=case_sensitive
            )
            
            yield await self._send_progress(1, 1, f"Replaced {len(matches)} matches")
            yield await self._send_result(result)
            yield await self._send_completion()
            
        except Exception as e:
            logger.error(f"Error in stream_find_replace: {e}")
            yield await self._send_error(str(e))
    
    async def stream_fuzzy_delete(
        self,
        text: str,
        target: str,
        similarity_threshold: float = 0.8,
        algorithm: str = "levenshtein"
    ) -> AsyncGenerator[str, None]:
        """Stream fuzzy deletion operation"""
        
        try:
            yield await self._send_progress(0, 1, f"Starting fuzzy deletion with {algorithm} algorithm...")
            
            # Split text for analysis
            words = text.split()
            yield await self._send_event("analysis", {
                "word_count": len(words),
                "target": target,
                "threshold": similarity_threshold
            })
            
            # Find matches
            matches = []
            for i, word in enumerate(words):
                similarity = self.text_processor._calculate_similarity(word, target, algorithm)
                if similarity >= similarity_threshold:
                    matches.append({"index": i, "word": word, "similarity": similarity})
                
                # Send progress for long texts
                if i % 100 == 0:
                    yield await self._send_progress(i, len(words), f"Analyzing word {i+1}/{len(words)}")
                    await asyncio.sleep(0.01)
            
            yield await self._send_event("matches_found", {
                "count": len(matches),
                "matches": matches[:10]  # Limit to first 10
            })
            
            # Perform deletion
            result = await self.text_processor.fuzzy_delete(
                text, target, similarity_threshold=similarity_threshold, algorithm=algorithm
            )
            
            yield await self._send_progress(1, 1, f"Deleted {len(matches)} fuzzy matches")
            yield await self._send_result(result)
            yield await self._send_completion()
            
        except Exception as e:
            logger.error(f"Error in stream_fuzzy_delete: {e}")
            yield await self._send_error(str(e))
    
    async def stream_search(
        self,
        text: str,
        query: str,
        search_type: str = "exact",
        case_sensitive: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream search operation"""
        
        try:
            yield await self._send_progress(0, 1, f"Starting {search_type} search...")
            
            # Perform search
            results = await self.text_processor.search(
                text, query, search_type=search_type, case_sensitive=case_sensitive
            )
            
            yield await self._send_event("search_info", {
                "query": query,
                "search_type": search_type,
                "case_sensitive": case_sensitive,
                "results_count": len(results)
            })
            
            # Stream results
            for i, result in enumerate(results):
                yield await self._send_event("search_result", {
                    "index": i,
                    "text": result.text,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "match_type": result.match_type
                })
                await asyncio.sleep(0.01)
            
            yield await self._send_progress(1, 1, f"Found {len(results)} matches")
            yield await self._send_result([{
                "text": r.text,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "match_type": r.match_type
            } for r in results])
            yield await self._send_completion()
            
        except Exception as e:
            logger.error(f"Error in stream_search: {e}")
            yield await self._send_error(str(e))
    
    async def stream_batch_process(
        self,
        texts: List[str],
        operation: str,
        parameters: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream batch processing operation"""
        
        try:
            total_texts = len(texts)
            yield await self._send_progress(0, total_texts, f"Starting batch {operation} on {total_texts} texts...")
            
            yield await self._send_event("batch_info", {
                "operation": operation,
                "parameters": parameters,
                "total_texts": total_texts
            })
            
            # Process texts one by one
            results = []
            for i, text in enumerate(texts):
                try:
                    yield await self._send_progress(i, total_texts, f"Processing text {i+1}/{total_texts}")
                    
                    # Process individual text
                    if operation == "split":
                        result = await self.text_processor.split_text(text, **parameters)
                    elif operation == "find_replace":
                        result = await self.text_processor.find_replace(text, **parameters)
                    elif operation == "fuzzy_delete":
                        result = await self.text_processor.fuzzy_delete(text, **parameters)
                    elif operation == "search":
                        result = await self.text_processor.search(text, **parameters)
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                    
                    # Send individual result
                    yield await self._send_event("batch_result", {
                        "index": i,
                        "original_length": len(text),
                        "result": result if isinstance(result, (str, list)) else str(result),
                        "processed_length": len(str(result))
                    })
                    
                    results.append(result)
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error processing text {i}: {e}")
                    yield await self._send_event("batch_error", {
                        "index": i,
                        "error": str(e)
                    })
                    results.append(None)
            
            yield await self._send_progress(total_texts, total_texts, "Batch processing completed")
            yield await self._send_result(results)
            yield await self._send_completion()
            
        except Exception as e:
            logger.error(f"Error in stream_batch_process: {e}")
            yield await self._send_error(str(e))