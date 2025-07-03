"""
Advanced Text Processing Module
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import difflib

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a search operation"""
    text: str
    start: int
    end: int
    score: float = 1.0
    match_type: str = "exact"


@dataclass
class ProcessResult:
    """Result of a text processing operation"""
    original_text: str
    processed_text: str
    operation: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class TextProcessor:
    """Advanced text processing with support for various operations"""
    
    def __init__(self):
        self.semantic_model = None
        self._initialize_semantic_model()
    
    def _initialize_semantic_model(self):
        """Initialize semantic similarity model if available"""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic model loaded successfully")
        except ImportError:
            logger.warning("Sentence transformers not available, semantic matching disabled")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
    
    async def split_text(
        self,
        text: str,
        delimiter: Optional[str] = None,
        regex_pattern: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> List[str]:
        """Split text with customizable delimiters and regex patterns"""
        
        if not text:
            return []
        
        # Default splitting behavior
        if not delimiter and not regex_pattern:
            # Smart splitting on sentences and paragraphs
            chunks = re.split(r'[.!?]+\s+|\n\s*\n', text)
        elif regex_pattern:
            # Use regex pattern for splitting
            chunks = re.split(regex_pattern, text)
        else:
            # Use simple delimiter
            chunks = text.split(delimiter)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Limit chunks if specified
        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        
        return chunks
    
    async def find_replace(
        self,
        text: str,
        find: str,
        replace: str,
        regex: bool = False,
        case_sensitive: bool = True
    ) -> str:
        """Find and replace text with regex support"""
        
        if not text or not find:
            return text
        
        try:
            if regex:
                # Use regex for find and replace
                flags = 0 if case_sensitive else re.IGNORECASE
                return re.sub(find, replace, text, flags=flags)
            else:
                # Simple string replacement
                if case_sensitive:
                    return text.replace(find, replace)
                else:
                    # Case-insensitive replacement
                    pattern = re.escape(find)
                    return re.sub(pattern, replace, text, flags=re.IGNORECASE)
        except re.error as e:
            logger.error(f"Regex error in find_replace: {e}")
            raise ValueError(f"Invalid regex pattern: {e}")
    
    async def fuzzy_delete(
        self,
        text: str,
        target: str,
        similarity_threshold: float = 0.8,
        algorithm: str = "levenshtein"
    ) -> str:
        """Delete text using fuzzy matching algorithms"""
        
        if not text or not target:
            return text
        
        # Split text into words/phrases for fuzzy matching
        words = text.split()
        target_words = target.split()
        
        # Find fuzzy matches
        matches_to_remove = []
        
        for i, word in enumerate(words):
            similarity = self._calculate_similarity(word, target, algorithm)
            
            if similarity >= similarity_threshold:
                matches_to_remove.append(i)
        
        # Also check for phrase matches
        for i in range(len(words) - len(target_words) + 1):
            phrase = " ".join(words[i:i + len(target_words)])
            similarity = self._calculate_similarity(phrase, target, algorithm)
            
            if similarity >= similarity_threshold:
                matches_to_remove.extend(range(i, i + len(target_words)))
        
        # Remove duplicates and sort in reverse order
        matches_to_remove = sorted(set(matches_to_remove), reverse=True)
        
        # Remove matched words
        for idx in matches_to_remove:
            if idx < len(words):
                words.pop(idx)
        
        return " ".join(words)
    
    def _calculate_similarity(self, text1: str, text2: str, algorithm: str) -> float:
        """Calculate similarity between two texts using specified algorithm"""
        
        if algorithm == "levenshtein":
            # Use difflib for edit distance calculation
            return self._levenshtein_similarity(text1.lower(), text2.lower())
        
        elif algorithm == "fuzzy":
            # Use difflib sequence matcher for fuzzy matching
            return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        elif algorithm == "semantic" and self.semantic_model:
            # Use semantic similarity
            try:
                embeddings = self.semantic_model.encode([text1, text2])
                # Calculate cosine similarity without numpy
                dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
                norm_a = sum(a * a for a in embeddings[0]) ** 0.5
                norm_b = sum(b * b for b in embeddings[1]) ** 0.5
                similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
                return float(similarity)
            except Exception as e:
                logger.error(f"Semantic similarity error: {e}")
                return 0.0
        
        else:
            # Fallback to simple string matching
            return 1.0 if text1.lower() == text2.lower() else 0.0
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate Levenshtein similarity without external library"""
        if len(text1) == 0 and len(text2) == 0:
            return 1.0
        
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        distance = self._levenshtein_distance(text1, text2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, text1: str, text2: str) -> int:
        """Calculate Levenshtein distance using dynamic programming"""
        if len(text1) < len(text2):
            return self._levenshtein_distance(text2, text1)
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = list(range(len(text2) + 1))
        for i, char1 in enumerate(text1):
            current_row = [i + 1]
            for j, char2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (char1 != char2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    async def search(
        self,
        text: str,
        query: str,
        search_type: str = "exact",
        case_sensitive: bool = True
    ) -> List[SearchResult]:
        """Search text with various matching algorithms"""
        
        if not text or not query:
            return []
        
        results = []
        search_text = text if case_sensitive else text.lower()
        search_query = query if case_sensitive else query.lower()
        
        if search_type == "exact":
            # Exact string matching
            start = 0
            while True:
                pos = search_text.find(search_query, start)
                if pos == -1:
                    break
                results.append(SearchResult(
                    text=text[pos:pos + len(query)],
                    start=pos,
                    end=pos + len(query),
                    score=1.0,
                    match_type="exact"
                ))
                start = pos + 1
        
        elif search_type == "partial":
            # Partial matching (substring search)
            words = search_query.split()
            for word in words:
                start = 0
                while True:
                    pos = search_text.find(word, start)
                    if pos == -1:
                        break
                    results.append(SearchResult(
                        text=text[pos:pos + len(word)],
                        start=pos,
                        end=pos + len(word),
                        score=0.8,
                        match_type="partial"
                    ))
                    start = pos + 1
        
        elif search_type == "wildcard":
            # Wildcard pattern matching
            pattern = search_query.replace("*", ".*").replace("?", ".")
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                for match in re.finditer(pattern, text, flags):
                    results.append(SearchResult(
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        score=0.9,
                        match_type="wildcard"
                    ))
            except re.error as e:
                logger.error(f"Wildcard pattern error: {e}")
        
        elif search_type == "fuzzy":
            # Fuzzy matching
            words = text.split()
            for i, word in enumerate(words):
                similarity = self._calculate_similarity(word, search_query, "fuzzy")
                if similarity >= 0.6:  # Threshold for fuzzy matching
                    word_start = text.find(word, sum(len(w) + 1 for w in words[:i]))
                    results.append(SearchResult(
                        text=word,
                        start=word_start,
                        end=word_start + len(word),
                        score=similarity,
                        match_type="fuzzy"
                    ))
        
        # Sort results by position
        results.sort(key=lambda x: x.start)
        return results
    
    async def batch_process(
        self,
        texts: List[str],
        operation: str,
        parameters: Dict[str, Any]
    ) -> List[ProcessResult]:
        """Process multiple texts in batch"""
        
        results = []
        
        for text in texts:
            try:
                if operation == "split":
                    processed = await self.split_text(text, **parameters)
                elif operation == "find_replace":
                    processed = await self.find_replace(text, **parameters)
                elif operation == "fuzzy_delete":
                    processed = await self.fuzzy_delete(text, **parameters)
                elif operation == "search":
                    processed = await self.search(text, **parameters)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                results.append(ProcessResult(
                    original_text=text,
                    processed_text=processed if isinstance(processed, str) else str(processed),
                    operation=operation,
                    parameters=parameters,
                    metadata={"length": len(text), "processed_length": len(str(processed))}
                ))
                
                # Yield control to allow streaming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error processing text in batch: {e}")
                results.append(ProcessResult(
                    original_text=text,
                    processed_text=text,  # Return original on error
                    operation=operation,
                    parameters=parameters,
                    metadata={"error": str(e)}
                ))
        
        return results