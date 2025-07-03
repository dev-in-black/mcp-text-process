"""
Tests for TextProcessor module
"""

import pytest
import asyncio
from text_processor import TextProcessor, SearchResult


class TestTextProcessor:

    def setup_method(self):
        """Setup test fixtures"""
        self.processor = TextProcessor()

    @pytest.mark.asyncio
    async def test_split_text_default(self):
        """Test default text splitting"""
        text = "This is a test. Another sentence! Final question?"
        result = await self.processor.split_text(text)
        assert len(result) == 3
        assert "This is a test" in result[0]
        assert "Another sentence" in result[1]
        assert "Final question" in result[2]

    @pytest.mark.asyncio
    async def test_split_text_custom_delimiter(self):
        """Test text splitting with custom delimiter"""
        text = "apple,banana,orange,grape"
        result = await self.processor.split_text(text, delimiter=",")
        assert len(result) == 4
        assert result == ["apple", "banana", "orange", "grape"]

    @pytest.mark.asyncio
    async def test_split_text_regex(self):
        """Test text splitting with regex pattern"""
        text = "word1 word2\tword3\nword4"
        result = await self.processor.split_text(text, regex_pattern=r"\s+")
        assert len(result) == 4
        assert result == ["word1", "word2", "word3", "word4"]

    @pytest.mark.asyncio
    async def test_split_text_max_chunks(self):
        """Test text splitting with max chunks limit"""
        text = "one two three four five"
        result = await self.processor.split_text(text, delimiter=" ", max_chunks=3)
        assert len(result) == 3
        assert result == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_find_replace_simple(self):
        """Test simple find and replace"""
        text = "Hello world, hello universe"
        result = await self.processor.find_replace(text, "hello", "hi")
        assert result == "Hello world, hi universe"

    @pytest.mark.asyncio
    async def test_find_replace_case_insensitive(self):
        """Test case-insensitive find and replace"""
        text = "Hello world, hello universe"
        result = await self.processor.find_replace(
            text, "hello", "hi", case_sensitive=False
        )
        assert result == "hi world, hi universe"

    @pytest.mark.asyncio
    async def test_find_replace_regex(self):
        """Test regex find and replace"""
        text = "The date is 2023-12-25"
        result = await self.processor.find_replace(
            text, r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD", regex=True
        )
        assert result == "The date is YYYY-MM-DD"

    @pytest.mark.asyncio
    async def test_fuzzy_delete_exact(self):
        """Test fuzzy deletion with exact match"""
        text = "apple banana orange apple grape"
        result = await self.processor.fuzzy_delete(
            text, "apple", similarity_threshold=1.0
        )
        assert "apple" not in result
        assert "banana" in result
        assert "orange" in result

    @pytest.mark.asyncio
    async def test_fuzzy_delete_similar(self):
        """Test fuzzy deletion with similar words"""
        text = "apple aple banana ornge grape"
        result = await self.processor.fuzzy_delete(
            text, "apple", similarity_threshold=0.7
        )
        # Should remove both "apple" and "aple" (similar to apple)
        assert "apple" not in result
        assert "aple" not in result
        assert "banana" in result

    @pytest.mark.asyncio
    async def test_search_exact(self):
        """Test exact search"""
        text = "The quick brown fox jumps over the lazy dog"
        results = await self.processor.search(text, "fox")
        assert len(results) == 1
        assert results[0].text == "fox"
        assert results[0].start == 16
        assert results[0].match_type == "exact"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self):
        """Test case-insensitive search"""
        text = "The Quick Brown Fox"
        results = await self.processor.search(text, "quick", case_sensitive=False)
        assert len(results) == 1
        assert results[0].text == "Quick"

    @pytest.mark.asyncio
    async def test_search_partial(self):
        """Test partial search"""
        text = "The quick brown fox jumps"
        results = await self.processor.search(
            text, "quick brown", search_type="partial"
        )
        assert len(results) >= 2  # Should find both "quick" and "brown"

    @pytest.mark.asyncio
    async def test_search_wildcard(self):
        """Test wildcard search"""
        text = "cat bat rat mat"
        results = await self.processor.search(text, "*at", search_type="wildcard")
        # The wildcard pattern "*at" matches the entire string since * matches everything
        assert len(results) >= 1
        # Check that we get some matches with "at" pattern
        at_matches = [r for r in results if "at" in r.text]
        assert len(at_matches) >= 1

    @pytest.mark.asyncio
    async def test_search_fuzzy(self):
        """Test fuzzy search"""
        text = "apple aple banana"
        results = await self.processor.search(text, "apple", search_type="fuzzy")
        assert len(results) >= 1
        # Should find at least the exact match
        exact_matches = [r for r in results if r.text == "apple"]
        assert len(exact_matches) == 1

    @pytest.mark.asyncio
    async def test_batch_process_split(self):
        """Test batch processing with split operation"""
        texts = ["one two three", "four five six"]
        results = await self.processor.batch_process(texts, "split", {"delimiter": " "})

        assert len(results) == 2
        assert len(results[0].processed_text.split()) == 3
        assert len(results[1].processed_text.split()) == 3

    @pytest.mark.asyncio
    async def test_batch_process_find_replace(self):
        """Test batch processing with find_replace operation"""
        texts = ["hello world", "hello universe"]
        results = await self.processor.batch_process(
            texts, "find_replace", {"find": "hello", "replace": "hi"}
        )

        assert len(results) == 2
        assert "hi world" in results[0].processed_text
        assert "hi universe" in results[1].processed_text

    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty text"""
        result = await self.processor.split_text("")
        assert result == []

        result = await self.processor.find_replace("", "find", "replace")
        assert result == ""

        result = await self.processor.fuzzy_delete("", "target")
        assert result == ""

        results = await self.processor.search("", "query")
        assert results == []

    def test_similarity_calculation(self):
        """Test similarity calculation algorithms"""
        # Test exact match
        similarity = self.processor._calculate_similarity(
            "apple", "apple", "levenshtein"
        )
        assert similarity == 1.0

        # Test different strings
        similarity = self.processor._calculate_similarity(
            "apple", "banana", "levenshtein"
        )
        assert similarity < 1.0

        # Test fuzzy algorithm
        similarity = self.processor._calculate_similarity("apple", "aple", "fuzzy")
        assert similarity > 0.7  # Should be quite similar

    @pytest.mark.asyncio
    async def test_invalid_regex_handling(self):
        """Test handling of invalid regex patterns"""
        text = "test text"
        with pytest.raises(ValueError):
            await self.processor.find_replace(text, "[invalid", "replace", regex=True)


if __name__ == "__main__":
    pytest.main([__file__])
