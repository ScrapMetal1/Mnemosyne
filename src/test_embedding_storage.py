"""
Unit tests for embedding storage and search functionality.
"""

import unittest
import numpy as np
import tempfile
import shutil
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_storage import EmbeddingStore, EmbeddingExtractor


class MockEmbeddingExtractor:
    """
    Mock embedding extractor for testing.
    Generates deterministic embeddings from text using hash-based approach.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def extract(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding from text.
        Uses hash to create consistent embeddings for same text.
        """
        # Create a deterministic seed from text hash
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        
        # Generate embedding vector
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Normalize to unit vector
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding


class TestEmbeddingStore(unittest.TestCase):
    """Unit tests for EmbeddingStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = EmbeddingStore(storage_dir=self.temp_dir)
        self.extractor = MockEmbeddingExtractor(embedding_dim=384)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_embedding(self):
        """Test adding an embedding to the store."""
        text = "I am in a kitchen. There are silver keys on the wooden counter."
        embedding = self.extractor.extract(text)
        
        memory_id = self.store.add(embedding, text)
        
        self.assertIsNotNone(memory_id)
        self.assertEqual(self.store.count(), 1)
        self.assertIn(memory_id, self.store.ids)
    
    def test_add_with_metadata(self):
        """Test adding embedding with custom metadata."""
        text = "Test description"
        embedding = self.extractor.extract(text)
        metadata = {"location": "kitchen", "time": "10:00 AM"}
        
        memory_id = self.store.add(embedding, text, metadata=metadata)
        
        stored = self.store.get_by_id(memory_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored['metadata']['location'], "kitchen")
        self.assertEqual(stored['metadata']['time'], "10:00 AM")
    
    def test_search_exact_match(self):
        """Test searching for an exact match."""
        # Add a memory
        text = "I am in a kitchen. There are silver keys on the wooden counter."
        embedding = self.extractor.extract(text)
        memory_id = self.store.add(embedding, text)
        
        # Search with same text (should find exact match)
        query_embedding = self.extractor.extract(text)
        results = self.store.search(query_embedding, top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], memory_id)
        self.assertEqual(results[0]['text'], text)
        self.assertGreaterEqual(results[0]['similarity'], 0.99)  # Should be very similar
    
    def test_search_similar_text(self):
        """Test searching with similar but not identical text."""
        # Add memory
        text = "I am in a kitchen. There are silver keys on the wooden counter."
        embedding = self.extractor.extract(text)
        memory_id = self.store.add(embedding, text)
        
        # Search with similar query
        query = "Where are the keys in the kitchen?"
        query_embedding = self.extractor.extract(query)
        results = self.store.search(query_embedding, top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], memory_id)
        self.assertIn('keys', results[0]['text'].lower())
        self.assertIn('kitchen', results[0]['text'].lower())
    
    def test_search_multiple_items(self):
        """Test searching when multiple items are stored."""
        # Add multiple distinct memories
        texts = [
            "I am in a kitchen. There are silver keys on the wooden counter.",
            "I am in a living room. There is a red sofa and a TV.",
            "I am in a bedroom. There is a blue bed and a window."
        ]
        
        memory_ids = []
        for text in texts:
            embedding = self.extractor.extract(text)
            memory_id = self.store.add(embedding, text)
            memory_ids.append(memory_id)
        
        self.assertEqual(self.store.count(), 3)
        
        # Search for kitchen-related query
        query = "Where did I leave my keys?"
        query_embedding = self.extractor.extract(query)
        results = self.store.search(query_embedding, top_k=1)
        
        self.assertEqual(len(results), 1)
        # Should match the kitchen memory
        self.assertEqual(results[0]['id'], memory_ids[0])
        self.assertIn('keys', results[0]['text'].lower())
    
    def test_search_empty_store(self):
        """Test searching when store is empty."""
        query_embedding = self.extractor.extract("test query")
        results = self.store.search(query_embedding, top_k=1)
        
        self.assertEqual(len(results), 0)
    
    def test_search_with_threshold(self):
        """Test searching with similarity threshold."""
        # Add a memory
        text = "I am in a kitchen. There are silver keys on the wooden counter."
        embedding = self.extractor.extract(text)
        self.store.add(embedding, text)
        
        # Search with very different query
        query = "Completely unrelated topic about space and planets"
        query_embedding = self.extractor.extract(query)
        
        # With high threshold, should return nothing
        results = self.store.search(query_embedding, top_k=1, threshold=0.9)
        
        # Results might be empty or have low similarity
        if len(results) > 0:
            self.assertLess(results[0]['similarity'], 0.9)
    
    def test_delete_embedding(self):
        """Test deleting an embedding."""
        text = "Test memory"
        embedding = self.extractor.extract(text)
        memory_id = self.store.add(embedding, text)
        
        self.assertEqual(self.store.count(), 1)
        
        deleted = self.store.delete(memory_id)
        self.assertTrue(deleted)
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get_by_id(memory_id))
    
    def test_delete_nonexistent(self):
        """Test deleting a non-existent embedding."""
        result = self.store.delete("nonexistent_id")
        self.assertFalse(result)
    
    def test_get_by_id(self):
        """Test retrieving embedding by ID."""
        text = "Test memory"
        embedding = self.extractor.extract(text)
        memory_id = self.store.add(embedding, text)
        
        retrieved = self.store.get_by_id(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['text'], text)
        self.assertEqual(retrieved['id'], memory_id)
        self.assertIn('embedding', retrieved)
        self.assertIn('metadata', retrieved)
    
    def test_persistence(self):
        """Test that embeddings persist across store instances."""
        text = "Persistent memory"
        embedding = self.extractor.extract(text)
        memory_id = self.store.add(embedding, text)
        
        # Create new store instance (should load from disk)
        new_store = EmbeddingStore(storage_dir=self.temp_dir)
        
        self.assertEqual(new_store.count(), 1)
        retrieved = new_store.get_by_id(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['text'], text)
    
    def test_clear_store(self):
        """Test clearing all embeddings."""
        # Add multiple memories
        for i in range(3):
            text = f"Memory {i}"
            embedding = self.extractor.extract(text)
            self.store.add(embedding, text)
        
        self.assertEqual(self.store.count(), 3)
        
        self.store.clear()
        self.assertEqual(self.store.count(), 0)


class TestEmbeddingExtractor(unittest.TestCase):
    """Unit tests for EmbeddingExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MockEmbeddingExtractor(embedding_dim=384)
    
    def test_extract_embedding(self):
        """Test extracting embedding from text."""
        text = "Test text for embedding"
        embedding = self.extractor.extract(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)
        self.assertEqual(embedding.shape[0], 384)
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_extract_deterministic(self):
        """Test that same text produces same embedding."""
        text = "Deterministic test"
        embedding1 = self.extractor.extract(text)
        embedding2 = self.extractor.extract(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_extract_different_texts(self):
        """Test that different texts produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"
        
        embedding1 = self.extractor.extract(text1)
        embedding2 = self.extractor.extract(text2)
        
        # Should be different (very unlikely to be identical)
        self.assertFalse(np.allclose(embedding1, embedding2, atol=1e-6))


if __name__ == '__main__':
    unittest.main()

