"""
Tests for inference module.
"""

import numpy as np
import pytest


class TestOODDetector:
    """Tests for OOD detection."""

    def test_energy_score_normal(self):
        """Test energy score for confident prediction."""
        from inference.ood import OODDetector
        
        ood = OODDetector(threshold=10.0)
        
        # High confidence prediction (one hot-ish)
        logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        result = ood.detect(logits)
        
        # Should be low energy (confident)
        assert result.ood_score < 0
        assert result.is_anomaly == False

    def test_energy_score_uncertain(self):
        """Test energy score for uncertain prediction."""
        from inference.ood import OODDetector
        
        ood = OODDetector(threshold=-5.0)  # Lower threshold
        
        # Uniform distribution (uncertain)
        logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = ood.detect(logits)
        
        # Should be higher energy (less confident)
        assert result.ood_score > -10


class TestEmbeddingStore:
    """Tests for embedding store."""

    def test_add_and_search(self):
        """Test adding and searching embeddings."""
        from inference.embeddings import EmbeddingStore
        import tempfile
        import os
        
        # Use temp file for test
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["FAISS_INDEX_PATH"] = os.path.join(tmpdir, "test.index")
            
            store = EmbeddingStore(embedding_dim=768)
            
            # Add some embeddings
            emb1 = np.random.randn(768).astype(np.float32)
            emb2 = np.random.randn(768).astype(np.float32)
            
            store.add(1, emb1)
            store.add(2, emb2)
            
            # Search for similar to emb1
            ids, sims = store.search(emb1, k=2)
            
            assert len(ids) == 2
            assert 1 in ids  # Should find itself


class TestClassifier:
    """Tests for classifier (requires torch/timm)."""

    @pytest.mark.skipif(True, reason="Requires torch and weights")
    def test_classify_image(self):
        """Test image classification."""
        from inference.classifier import AstroClassifier
        
        classifier = AstroClassifier()
        # Would need a test image here
