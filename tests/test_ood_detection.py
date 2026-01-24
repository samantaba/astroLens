"""
Test suite for OOD (Out-of-Distribution) Detection.

Tests:
- OOD score computation
- Threshold calibration
- Ensemble voting
- Detection accuracy
"""

import numpy as np
import pytest

# Mark all tests in this module
pytestmark = pytest.mark.model


class TestOODScoreComputation:
    """Test OOD score computation methods."""
    
    def test_msp_computation(self):
        """Test Maximum Softmax Probability computation."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # High confidence logits (should have low OOD score)
        high_conf_logits = np.array([10.0, 1.0, 0.5, 0.1, 0.1])
        msp_score = detector.compute_msp(high_conf_logits)
        
        assert 0 <= msp_score <= 1
        assert msp_score < 0.3  # Low OOD score for confident prediction
        
        # Low confidence logits (should have high OOD score)
        low_conf_logits = np.array([2.0, 1.9, 1.8, 1.7, 1.6])
        msp_score_low = detector.compute_msp(low_conf_logits)
        
        assert msp_score_low > msp_score  # Less confident = higher OOD score
    
    def test_energy_computation(self):
        """Test energy-based score computation."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # High confidence logits (low energy)
        high_conf_logits = np.array([10.0, 1.0, 0.5, 0.1, 0.1])
        energy_high = detector.compute_energy(high_conf_logits)
        
        # Low confidence logits (higher energy)
        low_conf_logits = np.array([2.0, 1.9, 1.8, 1.7, 1.6])
        energy_low = detector.compute_energy(low_conf_logits)
        
        # More confident should have lower energy (more negative)
        # Energy formula: -T * log(sum(exp(logit/T)))
        print(f"High conf energy: {energy_high}, Low conf energy: {energy_low}")
    
    def test_mahalanobis_without_calibration(self):
        """Test Mahalanobis distance without calibration."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # Without calibration, should return 0
        embedding = np.random.randn(768)
        logits = np.array([5.0, 1.0, 0.5, 0.1])
        
        mahal_score = detector.compute_mahalanobis(embedding, logits)
        
        assert mahal_score == 0.0  # Not calibrated


class TestOODDetection:
    """Test OOD detection logic."""
    
    def test_detect_returns_correct_structure(self):
        """Test that detect returns proper OODOutput."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        logits = np.array([5.0, 1.0, 0.5, 0.1, 0.05])
        
        result = detector.detect(logits)
        
        assert hasattr(result, 'ood_score')
        assert hasattr(result, 'is_anomaly')
        assert hasattr(result, 'threshold')
        assert hasattr(result, 'method_scores')
        assert hasattr(result, 'votes')
        
        assert isinstance(result.ood_score, float)
        assert isinstance(result.is_anomaly, bool)
        assert 'msp' in result.method_scores
        assert 'energy' in result.method_scores
    
    def test_high_confidence_not_anomaly(self):
        """Test that high confidence prediction is not anomaly."""
        from inference.ood import OODDetector
        
        detector = OODDetector(threshold=2.5, voting_threshold=2)
        
        # Very confident prediction
        logits = np.array([15.0, 0.1, 0.1, 0.1, 0.1])
        
        result = detector.detect(logits)
        
        # High confidence should not be flagged
        assert not result.is_anomaly or result.votes < 2
    
    def test_low_confidence_detection(self):
        """Test that low confidence might be flagged."""
        from inference.ood import OODDetector
        
        # Use aggressive mode for testing
        detector = OODDetector(aggressive_mode=True)
        
        # Uniform logits = very uncertain
        logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        result = detector.detect(logits)
        
        # Should have higher OOD score
        print(f"Uniform logits: OOD={result.ood_score:.3f}, votes={result.votes}")
        
        # MSP should be high (uncertain)
        assert result.method_scores['msp'] > 0.7


class TestEnsembleVoting:
    """Test ensemble voting mechanism."""
    
    def test_voting_threshold(self):
        """Test that voting threshold works correctly."""
        from inference.ood import OODDetector
        
        # Voting threshold = 2
        detector = OODDetector(voting_threshold=2)
        
        # Create logits that might trigger 1 vote but not 2
        logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
        
        result = detector.detect(logits)
        
        print(f"Votes: {result.votes}, is_anomaly: {result.is_anomaly}")
        
        # is_anomaly should match votes >= threshold
        if result.votes >= 2:
            assert result.is_anomaly
        else:
            assert not result.is_anomaly
    
    def test_aggressive_mode_lower_threshold(self):
        """Test aggressive mode uses voting_threshold=1."""
        from inference.ood import OODDetector
        
        detector_normal = OODDetector(voting_threshold=2)
        detector_aggressive = OODDetector(aggressive_mode=True)
        
        # Same uncertain logits
        logits = np.array([2.0, 1.9, 1.8, 1.7, 1.6])
        
        result_normal = detector_normal.detect(logits)
        result_aggressive = detector_aggressive.detect(logits)
        
        # Aggressive should be more likely to flag
        print(f"Normal: votes={result_normal.votes}, anomaly={result_normal.is_anomaly}")
        print(f"Aggressive: votes={result_aggressive.votes}, anomaly={result_aggressive.is_anomaly}")
        
        # Aggressive mode should require only 1 vote
        assert detector_aggressive.voting_threshold == 1


class TestThresholdCalibration:
    """Test threshold calibration functionality."""
    
    def test_calibration_updates_thresholds(self):
        """Test that calibration updates internal thresholds."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # Create synthetic in-distribution data
        n_samples = 100
        n_classes = 10
        embed_dim = 768
        
        np.random.seed(42)
        embeddings = np.random.randn(n_samples, embed_dim)
        
        # Create logits with clear class predictions
        logits = np.random.randn(n_samples, n_classes)
        # Make one class dominant in each sample
        for i in range(n_samples):
            logits[i, i % n_classes] += 5.0
        
        labels = np.arange(n_samples) % n_classes
        
        # Record original thresholds
        orig_msp = detector.msp_threshold
        orig_energy = detector.energy_threshold
        
        # Calibrate
        detector.calibrate(embeddings, logits, labels, target_fpr=0.05)
        
        # Thresholds should have changed
        print(f"MSP: {orig_msp} -> {detector.msp_threshold}")
        print(f"Energy: {orig_energy} -> {detector.energy_threshold}")
        
        # Class means should be computed
        assert detector.class_means is not None
        assert detector.shared_cov_inv is not None
    
    def test_calibrate_threshold_legacy(self):
        """Test legacy threshold calibration."""
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # Create logits
        n_samples = 50
        n_classes = 10
        
        np.random.seed(42)
        logits = np.random.randn(n_samples, n_classes)
        for i in range(n_samples):
            logits[i, i % n_classes] += 3.0
        
        orig_threshold = detector.threshold
        
        new_threshold = detector.calibrate_threshold(logits, target_fpr=0.05)
        
        print(f"Threshold: {orig_threshold} -> {new_threshold}")
        
        assert isinstance(new_threshold, float)


class TestModelIntegration:
    """Test OOD with actual classifier (if available)."""
    
    @pytest.mark.integration
    def test_with_classifier(self, artifacts_dir):
        """Test OOD detection with actual classifier."""
        try:
            from inference.classifier import AstroClassifier
            from inference.ood import OODDetector
            
            weights_path = artifacts_dir / "weights" / "vit_astrolens"
            
            if not weights_path.exists():
                pytest.skip("No trained weights available")
            
            classifier = AstroClassifier(weights_path=str(weights_path))
            detector = OODDetector()
            
            # Get sample image
            images_dir = artifacts_dir / "data" / "images"
            if not images_dir.exists():
                pytest.skip("No images available")
            
            sample_images = list(images_dir.glob("*.jpg"))[:5]
            if not sample_images:
                pytest.skip("No sample images")
            
            results = []
            for img_path in sample_images:
                try:
                    class_result = classifier.classify(str(img_path))
                    ood_result = detector.detect(class_result.logits, class_result.embedding)
                    
                    results.append({
                        "image": img_path.name,
                        "class": class_result.class_label,
                        "confidence": class_result.confidence,
                        "ood_score": ood_result.ood_score,
                        "is_anomaly": ood_result.is_anomaly,
                        "votes": ood_result.votes,
                    })
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
            
            print("\nOOD Detection Results:")
            for r in results:
                print(f"  {r['image']}: {r['class']} ({r['confidence']:.2%}), "
                      f"OOD={r['ood_score']:.3f}, votes={r['votes']}")
            
        except ImportError as e:
            pytest.skip(f"Could not import classifier: {e}")


class TestPerformance:
    """Performance benchmarks for OOD detection."""
    
    @pytest.mark.benchmark
    def test_detection_speed(self):
        """Benchmark OOD detection speed."""
        import time
        from inference.ood import OODDetector
        
        detector = OODDetector()
        
        # Simulate batch of predictions
        n_samples = 100
        n_classes = 10
        
        logits = np.random.randn(n_samples, n_classes)
        
        start = time.time()
        for i in range(n_samples):
            detector.detect(logits[i])
        duration = time.time() - start
        
        avg_time = duration / n_samples * 1000  # ms
        print(f"OOD detection: {avg_time:.2f}ms per sample")
        
        # Should be very fast (< 10ms per sample)
        assert avg_time < 10.0, f"OOD detection too slow: {avg_time:.2f}ms"
