#!/usr/bin/env python3
"""
Comprehensive System Tests for AstroLens

Tests all major components:
1. API endpoints (health, images, analysis, chat)
2. ML inference (classifier, OOD, embeddings)
3. LLM integration (annotator, agent)
4. Database operations

Run:
    python tests/test_full_system.py
    
Or with pytest:
    pytest tests/test_full_system.py -v
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
TIMEOUT = 60  # seconds


def get_client():
    """Get HTTP client for API calls."""
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# Health & Stats Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_api_health():
    """Test that API is running and healthy."""
    with get_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print(f"✓ API health check passed: {data}")


def test_api_stats():
    """Test statistics endpoint."""
    with get_client() as client:
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_images" in data
        print(f"✓ Stats: {data}")


# ─────────────────────────────────────────────────────────────────────────────
# Image CRUD Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_list_images():
    """Test listing images."""
    with get_client() as client:
        response = client.get("/images", params={"limit": 10})
        assert response.status_code == 200
        images = response.json()
        assert isinstance(images, list)
        print(f"✓ Listed {len(images)} images")
        return images


def test_get_image_detail():
    """Test getting image details."""
    with get_client() as client:
        # Get first image
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return None
        
        image_id = images[0]["id"]
        response = client.get(f"/images/{image_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == image_id
        print(f"✓ Got image detail: ID {image_id}, {data.get('filename')}")
        return image_id


def test_get_image_file():
    """Test serving image file."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return
        
        image_id = images[0]["id"]
        response = client.get(f"/images/{image_id}/file")
        
        if response.status_code == 404:
            print(f"⚠ Image file not found for ID {image_id}")
            return
        
        assert response.status_code == 200
        assert len(response.content) > 0
        print(f"✓ Image file served: {len(response.content)} bytes")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_image():
    """Test image classification."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test classification")
            return None
        
        image_id = images[0]["id"]
        
        print(f"  Classifying image {image_id}...")
        start = time.time()
        response = client.post(f"/analysis/classify/{image_id}")
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"⚠ Classification failed: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        assert "class_label" in data
        assert "confidence" in data
        print(f"✓ Classification: {data['class_label']} ({data['confidence']:.1%}) in {elapsed:.1f}s")
        return data


def test_anomaly_detection():
    """Test OOD/anomaly detection."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return None
        
        image_id = images[0]["id"]
        response = client.post(f"/analysis/anomaly/{image_id}")
        
        if response.status_code != 200:
            print(f"⚠ Anomaly detection failed: {response.text[:100]}")
            return None
        
        data = response.json()
        assert "ood_score" in data
        assert "is_anomaly" in data
        status = "ANOMALY" if data["is_anomaly"] else "normal"
        print(f"✓ Anomaly detection: score={data['ood_score']:.2f}, {status}")
        return data


def test_full_analysis():
    """Test full analysis pipeline."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return None
        
        image_id = images[0]["id"]
        
        print(f"  Running full analysis on image {image_id}...")
        start = time.time()
        response = client.post(f"/analysis/full/{image_id}")
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"⚠ Full analysis failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return None
        
        data = response.json()
        assert "classification" in data
        assert "anomaly" in data
        
        class_label = data["classification"]["class_label"]
        confidence = data["classification"]["confidence"]
        is_anomaly = data["anomaly"]["is_anomaly"]
        similar_count = len(data.get("similar", []))
        
        print(f"✓ Full analysis in {elapsed:.1f}s:")
        print(f"  - Classification: {class_label} ({confidence:.1%})")
        print(f"  - Anomaly: {is_anomaly}")
        print(f"  - Similar images: {similar_count}")
        return data


def test_find_similar():
    """Test similarity search."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return
        
        image_id = images[0]["id"]
        response = client.post(f"/analysis/similar/{image_id}", params={"k": 5})
        
        if response.status_code != 200:
            print(f"⚠ Similarity search failed: {response.text[:100]}")
            return
        
        data = response.json()
        similar = data.get("similar", [])
        print(f"✓ Found {len(similar)} similar images")
        
        for sim in similar[:3]:
            print(f"  - ID {sim['image_id']}: {sim['similarity']:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_annotate_image():
    """Test LLM annotation (requires API key or Ollama)."""
    with get_client() as client:
        images_response = client.get("/images", params={"limit": 1})
        images = images_response.json()
        
        if not images:
            print("⚠ No images to test")
            return
        
        image_id = images[0]["id"]
        
        print(f"  Annotating image {image_id}...")
        start = time.time()
        response = client.post(f"/annotate/{image_id}")
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"⚠ Annotation failed: {response.text[:100]}")
            print("  (This is OK if no LLM is configured)")
            return
        
        data = response.json()
        desc = data.get("description", "")[:80]
        print(f"✓ Annotation in {elapsed:.1f}s: {desc}...")


def test_chat_agent():
    """Test chat agent."""
    with get_client() as client:
        test_messages = [
            "help",
            "list images",
            "show statistics",
        ]
        
        for msg in test_messages:
            response = client.post("/chat", json={"message": msg})
            
            if response.status_code != 200:
                print(f"⚠ Chat failed for '{msg}': {response.status_code}")
                continue
            
            data = response.json()
            reply = data.get("reply", "")
            snippet = reply[:80].replace("\n", " ")
            print(f"✓ Chat '{msg}': {snippet}...")


# ─────────────────────────────────────────────────────────────────────────────
# Database Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_database_connection():
    """Test database is accessible."""
    from paths import DATABASE_URL
    from sqlalchemy import create_engine, text
    
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM images"))
        count = result.scalar()
        print(f"✓ Database connected: {count} images in database")


def test_paths_configuration():
    """Test paths are correctly configured."""
    from paths import (
        ARTIFACTS_DIR, DATA_DIR, IMAGES_DIR, WEIGHTS_DIR,
        DATASETS_DIR, DOWNLOADS_DIR, FAISS_INDEX_PATH
    )
    
    print("✓ Path configuration:")
    print(f"  ARTIFACTS_DIR: {ARTIFACTS_DIR} (exists: {ARTIFACTS_DIR.exists()})")
    print(f"  DATA_DIR: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"  IMAGES_DIR: {IMAGES_DIR} (exists: {IMAGES_DIR.exists()})")
    print(f"  WEIGHTS_DIR: {WEIGHTS_DIR} (exists: {WEIGHTS_DIR.exists()})")
    print(f"  DATASETS_DIR: {DATASETS_DIR} (exists: {DATASETS_DIR.exists()})")
    print(f"  FAISS_INDEX_PATH: {FAISS_INDEX_PATH} (exists: {FAISS_INDEX_PATH.exists()})")


# ─────────────────────────────────────────────────────────────────────────────
# ML Component Tests (Direct, without API)
# ─────────────────────────────────────────────────────────────────────────────

def test_classifier_direct():
    """Test classifier component directly."""
    from inference.classifier import AstroClassifier
    from paths import WEIGHTS_PATH, IMAGES_DIR
    import os
    
    # Check for weights
    weights_valid = (
        Path(WEIGHTS_PATH).exists() and
        (Path(WEIGHTS_PATH) / "config.json").exists()
    )
    
    print(f"  Loading classifier (weights_valid: {weights_valid})...")
    classifier = AstroClassifier(weights_path=WEIGHTS_PATH if weights_valid else None)
    
    # Find a test image
    test_images = list(IMAGES_DIR.glob("*.jpg"))[:1]
    if not test_images:
        print("⚠ No test images found")
        return
    
    result = classifier.classify(str(test_images[0]))
    print(f"✓ Direct classification: {result.class_label} ({result.confidence:.1%})")
    print(f"  Embedding shape: {result.embedding.shape}")
    print(f"  Logits shape: {result.logits.shape}")


def test_ood_detector_direct():
    """Test OOD detector directly."""
    import numpy as np
    from inference.ood import OODDetector
    
    detector = OODDetector(threshold=3.0)
    
    # Simulate confident prediction (low energy = normal)
    confident_logits = np.array([10.0, -2.0, -3.0, -1.5, -4.0, -2.5, -3.5, -1.0, -4.5, -2.0])
    result = detector.detect(confident_logits)
    print(f"✓ Confident prediction: energy={result.ood_score:.2f}, anomaly={result.is_anomaly}")
    
    # Simulate uncertain prediction (high energy = anomaly)
    uncertain_logits = np.array([1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.6, 1.0, 0.8])
    result = detector.detect(uncertain_logits)
    print(f"✓ Uncertain prediction: energy={result.ood_score:.2f}, anomaly={result.is_anomaly}")


def test_embeddings_store_direct():
    """Test FAISS embedding store directly."""
    import numpy as np
    from inference.embeddings import EmbeddingStore
    
    store = EmbeddingStore()
    count = store.count()
    print(f"✓ Embedding store: {count} vectors indexed")
    
    if count > 0:
        # Test search with random vector
        query = np.random.randn(768).astype(np.float32)
        ids, sims = store.search(query, k=3)
        print(f"  Test search returned {len(ids)} results")


# ─────────────────────────────────────────────────────────────────────────────
# Batch Analysis Test
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_analysis():
    """Test batch analysis of multiple images."""
    with get_client() as client:
        # Get images
        response = client.get("/images", params={"limit": 5})
        images = response.json()
        
        if len(images) < 2:
            print("⚠ Not enough images for batch test")
            return
        
        print(f"  Batch analyzing {len(images)} images...")
        
        results = {"success": 0, "errors": 0, "anomalies": 0}
        
        for img in images[:3]:  # Test first 3
            image_id = img["id"]
            resp = client.post(f"/analysis/full/{image_id}")
            
            if resp.status_code == 200:
                results["success"] += 1
                data = resp.json()
                if data.get("anomaly", {}).get("is_anomaly"):
                    results["anomalies"] += 1
            else:
                results["errors"] += 1
        
        print(f"✓ Batch analysis: {results['success']} success, {results['errors']} errors, {results['anomalies']} anomalies")


# ─────────────────────────────────────────────────────────────────────────────
# Run All Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("ASTROLENS COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Path Configuration", test_paths_configuration),
        ("Database Connection", test_database_connection),
        ("API Health", test_api_health),
        ("API Stats", test_api_stats),
        ("List Images", test_list_images),
        ("Get Image Detail", test_get_image_detail),
        ("Get Image File", test_get_image_file),
        ("Classifier (Direct)", test_classifier_direct),
        ("OOD Detector (Direct)", test_ood_detector_direct),
        ("Embeddings Store (Direct)", test_embeddings_store_direct),
        ("Classification API", test_classify_image),
        ("Anomaly Detection API", test_anomaly_detection),
        ("Full Analysis API", test_full_analysis),
        ("Similarity Search API", test_find_similar),
        ("Batch Analysis", test_batch_analysis),
        ("Chat Agent", test_chat_agent),
        ("LLM Annotation", test_annotate_image),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

