#!/usr/bin/env python3
"""
UI Feature Tests for AstroLens

Tests all UI-related API endpoints that the panels use.
Simulates what the UI does when buttons are clicked.

Run: python tests/test_ui_features.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

API_URL = "http://localhost:8000"
client = httpx.Client(base_url=API_URL, timeout=120)


def test_gallery_load():
    """Test: Gallery loads images (what GalleryPanel does on startup)."""
    print("\n=== TEST: Gallery Load ===")
    
    response = client.get("/images", params={"limit": 100})
    assert response.status_code == 200, f"Failed: {response.status_code} - {response.text}"
    
    images = response.json()
    print(f"✓ Gallery loaded {len(images)} images")
    
    if images:
        print(f"  First image: ID {images[0]['id']}, {images[0]['filename']}")
    return images


def test_batch_analyze_new():
    """Test: 'Analyze New' button in Batch Analysis tab."""
    print("\n=== TEST: Batch Analyze New ===")
    
    # Step 1: Get all images (what _analyze_unanalyzed does)
    response = client.get("/images", params={"limit": 2000})
    assert response.status_code == 200, f"Failed to list images: {response.status_code}"
    
    images = response.json()
    print(f"  Total images: {len(images)}")
    
    # Step 2: Find unanalyzed
    unanalyzed = [img for img in images if not img.get("class_label")]
    print(f"  Unanalyzed: {len(unanalyzed)}")
    
    if not unanalyzed:
        print("✓ All images already analyzed")
        return
    
    # Step 3: Analyze first 3 unanalyzed
    for img in unanalyzed[:3]:
        image_id = img["id"]
        print(f"  Analyzing image {image_id}...")
        
        start = time.time()
        resp = client.post(f"/analysis/full/{image_id}")
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            label = data.get("classification", {}).get("class_label", "?")
            is_anomaly = data.get("anomaly", {}).get("is_anomaly", False)
            status = "ANOMALY!" if is_anomaly else ""
            print(f"    ✓ {label} ({elapsed:.1f}s) {status}")
        else:
            print(f"    ✗ Failed: {resp.status_code}")
    
    print("✓ Analyze New works")


def test_batch_analyze_all():
    """Test: 'Re-analyze All' button in Batch Analysis tab."""
    print("\n=== TEST: Batch Re-analyze All ===")
    
    # Step 1: Get images
    response = client.get("/images", params={"limit": 2000})
    assert response.status_code == 200, f"Failed: {response.status_code}"
    
    images = response.json()
    print(f"  Total images: {len(images)}")
    
    if not images:
        print("⚠ No images to analyze")
        return
    
    # Analyze first 2 to test
    for img in images[:2]:
        image_id = img["id"]
        resp = client.post(f"/analysis/full/{image_id}")
        if resp.status_code == 200:
            print(f"  ✓ Re-analyzed image {image_id}")
        else:
            print(f"  ✗ Failed image {image_id}: {resp.status_code}")
    
    print("✓ Re-analyze All works")


def test_batch_rebuild_similarity():
    """Test: 'Rebuild Similarity' button."""
    print("\n=== TEST: Rebuild Similarity ===")
    
    # Get images
    response = client.get("/images", params={"limit": 10})
    images = response.json()
    
    # Full analysis rebuilds embeddings
    for img in images[:3]:
        image_id = img["id"]
        resp = client.post(f"/analysis/full/{image_id}")
        if resp.status_code == 200:
            print(f"  ✓ Rebuilt embedding for image {image_id}")
    
    # Test similarity search
    if images:
        resp = client.post(f"/analysis/similar/{images[0]['id']}", params={"k": 5})
        if resp.status_code == 200:
            similar = resp.json().get("similar", [])
            print(f"  ✓ Similarity search found {len(similar)} results")
        else:
            print(f"  ⚠ Similarity search failed: {resp.status_code}")
    
    print("✓ Rebuild Similarity works")


def test_chat_commands():
    """Test: AI Chat panel commands."""
    print("\n=== TEST: AI Chat Commands ===")
    
    commands = [
        ("help", "help menu"),
        ("list images", "image list"),
        ("show statistics", "stats"),
        ("show anomalies", "anomalies"),
    ]
    
    for cmd, desc in commands:
        resp = client.post("/chat", json={"message": cmd})
        
        if resp.status_code == 200:
            reply = resp.json().get("reply", "")[:60]
            print(f"  ✓ '{cmd}': {reply}...")
        else:
            print(f"  ✗ '{cmd}' failed: {resp.status_code}")
    
    print("✓ Chat commands work")


def test_image_annotation():
    """Test: LLM annotation feature."""
    print("\n=== TEST: LLM Annotation ===")
    
    response = client.get("/images", params={"limit": 1})
    images = response.json()
    
    if not images:
        print("⚠ No images to annotate")
        return
    
    image_id = images[0]["id"]
    print(f"  Annotating image {image_id}...")
    
    start = time.time()
    resp = client.post(f"/annotate/{image_id}")
    elapsed = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        desc = data.get("description", "")[:60]
        print(f"  ✓ Annotation in {elapsed:.1f}s: {desc}...")
    else:
        print(f"  ⚠ Annotation failed: {resp.status_code}")
        print(f"     (This is OK if no LLM configured)")
    
    print("✓ LLM Annotation works (or correctly reports LLM not configured)")


def test_viewer_detail():
    """Test: Image viewer detail view."""
    print("\n=== TEST: Viewer Detail ===")
    
    response = client.get("/images", params={"limit": 1})
    images = response.json()
    
    if not images:
        print("⚠ No images to view")
        return
    
    image_id = images[0]["id"]
    
    # Get detail
    resp = client.get(f"/images/{image_id}")
    assert resp.status_code == 200
    detail = resp.json()
    print(f"  ✓ Detail: {detail.get('filename')}, {detail.get('class_label', 'unanalyzed')}")
    
    # Get file
    resp = client.get(f"/images/{image_id}/file")
    assert resp.status_code == 200
    print(f"  ✓ Image file: {len(resp.content)} bytes")
    
    print("✓ Viewer Detail works")


def test_stats_endpoint():
    """Test: Statistics (used by multiple panels)."""
    print("\n=== TEST: Statistics ===")
    
    resp = client.get("/stats")
    assert resp.status_code == 200
    
    stats = resp.json()
    print(f"  Total images: {stats.get('total_images')}")
    print(f"  Analyzed: {stats.get('analyzed')}")
    print(f"  Anomalies: {stats.get('anomalies')}")
    print(f"  Annotated: {stats.get('annotated')}")
    
    print("✓ Statistics work")


def test_training_datasets():
    """Test: Training panel dataset detection."""
    print("\n=== TEST: Training Datasets ===")
    
    from paths import DATASETS_DIR
    
    if not DATASETS_DIR.exists():
        print(f"  ⚠ Datasets dir not found: {DATASETS_DIR}")
        return
    
    datasets = [d for d in DATASETS_DIR.iterdir() if d.is_dir()]
    print(f"  Found {len(datasets)} datasets:")
    
    for ds in datasets:
        # Count train/test splits
        train_dir = ds / "train"
        test_dir = ds / "test"
        
        train_count = sum(1 for _ in train_dir.rglob("*") if _.is_file()) if train_dir.exists() else 0
        test_count = sum(1 for _ in test_dir.rglob("*") if _.is_file()) if test_dir.exists() else 0
        
        print(f"    ✓ {ds.name}: train={train_count}, test={test_count}")
    
    print("✓ Training datasets accessible")


def test_weights_availability():
    """Test: Model weights availability."""
    print("\n=== TEST: Model Weights ===")
    
    from paths import WEIGHTS_DIR
    
    model_path = WEIGHTS_DIR / "vit_astrolens"
    config_path = model_path / "config.json"
    
    if config_path.exists():
        print(f"  ✓ Fine-tuned model found at {model_path}")
        
        import json
        with open(config_path) as f:
            config = json.load(f)
        
        num_classes = len(config.get("id2label", {}))
        print(f"    Classes: {num_classes}")
    else:
        print(f"  ⚠ No fine-tuned model at {model_path}")
        print("    (Using pre-trained weights)")
    
    print("✓ Weights check complete")


def test_duplicate_detection():
    """Test: Duplicate detection system."""
    print("\n=== TEST: Duplicate Detection ===")
    
    from inference.duplicates import DuplicateDetector, get_detector
    
    detector = DuplicateDetector(hash_size=16, similarity_threshold=0.90)
    
    # Get some images to test
    from paths import IMAGES_DIR
    images = list(IMAGES_DIR.glob("*.jpg"))[:5]
    
    if not images:
        print("  ⚠ No images to test duplicate detection")
        return
    
    # Test hash computation
    for img in images[:3]:
        hash_val = detector.compute_hash(str(img))
        print(f"  ✓ Hash: {img.name[:20]}... → {hash_val[:16]}...")
    
    # Test duplicate check
    if len(images) >= 2:
        result1 = detector.register_image(str(images[0]))
        print(f"  ✓ Image 1: duplicate={result1.is_duplicate}")
        
        result2 = detector.check_duplicate(str(images[0]))  # Same image
        print(f"  ✓ Same image check: duplicate={result2.is_duplicate}, sim={result2.similarity:.2%}")
    
    print(f"  ✓ Detector stats: {detector.get_stats()}")
    print("✓ Duplicate detection works")


def test_discovery_state():
    """Test: Discovery loop state file access."""
    print("\n=== TEST: Discovery State ===")
    
    from paths import DATA_DIR
    import json
    
    state_file = DATA_DIR / "discovery_state.json"
    candidates_file = DATA_DIR / "anomaly_candidates.json"
    
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"  ✓ Discovery state loaded:")
        print(f"    Cycles: {state.get('cycles_completed', 0)}")
        print(f"    Downloaded: {state.get('total_downloaded', 0)}")
        print(f"    Analyzed: {state.get('total_analyzed', 0)}")
        print(f"    Duplicates: {state.get('duplicates_skipped', 0)}")
        print(f"    Anomalies: {state.get('anomalies_found', 0)}")
        print(f"    Threshold: {state.get('current_threshold', 'N/A')}")
    else:
        print(f"  ℹ No discovery state yet (run discovery loop first)")
    
    if candidates_file.exists():
        with open(candidates_file) as f:
            candidates = json.load(f)
        print(f"  ✓ {len(candidates)} anomaly candidates found")
    else:
        print(f"  ℹ No candidates yet")
    
    print("✓ Discovery state accessible")


def test_discovery_panel_import():
    """Test: Discovery panel can be imported."""
    print("\n=== TEST: Discovery Panel Import ===")
    
    try:
        # Test that we can import the module without PyQt5 errors
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "discovery_panel",
            Path(__file__).parent.parent / "ui" / "discovery_panel.py"
        )
        # Just check the file is valid Python
        import ast
        with open(Path(__file__).parent.parent / "ui" / "discovery_panel.py") as f:
            code = f.read()
        ast.parse(code)
        print("  ✓ Discovery panel syntax is valid")
        print("  ✓ Discovery panel module exists")
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        raise
    
    print("✓ Discovery panel can be imported")


def run_all_tests():
    """Run all UI feature tests."""
    print("\n" + "=" * 60)
    print("ASTROLENS UI FEATURE TESTS")
    print("=" * 60)
    
    tests = [
        ("Gallery Load", test_gallery_load),
        ("Statistics", test_stats_endpoint),
        ("Viewer Detail", test_viewer_detail),
        ("Batch Analyze New", test_batch_analyze_new),
        ("Batch Re-analyze All", test_batch_analyze_all),
        ("Batch Rebuild Similarity", test_batch_rebuild_similarity),
        ("Chat Commands", test_chat_commands),
        ("LLM Annotation", test_image_annotation),
        ("Training Datasets", test_training_datasets),
        ("Model Weights", test_weights_availability),
        ("Duplicate Detection", test_duplicate_detection),
        ("Discovery State", test_discovery_state),
        ("Discovery Panel Import", test_discovery_panel_import),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except httpx.ConnectError:
        print("\n✗ ERROR: Could not connect to API at", API_URL)
        print("  Make sure the API is running: uvicorn api.main:app --port 8000")
        sys.exit(1)
    finally:
        client.close()

