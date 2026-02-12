
import sys
import os
import shutil
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from embedding_storage import EmbeddingStore

def test_time_filter():
    print("=" * 60)
    print("TEST: Time Filtering Logic")
    print("=" * 60)

    # Setup
    temp_dir = os.path.join(os.path.dirname(__file__), "test_time_filter_db")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    store = EmbeddingStore(storage_dir=temp_dir)

    # 1. Create Mock Embeddings
    # We don't need a real model, just random vectors
    vec_old = np.random.rand(384)
    vec_new = np.random.rand(384)

    # 2. Add OLD Memory (2020)
    print("\n[Action] Adding OLD memory (2020-01-01)...")
    id_old = store.add(
        embedding=vec_old,
        text="I am an old memory from the past.",
        metadata={"timestamp": "2020-01-01T12:00:00"} # Manual override
    )

    # 3. Add NEW Memory (2026)
    print("[Action] Adding NEW memory (2026-02-12)...")
    id_new = store.add(
        embedding=vec_new,
        text="I am a fresh memory from today.",
        metadata={"timestamp": "2026-02-12T12:00:00"}
    )
    
    # Check count
    print(f"Total memories: {store.count()}")

    # 4. Search with Filter (After 2025)
    print("\n[Test 1] Searching for memories AFTER 2025...")
    start_time = datetime(2025, 1, 1)
    
    # We query with the NEW vector to ensure high similarity isn't the issue
    # We expect ONLY the new one to return
    results = store.search(vec_new, top_k=5, filter={'start_time': start_time})
    
    print(f"Found {len(results)} results.")
    
    passed = True
    if len(results) != 1:
        print("❌ FAILED: Expected exactly 1 result.")
        passed = False
    elif results[0]['id'] != id_new:
        print(f"❌ FAILED: Expected ID {id_new}, got {results[0]['id']}")
        passed = False
    else:
        print("✅ PASSED: Only the new memory was returned.")

    # 5. Search without Filter
    print("\n[Test 2] Searching WITHOUT filter...")
    results_all = store.search(vec_new, top_k=5)
    print(f"Found {len(results_all)} results.")
    
    if len(results_all) == 2:
        print("✅ PASSED: Both memories returned.")
    else:
        print("❌ FAILED: Expected 2 results.")
        passed = False

    # Cleanup
    # shutil.rmtree(temp_dir)
    
    if passed:
        print("\n🎉 TIME FILTER LOGIC VERIFIED!")
    else:
        print("\n💥 TESTS FAILED.")

if __name__ == "__main__":
    test_time_filter()
