"""
Quick script to test and fix the model loading issue
"""
import sys
print("Python:", sys.executable)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"PyTorch error: {e}")

try:
    import sentence_transformers
    print(f"Sentence Transformers version: {sentence_transformers.__version__}")
except Exception as e:
    print(f"Sentence Transformers error: {e}")

print("\nAttempting to load model...")
try:
    # Try loading without any device specification
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device=None)
    print("✓ Model loaded successfully with device=None")
except Exception as e:
    print(f"✗ Failed with device=None: {type(e).__name__}: {e}")

print("\nAttempting with explicit CPU device...")
try:
    import torch
    from sentence_transformers import SentenceTransformer
    # Clear CUDA cache if exists
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to('cpu')
    print("✓ Model loaded and moved to CPU successfully")

    # Test encoding
    result = model.encode("test")
    print(f"✓ Test encoding successful, shape: {result.shape}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
