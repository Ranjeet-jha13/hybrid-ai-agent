# Quick test of all main packages

print("Testing installations...\n")

# Test 1: PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch FAILED: {e}")

# Test 2: Gymnasium
try:
    import gymnasium
    print(f"✓ Gymnasium {gymnasium.__version__}")
except Exception as e:
    print(f"✗ Gymnasium FAILED: {e}")

# Test 3: Pygame
try:
    import pygame
    print(f"✓ Pygame {pygame.version.ver}")
except Exception as e:
    print(f"✗ Pygame FAILED: {e}")

# Test 4: spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"✓ spaCy {spacy.__version__} with model")
except Exception as e:
    print(f"✗ spaCy FAILED: {e}")

# Test 5: Web frameworks
try:
    import fastapi
    import streamlit
    print(f"✓ FastAPI {fastapi.__version__}")
    print(f"✓ Streamlit {streamlit.__version__}")
except Exception as e:
    print(f"✗ Web frameworks FAILED: {e}")

# Test 6: Data science
try:
    import numpy
    import pandas
    print(f"✓ NumPy {numpy.__version__}")
    print(f"✓ Pandas {pandas.__version__}")
except Exception as e:
    print(f"✗ Data packages FAILED: {e}")

print("\n✅ Installation test complete!")