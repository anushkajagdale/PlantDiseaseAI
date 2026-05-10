"""
Test script to verify Plant Disease AI setup
Run this after installing all dependencies to ensure everything works correctly
"""

import os
import sys
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ ERROR: Python 3.9+ required (preferably 3.11)")
        return False
    return True

def check_required_packages():
    """Verify all required packages are installed"""
    required_packages = {
        'tensorflow': 'TensorFlow',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
    }
    
    print("\n📦 Checking required packages:")
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_model_file():
    """Check if model file exists and is readable"""
    model_path = 'model/plant_disease_model.keras'
    print(f"\n🤖 Checking model file:")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return False
    
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  ✓ Model found: {model_path}")
    print(f"  ✓ Size: {size_mb:.2f} MB")
    return True

def check_dataset_structure():
    """Check if dataset directories exist with correct structure"""
    print(f"\n📂 Checking dataset structure:")
    
    required_dirs = [
        'dataset/train/Healthy',
        'dataset/train/Diseased',
        'dataset/valid/Healthy',
        'dataset/valid/Diseased',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            count = len(os.listdir(dir_path))
            print(f"  ✓ {dir_path}: {count} files")
        else:
            print(f"  ⚠ {dir_path}: NOT FOUND (optional for inference)")
            all_exist = False
    
    return True  # Dataset not required for inference only

def test_model_loading():
    """Test if model can be loaded and used for inference"""
    print(f"\n🧪 Testing model loading and inference:")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        model_path = 'model/plant_disease_model.keras'
        model = tf.keras.models.load_model(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Input shape: {model.input_shape}")
        print(f"  ✓ Output shape: {model.output_shape}")
        
        # Test with dummy data
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        print(f"  ✓ Test inference successful")
        print(f"  ✓ Predicted class index: {class_index}")
        print(f"  ✓ Confidence: {confidence:.4f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading error: {str(e)}")
        return False

def test_utils_predict():
    """Test if the predict_image utility function works"""
    print(f"\n📸 Testing prediction utility function:")
    
    try:
        from utils import predict_image, CLASS_NAMES
        print(f"  ✓ Utils imported successfully")
        print(f"  ✓ Classes detected: {CLASS_NAMES}")
        
        # Note: We can't test actual prediction without an image file
        # But we verified the module loads correctly
        return True
        
    except Exception as e:
        print(f"  ❌ Utils import error: {str(e)}")
        return False

def test_streamlit():
    """Test if Streamlit is properly installed"""
    print(f"\n🌐 Testing Streamlit installation:")
    
    try:
        import streamlit as st
        version = st.__version__
        print(f"  ✓ Streamlit version: {version}")
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit error: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("🌿 Plant Disease AI - System Check")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Model File", check_model_file),
        ("Dataset Structure", check_dataset_structure),
        ("Model Loading", test_model_loading),
        ("Utils Module", test_utils_predict),
        ("Streamlit", test_streamlit),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Unexpected error: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("✅ SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready.")
        print("   Run: streamlit run app.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
