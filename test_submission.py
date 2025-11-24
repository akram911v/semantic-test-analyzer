#!/usr/bin/env python3
"""
Quick test script to verify Assignment #4 submission
"""

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from lsa_analyzer import LSASemanticAnalyzer
        print("✅ LSA analyzer imports successfully")
    except ImportError as e:
        print(f"❌ LSA analyzer import failed: {e}")
    
    try:
        from word2vec_model import Word2VecModel
        print("✅ Word2Vec model imports successfully")
    except ImportError as e:
        print(f"❌ Word2Vec model import failed: {e}")
    
    try:
        from doc2vec_model import Doc2VecModel
        print("✅ Doc2Vec model imports successfully")
    except ImportError as e:
        print(f"❌ Doc2Vec model import failed: {e}")
    
    try:
        from cnn_analyzer import CNNSemanticAnalyzer
        print("✅ CNN analyzer imports successfully")
    except ImportError as e:
        print(f"❌ CNN analyzer import failed: {e}")
    
    try:
        from rnn_analyzer import RNNSemanticAnalyzer
        print("✅ RNN analyzer imports successfully")
    except ImportError as e:
        print(f"❌ RNN analyzer import failed: {e}")

def test_initialization():
    """Test if all classes can be initialized"""
    print("\nTesting initialization...")
    
    try:
        from cnn_analyzer import CNNSemanticAnalyzer
        cnn = CNNSemanticAnalyzer()
        print("✅ CNN analyzer initializes successfully")
    except Exception as e:
        print(f"❌ CNN analyzer initialization failed: {e}")
    
    try:
        from rnn_analyzer import RNNSemanticAnalyzer
        rnn = RNNSemanticAnalyzer()
        print("✅ RNN analyzer initializes successfully")
    except Exception as e:
        print(f"❌ RNN analyzer initialization failed: {e}")

def test_requirements():
    """Check if requirements.txt is properly set up"""
    print("\nChecking requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            if 'tensorflow' in requirements or 'keras' in requirements:
                print("✅ TensorFlow/Keras found in requirements")
            else:
                print("❌ TensorFlow/Keras missing from requirements")
                
            if 'scikit-learn' in requirements or 'sklearn' in requirements:
                print("✅ scikit-learn found in requirements")
            else:
                print("❌ scikit-learn missing from requirements")
    except FileNotFoundError:
        print("❌ requirements.txt file not found")

def test_main_file():
    """Check if main.py exists and runs"""
    print("\nChecking main.py...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
            if 'CNNSemanticAnalyzer' in content and 'RNNSemanticAnalyzer' in content:
                print("✅ main.py includes CNN and RNN analyzers")
            else:
                print("❌ main.py missing CNN or RNN references")
    except FileNotFoundError:
        print("❌ main.py file not found")

if __name__ == "__main__":
    print("Running Assignment #4 Submission Test...")
    print("=" * 50)
    
    test_imports()
    test_initialization()
    test_requirements()
    test_main_file()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("If you see any ❌ marks, fix those issues before submission!")
