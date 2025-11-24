from lsa_analyzer import LSAAnalyzer
from word2vec_model import Word2VecModel
from doc2vec_model import Doc2VecModel
from cnn_analyzer import CNNSemanticAnalyzer
from rnn_analyzer import RNNSemanticAnalyzer

def main():
    print("=== Semantic Analyzers Comparison ===")
    
    # Sample documents for testing
    doc1 = "Machine learning is a subset of artificial intelligence"
    doc2 = "AI and machine learning are transforming technology"
    doc3 = "The weather today is sunny and warm"
    
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Document 3: {doc3}")
    print()
    
    # Initialize all analyzers
    print("Initializing analyzers...")
    
    # LSA Analyzer
    print("\n1. LSA Analyzer:")
    lsa = LSAAnalyzer()
    # Add your LSA comparison code here
    
    # Word2Vec Analyzer  
    print("\n2. Word2Vec Analyzer:")
    w2v = Word2VecModel()
    # Add your Word2Vec comparison code here
    
    # Doc2Vec Analyzer
    print("\n3. Doc2Vec Analyzer:")
    d2v = Doc2VecModel()
    # Add your Doc2Vec comparison code here
    
    # CNN Analyzer
    print("\n4. CNN Semantic Analyzer:")
    cnn = CNNSemanticAnalyzer()
    print("CNN model architecture ready")
    # Note: CNN needs training data for meaningful results
    
    # RNN Analyzer
    print("\n5. RNN Semantic Analyzer:")
    rnn = RNNSemanticAnalyzer() 
    print("RNN model architecture ready")
    # Note: RNN needs training data for meaningful results
    
    print("\n=== All 5 semantic analyzers are ready ===")
    print("LSA, Word2Vec, Doc2Vec, CNN, and RNN analyzers initialized successfully!")

if __name__ == "__main__":
    main()
